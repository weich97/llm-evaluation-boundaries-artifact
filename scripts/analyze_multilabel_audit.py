"""Strict integrity gate and analysis for the paper-4 multi-label benchmark."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
_runner_spec = importlib.util.spec_from_file_location(
    "run_multilabel_audit_eval", ROOT / "scripts" / "run_multilabel_audit_eval.py"
)
assert _runner_spec and _runner_spec.loader
audit_runner = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(audit_runner)
_generator_path = ROOT / "scripts" / "generate_multilabel_audit_tasks.py"
_generator_spec = importlib.util.spec_from_file_location(
    "generate_multilabel_audit_tasks_for_analysis", _generator_path
)
assert _generator_spec and _generator_spec.loader
multilabel_generator = importlib.util.module_from_spec(_generator_spec)
_generator_spec.loader.exec_module(multilabel_generator)

SCHEMA_VERSION = "finaudit.multilabel.v1"
RESULT_SCHEMA_VERSION = "finaudit.findings.v1"
RESPONSE_LEDGER_SCHEMA_VERSION = "finaudit.response-ledger.v1"
RESULT_FIELDS = {
    "schema_version",
    "model",
    "task_id",
    "sample",
    "parse_ok",
    "parse_error",
    "finding_count",
    "findings",
    "response_sha256",
    "prompt_hash",
    "decoding_hash",
    "request_hash",
    "cache_key",
}
FINDING_FIELDS = {"step_index", "kind"}
CONDITIONS = ("zero", "violation_only", "violation_plus_edit")
TRADING_PRODUCERS = ("deepseek_v4_pro", "glm_5_direct")
FROZEN_MODELS = ["deepseek:deepseek-v4-pro", "glm:glm-5"]
FROZEN_PRIMARY_CELLS = [
    "deepseek:deepseek-v4-pro|trading:deepseek_v4_pro",
    "deepseek:deepseek-v4-pro|trading:glm_5_direct",
    "deepseek:deepseek-v4-pro|tooluse",
    "glm:glm-5|trading:deepseek_v4_pro",
    "glm:glm-5|trading:glm_5_direct",
    "glm:glm-5|tooluse",
]
FROZEN_SECONDARY_OUTCOMES = [
    "zero exact-set accuracy and any-finding false-positive rate",
    "single and dual violation recall",
    "dual edit and joint-both recall",
    "dual both/violation-only/edit-only/neither patterns",
    "condition exact-set and cardinality accuracy",
    "overall micro precision, recall, and F1",
    "parse failures, invalid kinds, and duplicate findings",
]
FROZEN_PROMPT_SHA256 = {
    "trading": "9cb89c85a77342c4a86954742743673547c3a5261f4621dd9284259b24c3bef3",
    "tooluse": "c5253944e08eab9d42009ecf6a11ad7890d272aa6fb72b369a5ac8b715d4209e",
}
TOOL_CAPS = {
    "wire_transfer": ("amount_usd", 10000),
    "provision_vm": ("instance_count", 20),
    "send_batch": ("recipient_count", 5000),
    "grant_quota": ("quota_gb", 500),
}
DOMAIN_LABELS = {
    "trading": ("unclipped_position", "silent_risk_edit"),
    "tooluse": ("over_limit_call", "silent_arg_edit"),
}
ALLOWED_KINDS = {
    "trading": {"unclipped_position", "silent_risk_edit", "provenance_drift", "tampered_fill_price"},
    "tooluse": {"over_limit_call", "silent_arg_edit", "version_drift", "tampered_result"},
}
OPAQUE_ID = re.compile(r"^ml_[0-9a-f]{16}$")
EPS = 1e-9
PARSING_RULE = (
    "Accept a bare JSON array or a complete json code fence. Malformed top-level output yields no findings. "
    "Within a parsed array, retain structurally valid findings for label TP/FP even if another item makes "
    "parse_ok false; exact-set and cardinality accuracy require parse_ok."
)


class IntegrityError(ValueError):
    """Raised when a benchmark or result file is not an exact expected set."""


class IncompleteGridError(IntegrityError):
    """Raised only when valid checkpoint rows have not yet filled the frozen grid."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text_normalized(path: Path) -> str:
    """Hash a text asset with line endings normalized to LF."""

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def text_hash_matches(path: Path, recorded: object) -> bool:
    """Compare a text asset against a recorded hash, ignoring line-ending convention.

    A raw byte hash of a tracked text file is not portable: a checkout under
    ``core.autocrlf`` rewrites LF to CRLF, so the same content hashes differently
    on Windows than on the machine that wrote the manifest. Accepting either the
    raw or the LF-normalized digest keeps the gate reproducible on a reviewer's
    machine while still failing on any real content change.
    """

    if not isinstance(recorded, str):
        return False
    return recorded in {sha256_file(path), sha256_text_normalized(path)}


def opaque_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:16]}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"Malformed JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise IntegrityError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def domain_cell(row: dict[str, Any]) -> str:
    if row["domain"] == "tooluse":
        return "tooluse"
    return f"trading:{row['producer']}"


def _trading_oracle(trajectory: dict[str, Any]) -> set[tuple[int, str]]:
    found: set[tuple[int, str]] = set()
    steps = trajectory.get("steps", [])
    pinned_version = None
    if steps:
        pinned_version = (steps[0].get("reproducibility_state") or {}).get("model_version")
    for step_index, step in enumerate(steps):
        intended = {
            str(item.get("symbol")): item
            for item in step.get("decisions", [])
            if item.get("target_weight") is not None
        }
        for approved in step.get("approved_decisions", []):
            if approved.get("target_weight") is None:
                continue
            symbol = str(approved.get("symbol"))
            approved_weight = float(approved["target_weight"])
            has_evidence = "risk_clipped_from" in (approved.get("metadata") or {})
            if abs(approved_weight) > 0.35 + EPS and not has_evidence:
                found.add((step_index, "unclipped_position"))
            source = intended.get(symbol)
            if source is not None:
                intended_weight = float(source["target_weight"])
                if abs(approved_weight - intended_weight) > EPS and not has_evidence:
                    found.add((step_index, "silent_risk_edit"))
        if (step.get("reproducibility_state") or {}).get("model_version") != pinned_version:
            found.add((step_index, "provenance_drift"))
        prices = (step.get("observation") or {}).get("prices") or {}
        for fill in step.get("fills", []):
            symbol = str(fill.get("symbol", ""))
            if symbol not in prices:
                continue
            try:
                residual = float(fill["price"]) - float(prices[symbol]) - float(fill["slippage"])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(residual) > 1e-7:
                found.add((step_index, "tampered_fill_price"))
    return found


def _tool_oracle(trajectory: dict[str, Any]) -> set[tuple[int, str]]:
    found: set[tuple[int, str]] = set()
    expected_caps = {tool: {"arg": argument, "cap": cap} for tool, (argument, cap) in TOOL_CAPS.items()}
    if trajectory.get("caps") != expected_caps:
        raise IntegrityError("Tool trajectory cap table differs from the frozen public constraints")
    for step_index, step in enumerate(trajectory.get("steps", [])):
        tool = str(step.get("tool", ""))
        if tool not in TOOL_CAPS:
            raise IntegrityError(f"Unknown tool in trajectory: {tool!r}")
        argument, cap = TOOL_CAPS[tool]
        requested = float(step["requested_args"][argument])
        approved = float(step["approved_args"][argument])
        has_evidence = "clipped_from" in (step.get("approval_meta") or {})
        if approved > cap and not has_evidence:
            found.add((step_index, "over_limit_call"))
        if abs(approved - requested) > EPS and not has_evidence:
            found.add((step_index, "silent_arg_edit"))
        if step.get("tool_version") != trajectory.get("tool_version"):
            found.add((step_index, "version_drift"))
        result = step.get("result") or {}
        if result.get("total") != sum(result.get("line_items") or []):
            found.add((step_index, "tampered_result"))
    return found


def _one_symbol_record(step: dict[str, Any], key: str, symbol: str) -> dict[str, Any]:
    records = [record for record in step.get(key, []) if record.get("symbol") == symbol]
    if len(records) != 1:
        raise IntegrityError(f"Expected one {key} record for {symbol}; found {len(records)}")
    return records[0]


def _validate_full_source_provenance(
    manifest: dict[str, Any],
    triples: dict[str, list[dict[str, Any]]],
    trajectory_by_task: dict[str, dict[str, Any]],
    trading_source_root: Path,
) -> None:
    """Rebuild every frozen triplet from its independently stored source."""

    if not text_hash_matches(_generator_path, manifest.get("generator_sha256")):
        raise IntegrityError("Generator hash does not match the code used by the integrity gate")
    recorded_source_hashes = manifest.get("source_answer_key_sha256")
    if not isinstance(recorded_source_hashes, dict) or set(recorded_source_hashes) != set(
        TRADING_PRODUCERS
    ):
        raise IntegrityError("Source answer-key hash map does not name the two frozen producers")

    selected_sources: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for producer in TRADING_PRODUCERS:
        source_dir = trading_source_root / producer
        source_truth_path = source_dir / "ground_truth.jsonl"
        if not source_truth_path.is_file():
            raise IntegrityError(f"Missing frozen trading source answer key: {source_truth_path}")
        if not text_hash_matches(source_truth_path, recorded_source_hashes.get(producer)):
            raise IntegrityError(f"Trading source answer-key hash mismatch for {producer}")
        source_rows = [
            row for row in _read_jsonl(source_truth_path) if row.get("kind") == "unclipped_position"
        ]
        source_rows.sort(key=lambda row: (int(row.get("source_seed", -1)), str(row.get("task_id", ""))))
        if len(source_rows) < 30:
            raise IntegrityError(f"Trading source {producer} has only {len(source_rows)} eligible rows")
        selected = source_rows[:30]
        by_source: dict[tuple[str, int], dict[str, Any]] = {}
        for source in selected:
            key = (str(source.get("task_id", "")), int(source.get("source_seed", -1)))
            if not key[0] or key in by_source:
                raise IntegrityError(f"Duplicate or empty frozen source key for {producer}: {key}")
            by_source[key] = source
        selected_sources[producer] = by_source

    actual_trading: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    actual_tool: dict[str, list[dict[str, Any]]] = {}
    for triple_id, rows in triples.items():
        first = rows[0]
        if first["domain"] == "trading":
            actual_trading[str(first["producer"])].append((triple_id, rows))
        else:
            actual_tool[triple_id] = rows

    for producer in TRADING_PRODUCERS:
        expected_keys = set(selected_sources[producer])
        observed_keys = {
            (str(rows[0]["source_task_id"]), int(rows[0]["source_seed"]))
            for _, rows in actual_trading[producer]
        }
        if len(actual_trading[producer]) != 30 or observed_keys != expected_keys:
            raise IntegrityError(
                f"Trading source selection differs from the frozen first-30 rule for {producer}"
            )
        for triple_id, rows in actual_trading[producer]:
            first = rows[0]
            source_key = (str(first["source_task_id"]), int(first["source_seed"]))
            source = selected_sources[producer][source_key]
            if int(first["target_step_index"]) != int(source.get("step_index", -1)):
                raise IntegrityError(f"Trading target step is not source-derived for {triple_id}")
            source_trajectory_path = (
                trading_source_root
                / producer
                / "tasks"
                / source_key[0]
                / "trajectory.json"
            )
            if not source_trajectory_path.is_file():
                raise IntegrityError(f"Missing frozen source trajectory: {source_trajectory_path}")
            try:
                source_trajectory = json.loads(source_trajectory_path.read_text(encoding="utf-8"))
                variants = multilabel_generator.build_trading_triplet(source_trajectory, source)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise IntegrityError(f"Could not rebuild trading source for {triple_id}: {exc}") from exc
            for row in rows:
                candidate, defects = variants[str(row["condition"])]
                if trajectory_by_task[str(row["task_id"])] != candidate or row["defects"] != defects:
                    raise IntegrityError(f"Trading task does not exactly rebuild from its source: {row['task_id']}")

    expected_tool: dict[
        str,
        tuple[int, int, dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]],
    ] = {}
    seed = int((manifest.get("protocol") or {}).get("tool_base_seed", -1))
    while len(expected_tool) < 40:
        trajectory = multilabel_generator.generate_trajectory(seed)
        try:
            target_step = multilabel_generator._tool_target_step(trajectory, seed)
        except ValueError:
            seed += 1
            continue
        triple_id = opaque_id("tr", SCHEMA_VERSION, "tooluse", seed)
        expected_tool[triple_id] = (
            seed,
            target_step,
            multilabel_generator.build_tool_triplet(trajectory, target_step),
        )
        seed += 1
    if set(actual_tool) != set(expected_tool):
        raise IntegrityError("Tool-use source seeds differ from the frozen deterministic generation stream")
    for triple_id, rows in actual_tool.items():
        source_seed, target_step, variants = expected_tool[triple_id]
        first = rows[0]
        if (
            int(first["source_seed"]) != source_seed
            or first["source_task_id"] != f"tool_source_{source_seed}"
            or int(first["target_step_index"]) != target_step
        ):
            raise IntegrityError(f"Tool-use provenance does not match regeneration for {triple_id}")
        for row in rows:
            candidate, defects = variants[str(row["condition"])]
            if trajectory_by_task[str(row["task_id"])] != candidate or row["defects"] != defects:
                raise IntegrityError(f"Tool-use task does not exactly regenerate: {row['task_id']}")


def load_dataset(
    tasks_root: Path,
    *,
    require_full_scale: bool = True,
    trading_source_root: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate task dirs, manifest, hashes, private truth, and triplet structure."""

    manifest_path = tasks_root / "manifest.json"
    truth_path = tasks_root / "ground_truth.jsonl"
    if not manifest_path.is_file() or not truth_path.is_file():
        raise IntegrityError(f"Missing manifest or answer key under {tasks_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise IntegrityError(f"Unexpected manifest schema: {manifest.get('schema_version')!r}")
    analysis_plan_path = tasks_root / "analysis_plan.json"
    if not analysis_plan_path.is_file():
        raise IntegrityError("Missing frozen analysis_plan.json")
    if not text_hash_matches(analysis_plan_path, manifest.get("analysis_plan_sha256")):
        raise IntegrityError("Frozen analysis-plan hash does not match the manifest")
    analysis_plan = json.loads(analysis_plan_path.read_text(encoding="utf-8"))
    if analysis_plan.get("schema_version") != "finaudit.multilabel.plan.v1":
        raise IntegrityError("Unexpected analysis-plan schema")
    expected_plan_fields = {
        "schema_version",
        "auditors",
        "primary_cells",
        "primary_estimand",
        "primary_test",
        "multiplicity",
        "interval",
        "temperature",
        "samples_per_task",
        "parsing_rule",
        "stop_rule",
        "secondary_outcomes",
    }
    if require_full_scale and set(analysis_plan) != expected_plan_fields:
        raise IntegrityError(f"Analysis plan fields differ from the frozen schema: {sorted(analysis_plan)}")
    entries = manifest.get("tasks")
    if not isinstance(entries, list) or not entries:
        raise IntegrityError("Manifest task list is empty or malformed")

    manifest_by_task: dict[str, dict[str, Any]] = {}
    trajectory_by_task: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise IntegrityError("Manifest contains a non-object task entry")
        allowed_entry_fields = {
            "task_id",
            "domain",
            "producer",
            "trajectory_sha256",
            "prompt_sha256",
        }
        if set(entry) != allowed_entry_fields:
            raise IntegrityError(f"Manifest task entry has unexpected or missing fields: {sorted(set(entry))}")
        task_id = str(entry.get("task_id", ""))
        if not OPAQUE_ID.fullmatch(task_id):
            raise IntegrityError(f"Task id is not opaque: {task_id!r}")
        if task_id in manifest_by_task:
            raise IntegrityError(f"Duplicate manifest task id: {task_id}")
        domain = str(entry.get("domain", ""))
        if domain not in DOMAIN_LABELS:
            raise IntegrityError(f"Unknown domain for {task_id}: {domain!r}")
        producer = str(entry.get("producer", ""))
        if domain == "trading" and producer not in TRADING_PRODUCERS:
            raise IntegrityError(f"Unknown trading producer for {task_id}: {producer!r}")
        if domain == "tooluse" and producer != "rule_based_tool":
            raise IntegrityError(f"Unexpected tool-use producer for {task_id}: {producer!r}")
        task_dir = tasks_root / "tasks" / task_id
        trajectory_path = task_dir / "trajectory.json"
        prompt_path = task_dir / "prompt.md"
        if not trajectory_path.is_file() or not prompt_path.is_file():
            raise IntegrityError(f"Missing trajectory or prompt for {task_id}")
        if not text_hash_matches(trajectory_path, entry.get("trajectory_sha256")):
            raise IntegrityError(f"Trajectory hash mismatch for {task_id}")
        if not text_hash_matches(prompt_path, entry.get("prompt_sha256")):
            raise IntegrityError(f"Prompt hash mismatch for {task_id}")
        if entry.get("prompt_sha256") != FROZEN_PROMPT_SHA256[domain]:
            raise IntegrityError(f"Prompt for {task_id} differs from the frozen domain prompt")
        try:
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IntegrityError(f"Malformed trajectory JSON for {task_id}") from exc
        if not isinstance(trajectory, dict):
            raise IntegrityError(f"Trajectory is not an object for {task_id}")
        prompt = prompt_path.read_text(encoding="utf-8").lower()
        normalized_prompt = " ".join(prompt.split())
        if "exactly one" in normalized_prompt or "single injected defect" in normalized_prompt:
            raise IntegrityError(f"Cardinality leak in prompt for {task_id}")
        if "zero, one, or multiple" not in normalized_prompt or "multiple defects may coexist" not in normalized_prompt:
            raise IntegrityError(f"Unknown-cardinality instruction missing for {task_id}")
        manifest_by_task[task_id] = entry
        trajectory_by_task[task_id] = trajectory

    actual_dirs = {path.name for path in (tasks_root / "tasks").iterdir() if path.is_dir()}
    if actual_dirs != set(manifest_by_task):
        missing = sorted(set(manifest_by_task) - actual_dirs)
        extra = sorted(actual_dirs - set(manifest_by_task))
        raise IntegrityError(f"Task directory set mismatch: missing={missing[:3]}, extra={extra[:3]}")
    if int(manifest.get("task_count", -1)) != len(manifest_by_task):
        raise IntegrityError("Manifest task_count does not equal its task entries")

    truth_by_task: dict[str, dict[str, Any]] = {}
    triples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(truth_path):
        expected_truth_fields = {
            "schema_version",
            "task_id",
            "triple_id",
            "domain",
            "producer",
            "source_task_id",
            "source_seed",
            "condition",
            "target_step_index",
            "defects",
        }
        if set(row) != expected_truth_fields:
            raise IntegrityError(f"Answer-key row fields differ from the frozen schema: {sorted(row)}")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise IntegrityError(f"Unexpected answer-key schema for {row.get('task_id')}")
        task_id = str(row.get("task_id", ""))
        if task_id in truth_by_task:
            raise IntegrityError(f"Duplicate answer-key task id: {task_id}")
        if task_id not in manifest_by_task:
            raise IntegrityError(f"Answer key has unknown task id: {task_id}")
        manifest_entry = manifest_by_task[task_id]
        if row.get("domain") != manifest_entry.get("domain") or row.get("producer") != manifest_entry.get("producer"):
            raise IntegrityError(f"Manifest/answer-key domain mismatch for {task_id}")
        condition = str(row.get("condition", ""))
        if condition not in CONDITIONS:
            raise IntegrityError(f"Invalid condition for {task_id}: {condition!r}")
        target_step = row.get("target_step_index")
        source_seed = row.get("source_seed")
        source_task_id = row.get("source_task_id")
        if isinstance(target_step, bool) or not isinstance(target_step, int) or target_step < 0:
            raise IntegrityError(f"Invalid target_step_index for {task_id}")
        if isinstance(source_seed, bool) or not isinstance(source_seed, int):
            raise IntegrityError(f"Invalid source_seed for {task_id}")
        if not isinstance(source_task_id, str) or not source_task_id:
            raise IntegrityError(f"Invalid source_task_id for {task_id}")
        if re.fullmatch(r"tr_[0-9a-f]{16}", str(row.get("triple_id", ""))) is None:
            raise IntegrityError(f"Invalid opaque triple_id for {task_id}")
        defects = row.get("defects")
        if not isinstance(defects, list):
            raise IntegrityError(f"Defects are not a list for {task_id}")
        expected_count = CONDITIONS.index(condition)
        if len(defects) != expected_count:
            raise IntegrityError(
                f"Condition/cardinality mismatch for {task_id}: {condition} has {len(defects)} defects"
            )
        domain = str(row["domain"])
        violation_kind, edit_kind = DOMAIN_LABELS[domain]
        expected_kinds = {
            "zero": set(),
            "violation_only": {violation_kind},
            "violation_plus_edit": {violation_kind, edit_kind},
        }[condition]
        observed_pairs: set[tuple[int, str]] = set()
        for defect in defects:
            if not isinstance(defect, dict):
                raise IntegrityError(f"Non-object defect for {task_id}")
            if set(defect) != {"kind", "difficulty", "step_index", "detail"}:
                raise IntegrityError(f"Defect fields differ from the frozen schema for {task_id}")
            if not isinstance(defect.get("detail"), dict):
                raise IntegrityError(f"Defect detail is not an object for {task_id}")
            kind = str(defect.get("kind", ""))
            step_index = defect.get("step_index")
            if kind not in ALLOWED_KINDS[domain]:
                raise IntegrityError(f"Invalid defect kind for {task_id}: {kind!r}")
            if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
                raise IntegrityError(f"Invalid defect step for {task_id}")
            pair = (step_index, kind)
            if pair in observed_pairs:
                raise IntegrityError(f"Duplicate ground-truth pair for {task_id}: {pair}")
            observed_pairs.add(pair)
        if {kind for _, kind in observed_pairs} != expected_kinds:
            raise IntegrityError(f"Wrong defect kinds for {task_id}: {observed_pairs}")
        if observed_pairs and {step for step, _ in observed_pairs} != {int(row["target_step_index"])}:
            raise IntegrityError(f"Defects do not share the target step for {task_id}")
        oracle = _trading_oracle if domain == "trading" else _tool_oracle
        oracle_pairs = oracle(trajectory_by_task[task_id])
        if oracle_pairs != observed_pairs:
            raise IntegrityError(
                f"Rule-oracle mismatch for {task_id}: expected={observed_pairs}, observed={oracle_pairs}"
            )
        truth_by_task[task_id] = row
        triples[str(row.get("triple_id", ""))].append(row)

    if set(truth_by_task) != set(manifest_by_task):
        missing = sorted(set(manifest_by_task) - set(truth_by_task))
        raise IntegrityError(f"Answer key is missing manifest tasks: {missing[:3]}")
    for triple_id, rows in triples.items():
        if not triple_id or len(rows) != 3:
            raise IntegrityError(f"Triple {triple_id!r} has {len(rows)} tasks")
        if {str(row["condition"]) for row in rows} != set(CONDITIONS):
            raise IntegrityError(f"Triple {triple_id} does not contain all three conditions")
        shared = {
            (row["domain"], row["producer"], row["source_task_id"], int(row["source_seed"])) for row in rows
        }
        if len(shared) != 1:
            raise IntegrityError(f"Triple {triple_id} mixes source provenance")
        domain, producer, source_task_id, source_seed = next(iter(shared))
        if domain == "trading":
            expected_triple_id = opaque_id(
                "tr", SCHEMA_VERSION, "trading", producer, source_seed, source_task_id
            )
        else:
            if source_task_id != f"tool_source_{source_seed}":
                raise IntegrityError(f"Tool triple {triple_id} has a non-deterministic source id")
            expected_triple_id = opaque_id("tr", SCHEMA_VERSION, "tooluse", source_seed)
        if triple_id != expected_triple_id:
            raise IntegrityError(
                f"Triple id does not match its frozen provenance: {triple_id} != {expected_triple_id}"
            )
        for row in rows:
            expected_task_id = opaque_id("ml", expected_triple_id, row["condition"])
            if row["task_id"] != expected_task_id:
                raise IntegrityError(
                    f"Task id does not match its frozen triple/condition: {row['task_id']} != {expected_task_id}"
                )
        if len({int(row["target_step_index"]) for row in rows}) != 1:
            raise IntegrityError(f"Triple {triple_id} does not share one target step")
        if len({manifest_by_task[str(row["task_id"])]["prompt_sha256"] for row in rows}) != 1:
            raise IntegrityError(f"Triple {triple_id} does not use one cardinality-blind prompt")
        by_condition = {str(row["condition"]): row for row in rows}
        single_row = by_condition["violation_only"]
        dual_row = by_condition["violation_plus_edit"]
        zero_row = by_condition["zero"]
        single = copy.deepcopy(trajectory_by_task[str(single_row["task_id"])])
        dual = trajectory_by_task[str(dual_row["task_id"])]
        zero = trajectory_by_task[str(zero_row["task_id"])]
        target_step = int(single_row["target_step_index"])
        if single_row["domain"] == "trading":
            symbol = str(single_row["defects"][0]["detail"]["symbol"])
            single_record = _one_symbol_record(single["steps"][target_step], "decisions", symbol)
            dual_record = _one_symbol_record(dual["steps"][target_step], "decisions", symbol)
            single_approved = _one_symbol_record(single["steps"][target_step], "approved_decisions", symbol)
            dual_approved = _one_symbol_record(dual["steps"][target_step], "approved_decisions", symbol)
            high = float(single_approved["target_weight"])
            expected_side = "buy" if high > 0 else "sell"
            expected_low = math.copysign(0.28, high)
            if (
                abs(abs(high) - 0.56) > EPS
                or float(single_record["target_weight"]) != high
                or float(dual_approved["target_weight"]) != high
                or abs(float(dual_record["target_weight"]) - expected_low) > EPS
                or single_record.get("side") != expected_side
                or dual_record.get("side") != expected_side
                or single_approved.get("side") != expected_side
                or dual_approved.get("side") != expected_side
                or min(float(record.get("confidence", 0.0)) for record in (
                    single_record,
                    dual_record,
                    single_approved,
                    dual_approved,
                ))
                < 0.4
            ):
                raise IntegrityError(f"Trading triple {triple_id} differs from the frozen 0.28/0.56 protocol")
            for record in (single_record, dual_record, single_approved, dual_approved):
                metadata = record.get("metadata") or {}
                if "risk_clipped_from" in metadata or "risk_blocked" in metadata:
                    raise IntegrityError(f"Trading pair {triple_id} retains target clip/block metadata")
            for candidate in (single, dual):
                if any(
                    violation.get("symbol") == symbol and violation.get("constraint") == "min_confidence"
                    for violation in candidate["steps"][target_step].get("risk_violations", [])
                ):
                    raise IntegrityError(f"Trading pair {triple_id} retains target min-confidence record")
            single_record["target_weight"] = dual_record["target_weight"]
        else:
            argument = str(single_row["defects"][0]["detail"]["argument"])
            tool = str(single["steps"][target_step]["tool"])
            frozen_argument, cap = TOOL_CAPS[tool]
            if argument != frozen_argument:
                raise IntegrityError(f"Tool pair {triple_id} targets the wrong governed argument")
            zero_step = zero["steps"][target_step]
            single_step = single["steps"][target_step]
            dual_step = dual["steps"][target_step]
            high = round(cap * 1.6)
            original = zero_step["requested_args"][argument]
            if (
                zero_step["approved_args"][argument] != original
                or original < 0
                or original > cap
                or single_step["requested_args"][argument] != high
                or single_step["approved_args"][argument] != high
                or dual_step["requested_args"][argument] != original
                or dual_step["approved_args"][argument] != high
            ):
                raise IntegrityError(f"Tool triple {triple_id} differs from the frozen original/1.6x protocol")
            single["steps"][target_step]["requested_args"][argument] = dual["steps"][target_step][
                "requested_args"
            ][argument]
        if single != dual:
            raise IntegrityError(f"Single/dual pair {triple_id} differs outside the intended request field")
    if require_full_scale:
        expected_manifest_fields = {
            "schema_version",
            "task_count",
            "trading_triples_per_producer",
            "tool_triples",
            "protocol",
            "generator_sha256",
            "source_answer_key_sha256",
            "analysis_plan_sha256",
            "task_id_note",
            "answer_key_note",
            "tasks",
        }
        if set(manifest) != expected_manifest_fields:
            raise IntegrityError(f"Manifest fields differ from the frozen public schema: {sorted(manifest)}")
        if manifest.get("trading_triples_per_producer") != 30 or manifest.get("tool_triples") != 40:
            raise IntegrityError("Manifest does not freeze the registered 30/30/40 triplet design")
        if len(manifest_by_task) != 300 or len(triples) != 100:
            raise IntegrityError(
                f"Full benchmark must contain 300 tasks/100 triples; found {len(manifest_by_task)}/{len(triples)}"
            )
        protocol = manifest.get("protocol") or {}
        if (
            protocol.get("conditions") != list(CONDITIONS)
            or protocol.get("trading_cap") != 0.35
            or protocol.get("trading_low_factor") != 0.8
            or protocol.get("high_factor") != 1.6
            or protocol.get("min_pair_confidence") != 0.4
            or protocol.get("remove_target_min_confidence_record") is not True
            or protocol.get("tool_base_seed") != 9000
            or protocol.get("expected_result_keys") != 600
            or protocol.get("prompt_sha256_by_domain") != FROZEN_PROMPT_SHA256
        ):
            raise IntegrityError(f"Manifest protocol differs from the frozen design: {protocol}")
        if (
            analysis_plan.get("auditors") != FROZEN_MODELS
            or analysis_plan.get("primary_cells") != FROZEN_PRIMARY_CELLS
            or analysis_plan.get("primary_estimand")
            != "violation recall in violation_only minus violation_plus_edit"
            or analysis_plan.get("primary_test") != "two-sided exact McNemar within matched triples"
            or analysis_plan.get("multiplicity") != "Holm step-down across all six primary cells"
            or analysis_plan.get("interval")
            != "paired nonparametric bootstrap, 10000 draws, deterministic cell seed"
            or analysis_plan.get("temperature") != 0.0
            or analysis_plan.get("samples_per_task") != 1
            or analysis_plan.get("parsing_rule") != PARSING_RULE
            or analysis_plan.get("stop_rule")
            != "exactly 600 unique auditor/task/sample keys; no missing, duplicate, or extra keys"
            or analysis_plan.get("secondary_outcomes") != FROZEN_SECONDARY_OUTCOMES
        ):
            raise IntegrityError("Analysis plan differs from the registered two-auditor, six-cell design")
        triple_rows = [rows[0] for rows in triples.values()]
        triples_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in triple_rows:
            triples_by_cell[domain_cell(row)].append(row)
        expected_counts = {
            "trading:deepseek_v4_pro": 30,
            "trading:glm_5_direct": 30,
            "tooluse": 40,
        }
        observed_counts = {cell: len(rows) for cell, rows in triples_by_cell.items()}
        if observed_counts != expected_counts:
            raise IntegrityError(f"Triplet cell counts differ from the registered design: {observed_counts}")
        for cell, rows in triples_by_cell.items():
            seeds = [int(row["source_seed"]) for row in rows]
            if len(seeds) != len(set(seeds)):
                raise IntegrityError(f"Source seeds are duplicated within {cell}")
        deepseek_seeds = {int(row["source_seed"]) for row in triples_by_cell["trading:deepseek_v4_pro"]}
        glm_seeds = {int(row["source_seed"]) for row in triples_by_cell["trading:glm_5_direct"]}
        if deepseek_seeds != glm_seeds:
            raise IntegrityError("Trading producer cells do not share the same 30 source seeds")
        _validate_full_source_provenance(
            manifest,
            triples,
            trajectory_by_task,
            trading_source_root or ROOT / "outputs" / "audit_self",
        )
    return manifest_by_task, truth_by_task


def _load_cache_records(cache_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not cache_dir.is_dir():
        raise IntegrityError(f"Raw response cache directory is missing: {cache_dir}")
    for path in sorted(cache_dir.glob("auditml_*.jsonl")):
        for row in _read_jsonl(path):
            cache_key = row.get("cache_key")
            if not isinstance(cache_key, str) or not cache_key.startswith("auditmlv1:"):
                continue
            if cache_key in records:
                raise IntegrityError(f"Duplicate raw cache key across cache files: {cache_key}")
            records[cache_key] = row
    return records


def _load_response_ledger(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    expected_fields = {
        "schema_version",
        "model",
        "provider",
        "api_model",
        "task_id",
        "sample",
        "cache_key",
        "request_hash",
        "prompt_hash",
        "decoding_hash",
        "response_sha256",
        "parse_ok",
        "parse_error",
        "findings_sha256",
        "raw_cache_record_sha256",
        "source_cache_file",
        "source_cache_file_sha256",
    }
    for row in _read_jsonl(path):
        if set(row) != expected_fields or row.get("schema_version") != RESPONSE_LEDGER_SCHEMA_VERSION:
            raise IntegrityError("Response-ledger row differs from the frozen schema")
        sample = row.get("sample")
        if isinstance(sample, bool) or not isinstance(sample, int):
            raise IntegrityError("Response-ledger sample is not an integer")
        key = (str(row.get("model", "")), str(row.get("task_id", "")), sample)
        if key in records:
            raise IntegrityError(f"Duplicate response-ledger key: {key}")
        for field in (
            "request_hash",
            "prompt_hash",
            "decoding_hash",
            "response_sha256",
            "findings_sha256",
            "raw_cache_record_sha256",
            "source_cache_file_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", str(row.get(field, ""))) is None:
                raise IntegrityError(f"Invalid response-ledger {field} for {key}")
        if not isinstance(row.get("parse_ok"), bool) or not isinstance(row.get("parse_error"), str):
            raise IntegrityError(f"Invalid response-ledger parser fields for {key}")
        if not isinstance(row.get("source_cache_file"), str) or Path(row["source_cache_file"]).name != row[
            "source_cache_file"
        ]:
            raise IntegrityError(f"Unsafe response-ledger cache filename for {key}")
        records[key] = row
    return records


def validate_results(
    results_path: Path,
    truth_by_task: dict[str, dict[str, Any]],
    models: Iterable[str],
    *,
    tasks_root: Path | None = None,
    cache_dir: Path | None = None,
    response_ledger_path: Path | None = None,
    verify_cache: bool = True,
    require_frozen_models: bool = True,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Require the result keys to equal ``models x tasks x {sample 0}`` exactly."""

    model_list = list(models)
    if not model_list or len(model_list) != len(set(model_list)):
        raise IntegrityError("Model list is empty or duplicated")
    if require_frozen_models and model_list != FROZEN_MODELS:
        raise IntegrityError(f"Models differ from the frozen auditor list: {model_list}")
    if verify_cache and response_ledger_path is not None:
        raise IntegrityError("Choose raw-cache replay or released-ledger verification, not both")
    if verify_cache and (tasks_root is None or cache_dir is None):
        raise IntegrityError("Strict result verification requires tasks_root and raw cache_dir")
    if response_ledger_path is not None and tasks_root is None:
        raise IntegrityError("Released-ledger verification requires tasks_root")
    cache_records = _load_cache_records(cache_dir) if verify_cache and cache_dir is not None else {}
    ledger_records = (
        _load_response_ledger(response_ledger_path) if response_ledger_path is not None else {}
    )
    expected = {(model, task_id, 0) for model in model_list for task_id in truth_by_task}
    rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in _read_jsonl(results_path):
        if set(row) != RESULT_FIELDS:
            raise IntegrityError("Result row differs from the frozen public schema")
        model = str(row.get("model", ""))
        task_id = str(row.get("task_id", ""))
        sample = row.get("sample")
        if isinstance(sample, bool) or not isinstance(sample, int):
            raise IntegrityError(f"Invalid sample in result row for {(model, task_id)}")
        key = (model, task_id, sample)
        if key not in expected:
            raise IntegrityError(f"Unexpected result key: {key}")
        if key in rows_by_key:
            raise IntegrityError(f"Duplicate result key: {key}")
        if row.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise IntegrityError(f"Unexpected result schema for {key}")
        for hash_field in ("response_sha256", "request_hash", "prompt_hash", "decoding_hash"):
            value = row.get(hash_field)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise IntegrityError(f"Invalid {hash_field} for {key}")
        cache_key = row.get("cache_key")
        if not isinstance(cache_key, str) or not cache_key.startswith("auditmlv1:"):
            raise IntegrityError(f"Invalid cache_key for {key}")
        if not isinstance(row.get("parse_ok"), bool):
            raise IntegrityError(f"parse_ok is not boolean for {key}")
        findings = row.get("findings")
        if not isinstance(findings, list):
            raise IntegrityError(f"findings is not a list for {key}")
        domain = str(truth_by_task[task_id]["domain"])
        for finding in findings:
            if not isinstance(finding, dict) or set(finding) != FINDING_FIELDS:
                raise IntegrityError(f"Finding differs from the frozen public schema for {key}")
            step_index = finding.get("step_index")
            kind = finding.get("kind")
            if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
                raise IntegrityError(f"Invalid finding step for {key}")
            if not isinstance(kind, str) or not kind:
                raise IntegrityError(f"Invalid finding kind for {key}")
            if row["parse_ok"] and kind not in ALLOWED_KINDS[domain]:
                raise IntegrityError(f"parse_ok row contains an unknown kind for {key}: {kind}")
        if row.get("finding_count") != len(findings):
            raise IntegrityError(f"finding_count mismatch for {key}")
        if not isinstance(row.get("parse_error"), str):
            raise IntegrityError(f"parse_error is not a string for {key}")
        if tasks_root is not None:
            provider, api_model = model.split(":", 1)
            prompt = audit_runner.build_prompt(tasks_root / "tasks" / task_id, domain)
            fingerprint = audit_runner.request_fingerprint(provider, api_model, prompt)
            expected_cache_key = (
                f"auditmlv1:{provider}:{api_model}:{task_id}:"
                f"{fingerprint['request_hash']}:s0"
            )
            expected_fields = {
                "prompt_hash": fingerprint["prompt_hash"],
                "decoding_hash": fingerprint["decoding_hash"],
                "request_hash": fingerprint["request_hash"],
                "cache_key": expected_cache_key,
            }
            for field, expected_value in expected_fields.items():
                if row.get(field) != expected_value:
                    raise IntegrityError(f"Result {field} does not match the frozen request for {key}")
        if verify_cache and tasks_root is not None:
            cache_record = cache_records.get(expected_cache_key)
            if cache_record is None:
                raise IntegrityError(f"Raw response cache record is missing for {key}")
            if (
                cache_record.get("provider") != provider
                or cache_record.get("model") != api_model
                or cache_record.get("task_id") != task_id
                or cache_record.get("sample") != 0
                or cache_record.get("prompt") != prompt
                or cache_record.get("prompt_hash") != fingerprint["prompt_hash"]
                or cache_record.get("decoding_hash") != fingerprint["decoding_hash"]
                or cache_record.get("request_hash") != fingerprint["request_hash"]
            ):
                raise IntegrityError(f"Raw cache provenance mismatch for {key}")
            response_text = cache_record.get("response_text")
            if not isinstance(response_text, str):
                raise IntegrityError(f"Raw cache response is missing for {key}")
            if hashlib.sha256(response_text.encode("utf-8")).hexdigest() != row["response_sha256"]:
                raise IntegrityError(f"Raw response hash mismatch for {key}")
            parsed_ok, parsed_findings, parsed_error = audit_runner.parse_findings_strict(
                response_text, audit_runner.ALLOWED_KINDS[domain]
            )
            if (
                parsed_ok != row["parse_ok"]
                or parsed_findings != findings
                or parsed_error != row["parse_error"]
            ):
                raise IntegrityError(f"Stored findings do not reproduce from the raw response for {key}")
        elif response_ledger_path is not None:
            ledger = ledger_records.get(key)
            if ledger is None:
                raise IntegrityError(f"Response-ledger record is missing for {key}")
            findings_sha256 = hashlib.sha256(
                json.dumps(findings, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            expected_ledger = {
                "model": model,
                "provider": provider,
                "api_model": api_model,
                "task_id": task_id,
                "sample": sample,
                "cache_key": expected_cache_key,
                "request_hash": fingerprint["request_hash"],
                "prompt_hash": fingerprint["prompt_hash"],
                "decoding_hash": fingerprint["decoding_hash"],
                "response_sha256": row["response_sha256"],
                "parse_ok": row["parse_ok"],
                "parse_error": row["parse_error"],
                "findings_sha256": findings_sha256,
            }
            for field, expected_value in expected_ledger.items():
                if ledger.get(field) != expected_value:
                    raise IntegrityError(f"Response ledger {field} mismatch for {key}")
        rows_by_key[key] = row
    if set(rows_by_key) != expected:
        missing = sorted(expected - set(rows_by_key))
        raise IncompleteGridError(f"Result grid is incomplete: missing {len(missing)} keys; first={missing[:3]}")
    if response_ledger_path is not None and set(ledger_records) != expected:
        missing = sorted(expected - set(ledger_records))
        extra = sorted(set(ledger_records) - expected)
        raise IntegrityError(
            f"Response-ledger grid mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    return rows_by_key


def exact_mcnemar_p(single_only: int, dual_only: int) -> float:
    """Two-sided exact McNemar/binomial p-value for discordant pairs."""

    discordant = single_only + dual_only
    if discordant == 0:
        return 1.0
    tail = min(single_only, dual_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down family-wise adjusted p-values."""

    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_bootstrap_ci(differences: list[int], *, draws: int, seed: int) -> tuple[float, float]:
    if not differences:
        raise ValueError("Cannot bootstrap an empty paired sample")
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    rng = random.Random(seed)
    size = len(differences)
    estimates = [
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(draws)
    ]
    estimates.sort()
    lower = estimates[int(0.025 * (draws - 1))]
    upper = estimates[int(0.975 * (draws - 1))]
    return lower, upper


def _evaluated_row(result: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    domain = str(truth["domain"])
    raw_pairs = [(int(item["step_index"]), str(item["kind"])) for item in result["findings"]]
    duplicate_findings = len(raw_pairs) - len(set(raw_pairs))
    invalid_kinds = sum(kind not in ALLOWED_KINDS[domain] for _, kind in raw_pairs)
    predicted = set(raw_pairs)
    expected = {(int(item["step_index"]), str(item["kind"])) for item in truth["defects"]}
    remaining = set(expected)
    true_positives = 0
    for pair in raw_pairs:
        if pair in remaining:
            true_positives += 1
            remaining.remove(pair)
    false_positives = len(raw_pairs) - true_positives
    false_negatives = len(expected) - true_positives
    violation_kind, edit_kind = DOMAIN_LABELS[domain]
    target_step = int(truth["target_step_index"])
    return {
        "model": result["model"],
        "task_id": truth["task_id"],
        "triple_id": truth["triple_id"],
        "domain": domain,
        "producer": truth["producer"],
        "cell": domain_cell(truth),
        "condition": truth["condition"],
        "parse_ok": bool(result["parse_ok"]),
        "raw_finding_count": len(raw_pairs),
        "duplicate_findings": duplicate_findings,
        "invalid_kinds": invalid_kinds,
        "predicted": predicted,
        "expected": expected,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "exact_set": bool(result["parse_ok"]) and duplicate_findings == 0 and predicted == expected,
        "cardinality_correct": (
            bool(result["parse_ok"])
            and duplicate_findings == 0
            and len(predicted) == len(expected)
        ),
        "violation_hit": (target_step, violation_kind) in predicted,
        "edit_hit": (target_step, edit_kind) in predicted,
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _metric_rows(evaluated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        groups[(str(row["model"]), str(row["cell"]), str(row["condition"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, cell, condition), rows in sorted(groups.items()):
        tp = sum(int(row["tp"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        both = sum(bool(row["violation_hit"]) and bool(row["edit_hit"]) for row in rows)
        violation_only = sum(bool(row["violation_hit"]) and not bool(row["edit_hit"]) for row in rows)
        edit_only = sum(not bool(row["violation_hit"]) and bool(row["edit_hit"]) for row in rows)
        neither = len(rows) - both - violation_only - edit_only
        output.append(
            {
                "model": model,
                "cell": cell,
                "condition": condition,
                "tasks": len(rows),
                "parse_failures": sum(not bool(row["parse_ok"]) for row in rows),
                "invalid_kinds": sum(int(row["invalid_kinds"]) for row in rows),
                "duplicate_findings": sum(int(row["duplicate_findings"]) for row in rows),
                "exact_set_accuracy": _safe_ratio(sum(bool(row["exact_set"]) for row in rows), len(rows)),
                "cardinality_accuracy": _safe_ratio(
                    sum(bool(row["cardinality_correct"]) for row in rows), len(rows)
                ),
                "any_finding_rate": _safe_ratio(sum(int(row["raw_finding_count"]) > 0 for row in rows), len(rows)),
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": f1,
                "violation_recall": _safe_ratio(sum(bool(row["violation_hit"]) for row in rows), len(rows))
                if condition != "zero"
                else "",
                "edit_recall": _safe_ratio(sum(bool(row["edit_hit"]) for row in rows), len(rows))
                if condition == "violation_plus_edit"
                else "",
                "dual_both": both if condition == "violation_plus_edit" else "",
                "dual_violation_only": violation_only if condition == "violation_plus_edit" else "",
                "dual_edit_only": edit_only if condition == "violation_plus_edit" else "",
                "dual_neither": neither if condition == "violation_plus_edit" else "",
            }
        )
    return output


def _overall_rows(evaluated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        groups[str(row["model"])].append(row)
    output: list[dict[str, Any]] = []
    for model, rows in sorted(groups.items()):
        tp = sum(int(row["tp"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        output.append(
            {
                "model": model,
                "tasks": len(rows),
                "parse_failures": sum(not bool(row["parse_ok"]) for row in rows),
                "invalid_kinds": sum(int(row["invalid_kinds"]) for row in rows),
                "duplicate_findings": sum(int(row["duplicate_findings"]) for row in rows),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": _safe_ratio(2 * precision * recall, precision + recall),
                "exact_set_accuracy": _safe_ratio(sum(bool(row["exact_set"]) for row in rows), len(rows)),
            }
        )
    return output


def _primary_rows(evaluated: list[dict[str, Any]], *, bootstrap_draws: int) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in evaluated:
        key = (str(row["model"]), str(row["cell"]), str(row["triple_id"]), str(row["condition"]))
        if key in by_key:
            raise IntegrityError(f"Duplicate evaluated condition key: {key}")
        by_key[key] = row
    groups = sorted({(key[0], key[1]) for key in by_key})
    output: list[dict[str, Any]] = []
    for model, cell in groups:
        triple_ids = sorted({key[2] for key in by_key if key[0] == model and key[1] == cell})
        differences: list[int] = []
        single_hits: list[bool] = []
        dual_hits: list[bool] = []
        for triple_id in triple_ids:
            single = by_key.get((model, cell, triple_id, "violation_only"))
            dual = by_key.get((model, cell, triple_id, "violation_plus_edit"))
            zero = by_key.get((model, cell, triple_id, "zero"))
            if single is None or dual is None or zero is None:
                raise IntegrityError(f"Incomplete evaluated triple: {(model, cell, triple_id)}")
            single_hit = bool(single["violation_hit"])
            dual_hit = bool(dual["violation_hit"])
            single_hits.append(single_hit)
            dual_hits.append(dual_hit)
            differences.append(int(single_hit) - int(dual_hit))
        single_only = sum(single and not dual for single, dual in zip(single_hits, dual_hits))
        dual_only = sum(not single and dual for single, dual in zip(single_hits, dual_hits))
        seed = int(hashlib.sha256(f"{model}|{cell}".encode()).hexdigest()[:16], 16)
        ci_low, ci_high = paired_bootstrap_ci(differences, draws=bootstrap_draws, seed=seed)
        output.append(
            {
                "model": model,
                "cell": cell,
                "triples": len(triple_ids),
                "single_violation_recall": _safe_ratio(sum(single_hits), len(single_hits)),
                "dual_violation_recall": _safe_ratio(sum(dual_hits), len(dual_hits)),
                "recall_drop": _safe_ratio(sum(differences), len(differences)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "single_only": single_only,
                "dual_only": dual_only,
                "mcnemar_p": exact_mcnemar_p(single_only, dual_only),
            }
        )
    expected_cells = {"trading:deepseek_v4_pro", "trading:glm_5_direct", "tooluse"}
    if {str(row["cell"]) for row in output} != expected_cells or len(output) != 6:
        raise IntegrityError(f"Primary family must contain exactly 2 models x 3 cells; found {len(output)} rows")
    adjusted = holm_adjust([float(row["mcnemar_p"]) for row in output])
    for row, value in zip(output, adjusted):
        row["holm_p"] = value
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def analyze(
    tasks_root: Path,
    results_path: Path,
    cache_dir: Path,
    output_dir: Path,
    *,
    models: list[str],
    bootstrap_draws: int,
    trading_source_root: Path | None = None,
    response_ledger_path: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if bootstrap_draws != 10000:
        raise IntegrityError(f"Bootstrap draws differ from the frozen plan: {bootstrap_draws}")
    _, truth_by_task = load_dataset(tasks_root, trading_source_root=trading_source_root)
    results = validate_results(
        results_path,
        truth_by_task,
        models,
        tasks_root=tasks_root,
        cache_dir=cache_dir,
        response_ledger_path=response_ledger_path,
        verify_cache=response_ledger_path is None,
    )
    evaluated = [
        _evaluated_row(results[(model, task_id, 0)], truth_by_task[task_id])
        for model in models
        for task_id in sorted(truth_by_task)
    ]
    primary = _primary_rows(evaluated, bootstrap_draws=bootstrap_draws)
    conditions = _metric_rows(evaluated)
    overall = _overall_rows(evaluated)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "primary_violation_drop.csv", primary)
    _write_csv(output_dir / "condition_metrics.csv", conditions)
    _write_csv(output_dir / "model_overall.csv", overall)

    lines = [
        "# Multi-label audit analysis",
        "",
        f"Strict grid verified: {len(results)} unique model/task/sample keys; no missing, duplicate, or extra rows.",
        "Primary family: six paired violation-recall contrasts with exact two-sided McNemar tests and Holm correction.",
        "",
        "| Model | Cell | n | Single recall | Dual recall | Drop | 95% CI | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row['model']} | {row['cell']} | {row['triples']} | "
            f"{float(row['single_violation_recall']):.3f} | {float(row['dual_violation_recall']):.3f} | "
            f"{float(row['recall_drop']):+.3f} | "
            f"[{float(row['ci95_low']):+.3f}, {float(row['ci95_high']):+.3f}] | "
            f"{float(row['holm_p']):.4g} |"
        )
    lines.extend(
        [
            "",
            "Malformed responses never count as a correct empty set; parse failures, invalid kinds, and duplicate findings are reported separately.",
        ]
    )
    _atomic_write_text(output_dir / "multilabel_analysis.md", "\n".join(lines) + "\n")
    analysis_manifest: dict[str, Any] = {
        "schema_version": "finaudit.multilabel.analysis.v1",
        "tasks_manifest_sha256": sha256_file(tasks_root / "manifest.json"),
        "ground_truth_sha256": sha256_file(tasks_root / "ground_truth.jsonl"),
        "results_sha256": sha256_file(results_path),
        "verification_mode": "released-response-ledger" if response_ledger_path else "raw-cache-replay",
        "models": models,
        "expected_result_keys": len(models) * len(truth_by_task),
        "observed_result_keys": len(results),
        "primary_family_size": len(primary),
        "bootstrap_draws": bootstrap_draws,
    }
    if response_ledger_path is None:
        analysis_manifest["cache_files_sha256"] = {
            path.name: sha256_file(path) for path in sorted(cache_dir.glob("auditml_*.jsonl"))
        }
    else:
        analysis_manifest["response_ledger_sha256"] = sha256_file(response_ledger_path)
    _atomic_write_text(
        output_dir / "analysis_manifest.json",
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
    )
    return {"primary": primary, "conditions": conditions, "overall": overall}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and analyze paper-4 multi-label results.")
    parser.add_argument("--tasks-dir", default="outputs/audit_multilabel_tasks")
    parser.add_argument(
        "--results", default="outputs/audit_multilabel_eval/multilabel_audit_results.jsonl"
    )
    parser.add_argument("--cache-dir", default="outputs/llm_cache/audit_multilabel_v1")
    parser.add_argument(
        "--response-ledger",
        help="Verify released structured results against a redacted provenance ledger instead of raw cache.",
    )
    parser.add_argument("--trading-source-root", default="outputs/audit_self")
    parser.add_argument("--output-dir", default="docs/results/finaudit_multilabel")
    parser.add_argument("--models", default="deepseek:deepseek-v4-pro,glm:glm-5")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument(
        "--check-dataset-only",
        action="store_true",
        help="Validate task hashes, answer key, rule oracle, triplets, and frozen scale without results.",
    )
    args = parser.parse_args(argv)
    if args.bootstrap_draws != 10000:
        parser.error("--bootstrap-draws is frozen at 10000")

    tasks_root = Path(args.tasks_dir)
    results_path = Path(args.results)
    cache_dir = Path(args.cache_dir)
    response_ledger_path = Path(args.response_ledger) if args.response_ledger else None
    trading_source_root = Path(args.trading_source_root)
    output_dir = Path(args.output_dir)
    if not tasks_root.is_absolute():
        tasks_root = ROOT / tasks_root
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    if response_ledger_path is not None and not response_ledger_path.is_absolute():
        response_ledger_path = ROOT / response_ledger_path
    if not trading_source_root.is_absolute():
        trading_source_root = ROOT / trading_source_root
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if args.check_dataset_only:
        try:
            manifest, truth = load_dataset(
                tasks_root, trading_source_root=trading_source_root
            )
        except (
            FileNotFoundError,
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            print(f"INTEGRITY ERROR: {exc}", file=sys.stderr)
            return 3
        print(f"Dataset gate passed: {len(manifest)} manifest tasks and {len(truth)} answer-key rows", flush=True)
        return 0
    if not results_path.is_file():
        print(f"INCOMPLETE: result file does not exist: {results_path}", file=sys.stderr)
        return 2
    try:
        result = analyze(
            tasks_root,
            results_path,
            cache_dir,
            output_dir,
            models=models,
            bootstrap_draws=args.bootstrap_draws,
            trading_source_root=trading_source_root,
            response_ledger_path=response_ledger_path,
        )
    except IncompleteGridError as exc:
        print(f"INCOMPLETE: {exc}", file=sys.stderr)
        return 2
    except (
        FileNotFoundError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"INTEGRITY ERROR: {exc}", file=sys.stderr)
        return 3
    print(
        f"Verified and analyzed {sum(int(row['tasks']) for row in result['overall'])} model-task rows -> {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
