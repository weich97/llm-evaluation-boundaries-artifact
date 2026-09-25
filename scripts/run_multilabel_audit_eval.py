"""Collect auditor findings for the paper-4 multi-label benchmark.

The runner never reads the private answer key and never writes model response
text to the result file.  It checkpoints one strict, schema-validated findings
record per ``(model, task, sample)`` and uses a fresh cache namespace whose key
covers the system prompt, user prompt, model, temperature, and output schema.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_legacy_spec = importlib.util.spec_from_file_location("run_audit_eval", ROOT / "scripts" / "run_audit_eval.py")
assert _legacy_spec and _legacy_spec.loader
run_audit_eval = importlib.util.module_from_spec(_legacy_spec)
_legacy_spec.loader.exec_module(run_audit_eval)

from tradearena.agents.llm import DeepSeekLLMAnalyst
from tradearena.evaluation.toolaudit import tool_compact_view

CACHE_NAMESPACE = "auditmlv1"
OUTPUT_SCHEMA_VERSION = "finaudit.findings.v1"
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
SYSTEM_PROMPT = run_audit_eval.AUDIT_SYSTEM_PROMPT
ALLOWED_KINDS = {
    "trading": {"unclipped_position", "silent_risk_edit", "provenance_drift", "tampered_fill_price"},
    "tooluse": {"over_limit_call", "silent_arg_edit", "version_drift", "tampered_result"},
}
COMPACT_NOTES = {
    "trading": (
        "The artifact below is a compact audit view: each step contains intended and risk-approved "
        "decisions, fills, recorded risk violations, model provenance, and observed close prices."
    ),
    "tooluse": (
        "The artifact below is a compact action log: each step contains requested and approved "
        "arguments, approval metadata, the result, tool version, and recorded policy flags."
    ),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def request_fingerprint(
    provider: str,
    model: str,
    prompt: str,
    *,
    sample: int = 0,
    temperature: float = 0.0,
    system_prompt: str = SYSTEM_PROMPT,
    thinking: str = "disabled",
    use_response_format: bool = False,
) -> dict[str, str]:
    """Return collision-resistant cache identifiers for the complete request."""

    decoding = {
        "system_prompt": system_prompt,
        "temperature": float(temperature),
        "thinking": thinking,
        "use_response_format": use_response_format,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
    }
    decoding_hash = hashlib.sha256(_canonical(decoding).encode("utf-8")).hexdigest()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    request = {
        "provider": provider,
        "model": model,
        "prompt_hash": prompt_hash,
        "decoding_hash": decoding_hash,
        "sample": int(sample),
    }
    request_hash = hashlib.sha256(_canonical(request).encode("utf-8")).hexdigest()
    return {
        "prompt_hash": prompt_hash,
        "decoding_hash": decoding_hash,
        "request_hash": request_hash,
    }


def parse_findings_strict(
    response_text: str, allowed_kinds: set[str]
) -> tuple[bool, list[dict[str, Any]], str]:
    """Parse a raw JSON findings array without treating malformed text as ``[]``."""

    text = response_text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False, [], "invalid_json"
    if not isinstance(value, list):
        return False, [], "top_level_not_array"

    findings: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"item_{index}_not_object")
            continue
        step_index = item.get("step_index")
        kind = item.get("kind")
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            errors.append(f"item_{index}_invalid_step")
            continue
        if not isinstance(kind, str) or not kind:
            errors.append(f"item_{index}_invalid_kind")
            continue
        findings.append({"step_index": step_index, "kind": kind})
        if kind not in allowed_kinds:
            errors.append(f"item_{index}_unknown_kind")
    return not errors, findings, ";".join(errors)


def build_prompt(task_dir: Path, domain: str) -> str:
    instructions = (task_dir / "prompt.md").read_text(encoding="utf-8")
    trajectory = json.loads((task_dir / "trajectory.json").read_text(encoding="utf-8"))
    if domain == "trading":
        compact = run_audit_eval.compact_trajectory(trajectory)
    elif domain == "tooluse":
        compact = tool_compact_view(trajectory)
    else:
        raise ValueError(f"Unknown audit domain: {domain!r}")
    return f"{instructions}\n\n{COMPACT_NOTES[domain]}\n\n```json\n{_canonical(compact)}\n```\n"


def _make_analyst(provider: str, model: str, cache_dir: Path) -> DeepSeekLLMAnalyst:
    slug = f"{provider}_{model}".replace("-", "_").replace(".", "_").replace(":", "_")
    common: dict[str, Any] = {
        "model": model,
        "cache_path": str(cache_dir / f"auditml_{slug}.jsonl"),
        "timeout_seconds": 180,
        "temperature": 0.0,
        "thinking": "disabled",
        "use_response_format": False,
        "system_prompt_override": SYSTEM_PROMPT,
    }
    if provider == "glm":
        return DeepSeekLLMAnalyst(
            **common,
            api_key_env="GLM_API_KEY",
            fallback_api_key_env="",
            api_base_url="https://open.bigmodel.cn/api/paas/v4",
            provider="glm",
            api_protocol="openai_chat_completions",
        )
    if provider == "poe":
        poe_common = {**common, "thinking": ""}
        return DeepSeekLLMAnalyst(
            **poe_common,
            api_key_env="POE_API_KEY",
            fallback_api_key_env="",
            api_base_url="https://api.poe.com/v1",
            provider="poe",
            api_protocol="openai_chat_completions",
        )
    if provider != "deepseek":
        raise ValueError(f"Unsupported provider: {provider!r}")
    return DeepSeekLLMAnalyst(**common, provider="deepseek")


def call_model(
    provider: str,
    model: str,
    prompt: str,
    cache_dir: Path,
    task_id: str,
    *,
    sample: int = 0,
    temperature: float = 0.0,
    analyst: DeepSeekLLMAnalyst | None = None,
) -> tuple[str, dict[str, str]]:
    """Call one auditor through the existing transport under a fresh cache key."""

    if temperature != 0.0:
        raise ValueError("The primary multi-label protocol is frozen at temperature 0")
    analyst = analyst or _make_analyst(provider, model, cache_dir)
    fingerprint = request_fingerprint(
        provider,
        model,
        prompt,
        sample=sample,
        temperature=temperature,
        thinking=analyst.thinking,
        use_response_format=analyst.use_response_format,
    )
    cache_key = (
        f"{CACHE_NAMESPACE}:{provider}:{model}:{task_id}:"
        f"{fingerprint['request_hash']}:s{sample}"
    )
    fingerprint["cache_key"] = cache_key
    cached = analyst._cache().get(cache_key)
    if cached is not None:
        return str(cached["response_text"]), fingerprint
    response_text = analyst._call_deepseek(prompt)
    analyst._append_cache(
        {
            "cache_key": cache_key,
            "namespace": CACHE_NAMESPACE,
            "provider": provider,
            "model": model,
            "task_id": task_id,
            "sample": sample,
            "temperature": temperature,
            **fingerprint,
            "prompt": prompt,
            "response_text": response_text,
        }
    )
    return response_text, fingerprint


def _load_manifest(tasks_root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((tasks_root / "manifest.json").read_text(encoding="utf-8"))
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Manifest has no task entries")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for entry in tasks:
        if not isinstance(entry, dict):
            raise ValueError("Manifest task entry is not an object")
        task_id = str(entry.get("task_id", ""))
        domain = str(entry.get("domain", ""))
        if not task_id or task_id in seen:
            raise ValueError(f"Duplicate or empty manifest task id: {task_id!r}")
        if domain not in ALLOWED_KINDS:
            raise ValueError(f"Invalid domain for {task_id}: {domain!r}")
        task_dir = tasks_root / "tasks" / task_id
        if not (task_dir / "trajectory.json").is_file() or not (task_dir / "prompt.md").is_file():
            raise ValueError(f"Missing artifact files for task {task_id}")
        seen.add(task_id)
        normalized.append({**entry, "task_id": task_id, "domain": domain})
    return sorted(normalized, key=lambda entry: str(entry["task_id"]))


def _load_done(
    results_path: Path,
    *,
    allowed_models: set[str],
    allowed_tasks: set[str],
) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not results_path.exists():
        return done
    with results_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed checkpoint JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Checkpoint line {line_number} is not an object")
            if set(row) != RESULT_FIELDS:
                raise ValueError(f"Checkpoint line {line_number} differs from the frozen result schema")
            findings = row.get("findings")
            if not isinstance(findings, list) or any(
                not isinstance(finding, dict) or set(finding) != FINDING_FIELDS
                for finding in findings
            ):
                raise ValueError(f"Checkpoint line {line_number} has a non-canonical finding schema")
            model = str(row.get("model", ""))
            task_id = str(row.get("task_id", ""))
            sample = row.get("sample")
            if model not in allowed_models or task_id not in allowed_tasks or sample != 0:
                raise ValueError(f"Unexpected checkpoint key at line {line_number}: {(model, task_id, sample)}")
            key = (model, task_id, 0)
            if key in done:
                raise ValueError(f"Duplicate checkpoint key at line {line_number}: {key}")
            done.add(key)
    return done


def _canonicalize_complete_results(results_path: Path, expected_count: int) -> None:
    """Atomically sort a complete checkpoint so its released bytes are deterministic."""

    rows: list[dict[str, Any]] = []
    with results_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != RESULT_FIELDS:
                raise ValueError(
                    f"Cannot canonicalize non-frozen result schema at line {line_number}"
                )
            findings = row.get("findings")
            if not isinstance(findings, list) or any(
                not isinstance(finding, dict) or set(finding) != FINDING_FIELDS
                for finding in findings
            ):
                raise ValueError(
                    f"Cannot canonicalize non-frozen finding schema at line {line_number}"
                )
            rows.append(row)
    if len(rows) != expected_count:
        raise ValueError(f"Cannot canonicalize incomplete results: {len(rows)}/{expected_count}")
    keys = [(str(row["model"]), str(row["task_id"]), int(row["sample"])) for row in rows]
    if len(set(keys)) != expected_count:
        raise ValueError("Cannot canonicalize duplicated result keys")
    rows = [row for _, row in sorted(zip(keys, rows, strict=True), key=lambda item: item[0])]
    temporary = results_path.with_suffix(results_path.suffix + ".canonical.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(_canonical(row) + "\n" for row in rows))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, results_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run direct auditors on paper-4 multi-label tasks.")
    parser.add_argument("--tasks-dir", default="outputs/audit_multilabel_tasks")
    parser.add_argument("--models", default="deepseek:deepseek-v4-pro,glm:glm-5")
    parser.add_argument("--cache-dir", default="outputs/llm_cache/audit_multilabel_v1")
    parser.add_argument("--output-dir", default="outputs/audit_multilabel_eval")
    args = parser.parse_args(argv)

    tasks_root = Path(args.tasks_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    if not tasks_root.is_absolute():
        tasks_root = ROOT / tasks_root
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    task_entries = _load_manifest(tasks_root)
    model_specs = [item.strip() for item in args.models.split(",") if item.strip()]
    if not model_specs or any(":" not in item for item in model_specs):
        parser.error("--models must contain comma-separated provider:model entries")
    if len(model_specs) != len(set(model_specs)):
        parser.error("--models contains duplicates")
    analysts: dict[str, DeepSeekLLMAnalyst] = {}
    for spec in model_specs:
        provider, model = spec.split(":", 1)
        analysts[spec] = _make_analyst(provider, model, cache_dir)

    results_path = output_dir / "multilabel_audit_results.jsonl"
    done = _load_done(
        results_path,
        allowed_models=set(model_specs),
        allowed_tasks={str(entry["task_id"]) for entry in task_entries},
    )
    if done:
        print(f"Resuming after {len(done)} strict checkpoint rows", flush=True)

    with results_path.open("a", encoding="utf-8", newline="\n") as handle:
        for spec in model_specs:
            provider, model = spec.split(":", 1)
            for entry in task_entries:
                task_id = str(entry["task_id"])
                key = (spec, task_id, 0)
                if key in done:
                    continue
                domain = str(entry["domain"])
                prompt = build_prompt(tasks_root / "tasks" / task_id, domain)
                try:
                    response, fingerprint = call_model(
                        provider,
                        model,
                        prompt,
                        cache_dir,
                        task_id,
                        analyst=analysts[spec],
                    )
                except Exception as exc:
                    print(
                        f"FAILED {spec} {task_id}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                parse_ok, findings, parse_error = parse_findings_strict(response, ALLOWED_KINDS[domain])
                record = {
                    "schema_version": OUTPUT_SCHEMA_VERSION,
                    "model": spec,
                    "task_id": task_id,
                    "sample": 0,
                    "parse_ok": parse_ok,
                    "parse_error": parse_error,
                    "finding_count": len(findings),
                    "findings": findings,
                    "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                    **fingerprint,
                }
                handle.write(_canonical(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                done.add(key)
                print(
                    f"OK {len(done)}/{len(model_specs) * len(task_entries)} "
                    f"{spec} {task_id} parse_ok={parse_ok} findings={len(findings)}",
                    flush=True,
                )
    expected_count = len(model_specs) * len(task_entries)
    if len(done) != expected_count:
        print(
            f"INCOMPLETE: collected {len(done)}/{expected_count} exact keys; rerun to resume",
            file=sys.stderr,
            flush=True,
        )
        return 2
    _canonicalize_complete_results(results_path, expected_count)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"INTEGRITY ERROR: {exc}", file=sys.stderr, flush=True)
        exit_code = 3
    raise SystemExit(exit_code)
