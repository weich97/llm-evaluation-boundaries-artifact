"""Analyze the frozen-response v0.3 two-by-two replay without private text.

The causal execution contrast holds one provider response tape fixed and
verifies that the resulting pre-risk decision path is identical across E0/E1.
Risk approval and fills remain downstream, execution-endogenous quantities.
The between-origin contrast is descriptive: the two live response tapes can
differ because of sampling, closed-loop feedback, and provider-time drift.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradearena.core.redaction import scan_public_artifact_paths, scan_public_artifact_payload
from tradearena.core.reproducibility import canonical_json, sha256_text

PROTOCOL_ID = "trellm-v0.3-iclr-protocol"
PLAN_SCHEMA = "tradearena.fixed-intent-replay.plan.v1"
ANALYSIS_SCHEMA = "tradearena.fixed-response-replay.analysis.v1"
SOURCE_HASH_POLICY = "sha256 of text bytes after CRLF/CR -> LF normalization"
EXPECTED_INTENTS = 900
EXPECTED_BASE_PAIRS = 450
EXPECTED_REPLAYS = 1800
EXPECTED_DIVERGENCE = 450
EXPECTED_SEEDS = 10
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260719
BOOTSTRAP_UNIT = (
    "shared source seed; retain all models, scenarios, samples, origins, and destinations"
)
BOOTSTRAP_RNG_ALGORITHM = "CPython random.Random (MT19937)"
BOOTSTRAP_DRAW_METHOD = (
    "for each replicate, draw G randrange(G) indices with replacement and store "
    "multiplicity counts aligned to sorted seed clusters"
)
PERCENTILE_METHOD = "Hyndman-Fan type 7 (linear interpolation)"
PERCENTILE_FORMULA = (
    "sort x[0..B-1]; z=(B-1)*p; interpolate x[floor(z)] and x[ceil(z)] by z-floor(z)"
)
T_CRITICAL_DF9_975 = 2.2621571628540993
EXECUTION_LEVELS = ("E0", "E1")
METRICS = (
    "total_return",
    "sharpe",
    "max_drawdown",
    "execution_fill_rate",
    "total_slippage_cost",
    "rejected_order_count",
    "risk_clipped_decisions",
    "risk_violation_count",
    "trajectory_reproducibility_coverage",
)
FLOAT_METRICS = {
    "total_return",
    "sharpe",
    "max_drawdown",
    "execution_fill_rate",
    "total_slippage_cost",
    "trajectory_reproducibility_coverage",
}
INTEGER_METRICS = set(METRICS) - FLOAT_METRICS
ESTIMANDS = (
    "execution_within_I0",
    "execution_within_I1",
    "response_origin_within_X0",
    "response_origin_within_X1",
    "interaction",
    "execution_shapley",
    "response_origin_shapley",
    "observed_diagonal",
)
PLAN_ESTIMANDS = (
    "destination execution contrast within response origin I0",
    "destination execution contrast within response origin I1",
    "realized response-path contrast within X0",
    "realized response-path contrast within X1",
    "two-by-two interaction",
    "execution and realized-path Shapley decomposition of the observed diagonal difference",
)
ANALYSIS_CONTRACT = {
    "overall_weighting": (
        "equal weight per complete model-scenario-seed-sample base pair; provider mix follows "
        "the frozen design (GLM 60%, DeepSeek 40%)"
    ),
    "interval_interpretation": (
        "descriptive pointwise 95% shared-seed cluster intervals; no family-wise error control"
    ),
    "primary_summary": "total_return execution_shapley",
    "ranking": {
        "metric": "sharpe",
        "direction": "higher_is_better",
        "scope": "scenario_by_response_origin",
        "agreement": "Kendall tau-b between E0 and E1 destination rankings",
    },
}
INTENT_FIELDS = {
    "protocol_id",
    "base_key_sha256",
    "source_plan_id",
    "provider",
    "model_id",
    "model_version_or_release",
    "scenario_id",
    "contamination_tier",
    "seed",
    "sample_index",
    "intent_origin_execution",
    "periods",
    "response_count",
    "call_log_sha256",
    "prompt_calls_sha256",
    "response_calls_sha256",
    "normalized_parsed_response_path_sha256",
    "parse_coverage",
    "source_status",
}
REPLAY_FIELDS = {
    "protocol_id",
    "replay_id",
    "base_key_sha256",
    "source_plan_id",
    "provider",
    "model_id",
    "model_version_or_release",
    "scenario_id",
    "contamination_tier",
    "seed",
    "sample_index",
    "intent_origin_execution",
    "replay_execution_level",
    "periods",
    "response_count",
    "response_calls_sha256",
    "normalized_parsed_response_path_sha256",
    "decision_path_sha256",
    "generated_prompt_calls_sha256",
    "diagonal",
    "diagonal_check_applicable",
    "diagonal_reproduction_pass",
    "status",
    *METRICS,
    "hold_ratio",
    "mean_gross_target_exposure",
    "metrics_sha256",
}
DIVERGENCE_FIELDS = {
    "protocol_id",
    "base_key_sha256",
    "provider",
    "model_id",
    "model_version_or_release",
    "scenario_id",
    "contamination_tier",
    "seed",
    "sample_index",
    "first_prompt_equal",
    "first_response_hash_equal",
    "first_parsed_response_equal",
    "first_parsed_response_divergence_step",
    "full_response_path_equal",
    "full_parsed_response_path_equal",
}


class IntegrityError(ValueError):
    """Raised when released inputs do not match the frozen replay contract."""


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_lf_text(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"invalid required JSON: {path}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"required JSON is not an object: {path}")
    return value


def _read_csv(path: Path, fields: set[str], expected_rows: int) -> list[dict[str, str]]:
    try:
        handle = path.open(encoding="utf-8", newline="")
    except FileNotFoundError as exc:
        raise IntegrityError(f"missing required CSV: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != fields:
            raise IntegrityError(f"CSV schema differs from the frozen contract: {path.name}")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise IntegrityError(f"{path.name} has {len(rows)}/{expected_rows} rows")
    return rows


def _bool(value: str, *, allow_blank: bool = False) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    if allow_blank and value == "":
        return None
    raise IntegrityError(f"invalid serialized boolean: {value!r}")


def _finite(value: str, field: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise IntegrityError(f"non-numeric {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise IntegrityError(f"non-finite {field}: {value!r}")
    return number


def _int(value: str, field: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise IntegrityError(f"non-integer {field}: {value!r}") from exc
    return number


def _metric_values(row: dict[str, str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in METRICS:
        value = _finite(row[metric], metric)
        if metric in INTEGER_METRICS and not value.is_integer():
            raise IntegrityError(f"integer metric is fractional: {metric}={value}")
        values[metric] = value
    expected_hash = sha256_text(
        canonical_json(
            {
                metric: int(values[metric]) if metric in INTEGER_METRICS else values[metric]
                for metric in METRICS
            }
        )
    )
    if row["metrics_sha256"] != expected_hash:
        raise IntegrityError(f"metric hash mismatch for replay {row['replay_id']}")
    return values


def _validate_analysis_plan(plan: dict[str, Any]) -> None:
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("metrics") != list(METRICS)
    ):
        raise IntegrityError("analysis plan schema/protocol/metric family differs")
    if plan.get("expected") != {
        "base_pairs": EXPECTED_BASE_PAIRS,
        "execution_levels": list(EXECUTION_LEVELS),
        "intent_paths": EXPECTED_INTENTS,
        "periods_per_path": 24,
        "replay_rows": EXPECTED_REPLAYS,
        "samples_per_seed": 3,
        "seeds_per_model_scenario": EXPECTED_SEEDS,
        "source_rows": EXPECTED_INTENTS,
    }:
        raise IntegrityError("analysis plan expected grid differs")
    if plan.get("bootstrap") != {
        "draws": BOOTSTRAP_DRAWS,
        "resampling_unit": BOOTSTRAP_UNIT,
        "seed": BOOTSTRAP_SEED,
    }:
        raise IntegrityError("analysis plan bootstrap contract differs")
    if (
        plan.get("estimands") != list(PLAN_ESTIMANDS)
        or plan.get("analysis_contract") != ANALYSIS_CONTRACT
        or plan.get("source_hash_policy") != SOURCE_HASH_POLICY
        or plan.get("raw_prompt_public") is not False
        or plan.get("raw_response_public") is not False
    ):
        raise IntegrityError("analysis plan estimand/privacy/hash contract differs")
    source_hashes = plan.get("source_sha256")
    source_sets = plan.get("source_set_sha256")
    implementation = plan.get("implementation_sha256")
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in source_hashes.items()
        )
        or not isinstance(source_sets, dict)
        or len(source_sets) != 4
        or not isinstance(implementation, dict)
        or set(implementation) != {"src_tradearena_python_tree"}
    ):
        raise IntegrityError("analysis plan source/implementation bindings are incomplete")


def load_frozen_replay(
    input_dir: Path, analysis_plan_path: Path
) -> tuple[
    dict[str, dict[tuple[str, str], dict[str, Any]]],
    list[dict[str, str]],
    dict[str, Any],
]:
    """Validate the released manifest and return exact four-cell base groups."""

    plan = _read_json(analysis_plan_path)
    _validate_analysis_plan(plan)
    integrity_path = input_dir / "source_integrity.json"
    integrity = _read_json(integrity_path)
    if (
        integrity.get("schema_version") != "tradearena.fixed-intent-replay.integrity.v1"
        or integrity.get("protocol_id") != PROTOCOL_ID
        or integrity.get("analysis_plan_sha256") != _sha256(analysis_plan_path)
        or integrity.get("source_hash_policy") != SOURCE_HASH_POLICY
        or integrity.get("failed_count") != 0
        or integrity.get("source_ready") is not True
        or not integrity.get("checks")
        or not all(value is True for value in integrity["checks"].values())
    ):
        raise IntegrityError("source integrity manifest is not a complete passing generation")
    paths = {
        "intent_paths.csv": input_dir / "intent_paths.csv",
        "replay_rows.csv": input_dir / "replay_rows.csv",
        "intent_divergence.csv": input_dir / "intent_divergence.csv",
    }
    if integrity.get("output_sha256") != {name: _sha256(path) for name, path in paths.items()}:
        raise IntegrityError("released CSV hashes differ from source_integrity.json")
    intents = _read_csv(paths["intent_paths.csv"], INTENT_FIELDS, EXPECTED_INTENTS)
    replays = _read_csv(paths["replay_rows.csv"], REPLAY_FIELDS, EXPECTED_REPLAYS)
    divergences = _read_csv(
        paths["intent_divergence.csv"], DIVERGENCE_FIELDS, EXPECTED_DIVERGENCE
    )

    intent_by_source: dict[str, dict[str, str]] = {}
    origins_by_base: dict[str, set[str]] = defaultdict(set)
    for row in intents:
        if (
            row["protocol_id"] != PROTOCOL_ID
            or row["source_status"] != "ok"
            or _int(row["periods"], "periods") != 24
            or _int(row["response_count"], "response_count") != 24
            or _finite(row["parse_coverage"], "parse_coverage") != 1.0
            or row["intent_origin_execution"] not in EXECUTION_LEVELS
        ):
            raise IntegrityError(f"invalid intent path row: {row['source_plan_id']}")
        source_id = row["source_plan_id"]
        if source_id in intent_by_source:
            raise IntegrityError(f"duplicate intent source id: {source_id}")
        intent_by_source[source_id] = row
        origins_by_base[row["base_key_sha256"]].add(row["intent_origin_execution"])
    if len(origins_by_base) != EXPECTED_BASE_PAIRS or any(
        origins != set(EXECUTION_LEVELS) for origins in origins_by_base.values()
    ):
        raise IntegrityError("intent paths are not 450 exact E0/E1 origin pairs")

    groups: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    replay_ids: set[str] = set()
    for row in replays:
        source = intent_by_source.get(row["source_plan_id"])
        if source is None:
            raise IntegrityError(f"replay has no frozen source: {row['source_plan_id']}")
        origin = row["intent_origin_execution"]
        destination = row["replay_execution_level"]
        diagonal = _bool(row["diagonal"])
        applicable = _bool(row["diagonal_check_applicable"])
        passed = _bool(row["diagonal_reproduction_pass"], allow_blank=True)
        if (
            row["protocol_id"] != PROTOCOL_ID
            or row["status"] != "ok"
            or origin not in EXECUTION_LEVELS
            or destination not in EXECUTION_LEVELS
            or diagonal != (origin == destination)
            or applicable != diagonal
            or (passed is True) != diagonal
            or _int(row["periods"], "periods") != 24
            or _int(row["response_count"], "response_count") != 24
        ):
            raise IntegrityError(f"invalid replay gate fields: {row['replay_id']}")
        for field in (
            "base_key_sha256",
            "provider",
            "model_id",
            "model_version_or_release",
            "scenario_id",
            "contamination_tier",
            "seed",
            "sample_index",
            "intent_origin_execution",
            "response_calls_sha256",
            "normalized_parsed_response_path_sha256",
        ):
            if row[field] != source[field]:
                raise IntegrityError(f"replay/source mismatch for {field}: {row['replay_id']}")
        if row["replay_id"] in replay_ids:
            raise IntegrityError(f"duplicate replay id: {row['replay_id']}")
        replay_ids.add(row["replay_id"])
        cell = (origin, destination)
        if cell in groups[row["base_key_sha256"]]:
            raise IntegrityError(f"duplicate replay cell for {row['base_key_sha256']}: {cell}")
        groups[row["base_key_sha256"]][cell] = {
            **row,
            "seed_int": _int(row["seed"], "seed"),
            "sample_int": _int(row["sample_index"], "sample_index"),
            "metric_values": _metric_values(row),
        }
    expected_cells = {(origin, destination) for origin in EXECUTION_LEVELS for destination in EXECUTION_LEVELS}
    if len(groups) != EXPECTED_BASE_PAIRS or any(set(cells) != expected_cells for cells in groups.values()):
        raise IntegrityError("replay rows are not 450 exact two-by-two groups")
    for base_key, cells in groups.items():
        for field in (
            "provider",
            "model_id",
            "model_version_or_release",
            "scenario_id",
            "contamination_tier",
            "seed",
            "sample_index",
        ):
            if len({cell[field] for cell in cells.values()}) != 1:
                raise IntegrityError(f"cross-origin base identity differs for {base_key}: {field}")
        source_ids = {
            origin: {
                cells[(origin, destination)]["source_plan_id"]
                for destination in EXECUTION_LEVELS
            }
            for origin in EXECUTION_LEVELS
        }
        if any(len(ids) != 1 for ids in source_ids.values()) or (
            next(iter(source_ids["E0"])) == next(iter(source_ids["E1"]))
        ):
            raise IntegrityError(f"source-plan origin binding differs for {base_key}")
        for origin in EXECUTION_LEVELS:
            left = cells[(origin, "E0")]
            right = cells[(origin, "E1")]
            if (
                left["decision_path_sha256"] != right["decision_path_sha256"]
                or left["response_calls_sha256"] != right["response_calls_sha256"]
            ):
                raise IntegrityError(f"response/pre-risk decision path changed for {base_key}/{origin}")
    divergence_by_base = {row["base_key_sha256"]: row for row in divergences}
    if len(divergence_by_base) != EXPECTED_DIVERGENCE or set(divergence_by_base) != set(groups):
        raise IntegrityError("divergence rows do not match the exact base-pair set")
    for row in divergences:
        if row["protocol_id"] != PROTOCOL_ID or _bool(row["first_prompt_equal"]) is not True:
            raise IntegrityError(f"invalid divergence provenance for {row['base_key_sha256']}")
        for field in (
            "first_response_hash_equal",
            "first_parsed_response_equal",
            "full_response_path_equal",
            "full_parsed_response_path_equal",
        ):
            _bool(row[field])
        step = _int(row["first_parsed_response_divergence_step"], "divergence step")
        if not -1 <= step < 24:
            raise IntegrityError("parsed-response divergence step is out of range")
        cells = groups[row["base_key_sha256"]]
        e0 = cells[("E0", "E0")]
        e1 = cells[("E1", "E0")]
        for field in (
            "provider",
            "model_id",
            "model_version_or_release",
            "scenario_id",
            "contamination_tier",
            "seed",
            "sample_index",
        ):
            if row[field] != e0[field] or row[field] != e1[field]:
                raise IntegrityError(
                    f"divergence/base identity differs for {row['base_key_sha256']}: {field}"
                )
        if bool(_bool(row["full_response_path_equal"])) != (
            e0["response_calls_sha256"] == e1["response_calls_sha256"]
        ) or bool(_bool(row["full_parsed_response_path_equal"])) != (
            e0["normalized_parsed_response_path_sha256"]
            == e1["normalized_parsed_response_path_sha256"]
        ):
            raise IntegrityError("divergence equality flags do not match replay provenance")
        decision_equal = (
            cells[("E0", "E0")]["decision_path_sha256"]
            == cells[("E1", "E0")]["decision_path_sha256"]
        )
        if decision_equal:
            for destination in EXECUTION_LEVELS:
                if (
                    cells[("E0", destination)]["metric_values"]
                    != cells[("E1", destination)]["metric_values"]
                ):
                    raise IntegrityError(
                        "identical cross-origin pre-risk decisions produced different outcomes"
                    )
        row["_full_pre_risk_decision_path_equal"] = decision_equal
    seeds = {cells[("E0", "E0")]["seed_int"] for cells in groups.values()}
    if len(seeds) != EXPECTED_SEEDS:
        raise IntegrityError(f"expected 10 shared seed clusters, found {sorted(seeds)}")
    return dict(groups), divergences, integrity


def factorial_values(cells: dict[tuple[str, str], dict[str, Any]], metric: str) -> dict[str, float]:
    y00 = float(cells[("E0", "E0")]["metric_values"][metric])
    y01 = float(cells[("E0", "E1")]["metric_values"][metric])
    y10 = float(cells[("E1", "E0")]["metric_values"][metric])
    y11 = float(cells[("E1", "E1")]["metric_values"][metric])
    values = {
        "execution_within_I0": y01 - y00,
        "execution_within_I1": y11 - y10,
        "response_origin_within_X0": y10 - y00,
        "response_origin_within_X1": y11 - y01,
        "interaction": y11 - y10 - y01 + y00,
        "execution_shapley": 0.5 * ((y01 - y00) + (y11 - y10)),
        "response_origin_shapley": 0.5 * ((y10 - y00) + (y11 - y01)),
        "observed_diagonal": y11 - y00,
    }
    if not math.isclose(
        values["execution_shapley"] + values["response_origin_shapley"],
        values["observed_diagonal"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise IntegrityError("Shapley contributions do not reconstruct the diagonal contrast")
    return values


def bootstrap_weights(seeds: list[int], draws: int, rng_seed: int) -> list[tuple[int, ...]]:
    rng = random.Random(rng_seed)
    weights: list[tuple[int, ...]] = []
    for _ in range(draws):
        counts = [0] * len(seeds)
        for _ in seeds:
            counts[rng.randrange(len(seeds))] += 1
        weights.append(tuple(counts))
    return weights


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("percentile requires values and a probability in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_interval(
    cluster_values: dict[int, float],
    seeds: list[int],
    weights: list[tuple[int, ...]],
) -> tuple[float, float, float]:
    if set(cluster_values) != set(seeds):
        raise IntegrityError("cluster values do not cover the exact shared seed set")
    values = [cluster_values[seed] for seed in seeds]
    estimate = mean(values)
    bootstrapped = [
        sum(weight * value for weight, value in zip(draw, values, strict=True)) / len(seeds)
        for draw in weights
    ]
    return estimate, percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)


def cluster_sensitivity(cluster_values: dict[int, float], seeds: list[int]) -> dict[str, float]:
    if set(cluster_values) != set(seeds):
        raise IntegrityError("cluster values do not cover the exact shared seed set")
    values = [cluster_values[seed] for seed in seeds]
    estimate = mean(values)
    cluster_se = math.sqrt(
        sum((value - estimate) ** 2 for value in values)
        / (len(values) * (len(values) - 1))
    )
    leave_one_out = [
        mean(value for index, value in enumerate(values) if index != omitted)
        for omitted in range(len(values))
    ]
    return {
        "cluster_se": cluster_se,
        "t9_ci_low": estimate - T_CRITICAL_DF9_975 * cluster_se,
        "t9_ci_high": estimate + T_CRITICAL_DF9_975 * cluster_se,
        "leave_one_seed_low": min(leave_one_out),
        "leave_one_seed_high": max(leave_one_out),
    }


def _group_estimands(
    groups: dict[str, dict[tuple[str, str], dict[str, Any]]],
    metric: str,
    *,
    model_id: str | None = None,
    scenario_id: str | None = None,
) -> list[tuple[int, dict[str, float]]]:
    selected: list[tuple[int, dict[str, float]]] = []
    for cells in groups.values():
        identity = cells[("E0", "E0")]
        if model_id is not None and identity["model_id"] != model_id:
            continue
        if scenario_id is not None and identity["scenario_id"] != scenario_id:
            continue
        selected.append((int(identity["seed_int"]), factorial_values(cells, metric)))
    if not selected:
        raise IntegrityError("factorial scope selected no base pairs")
    return selected


def overall_factorial_rows(
    groups: dict[str, dict[tuple[str, str], dict[str, Any]]],
    seeds: list[int],
    weights: list[tuple[int, ...]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        selected = _group_estimands(groups, metric)
        by_seed: dict[int, list[dict[str, float]]] = defaultdict(list)
        for seed, values in selected:
            by_seed[seed].append(values)
        for estimand in ESTIMANDS:
            cluster_values = {
                seed: mean(row[estimand] for row in by_seed[seed]) for seed in seeds
            }
            estimate, low, high = cluster_interval(cluster_values, seeds, weights)
            rows.append(
                {
                    "metric": metric,
                    "estimand": estimand,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "seed_clusters": len(seeds),
                    "base_pairs": len(selected),
                    "bootstrap_draws": len(weights),
                    **cluster_sensitivity(cluster_values, seeds),
                }
            )
    return rows


def factorial_scope_rows(
    groups: dict[str, dict[tuple[str, str], dict[str, Any]]]
) -> list[dict[str, Any]]:
    identities = [cells[("E0", "E0")] for cells in groups.values()]
    models = sorted({str(row["model_id"]) for row in identities})
    scenarios = sorted({str(row["scenario_id"]) for row in identities})
    rows: list[dict[str, Any]] = []
    for model_id in models:
        for scenario_id in scenarios:
            for metric in ("total_return", "sharpe"):
                selected = _group_estimands(
                    groups, metric, model_id=model_id, scenario_id=scenario_id
                )
                for estimand in ESTIMANDS:
                    rows.append(
                        {
                            "model_id": model_id,
                            "scenario_id": scenario_id,
                            "metric": metric,
                            "estimand": estimand,
                            "estimate": mean(values[estimand] for _, values in selected),
                            "base_pairs": len(selected),
                            "seed_clusters": len({seed for seed, _ in selected}),
                        }
                    )
    return rows


def kendall_tau_b(first: dict[str, float], second: dict[str, float]) -> float:
    if set(first) != set(second) or len(first) < 2:
        raise ValueError("Kendall tau-b requires two mappings over the same keys")
    keys = sorted(first)
    concordant = discordant = ties_first = ties_second = 0
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            delta_first = first[left] - first[right]
            delta_second = second[left] - second[right]
            if delta_first == 0.0 and delta_second == 0.0:
                continue
            if delta_first == 0.0:
                ties_first += 1
            elif delta_second == 0.0:
                ties_second += 1
            elif delta_first * delta_second > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_first)
        * (concordant + discordant + ties_second)
    )
    return (concordant - discordant) / denominator if denominator else 0.0


def _ranking(values: dict[str, float]) -> str:
    return ">".join(sorted(values, key=lambda model: (-values[model], model)))


def _weighted_board(
    seed_model_means: dict[str, dict[int, dict[str, float]]],
    destination: str,
    draw: tuple[int, ...],
    seeds: list[int],
    models: list[str],
) -> dict[str, float]:
    return {
        model: sum(
            weight * seed_model_means[destination][seed][model]
            for seed, weight in zip(seeds, draw, strict=True)
        )
        / len(seeds)
        for model in models
    }


def ranking_rows(
    groups: dict[str, dict[tuple[str, str], dict[str, Any]]],
    seeds: list[int],
    weights: list[tuple[int, ...]],
) -> list[dict[str, Any]]:
    identities = [cells[("E0", "E0")] for cells in groups.values()]
    models = sorted({str(row["model_id"]) for row in identities})
    scenarios = sorted({str(row["scenario_id"]) for row in identities})
    rows: list[dict[str, Any]] = []
    for scenario_id in scenarios:
        for origin in EXECUTION_LEVELS:
            by_destination_seed_model: dict[str, dict[int, dict[str, list[float]]]] = {
                destination: defaultdict(lambda: defaultdict(list))
                for destination in EXECUTION_LEVELS
            }
            for cells in groups.values():
                identity = cells[(origin, "E0")]
                if identity["scenario_id"] != scenario_id:
                    continue
                for destination in EXECUTION_LEVELS:
                    row = cells[(origin, destination)]
                    by_destination_seed_model[destination][int(row["seed_int"])][
                        str(row["model_id"])
                    ].append(float(row["metric_values"]["sharpe"]))
            seed_model_means: dict[str, dict[int, dict[str, float]]] = {
                destination: {
                    seed: {
                        model: mean(
                            by_destination_seed_model[destination][seed][model]
                        )
                        for model in models
                    }
                    for seed in seeds
                }
                for destination in EXECUTION_LEVELS
            }

            point_weight = tuple(1 for _ in seeds)
            board_e0 = _weighted_board(
                seed_model_means, "E0", point_weight, seeds, models
            )
            board_e1 = _weighted_board(
                seed_model_means, "E1", point_weight, seeds, models
            )
            bootstrap_tau: list[float] = []
            exact_order_matches = 0
            winner_matches = 0
            for draw in weights:
                draw_e0 = _weighted_board(seed_model_means, "E0", draw, seeds, models)
                draw_e1 = _weighted_board(seed_model_means, "E1", draw, seeds, models)
                bootstrap_tau.append(kendall_tau_b(draw_e0, draw_e1))
                exact_order_matches += int(_ranking(draw_e0) == _ranking(draw_e1))
                winner_matches += int(
                    _ranking(draw_e0).split(">", 1)[0]
                    == _ranking(draw_e1).split(">", 1)[0]
                )
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "response_origin": origin,
                    "metric": "sharpe",
                    "kendall_tau_b": kendall_tau_b(board_e0, board_e1),
                    "ci95_low": percentile(bootstrap_tau, 0.025),
                    "ci95_high": percentile(bootstrap_tau, 0.975),
                    "winner_e0": _ranking(board_e0).split(">", 1)[0],
                    "winner_e1": _ranking(board_e1).split(">", 1)[0],
                    "ranking_e0": _ranking(board_e0),
                    "ranking_e1": _ranking(board_e1),
                    "exact_order_probability": exact_order_matches / len(weights),
                    "same_winner_probability": winner_matches / len(weights),
                    "models": len(models),
                    "seed_clusters": len(seeds),
                    "bootstrap_draws": len(weights),
                }
            )
    return rows


def divergence_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: list[tuple[str, str, list[dict[str, str]]]] = [("overall", "all", rows)]
    for model_id in sorted({row["model_id"] for row in rows}):
        grouped.append(("model", model_id, [row for row in rows if row["model_id"] == model_id]))
    for scenario_id in sorted({row["scenario_id"] for row in rows}):
        grouped.append(
            ("scenario", scenario_id, [row for row in rows if row["scenario_id"] == scenario_id])
        )
    output: list[dict[str, Any]] = []
    for scope, value, selected in grouped:
        divergent_steps = [
            _int(row["first_parsed_response_divergence_step"], "divergence step")
            for row in selected
            if _int(row["first_parsed_response_divergence_step"], "divergence step") >= 0
        ]
        output.append(
            {
                "scope": scope,
                "value": value,
                "pairs": len(selected),
                "first_prompt_equal_rate": mean(
                    int(bool(_bool(row["first_prompt_equal"]))) for row in selected
                ),
                "first_response_hash_equal_rate": mean(
                    int(bool(_bool(row["first_response_hash_equal"]))) for row in selected
                ),
                "first_parsed_response_equal_rate": mean(
                    int(bool(_bool(row["first_parsed_response_equal"]))) for row in selected
                ),
                "full_response_path_equal_rate": mean(
                    int(bool(_bool(row["full_response_path_equal"]))) for row in selected
                ),
                "full_parsed_response_path_equal_rate": mean(
                    int(bool(_bool(row["full_parsed_response_path_equal"]))) for row in selected
                ),
                "full_pre_risk_decision_path_equal_rate": mean(
                    int(bool(row["_full_pre_risk_decision_path_equal"])) for row in selected
                ),
                "mean_first_parsed_divergence_step": (
                    mean(divergent_steps) if divergent_steps else -1.0
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise IntegrityError(f"refusing to write empty analysis table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _validate_output_allowlist(output_dir: Path, input_dir: Path) -> None:
    allowed = {
        "factorial_estimands.csv",
        "factorial_by_model_scenario.csv",
        "ranking_stability.csv",
        "response_path_divergence.csv",
        "fixed_response_analysis.md",
        "analysis_manifest.json",
    }
    if output_dir.resolve() == input_dir.resolve():
        allowed.update(
            {
                "analysis_plan.json",
                "source_integrity.json",
                "intent_paths.csv",
                "replay_rows.csv",
                "intent_divergence.csv",
            }
        )
    entries = list(output_dir.iterdir()) if output_dir.is_dir() else []
    unexpected = sorted(path.name for path in entries if not path.is_file() or path.name not in allowed)
    if unexpected:
        raise IntegrityError(f"analysis output directory has unexpected entries: {unexpected}")


def _markdown(
    factorial: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    divergence: list[dict[str, Any]],
    inactivity: tuple[int, int],
) -> str:
    return_rows = {
        str(row["estimand"]): row
        for row in factorial
        if row["metric"] == "total_return"
    }
    overall = next(row for row in divergence if row["scope"] == "overall")
    lines = [
        "# Fixed-response two-by-two replay analysis",
        "",
        "The source gate fixes each raw provider response tape and verifies identical pre-risk decisions across replay destinations. Downstream risk approval and fills remain execution-endogenous. Response-origin contrasts are descriptive, not execution-randomized.",
        "",
        "## Overall total-return decomposition",
        "",
        "| Estimand | Estimate | Seed-cluster 95% CI |",
        "|---|---:|---:|",
    ]
    for name in ESTIMANDS:
        row = return_rows[name]
        lines.append(
            f"| {name} | {float(row['estimate']):+.6f} | "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] |"
        )
    lines.extend(
        [
            "",
            "Shared-seed bootstrap: 10 clusters, 10,000 resamples, seed 20260719, using CPython random.Random (MT19937). Each replicate makes 10 randrange(10) draws with replacement and retains every model, scenario, provider sample, response origin, and replay destination within each selected seed. CIs use Hyndman-Fan type 7 linear percentiles at 2.5% and 97.5%.",
            "",
            "## Source-arm divergence diagnostic",
            "",
            f"All 450 pairs share the first prompt. The first raw response hash agrees in {float(overall['first_response_hash_equal_rate']):.1%}, the first parsed response in {float(overall['first_parsed_response_equal_rate']):.1%}, the full parsed-response path in {float(overall['full_parsed_response_path_equal_rate']):.1%}, and the full pre-risk decision path in {float(overall['full_pre_risk_decision_path_equal_rate']):.1%}. The original diagonal E0/E1 contrast therefore cannot be called execution-only.",
            "",
            "## Sharpe ranking stability under fixed response tapes",
            "",
            "| Scenario | Response origin | tau-b(E0,E1) | 95% CI | Exact-order p | Winner E0 -> E1 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in rankings:
        lines.append(
            f"| {row['scenario_id']} | {row['response_origin']} | "
            f"{float(row['kendall_tau_b']):+.3f} | "
            f"[{float(row['ci95_low']):+.3f}, {float(row['ci95_high']):+.3f}] | "
            f"{float(row['exact_order_probability']):.3f} | "
            f"{row['winner_e0']} -> {row['winner_e1']} |"
        )
    lines.extend(
        [
            "",
            f"Ranking caveat: deepseek-v4-pro is inactive (zero return, all-hold, zero gross target exposure) in {inactivity[0]}/{inactivity[1]} replay rows; inactivity must not be interpreted as robustness.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(input_dir: Path, analysis_plan_path: Path, output_dir: Path) -> dict[str, Any]:
    groups, divergence_input, source_integrity = load_frozen_replay(
        input_dir, analysis_plan_path
    )
    seeds = sorted({cells[("E0", "E0")]["seed_int"] for cells in groups.values()})
    weights = bootstrap_weights(seeds, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    factorial = overall_factorial_rows(groups, seeds, weights)
    scoped = factorial_scope_rows(groups)
    rankings = ranking_rows(groups, seeds, weights)
    divergence = divergence_rows(divergence_input)
    pro_rows = [
        cell
        for cells in groups.values()
        for cell in cells.values()
        if cell["model_id"] == "deepseek-v4-pro"
    ]
    inactive_pro_rows = sum(
        cell["metric_values"]["total_return"] == 0.0
        and _finite(cell["hold_ratio"], "hold_ratio") == 1.0
        and _finite(cell["mean_gross_target_exposure"], "mean_gross_target_exposure") == 0.0
        for cell in pro_rows
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_output_allowlist(output_dir, input_dir)
    with tempfile.TemporaryDirectory(prefix="v03_fixed_response_analysis_", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        csv_findings = scan_public_artifact_payload(
            {
                "factorial_estimands": factorial,
                "factorial_by_model_scenario": scoped,
                "ranking_stability": rankings,
                "response_path_divergence": divergence,
            }
        )
        if csv_findings:
            raise IntegrityError("analysis CSV privacy scan failed: " + csv_findings[0])
        _write_csv(stage / "factorial_estimands.csv", factorial)
        _write_csv(stage / "factorial_by_model_scenario.csv", scoped)
        _write_csv(stage / "ranking_stability.csv", rankings)
        _write_csv(stage / "response_path_divergence.csv", divergence)
        _write_text_lf(
            stage / "fixed_response_analysis.md",
            _markdown(
                factorial,
                rankings,
                divergence,
                (inactive_pro_rows, len(pro_rows)),
            ),
        )
        output_names = (
            "factorial_estimands.csv",
            "factorial_by_model_scenario.csv",
            "ranking_stability.csv",
            "response_path_divergence.csv",
            "fixed_response_analysis.md",
        )
        manifest = {
            "schema_version": ANALYSIS_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "analysis_plan_sha256": _sha256(analysis_plan_path),
            "source_integrity_sha256": _sha256(input_dir / "source_integrity.json"),
            "source_output_sha256": source_integrity["output_sha256"],
            "analyzer_sha256": _sha256_lf_text(Path(__file__)),
            "analyzer_hash_policy": SOURCE_HASH_POLICY,
            "bootstrap": {
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "clusters": seeds,
                "unit": "shared source seed",
                "rng_algorithm": BOOTSTRAP_RNG_ALGORITHM,
                "draw_method": BOOTSTRAP_DRAW_METHOD,
                "weight_matrix_sha256": sha256_text(canonical_json(weights)),
                "percentile_method": PERCENTILE_METHOD,
                "percentile_formula": PERCENTILE_FORMULA,
                "percentile_probabilities": [0.025, 0.975],
            },
            "observed": {
                "base_pairs": len(groups),
                "replay_rows": sum(len(cells) for cells in groups.values()),
                "factorial_rows": len(factorial),
                "scoped_factorial_rows": len(scoped),
                "ranking_rows": len(rankings),
                "divergence_summary_rows": len(divergence),
            },
            "claim_boundary": {
                "execution": "response tape fixed and pre-risk decision path verified fixed",
                "downstream": "risk approval and fills remain execution-endogenous",
                "response_origin": "descriptive realized-path contrast",
            },
            "output_sha256": {name: _sha256(stage / name) for name in output_names},
            "failed_count": 0,
        }
        _write_text_lf(
            stage / "analysis_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        findings = scan_public_artifact_paths([stage])
        if findings:
            raise IntegrityError("analysis privacy scan failed: " + findings[0])
        for name in output_names:
            os.replace(stage / name, output_dir / name)
        os.replace(stage / "analysis_manifest.json", output_dir / "analysis_manifest.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze the v0.3 fixed-response two-by-two replay.")
    parser.add_argument("--input-dir", default="docs/results/v0_3_fixed_intent_replay")
    parser.add_argument(
        "--analysis-plan",
        default="docs/results/v0_3_fixed_intent_replay/analysis_plan.json",
    )
    parser.add_argument("--output-dir", default="docs/results/v0_3_fixed_intent_replay")
    args = parser.parse_args(argv)
    input_dir = Path(args.input_dir)
    analysis_plan_path = Path(args.analysis_plan)
    output_dir = Path(args.output_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    if not analysis_plan_path.is_absolute():
        analysis_plan_path = ROOT / analysis_plan_path
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    try:
        manifest = analyze(input_dir, analysis_plan_path, output_dir)
    except (
        FileNotFoundError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"INTEGRITY ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"analyzed {manifest['observed']['replay_rows']} replay rows with "
        f"{manifest['bootstrap']['draws']} shared-seed bootstrap draws -> {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
