"""Required evidence grids and descriptive checks for the TMLR A+B artifact.

No collection, network calls, or changes to the frozen primary analysis plan.
Legacy tables start from structured findings, not withheld response text.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("deepseek:deepseek-v4-pro", "glm:glm-5")
PRODUCERS = ("deepseek_v4_pro", "glm_5_direct")
LEGACY_FIELDS = ("model", "task_id", "sample", "findings", "true_positives")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def legacy_specs() -> list[tuple[str, str, int, int]]:
    """(truth, results, number of truth tasks, samples per model/task)."""
    specs = []
    for producer in PRODUCERS:
        for arm in ("v2", "constraint_v2", "cot_v2", "sc_v2"):
            specs.append((
                f"outputs/audit_pairs/{producer}/ground_truth.jsonl",
                f"outputs/audit_pairs_eval_{arm}/{producer}/audit_eval_results.jsonl",
                60, 3 if arm == "sc_v2" else 1,
            ))
        specs.append((
            f"outputs/audit_self/{producer}/ground_truth.jsonl",
            f"outputs/audit_self_eval_v2/{producer}/audit_eval_results.jsonl", 120, 1,
        ))
    specs.append(("outputs/toolaudit_pairs/ground_truth.jsonl",
                  "outputs/toolaudit_pairs_eval_v2/toolaudit_eval_results.jsonl", 80, 1))
    return specs


def validate_legacy(root: Path, *, allow_unreleased_models: bool = False) -> int:
    """No implicit intersections, missing cells, duplicate samples, or trusted hit flags."""
    count = 0
    for truth_name, results_name, task_count, samples in legacy_specs():
        truth_rows = read_jsonl(root / truth_name)
        truth = {r["task_id"]: r for r in truth_rows}
        if len(truth) != len(truth_rows) or len(truth) != task_count:
            raise ValueError(f"Wrong/duplicate truth grid: {truth_name}")
        expected = {(m, t, s) for m in MODELS for t in truth for s in range(samples)}
        seen = set()
        for row in read_jsonl(root / results_name):
            if row["model"] not in MODELS:
                if allow_unreleased_models:
                    continue
                raise ValueError(f"Unexpected model in released legacy rows: {results_name}")
            key = (row["model"], row["task_id"], row.get("sample", 0))
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected/duplicate legacy key: {results_name}: {key}")
            seen.add(key)
            target = truth[row["task_id"]]
            hit = any(
                isinstance(f.get("step_index"), int)
                and f["step_index"] == target["step_index"] and f.get("kind") == target["kind"]
                for f in row["findings"]
            )
            if int(hit) != int(row["true_positives"]):
                raise ValueError(f"Legacy stored hit does not match findings: {key}")
        if seen != expected:
            raise ValueError(f"Missing legacy keys: {results_name}: {len(expected - seen)}")
        count += len(seen)
    return count


def diagnostics(root: Path) -> list[dict[str, Any]]:
    """Post-hoc descriptive decomposition; no additional hypothesis tests."""
    truth = {r["task_id"]: r for r in read_jsonl(
        root / "outputs/audit_multilabel_tasks/ground_truth.jsonl")}
    results = read_jsonl(root / "outputs/audit_multilabel_eval/multilabel_audit_results.jsonl")
    output = []
    for model in MODELS:
        counts: Counter[str] = Counter()
        for row in results:
            if row["model"] != model:
                continue
            target = truth[row["task_id"]]
            pred = [(f["step_index"], f["kind"]) for f in row["findings"]]
            if target["condition"] == "zero":
                counts["zero_findings"] += len(pred)
            if target["condition"] != "violation_plus_edit":
                continue
            counts["dual_tasks"] += 1
            gold = {(f["step_index"], f["kind"]) for f in target["defects"]}
            if gold <= set(pred):
                counts["both_targets"] += 1
                counts["both_with_extra_findings"] += len(pred) > len(gold)
                counts["both_parse_failures"] += not row["parse_ok"]
                counts["exact_dual_sets"] += row["parse_ok"] and len(pred) == len(gold)
        output.append({"model": model, **{k: counts[k] for k in (
            "dual_tasks", "both_targets", "both_with_extra_findings", "both_parse_failures",
            "exact_dual_sets", "zero_findings",
        )}})
    return output


def legacy_quantity_counterexample(root: Path) -> int:
    count = 0
    for producer in PRODUCERS:
        truth = {r["task_id"]: r for r in read_jsonl(
            root / f"outputs/audit_pairs/{producer}/ground_truth.jsonl")}
        for row in read_jsonl(root / f"outputs/audit_pairs_eval_v2/{producer}/audit_eval_results.jsonl"):
            target = truth[row["task_id"]]
            if row["model"] != MODELS[0] or target["detail"]["ambiguity"] != "confounded":
                continue
            labels = {f["kind"] for f in row["findings"]
                      if f["step_index"] == target["step_index"]
                      and f["kind"] in {"unclipped_position", "silent_risk_edit"}}
            count += len(labels) == 1 and len(row["findings"]) > 1
    return count


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    count = validate_legacy(ROOT, allow_unreleased_models=not (ROOT / "artifact_manifest.json").exists())
    import build_corrected_key_rescore
    import build_intervention_v2

    if build_corrected_key_rescore.main() or build_intervention_v2.main():
        raise ValueError("Legacy reanalysis failed")
    rows = diagnostics(ROOT)
    expected = [(78, 57, 21, 850), (23, 16, 7, 378)]
    for row, values in zip(rows, expected, strict=True):
        observed = tuple(row[k] for k in (
            "both_targets", "both_with_extra_findings", "exact_dual_sets", "zero_findings"))
        if observed != values or row["dual_tasks"] != 100 or row["both_parse_failures"] != 0:
            raise ValueError("Descriptive decomposition differs from manuscript")
    if legacy_quantity_counterexample(ROOT) != 4:
        raise ValueError("Legacy quantity-cue counterexample count differs")
    target = ROOT / "docs/results/finaudit_multilabel/coverage_decomposition.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {count} legacy model/task/sample keys; descriptive decomposition reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
