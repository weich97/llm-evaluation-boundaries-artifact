"""Intervention arms and the self-audit control, on neutral-prompt (v2) runs.

The earlier tables under docs/results/finaudit/ (ablation_family.csv,
paired_intervention.csv, self_bias_matrix.csv) were built from runs collected
under the trading-analyst system prompt and from a four-producer matrix that no
longer exists. They are superseded and must not be cited: this script recomputes
the same quantities from the v2 trees, which are the ones the report uses.

Two blocks:

1. Interventions. On the matched trading pairs, confounded-arm violation recall
   under the default prompt versus each modified prompt (constraint, cot,
   self-consistency), with a within-pair exact McNemar against the default arm.
   The self-consistency arm carries three samples per task and is aggregated by
   the specified majority rule; the any-of-three alternative is reported
   alongside because it inflates recall mechanically and the difference between
   the two is worth seeing.

2. Self-audit. Recall on each auditor's own producer set versus the other's,
   with a Fisher exact test and the design's approximate minimum detectable
   effect, since a null at this n is a statement about resolution.

Writes docs/results/finaudit/intervention_v2.csv and self_audit_v2.csv.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/results/finaudit"
PRODUCERS = ("deepseek_v4_pro", "glm_5_direct")
MODELS = ("deepseek:deepseek-v4-pro", "glm:glm-5")
ARMS = {
    "default": "audit_pairs_eval_v2",
    "constraint": "audit_pairs_eval_constraint_v2",
    "cot": "audit_pairs_eval_cot_v2",
    "selfcons": "audit_pairs_eval_sc_v2",
}
VIOLATION_KIND = "unclipped_position"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant counts."""

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""

    def weight(x: int) -> float:
        return (
            math.comb(a + b, x)
            * math.comb(c + d, a + c - x)
            / math.comb(a + b + c + d, a + c)
        )

    observed = weight(a)
    lo = max(0, a + c - (c + d))
    hi = min(a + b, a + c)
    return min(1.0, sum(weight(x) for x in range(lo, hi + 1) if weight(x) <= observed * (1 + 1e-9)))


def load_arm(tree: str, model: str, ambiguity: str, vote: str | None = None) -> dict[str, bool]:
    """source key -> hit, for one arm/model/arm-of-pair."""

    samples: dict[str, list[bool]] = defaultdict(list)
    for producer in PRODUCERS:
        truth_path = ROOT / f"outputs/audit_pairs/{producer}/ground_truth.jsonl"
        results_path = ROOT / f"outputs/{tree}/{producer}/audit_eval_results.jsonl"
        if not (truth_path.exists() and results_path.exists()):
            continue
        truth = {
            json.loads(line)["task_id"]: json.loads(line)
            for line in truth_path.open(encoding="utf-8")
            if line.strip()
        }
        for line in results_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["model"] != model:
                continue
            entry = truth.get(record["task_id"])
            if not entry or entry["kind"] != VIOLATION_KIND:
                continue
            if entry["detail"].get("ambiguity") != ambiguity:
                continue
            key = f"{producer}:{entry['detail']['pair_id']}"
            samples[key].append(int(record.get("true_positives", 0)) > 0)
    if vote == "majority":
        return {k: sum(v) * 2 > len(v) for k, v in samples.items()}
    if vote == "any":
        return {k: any(v) for k, v in samples.items()}
    return {k: v[0] for k, v in samples.items()}


def interventions() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        base = load_arm(ARMS["default"], model, "confounded")
        for arm, tree in ARMS.items():
            votes = ["majority", "any"] if arm == "selfcons" else [None]
            for vote in votes:
                arm_hits = load_arm(tree, model, ambiguity="confounded", vote=vote)
                shared = sorted(set(base) & set(arm_hits))
                if not shared:
                    continue
                fixed = sum(1 for k in shared if arm_hits[k] and not base[k])
                regressed = sum(1 for k in shared if base[k] and not arm_hits[k])
                clean = load_arm(tree, model, ambiguity="clean", vote=vote)
                rows.append(
                    {
                        "auditor": model,
                        "arm": arm if vote is None else f"{arm}_{vote}",
                        "n_pairs": len(shared),
                        "default_confounded_recall": round(
                            sum(base[k] for k in shared) / len(shared), 3
                        ),
                        "arm_confounded_recall": round(
                            sum(arm_hits[k] for k in shared) / len(shared), 3
                        ),
                        "arm_clean_recall": round(sum(clean.values()) / len(clean), 3) if clean else "",
                        "fixed": fixed,
                        "regressed": regressed,
                        "mcnemar_p": "" if arm == "default" else f"{mcnemar_exact(regressed, fixed):.3e}",
                    }
                )
    return rows


def self_audit() -> list[dict[str, object]]:
    """Recall on own vs other producer's task set, under the neutral prompt."""

    own = {"deepseek:deepseek-v4-pro": "deepseek_v4_pro", "glm:glm-5": "glm_5_direct"}
    cells: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for producer in PRODUCERS:
        truth_path = ROOT / f"outputs/audit_self/{producer}/ground_truth.jsonl"
        results_path = ROOT / f"outputs/audit_self_eval_v2/{producer}/audit_eval_results.jsonl"
        if not (truth_path.exists() and results_path.exists()):
            continue
        truth = {
            json.loads(line)["task_id"]
            for line in truth_path.open(encoding="utf-8")
            if line.strip()
        }
        for line in results_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["model"] not in MODELS or record["task_id"] not in truth:
                continue
            if int(record.get("sample", 0)) != 0:
                continue
            cells[(record["model"], producer)].append(int(record.get("true_positives", 0)) > 0)

    rows: list[dict[str, object]] = []
    for model in MODELS:
        self_hits = cells.get((model, own[model]), [])
        cross_producer = next(p for p in PRODUCERS if p != own[model])
        cross_hits = cells.get((model, cross_producer), [])
        if not self_hits or not cross_hits:
            continue
        a, b = sum(self_hits), len(self_hits) - sum(self_hits)
        c, d = sum(cross_hits), len(cross_hits) - sum(cross_hits)
        self_rate, cross_rate = a / len(self_hits), c / len(cross_hits)
        pooled = (a + c) / (len(self_hits) + len(cross_hits))
        # Rough two-proportion MDE at alpha .05, 80% power; ignores task-level
        # clustering, so it is a floor rather than an exact figure.
        mde = (1.96 + 0.84) * math.sqrt(2 * pooled * (1 - pooled) / len(self_hits))
        rows.append(
            {
                "auditor": model,
                "n_per_cell": len(self_hits),
                "self_recall": round(self_rate, 3),
                "cross_recall": round(cross_rate, 3),
                "gap_cross_minus_self": round(cross_rate - self_rate, 3),
                "fisher_p": f"{fisher_exact_2x2(a, b, c, d):.3f}",
                "approx_mde": round(mde, 3),
            }
        )
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def main() -> int:
    intervention_rows = interventions()
    audit_rows = self_audit()
    if not intervention_rows or not audit_rows:
        print("missing v2 inputs")
        return 1
    _write(OUT_DIR / "intervention_v2.csv", intervention_rows)
    _write(OUT_DIR / "self_audit_v2.csv", audit_rows)
    for row in intervention_rows:
        print(
            f"  {row['auditor']:26s}{row['arm']!s:18s}"
            f"{row['default_confounded_recall']} -> {row['arm_confounded_recall']}"
            f"  fixed={row['fixed']} regr={row['regressed']} p={row['mcnemar_p']}"
        )
    for row in audit_rows:
        print(
            f"  {row['auditor']:26s}self={row['self_recall']} cross={row['cross_recall']} "
            f"gap={row['gap_cross_minus_self']:+} p={row['fisher_p']} mde={row['approx_mde']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
