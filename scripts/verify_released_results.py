"""Executable consistency checks over the released result artifacts.

Each check recomputes a published quantity from the data under
``docs/results/`` (or, where the input is a run artifact that is not in version
control, says so and skips) and compares it with the value the released tables
and figures carry. A mismatch is an error, not a warning.

The point is not that a number was wrong once. It is that analysis code and the
artifacts under it drift silently: an arm gets re-collected, an estimator is
corrected, a generator is regenerated, and a table that was true stops being
true without anything failing. This file is the standing check.

Scope, stated honestly: this covers the quantities that carry an argument, not
every number in every CSV.

A missing input is reported as a skip, never as a disagreement. On a clean
clone the run directory holding raw provider responses is absent, so some
checks cannot run; they name the input they need.

Usage:
    python scripts/verify_released_results.py            # all checks
    python scripts/verify_released_results.py --study B  # one study
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class ClaimFailure(AssertionError):
    """Raised when a registered claim does not match its source."""


class DataUnavailable(Exception):
    """Raised when a check's input is not in this checkout.

    Distinct from ClaimFailure on purpose. The run directory holding raw
    provider responses is outside version control, so on a public checkout some
    inputs are simply absent. Treating that as a failed claim would be wrong,
    and treating a missing file as an empty one -- which is what the previous
    ``return []`` did -- would be worse: it turns "I could not check" into "I
    checked and the data disagrees".
    """


def _require_dir(path: Path) -> Path:
    """A directory a check reads by globbing, rather than by loading one file."""

    if not path.is_dir():
        raise DataUnavailable(str(path.relative_to(ROOT)) + "/")
    return path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DataUnavailable(str(path.relative_to(ROOT)))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise DataUnavailable(str(path.relative_to(ROOT)))
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _close(actual: float, expected: float, tol: float = 5e-4) -> bool:
    return abs(actual - expected) <= tol


def _mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


# --------------------------------------------------------------------------
# Study A
# --------------------------------------------------------------------------

def check_matrix_completeness() -> tuple[str, str]:
    rows = _load_csv(ROOT / "outputs/v0_3_direct_api_matrix/direct_api_submission_runs.csv")
    ok = sum(1 for r in rows if r["status"] == "ok")
    models = len({r["model_id"] for r in rows})
    if len(rows) != 900 or ok != 900:
        raise ClaimFailure(f"expected 900 runs all ok, got {len(rows)} rows / {ok} ok")
    if models != 5:
        raise ClaimFailure(f"expected 5 models, got {models}")
    return "900 runs, all ok, 5 models", f"{len(rows)} rows, {ok} ok, {models} models"


def check_matrix_temperature() -> tuple[str, str]:
    """The released matrix uses one sampling envelope, temperature 0.2, across the matrix."""

    temps: set[str] = set()
    manifests = _require_dir(ROOT / "outputs/v0_3_direct_api_matrix/provider_manifests")
    for path in sorted(manifests.glob("*.json")):
        found = re.findall(r'"temperature"\s*:\s*([0-9.]+)', path.read_text(encoding="utf-8"))
        temps.update(found)
    if temps != {"0.2"}:
        raise ClaimFailure(f"expected only temperature 0.2, found {sorted(temps)}")
    return "temperature 0.2 across the matrix", f"manifest temperatures {sorted(temps)}"


def check_execution_decomposition() -> tuple[str, str]:
    rows = _load_csv(ROOT / "docs/results/v0_3_fixed_intent_replay/factorial_estimands.csv")
    wanted = {
        "execution_shapley": (-0.0170, "$-0.0170$"),
        "response_origin_shapley": (0.0035, "$+0.0035$"),
        "observed_diagonal": (-0.0135, "$-0.0135$"),
    }
    seen: dict[str, float] = {}
    for row in rows:
        metric = row.get("metric") or row.get("outcome") or ""
        name = row.get("estimand") or row.get("name") or ""
        if metric != "total_return":
            continue
        if name in wanted:
            seen[name] = float(row.get("estimate") or row.get("value"))
    missing = set(wanted) - set(seen)
    if missing:
        raise ClaimFailure(f"estimands absent from CSV: {sorted(missing)}")
    for name, (expected, printed) in wanted.items():
        if not _close(seen[name], expected, tol=5e-4):
            raise ClaimFailure(f"{name}: released value {printed}, data says {seen[name]:+.4f}")
    return (
        "exec -0.0170, origin +0.0035, diagonal -0.0135",
        ", ".join(f"{k} {v:+.4f}" for k, v in sorted(seen.items())),
    )


def check_forward_window_declaration() -> tuple[str, str]:
    """The pre-registered window the report names but draws no result from.

    The report tells a reader the declaration exists, is hash-committed, and can
    be checked. That is a claim about a file, so it is asserted like any other:
    the file is present, its recorded digest matches its own contents, and the
    window it governs is the one the report states.
    """

    path = ROOT / "docs/results/forward_window_commitment_2026q3.json"
    if not path.exists():
        raise DataUnavailable(str(path.relative_to(ROOT)))
    declaration = json.loads(path.read_text(encoding="utf-8"))

    # The project already defines which fields the commitment covers; recomputing
    # that here would be a second implementation of the thing being verified.
    sys.path.insert(0, str(ROOT / "scripts"))
    from freeze_forward_window import declaration_hash

    recorded = str(declaration.get("declaration_hash", ""))
    recomputed = declaration_hash(declaration)
    if recorded != recomputed:
        raise ClaimFailure(
            f"declaration digest does not verify: recorded {recorded}, recomputed {recomputed}"
        )

    expected = {
        "window_start": "2026-07-01",
        "window_end": "2026-09-15",
        "rank_metric": "sharpe",
        "execution_mode": "realistic-stress",
        "frequency": "weekly",
    }
    for key, want in expected.items():
        if declaration.get(key) != want:
            raise ClaimFailure(f"{key}: released {want!r}, declaration {declaration.get(key)!r}")
    symbols = list(declaration.get("symbols") or [])
    if symbols != ["BTC-USD", "BTC=F", "GSPC"]:
        raise ClaimFailure(f"symbols: released BTC-USD/BTC=F/GSPC, declaration {symbols}")

    return (
        "forward window 2026-07-01..2026-09-15 declared and digest-verifying",
        f"{recorded[:23]}..., {len(symbols)} symbols",
    )


def check_robustness_intervals() -> tuple[str, str]:
    """The t9 and leave-one-seed columns quoted alongside the bootstrap.

    These sit in the CSV next to the headline interval and are easy to quote
    without recomputing. The leave-one-seed range for the tape-origin term
    excludes zero while its bootstrap interval does not, so if that pair ever
    stops holding, the paragraph explaining the difference is wrong too.
    """

    rows = _load_csv(ROOT / "docs/results/v0_3_fixed_intent_replay/factorial_estimands.csv")
    wanted = {
        ("total_return", "execution_shapley"): {
            "cluster_se": 0.0031,
            "t9": (-0.0239, -0.0101),
            "loo": (-0.0181, -0.0152),
        },
        ("sharpe", "response_origin_shapley"): {
            "t9": (0.3228, 1.3039),
        },
        ("total_return", "response_origin_shapley"): {
            "ci": (-0.0061, 0.0136),
            "loo": (0.0005, 0.0061),
        },
    }
    seen: dict[tuple[str, str], dict[str, str]] = {
        (row["metric"], row["estimand"]): row for row in rows
    }
    for key, expectations in wanted.items():
        row = seen.get(key)
        if row is None:
            raise ClaimFailure(f"{key} absent from the estimands CSV")
        for field, expected in expectations.items():
            if field == "cluster_se":
                if not _close(float(row["cluster_se"]), expected, tol=5e-5):
                    raise ClaimFailure(f"{key} cluster SE: released {expected}, data {row['cluster_se']}")
                continue
            prefix = {"t9": ("t9_ci_low", "t9_ci_high"),
                      "loo": ("leave_one_seed_low", "leave_one_seed_high"),
                      "ci": ("ci95_low", "ci95_high")}[field]
            low, high = float(row[prefix[0]]), float(row[prefix[1]])
            if not _close(low, expected[0], tol=5e-4) or not _close(high, expected[1], tol=5e-4):
                raise ClaimFailure(
                    f"{key} {field}: released [{expected[0]}, {expected[1]}], "
                    f"data [{low:.4f}, {high:.4f}]"
                )

    origin = seen[("total_return", "response_origin_shapley")]
    bootstrap_spans_zero = float(origin["ci95_low"]) < 0 < float(origin["ci95_high"])
    loo_excludes_zero = float(origin["leave_one_seed_low"]) > 0
    if not (bootstrap_spans_zero and loo_excludes_zero):
        raise ClaimFailure(
            "the released text explains a bootstrap that spans zero beside a "
            "leave-one-seed range that does not; the data no longer shows that pattern"
        )

    return (
        "t9 and leave-one-seed as released; tape-origin bootstrap spans zero, leave-one-seed does not",
        "all four robustness columns match",
    )


def check_path_divergence() -> tuple[str, str]:
    """19.8% full-path agreement, and every agreeing pair is the inactive model."""

    rows = _load_csv(ROOT / "docs/results/v0_3_fixed_intent_replay/response_path_divergence.csv")
    overall = [r for r in rows if (r.get("scope") or r.get("group") or "overall") in ("overall", "")]
    rate = None
    for row in overall:
        for key, value in row.items():
            if "decision" in key and "rate" in key and value:
                rate = float(value)
    if rate is None:
        for row in rows:
            for key, value in row.items():
                if "decision_path" in key and "rate" in key and value:
                    rate = float(value)
                    break
    if rate is None:
        raise ClaimFailure("no full pre-risk decision-path agreement rate in CSV")
    if not _close(rate, 0.1978, tol=1e-3):
        raise ClaimFailure(f"released value is 19.8%, data says {rate:.4%}")
    return "19.8% decision-path agreement", f"{rate:.4%}"


def check_ranking_stability() -> tuple[str, str]:
    """The ranking table's six cells: tau_b, exact-order probability, and the single flip.

    These numbers used to exist only as pixels inside the Study A figure, which
    put them beyond both citation and this gate. They are asserted here against the CSV that produced them.
    """

    rows = _load_csv(ROOT / "docs/results/v0_3_fixed_intent_replay/ranking_stability.csv")
    sharpe = [row for row in rows if row.get("metric") == "sharpe"]
    if len(sharpe) != 6:
        raise ClaimFailure(f"expected 6 scenario x tape-origin cells, found {len(sharpe)}")

    taus = sorted(float(row["kendall_tau_b"]) for row in sharpe)
    if taus.count(1.0) != 5 or not _close(taus[0], 0.8):
        raise ClaimFailure(f"released tau_b is five cells at 1.000 and one at 0.800, data says {taus}")

    probabilities = [float(row["exact_order_probability"]) for row in sharpe]
    if not _close(min(probabilities), 0.2608) or not _close(max(probabilities), 0.7635):
        raise ClaimFailure(
            "released exact-order probability spans 0.26 to 0.76, data spans "
            f"{min(probabilities):.4f} to {max(probabilities):.4f}"
        )

    flips = [row for row in sharpe if row["winner_e0"] != row["winner_e1"]]
    if len(flips) != 1:
        raise ClaimFailure(f"released results record exactly one winner flip, data has {len(flips)}")
    flip = flips[0]
    if "jump_tail" not in flip["scenario_id"] or flip["response_origin"] != "E1":
        raise ClaimFailure(f"released results place the flip at jump-tail/E1, data has {flip['scenario_id']}/{flip['response_origin']}")
    if flip["winner_e0"] != "glm-5.2" or flip["winner_e1"] != "glm-5":
        raise ClaimFailure(f"released winner flip is glm-5.2 -> glm-5, data says {flip['winner_e0']} -> {flip['winner_e1']}")

    return (
        "5x tau_b 1.000 + 1x 0.800; exact-order 0.2608-0.7635; one flip at jump-tail/E1",
        f"taus {taus}; probabilities {sorted(probabilities)}",
    )


def check_inactive_row() -> tuple[str, str]:
    """The 358-of-360 all-hold rows that inflate tau_b, and that only one model does it."""

    rows = _load_csv(ROOT / "docs/results/v0_3_fixed_intent_replay/replay_rows.csv")
    idle: dict[str, int] = {}
    total: dict[str, int] = {}
    for row in rows:
        model = row["model_id"]
        total[model] = total.get(model, 0) + 1
        if (
            float(row["total_return"]) == 0.0
            and float(row["mean_gross_target_exposure"]) == 0.0
            and float(row["hold_ratio"]) == 1.0
        ):
            idle[model] = idle.get(model, 0) + 1

    if total.get("deepseek-v4-pro") != 360:
        raise ClaimFailure(f"released results have 360 replay rows per model, data says {total.get('deepseek-v4-pro')}")
    if idle.get("deepseek-v4-pro") != 358:
        raise ClaimFailure(f"released results report 358 inactive rows, data says {idle.get('deepseek-v4-pro')}")
    others = {model: count for model, count in idle.items() if model != "deepseek-v4-pro"}
    if others:
        raise ClaimFailure(f"released results attribute inactivity to one model, data also has {others}")

    return "deepseek-v4-pro inactive in 358/360 rows, alone", f"{idle} of {total}"


# --------------------------------------------------------------------------
# Study B
# --------------------------------------------------------------------------

def check_legacy_prompts_assert_cardinality() -> tuple[str, str]:
    """Verify literal quantity cues, not a causal interpretation of those cues."""

    trading = list(_require_dir(ROOT / "outputs/audit_pairs").glob("*/tasks/*/prompt.md"))
    tooluse = list(_require_dir(ROOT / "outputs/toolaudit_pairs/tasks").glob("*/prompt.md"))
    t_hits = sum(1 for p in trading if "Exactly 1 record" in p.read_text(encoding="utf-8"))
    u_hits = sum(
        1 for p in tooluse
        if "Exactly one step contains a single injected defect" in p.read_text(encoding="utf-8")
    )
    if not trading or t_hits != len(trading):
        raise ClaimFailure(f"trading prompts asserting cardinality: {t_hits}/{len(trading)}")
    if not tooluse or u_hits != len(tooluse):
        raise ClaimFailure(f"tool-use prompts asserting cardinality: {u_hits}/{len(tooluse)}")
    if (len(trading), len(tooluse)) != (120, 80):
        raise ClaimFailure("legacy prompt grid must contain 120 trading and 80 tool-use tasks")
    return "120 one-record cues; 80 single-defect cues", f"{t_hits}/{len(trading)}, {u_hits}/{len(tooluse)}"


def check_frozen_prompt_is_neutral() -> tuple[str, str]:
    """And this: the frozen corpus prompt does not state the answer count."""

    prompts = list(_require_dir(ROOT / "outputs/audit_multilabel_tasks/tasks").glob("*/prompt.md"))
    if not prompts:
        raise ClaimFailure("frozen corpus prompts not found")
    neutral = sum(
        1 for p in prompts
        if "zero, one, or multiple defects" in p.read_text(encoding="utf-8")
    )
    asserting = sum(1 for p in prompts if "Exactly 1 record" in p.read_text(encoding="utf-8"))
    if asserting:
        raise ClaimFailure(f"{asserting} frozen prompts still assert a single defect")
    if neutral != len(prompts):
        raise ClaimFailure(f"only {neutral}/{len(prompts)} frozen prompts are cardinality-neutral")
    return "frozen prompts are cardinality-neutral", f"{neutral}/{len(prompts)} neutral, 0 asserting"


def check_frozen_grid_and_gate() -> tuple[str, str]:
    rows = _load_jsonl(ROOT / "outputs/audit_multilabel_eval/multilabel_audit_results.jsonl")
    keys = {(r["model"], r["task_id"], r.get("sample", 0)) for r in rows}
    if len(rows) != 600 or len(keys) != 600:
        raise ClaimFailure(f"expected 600 unique keys, got {len(rows)} rows / {len(keys)} keys")
    drops = [float(r["recall_drop"]) for r in _load_csv(
        ROOT / "docs/results/finaudit_multilabel/primary_violation_drop.csv")]
    holm = [float(r["holm_p"]) for r in _load_csv(
        ROOT / "docs/results/finaudit_multilabel/primary_violation_drop.csv")]
    positive = sum(1 for d in drops if d > 0)
    median = sorted(drops)[len(drops) // 2 - 1 : len(drops) // 2 + 1]
    median_value = sum(median) / 2
    significant = sum(1 for p in holm if p < 0.05)
    if positive != 5 or not _close(median_value, 0.267, 1e-3) or significant != 3:
        raise ClaimFailure(
            f"released results report 5 positive / median 0.267 / 3 significant; "
            f"data says {positive} / {median_value:.3f} / {significant}"
        )
    return "600/600 keys; 5 positive, median 0.267, 3 Holm-significant", \
           f"{len(keys)} keys; {positive}, {median_value:.3f}, {significant}"


def check_frozen_cardinality_split() -> tuple[str, str]:
    """The demoted claim's replacement: the two auditors diverge."""

    rows = [r for r in _load_csv(ROOT / "docs/results/finaudit_multilabel/condition_metrics.csv")
            if "violation_plus_edit" in " ".join(r.values())]
    both: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        total = sum(int(row[k]) for k in
                    ("dual_both", "dual_violation_only", "dual_edit_only", "dual_neither"))
        both[row["model"]].append((int(row["dual_both"]), total))
    shares = {m: [b / t for b, t in v] for m, v in both.items()}
    ds = shares.get("deepseek:deepseek-v4-pro", [])
    glm = shares.get("glm:glm-5", [])
    if not ds or not glm:
        raise ClaimFailure("condition metrics missing an auditor")
    if not (0.70 <= min(ds) and max(ds) <= 0.88):
        raise ClaimFailure(f"released range for deepseek is 70-88%; data {min(ds):.2f}-{max(ds):.2f}")
    if not (min(glm) == 0.0 and max(glm) <= 0.48):
        raise ClaimFailure(f"released range for glm is 0-48%; data {min(glm):.2f}-{max(glm):.2f}")
    return "ds both 70-88%, glm 0-48%", \
           f"ds {min(ds):.2f}-{max(ds):.2f}, glm {min(glm):.2f}-{max(glm):.2f}"


def check_specificity_and_exact_sets() -> tuple[str, str]:
    """Target presence must not be mistaken for a correct audit set."""

    overall = {
        row["model"]: row
        for row in _load_csv(ROOT / "docs/results/finaudit_multilabel/model_overall.csv")
    }
    expected_overall = {
        "deepseek:deepseek-v4-pro": (0.14874651810584957, 0.09666666666666666),
        "glm:glm-5": (0.199203187250996, 0.36),
    }
    for model, (precision, exact) in expected_overall.items():
        row = overall.get(model)
        if row is None:
            raise ClaimFailure(f"model_overall.csv is missing {model}")
        if not _close(float(row["micro_precision"]), precision, 1e-12):
            raise ClaimFailure(
                f"{model}: expected precision {precision}, got {row['micro_precision']}"
            )
        if not _close(float(row["exact_set_accuracy"]), exact, 1e-12):
            raise ClaimFailure(
                f"{model}: expected exact-set accuracy {exact}, got {row['exact_set_accuracy']}"
            )

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in _load_csv(ROOT / "docs/results/finaudit_multilabel/condition_metrics.csv"):
        model = row["model"]
        tasks = int(row["tasks"])
        condition = row["condition"]
        if condition == "zero":
            counts[model]["zero_any"] += round(float(row["any_finding_rate"]) * tasks)
            counts[model]["zero_exact"] += round(float(row["exact_set_accuracy"]) * tasks)
        elif condition == "violation_plus_edit":
            counts[model]["dual_both"] += int(row["dual_both"])
            counts[model]["dual_exact"] += round(float(row["exact_set_accuracy"]) * tasks)

    expected_counts = {
        "deepseek:deepseek-v4-pro": Counter(
            zero_any=98, zero_exact=1, dual_both=78, dual_exact=21
        ),
        "glm:glm-5": Counter(
            zero_any=45, zero_exact=55, dual_both=23, dual_exact=7
        ),
    }
    for model, expected in expected_counts.items():
        if counts.get(model) != expected:
            raise ClaimFailure(f"{model}: expected {dict(expected)}, got {dict(counts[model])}")

    return (
        "precision .149/.199; exact set 29/300 and 108/300; "
        "zero false-positive 98/100 and 45/100; dual exact 21/100 and 7/100",
        "all aggregate and condition-level counts match",
    )


def check_intervention_arms() -> tuple[str, str]:
    """The numbers in 'What moves the behaviour', from the committed script."""

    rows = _load_csv(ROOT / "docs/results/finaudit/intervention_v2.csv")
    if not rows:
        raise ClaimFailure("intervention_v2.csv absent -- run build_intervention_v2.py")
    got = {(r["auditor"], r["arm"]): r for r in rows}
    expected = {
        ("deepseek:deepseek-v4-pro", "constraint"): 0.833,
        ("deepseek:deepseek-v4-pro", "cot"): 0.333,
        ("deepseek:deepseek-v4-pro", "selfcons_majority"): 0.500,
        ("glm:glm-5", "constraint"): 0.933,
        ("glm:glm-5", "cot"): 0.617,
        ("glm:glm-5", "selfcons_majority"): 0.283,
    }
    for key, value in expected.items():
        row = got.get(key)
        if row is None:
            raise ClaimFailure(f"missing arm {key}")
        actual = float(row["arm_confounded_recall"])
        if not _close(actual, value, 1e-3):
            raise ClaimFailure(f"{key}: released value {value}, data says {actual}")
    return "constraint .833/.933, cot .333/.617, sc-majority .500/.283", "all six arms match"


def check_self_audit_null() -> tuple[str, str]:
    rows = _load_csv(ROOT / "docs/results/finaudit/self_audit_v2.csv")
    if not rows:
        raise ClaimFailure("self_audit_v2.csv absent -- run build_intervention_v2.py")
    gaps = {r["auditor"]: float(r["gap_cross_minus_self"]) for r in rows}
    if not _close(gaps.get("deepseek:deepseek-v4-pro", 9), -0.033, 1e-3):
        raise ClaimFailure(f"deepseek gap: released -0.033, data {gaps.get('deepseek:deepseek-v4-pro')}")
    if not _close(gaps.get("glm:glm-5", 9), 0.0, 1e-9):
        raise ClaimFailure(f"glm gap: released 0.000, data {gaps.get('glm:glm-5')}")
    return "self-audit gaps -0.033 and 0.000", f"{gaps}"


# --------------------------------------------------------------------------
# Study C
# --------------------------------------------------------------------------

def check_fault_corpus() -> tuple[str, str]:
    rows = _load_csv(ROOT / "docs/results/live_readiness_e1/e1_interception.csv")
    directed = sum(1 for r in rows if r["bucket"] == "directed")
    fuzz = sum(1 for r in rows if r["bucket"] == "fuzz")
    fields = len({r["target_field"] for r in rows})
    reasons = len({r["detail"] for r in rows})
    intercepted = sum(1 for r in rows if str(r["intercepted"]).lower() == "true")
    escapes = len(rows) - intercepted
    if (len(rows), directed, fuzz, fields, reasons, intercepted, escapes) != (
        365, 146, 219, 123, 255, 344, 21
    ):
        raise ClaimFailure(
            f"corpus mismatch: {len(rows)} faults, {directed} directed, {fuzz} fuzz, "
            f"{fields} fields, {reasons} reasons, {intercepted} intercepted, {escapes} escapes"
        )
    for row in rows:
        if row["bucket"] != "fuzz":
            continue
        selected_artifact = row["description"].rsplit(" on ", 1)[-1]
        resolved_artifact = row["target_field"].split(".", 1)[0]
        if selected_artifact != resolved_artifact:
            raise ClaimFailure(
                f"{row['fault_id']}: selected {selected_artifact}, mutated {resolved_artifact}"
            )
    return "365 = 146+219, 123 fields, 255 outcomes, 344 rejected, 21 accepted", "all match"


def check_interception_table() -> tuple[str, str]:
    """Verify first-rejection counts overall and by mutation source."""

    rows = _load_csv(ROOT / "docs/results/live_readiness_e1/e1_interception.csv")
    total = len(rows)
    intercepted = sum(1 for row in rows if row["intercepted"] == "True")
    if (total, intercepted) != (365, 344):
        raise ClaimFailure(f"overall: released 344/365, data says {intercepted}/{total}")

    first = Counter(row["first_layer"] for row in rows if row["intercepted"] == "True")
    vector = [
        first.get("schema_validation", 0),
        first.get("single_artifact_validator", 0),
        first.get("approval_hash_binding", 0),
        first.get("cross_artifact_preflight", 0),
        first.get("orchestrator_revalidation", 0),
    ]
    if vector != [261, 20, 36, 18, 9]:
        raise ClaimFailure(f"released vector is 261/20/36/18/9, data says {'/'.join(map(str, vector))}")
    if sum(vector) != intercepted:
        raise ClaimFailure(f"first-intercept vector sums to {sum(vector)}, not {intercepted}")

    by_source = {}
    for source in ("directed", "fuzz"):
        source_rows = [row for row in rows if row["bucket"] == source and row["intercepted"] == "True"]
        source_first = Counter(row["first_layer"] for row in source_rows)
        by_source[source] = [source_first.get(layer, 0) for layer in (
            "schema_validation",
            "single_artifact_validator",
            "approval_hash_binding",
            "cross_artifact_preflight",
            "orchestrator_revalidation",
        )]
    if by_source != {"directed": [63, 20, 32, 17, 9], "fuzz": [198, 0, 4, 1, 0]}:
        raise ClaimFailure(f"first-rejection source strata changed: {by_source}")

    return "344/365 rejected; first-rejection 261/20/36/18/9", f"source strata {by_source}"


def check_leave_one_out() -> tuple[str, str]:
    """First hits partition into later rejection, new acceptance, or exception."""

    rows = _load_csv(ROOT / "docs/results/live_readiness_e1/e1_leave_one_out.csv")
    by_layer = {row["layer"]: row for row in rows}
    expected = {
        "schema_validation": {"first": 261, "newly": 0, "absorbed": 250, "crashes": 11},
        "single_artifact_validator": {"first": 20, "newly": 0, "absorbed": 20, "crashes": 0},
        "approval_hash_binding": {"first": 36, "newly": 0, "absorbed": 36, "crashes": 0},
        "cross_artifact_preflight": {"first": 18, "newly": 18, "absorbed": 0, "crashes": 0},
        "orchestrator_revalidation": {"first": 9, "newly": 9, "absorbed": 0, "crashes": 0},
    }
    if set(by_layer) != set(expected):
        raise ClaimFailure(f"layers changed: {sorted(by_layer)}")

    for layer, want in expected.items():
        row = by_layer[layer]
        got = {
            "first": int(row["first_intercepts_when_present"]),
            "newly": int(row["newly_escaping"]),
            "absorbed": int(row["absorbed_by_later_layer"]),
            "crashes": int(row["downstream_crashes"]),
        }
        if got != want:
            raise ClaimFailure(f"{layer}: released {want}, data {got}")

        if got["first"] != got["newly"] + got["absorbed"] + got["crashes"]:
            raise ClaimFailure(f"{layer}: bypass fates do not partition first hits: {got}")

    ledger = _load_csv(ROOT / "docs/results/live_readiness_e1/e1_leave_one_out_ledger.csv")
    if len(ledger) != 5 * 365:
        raise ClaimFailure(f"bypass ledger has {len(ledger)} rows, expected 1825")
    fate_counts = Counter(
        (row["removed_layer"], row["fate"])
        for row in ledger
        if row["baseline_first_layer"] == row["removed_layer"]
    )
    for layer, want in expected.items():
        got = {
            "newly": fate_counts[(layer, "newly_escaping")],
            "absorbed": fate_counts[(layer, "absorbed_by_later_layer")],
            "crashes": fate_counts[(layer, "downstream_crash")],
        }
        if got != {key: want[key] for key in ("newly", "absorbed", "crashes")}:
            raise ClaimFailure(f"{layer}: aggregate/ledger mismatch: {got} vs {want}")

    return (
        "261/20/36/18/9 first hits; newly accepted 0/0/0/18/9; 11 schema-bypass exceptions",
        "all three bypass fates are mutually exclusive and exhaustive",
    )


def check_e4_week_counts() -> tuple[str, str]:
    """How many committed weeks are reconciled, against the event book.

    This one drifts on a schedule: the weekly job reconciles another week and
    the sentence quoting the count silently ages. It did exactly that once
    already -- the report said four while the book said five.
    """

    path = ROOT / "docs/results/live_readiness_e4/e4_event_book.csv"
    rows = _load_csv(path)
    states = Counter((row.get("state") or row.get("status") or "").strip().upper() for row in rows)
    reconciled = states.get("RECONCILED", 0)
    committed = len(rows)
    if committed != 12:
        raise ClaimFailure(f"released text says twelve committed weeks, book has {committed}")
    if reconciled != 5:
        raise ClaimFailure(
            f"released text says five weeks reconciled, book has {reconciled} "
            f"(states: {dict(states)}). The count advances weekly; update the sentence."
        )
    return (
        "5 of 12 committed weeks reconciled",
        f"{reconciled}/{committed}, states {dict(states)}",
    )


def check_escape_composition() -> tuple[str, str]:
    """Five escapes are on the approver identity -- the authority-bearing field."""

    rows = [r for r in _load_csv(ROOT / "docs/results/live_readiness_e1/e1_interception.csv")
            if str(r["intercepted"]).lower() != "true"]
    approver = sum(1 for r in rows if r["target_field"] == "approval.approved_by")
    directed_escapes = sum(1 for r in rows if r["bucket"] == "directed")
    if approver != 5 or directed_escapes != 5:
        raise ClaimFailure(
            f"released value 5 approver-id escapes, all directed; data says "
            f"{approver} approver-id, {directed_escapes} directed"
        )
    if len(rows) - approver != 16:
        raise ClaimFailure(f"released value 16 remaining accepted variants, data says {len(rows) - approver}")
    return "5 approver-id accepted variants (all directed), 16 others", "matches"


def check_artifact_binding_shape() -> tuple[str, str]:
    """Two of five artifacts carry a digest; one carries an expiry."""

    schemas = {
        "capability": "broker_adapter_capability.schema.json",
        "handoff": "broker_handoff_artifact.schema.json",
        "approval": "broker_approval_artifact.schema.json",
        "response": "broker_response_artifact.schema.json",
        "runbook": "operator_runbook_artifact.schema.json",
        "bundle": "live_readiness_preflight.schema.json",
    }
    with_digest, with_expiry, closed = [], [], 0
    for name, filename in schemas.items():
        spec = json.loads((ROOT / "schemas" / filename).read_text(encoding="utf-8"))
        props = spec.get("properties", {})
        if spec.get("additionalProperties") is False:
            closed += 1
        if any("hash" in k or "digest" in k for k in props):
            with_digest.append(name)
        if any(k == "expires_at" for k in props):
            with_expiry.append(name)
    if sorted(with_digest) != ["approval", "response"]:
        raise ClaimFailure(f"released value approval+response bind by digest; data says {with_digest}")
    if with_expiry != ["approval"]:
        raise ClaimFailure(f"released value only the approval expires; data says {with_expiry}")
    if closed != 6:
        raise ClaimFailure(f"released value all six are closed-world; only {closed} set additionalProperties false")
    return "6 closed-world; digest on approval+response; expiry on approval", \
           f"closed {closed}/6, digest {with_digest}, expiry {with_expiry}"


def check_authority_probe_hole() -> tuple[str, str]:
    """One authority probe escapes the gate and every monitor -- reported as a hole."""

    sys.path.insert(0, str(ROOT / "src"))
    from tradearena.evaluation.airlock_faults import LayeredInterceptor, build_clean_template
    from tradearena.evaluation.airlock_monitor import build_monitor_items

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        interceptor = LayeredInterceptor(build_clean_template(tmp_path / "tpl", variant="a"))
        escaped = []
        for item in build_monitor_items(tmp_path / "items", variant="a"):
            if item.tier != "authority":
                continue
            outcome = interceptor._detect(item.payloads)
            layer = outcome[0] if isinstance(outcome, tuple) else outcome
            if str(layer) == "escape":
                escaped.append(item.item_id)
    if escaped != ["authority_05"]:
        raise ClaimFailure(f"released value exactly authority_05 escapes; data says {escaped}")

    monitor = _load_jsonl(ROOT / "outputs/airlock_monitor/airlock_monitor_results.jsonl")
    flags = [r for r in monitor if r.get("item_id") == "authority_05"]
    if not flags or any(r.get("flagged") for r in flags):
        raise ClaimFailure(f"released value no monitor flags authority_05; data says {flags}")
    return "authority_05 escapes gate and all monitors", f"escaped={escaped}, monitors flagged=0/{len(flags)}"


def check_signature_end_to_end() -> tuple[str, str]:
    """A signed approval remains schema-compatible; an untrusted key does not verify."""

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "tests/test_live_session.py::test_signed_approval_passes_the_gate_it_authorizes",
         "tests/test_approval_signing.py::test_resigning_with_an_untrusted_key_is_rejected"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ClaimFailure(f"signature tests failed:\n{result.stdout[-800:]}")
    return "signed approval is accepted by the current path; standalone verifier rejects untrusted key", \
           "both tests pass"


def check_monitor_sample_sizes() -> tuple[str, str]:
    rows = _load_csv(ROOT / "docs/results/live_readiness_e6/e6_monitor.csv")
    sizes = {(r["tier"], int(r["n"])) for r in rows}
    tiers = dict(sizes)
    if tiers.get("clean") != 9 or tiers.get("semantic") != 9 or tiers.get("freetext") != 9:
        raise ClaimFailure(f"released value n=9 per tier; data says {tiers}")
    if tiers.get("authority") != 6:
        raise ClaimFailure(f"released value n=6 for authority; data says {tiers.get('authority')}")
    return "n=9 per tier, 6 for authority", f"{tiers}"


CHECKS: list[tuple[str, str, Callable[[], tuple[str, str]]]] = [
    ("A", "matrix completeness", check_matrix_completeness),
    ("A", "matrix temperature envelope", check_matrix_temperature),
    ("A", "execution decomposition", check_execution_decomposition),
    ("A", "robustness intervals", check_robustness_intervals),
    ("A", "forward window declaration", check_forward_window_declaration),
    ("A", "decision-path divergence", check_path_divergence),
    ("A", "ranking stability table", check_ranking_stability),
    ("A", "inactive row inflates tau_b", check_inactive_row),
    ("B", "legacy prompt quantity cues", check_legacy_prompts_assert_cardinality),
    ("B", "frozen prompt is neutral", check_frozen_prompt_is_neutral),
    ("B", "frozen grid and decision rule", check_frozen_grid_and_gate),
    ("B", "frozen cardinality split", check_frozen_cardinality_split),
    ("B", "specificity and exact sets", check_specificity_and_exact_sets),
    ("B", "intervention arms", check_intervention_arms),
    ("B", "self-audit null", check_self_audit_null),
]

#: Checks for the layered-validation work, which was cut from the report on
#: 2026-08-17 and is being carried toward a separate paper. The data and the
#: assertions are unchanged and still pass; they are held here rather than in
#: CHECKS because a gate that verifies claims no current manuscript makes would
#: report a coverage this report does not have. Run them with --gate.
GATE_CHECKS: list[tuple[str, str, Callable[[], tuple[str, str]]]] = [
    ("gate", "fault corpus", check_fault_corpus),
    ("gate", "first-rejection counts", check_interception_table),
    ("gate", "stage-bypass partition", check_leave_one_out),
    ("gate", "accepted-variant composition", check_escape_composition),
    ("gate", "artifact binding shape", check_artifact_binding_shape),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the released results.")
    parser.add_argument("--study", choices=["A", "B"], help="run one study's checks only")
    parser.add_argument("--strict", action="store_true", help="Fail if any required check is unavailable")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="run the approval-gate checks instead: results retained for a separate paper",
    )
    args = parser.parse_args(argv)

    if args.gate:
        selected = list(GATE_CHECKS)
        print("Layered-validation checks. These cover work cut from the report on 2026-08-17;")
        print("they are kept runnable so the results stay verifiable for a separate paper.\n")
    else:
        selected = [c for c in CHECKS if args.study is None or c[0] == args.study]
    failures: list[tuple[str, str]] = []
    unavailable: list[tuple[str, str]] = []
    print(f"{'study':6s}{'claim':34s}{'status':8s}asserted -> computed")
    print("-" * 100)
    for study, name, check in selected:
        try:
            asserted, computed = check()
            print(f"{study:6s}{name:34s}{'ok':8s}{asserted}  ->  {computed}")
        except DataUnavailable as exc:
            unavailable.append((name, str(exc)))
            print(f"{study:6s}{name:34s}{'SKIP':8s}input not in this checkout: {exc}")
        except ClaimFailure as exc:
            failures.append((name, str(exc)))
            print(f"{study:6s}{name:34s}{'FAIL':8s}{exc}")
        except Exception as exc:  # a check that cannot run is also a failure
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"{study:6s}{name:34s}{'ERROR':8s}{type(exc).__name__}: {exc}")

    print()
    runnable = len(selected) - len(unavailable)
    print(f"{runnable - len(failures)}/{runnable} runnable claims verified")
    if unavailable:
        print(
            f"\n{len(unavailable)} checks could not run here. Their inputs live under "
            "outputs/, which is outside version control; a checkout that holds the run "
            "artifacts verifies these too:"
        )
        for name, path in unavailable:
            print(f"  - {name}: {path}")
    if args.strict and unavailable:
        print("STRICT FAILURE: required evidence may not be skipped")
        return 1
    if failures:
        print(f"\n{len(failures)} FAILED")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
