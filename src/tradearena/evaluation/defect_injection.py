"""Trajectory defect injection for the audit-agent task family.

Research plan 04: instead of asking whether an LLM can trade, ask whether it
can audit a trade trajectory. Each injector takes a clean trajectory dict
(``Trajectory.to_dict()`` or a loaded artifact), corrupts one record, and
returns the defective copy plus a machine-checkable ground-truth label, so
detection tasks can be generated and scored without human annotation.

Defect kinds:

- ``unclipped_position`` (L1, single record): an approved decision exceeds
  the documented per-name cap with no clip evidence.
- ``silent_risk_edit`` (L2, cross record): the approved weight differs from
  the model's intent but the clip provenance is removed.
- ``provenance_drift`` (L2, cross record): one step's reproducibility state
  claims a different model version than the rest of the run.
- ``tampered_fill_price`` (L3, recompute): a fill price is moved without
  updating slippage or commission, breaking the execution-cost identity.
"""

from __future__ import annotations

import copy
import random
from typing import Any

DEFECT_KINDS = ("unclipped_position", "silent_risk_edit", "provenance_drift", "tampered_fill_price")
DEFECT_DIFFICULTY = {
    "unclipped_position": "L1",
    "silent_risk_edit": "L2",
    "provenance_drift": "L2",
    "tampered_fill_price": "L3",
}


def applicable_steps(trajectory: dict[str, Any], kind: str) -> list[int]:
    """Step indices where the requested defect can be injected."""

    steps = trajectory.get("steps", [])
    indices: list[int] = []
    for index, step in enumerate(steps):
        if kind == "unclipped_position":
            if any(_weight(decision) is not None for decision in step.get("approved_decisions", [])):
                indices.append(index)
        elif kind == "silent_risk_edit":
            if any(
                isinstance(decision.get("metadata"), dict) and "risk_clipped_from" in decision["metadata"]
                for decision in step.get("approved_decisions", [])
            ):
                indices.append(index)
        elif kind == "provenance_drift":
            if isinstance(step.get("reproducibility_state"), dict) and step["reproducibility_state"].get("model_version"):
                indices.append(index)
        elif kind == "tampered_fill_price":
            if any(_finite(fill.get("price")) for fill in step.get("fills", [])):
                indices.append(index)
        else:
            raise ValueError(f"Unknown defect kind: {kind!r}; expected one of {DEFECT_KINDS}")
    return indices


def inject_defect(
    trajectory: dict[str, Any],
    kind: str,
    *,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a defective deep copy of the trajectory plus its ground truth."""

    candidates = applicable_steps(trajectory, kind)
    if not candidates:
        raise ValueError(f"No step in this trajectory supports defect kind {kind!r}")
    rng = random.Random(seed)
    step_index = candidates[rng.randrange(len(candidates))]
    defective = copy.deepcopy(trajectory)
    step = defective["steps"][step_index]

    if kind == "unclipped_position":
        decisions = [d for d in step["approved_decisions"] if _weight(d) is not None]
        decision = decisions[rng.randrange(len(decisions))]
        original = float(decision["target_weight"])
        decision["target_weight"] = 0.8 if original >= 0.0 else -0.8
        if isinstance(decision.get("metadata"), dict):
            decision["metadata"].pop("risk_clipped_from", None)
        detail = {"symbol": decision.get("symbol", ""), "original_target_weight": original}
    elif kind == "silent_risk_edit":
        decisions = [
            d
            for d in step["approved_decisions"]
            if isinstance(d.get("metadata"), dict) and "risk_clipped_from" in d["metadata"]
        ]
        decision = decisions[rng.randrange(len(decisions))]
        removed = decision["metadata"].pop("risk_clipped_from")
        detail = {"symbol": decision.get("symbol", ""), "removed_risk_clipped_from": removed}
    elif kind == "provenance_drift":
        state = step["reproducibility_state"]
        original_version = state.get("model_version")
        state["model_version"] = "unverified-model-v9"
        detail = {"original_model_version": original_version}
    else:  # tampered_fill_price
        fills = [f for f in step["fills"] if _finite(f.get("price"))]
        fill = fills[rng.randrange(len(fills))]
        original_price = float(fill["price"])
        fill["price"] = original_price * 1.05
        detail = {"symbol": fill.get("symbol", ""), "original_price": original_price}

    ground_truth = {
        "kind": kind,
        "difficulty": DEFECT_DIFFICULTY[kind],
        "step_index": step_index,
        "detail": detail,
    }
    return defective, ground_truth


def score_findings(
    findings: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    step_tolerance: int = 0,
) -> dict[str, float | int]:
    """Precision/recall/F1 of reported findings against injected defects.

    A finding matches a defect when the kinds are equal and the reported step
    index is within ``step_tolerance`` of the injected step. Each defect can
    be claimed by at most one finding.
    """

    remaining = list(ground_truth)
    true_positives = 0
    for finding in findings:
        kind = str(finding.get("kind", ""))
        step_index = finding.get("step_index")
        if not isinstance(step_index, int):
            continue
        for defect in remaining:
            if defect["kind"] == kind and abs(int(defect["step_index"]) - step_index) <= step_tolerance:
                true_positives += 1
                remaining.remove(defect)
                break
    precision = true_positives / len(findings) if findings else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "findings": len(findings),
        "defects": len(ground_truth),
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _weight(decision: dict[str, Any]) -> float | None:
    return _finite(decision.get("target_weight"))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number
