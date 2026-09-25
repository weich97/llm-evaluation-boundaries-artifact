"""Figures for the audit study of the merged technical report.

Two panels, both driven by the corrected multi-label re-scoring of the stored
paired responses (docs/results/finaudit/corrected_key_rescore.csv):

  (a) Target-label coverage on the two-defect arm. Each auditor's responses are
      partitioned into "both true defects reported", "violation only", "edit
      only", and "neither". These categories do not count unrelated findings:
      reporting one target label does not imply reporting only one finding.
  (b) The estimand. Violation recall on the one-defect arm versus the
      two-defect arm, with the within-pair McNemar p-value annotated. The
      corrected key removes the false-positive penalty for a valid edit finding;
      it does not turn an unreported violation into a detected one.

Writes paper/tech_report/figures/audit_cardinality.pdf (and .png for drafts).
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from report_figure_style import FIGURE_WIDTH, LABEL_SIZE, SMALL_SIZE, TITLE_SIZE, apply_style

ROOT = Path(__file__).resolve().parents[1]
RESCORE = ROOT / "docs/results/finaudit/corrected_key_rescore.csv"
OUT_DIR = ROOT / "paper/tech_report/figures"
MODELS = ("deepseek:deepseek-v4-pro", "glm:glm-5")

DOMAINS = {
    "trading": {
        "sources": [
            (
                ROOT / f"outputs/audit_pairs/{producer}/ground_truth.jsonl",
                ROOT / f"outputs/audit_pairs_eval_v2/{producer}/audit_eval_results.jsonl",
                producer,
            )
            for producer in ("deepseek_v4_pro", "glm_5_direct")
        ],
        "violation_kind": "unclipped_position",
        "edit_kind": "silent_risk_edit",
    },
    "tooluse": {
        "sources": [
            (
                ROOT / "outputs/toolaudit_pairs/ground_truth.jsonl",
                ROOT / "outputs/toolaudit_pairs_eval_v2/toolaudit_eval_results.jsonl",
                "tooluse",
            )
        ],
        "violation_kind": "over_limit_call",
        "edit_kind": "silent_arg_edit",
    },
}

LABEL = {"deepseek:deepseek-v4-pro": "deepseek-v4-pro", "glm:glm-5": "glm-5"}
# Domain names as printed in the manuscript.
DOMAIN_LABEL = {"trading": "trading", "tooluse": "tool-use"}
# colour-blind safe, ordered worst-to-best reading left to right in the stack
COLORS = {
    "both": "#1b7837",
    "violation_only": "#7fbf7b",
    "edit_only": "#d6604d",
    "neither": "#b2182b",
}


def cardinality_breakdown() -> dict[tuple[str, str], dict[str, float]]:
    """(domain, model) -> share of dual-arm responses in each cardinality class."""

    out: dict[tuple[str, str], dict[str, float]] = {}
    for domain, spec in DOMAINS.items():
        vkind, ekind = spec["violation_kind"], spec["edit_kind"]
        reported: dict[tuple[str, str], set[str]] = {}
        for gt_path, res_path, tag in spec["sources"]:
            if not (gt_path.exists() and res_path.exists()):
                continue
            truth = {
                json.loads(line)["task_id"]: json.loads(line)
                for line in gt_path.open(encoding="utf-8")
                if line.strip()
            }
            for line in res_path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["model"] not in MODELS or int(record.get("sample", 0)) != 0:
                    continue
                entry = truth.get(record["task_id"])
                if not entry or entry["kind"] != vkind:
                    continue
                if entry["detail"].get("ambiguity") != "confounded":
                    continue
                step = int(entry["step_index"])
                kinds = {
                    str(f.get("kind", ""))
                    for f in record["findings"] or []
                    if str(f.get("step_index", "")) == str(step)
                }
                reported[(record["model"], f"{tag}:{entry['detail']['pair_id']}")] = kinds
        for model in MODELS:
            cells = [kinds for (m, _), kinds in reported.items() if m == model]
            if not cells:
                continue
            counts: collections.Counter[str] = collections.Counter()
            for kinds in cells:
                has_v, has_e = vkind in kinds, ekind in kinds
                if has_v and has_e:
                    counts["both"] += 1
                elif has_v:
                    counts["violation_only"] += 1
                elif has_e:
                    counts["edit_only"] += 1
                else:
                    counts["neither"] += 1
            total = sum(counts.values())
            out[(domain, model)] = {k: counts[k] / total for k in COLORS}
    return out


def main() -> int:
    if not RESCORE.exists():
        print(f"missing {RESCORE}; run build_corrected_key_rescore.py first")
        return 1
    rescore = {(r["domain"], r["auditor"]): r for r in csv.DictReader(RESCORE.open(encoding="utf-8"))}
    breakdown = cardinality_breakdown()
    cells = [(d, m) for d in DOMAINS for m in MODELS if (d, m) in breakdown]
    ticks = [f"{LABEL[m]}\n({DOMAIN_LABEL[d]})" for d, m in cells]
    x = range(len(cells))

    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 6.0))

    bottom = [0.0] * len(cells)
    for key in ("both", "violation_only", "edit_only", "neither"):
        vals = [breakdown[c][key] for c in cells]
        ax_a.bar(x, vals, bottom=bottom, color=COLORS[key], width=0.62,
                 edgecolor="white", linewidth=0.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    # the "both reported" share is the headline and is nearly invisible in the
    # stack precisely because it is small; label it explicitly.
    for i, cell in enumerate(cells):
        ax_a.annotate(
            f"both: {breakdown[cell]['both'] * 100:.0f}%",
            (i, 1.02), ha="center", fontsize=SMALL_SIZE, fontweight="bold",
        )
    ax_a.set_ylim(0, 1.30)
    ax_a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.set_ylabel("Share of two-defect responses", fontsize=LABEL_SIZE)
    ax_a.set_title("(a) Target-label coverage on two-defect tasks", fontsize=TITLE_SIZE,
                   loc="left", pad=9)
    ax_a.set_xticks(list(x)); ax_a.set_xticklabels(ticks, fontsize=SMALL_SIZE)
    ax_a.legend(
        handles=[
            Patch(facecolor=COLORS["both"], label="both reported"),
            Patch(facecolor=COLORS["violation_only"], label="violation only"),
            Patch(facecolor=COLORS["edit_only"], label="edit only"),
            Patch(facecolor=COLORS["neither"], label="neither"),
        ],
        fontsize=SMALL_SIZE, loc="upper center", framealpha=0.95, ncol=4,
        columnspacing=1.0, handlelength=1.4,
    )

    width = 0.34
    single = [float(rescore[c]["single_violation_recall"]) for c in cells]
    dual = [float(rescore[c]["dual_violation_recall"]) for c in cells]
    ax_b.bar([i - width / 2 for i in x], single, width, label="one true defect",
             color="#4393c3", edgecolor="white")
    ax_b.bar([i + width / 2 for i in x], dual, width, label="two true defects",
             color="#f4a582", edgecolor="white")
    for i, c in enumerate(cells):
        top = max(single[i], dual[i])
        ax_b.annotate(f"$p$={float(rescore[c]['mcnemar_p']):.0e}",
                      (i, top + 0.04), ha="center", fontsize=SMALL_SIZE)
    ax_b.set_ylim(0, 1.30)
    ax_b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_b.set_ylabel("Violation recall", fontsize=LABEL_SIZE)
    ax_b.set_title("(b) Target recall with one vs two defects", fontsize=TITLE_SIZE,
                   loc="left", pad=9)
    ax_b.set_xticks(list(x)); ax_b.set_xticklabels(ticks, fontsize=SMALL_SIZE)
    ax_b.legend(fontsize=SMALL_SIZE, loc="upper center", framealpha=0.95, ncol=2)
    ax_b.grid(axis="y", alpha=0.25, lw=0.4)

    fig.tight_layout(pad=0.8, h_pad=1.8)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        metadata = {"Creator": "Matplotlib", "CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(OUT_DIR / f"audit_cardinality.{suffix}", dpi=200, metadata=metadata)
    print(f"wrote {(OUT_DIR / 'audit_cardinality.pdf').relative_to(ROOT)}")
    for cell in cells:
        share = breakdown[cell]
        print(f"  {cell[0]:8s} {LABEL[cell[1]]:18s} both={share['both']:.2f} "
              f"v-only={share['violation_only']:.2f} e-only={share['edit_only']:.2f} "
              f"neither={share['neither']:.2f}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
