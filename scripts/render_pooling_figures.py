"""Two figures about what a pooled number hides.

Both are read-only over released analysis tables: no API call, no re-run.

  (a) Study A. Pre-risk decision-path agreement between the two execution
      destinations, per model and pooled. The pooled 19.8% that the body quotes
      is produced entirely by one model; the other four agree on exactly zero of
      their 90 pairs each. A reader who sees only the pooled figure would
      conclude the execution convention barely moves the decision path, which is
      the opposite of what the split shows.

  (b) Study B. The six internally hash-bound auditor x source cells of the
      multi-label experiment as a forest plot, with Holm-adjusted significance
      marked. The body quotes a median drop; the spread is what decides whether
      the effect is a property of auditing or of one auditor.

Writes paper/tech_report/figures/pooling_hides_structure.{pdf,png}.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from report_figure_style import FIGURE_WIDTH, LABEL_SIZE, SMALL_SIZE, TITLE_SIZE, apply_style

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/results"
OUT_DIR = ROOT / "paper/tech_report/figures"
STEM = "pooling_hides_structure"

INK = "#1b1b1b"
MUTED = "#8a8a8a"
HOT = "#b3423c"
COOL = "#3c6db3"
# Source names as printed in the manuscript's tables; the CSV keeps producer ids.
SOURCE_LABEL = {
    "tooluse": "tool-use",
    "trading:deepseek_v4_pro": "trading A",
    "trading:glm_5_direct": "trading B",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def panel_a(ax: plt.Axes) -> None:
    rows = _read(RESULTS / "v0_3_fixed_intent_replay/response_path_divergence.csv")
    models = [r for r in rows if r["scope"] == "model"]
    pooled = next(r for r in rows if r["scope"] == "overall")

    models.sort(key=lambda r: -float(r["full_pre_risk_decision_path_equal_rate"]))
    labels = [r["value"] for r in models]
    rates = [float(r["full_pre_risk_decision_path_equal_rate"]) for r in models]
    pairs = [int(r["pairs"]) for r in models]
    pooled_rate = float(pooled["full_pre_risk_decision_path_equal_rate"])

    positions = range(len(labels))
    colours = [HOT if rate > 0.5 else COOL for rate in rates]
    ax.barh(list(positions), rates, color=colours, height=0.62)

    for y, (rate, n) in enumerate(zip(rates, pairs)):
        hits = round(rate * n)
        ax.text(
            rate + 0.018,
            y,
            f"{hits}/{n}" if rate > 0 else f"0/{n}",
            va="center",
            fontsize=SMALL_SIZE,
            color=INK,
        )

    ax.axvline(pooled_rate, color=INK, linestyle="--", linewidth=1.1)
    ax.text(
        pooled_rate + 0.018,
        0.03,
        f"pooled {pooled_rate:.3f}",
        transform=mtransforms.blended_transform_factory(ax.transData, ax.transAxes),
        fontsize=SMALL_SIZE,
        color=INK,
        style="italic",
    )

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=SMALL_SIZE)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Pre-risk decision paths identical across E0 and E1", fontsize=LABEL_SIZE)
    ax.set_title("(a) Study A: one model supplies all path agreements", fontsize=TITLE_SIZE,
                 loc="left", pad=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], color=HOT, lw=6, label="nearly inactive (358/360)"),
            Line2D([0], [0], color=COOL, lw=6, label="trades"),
            Line2D([0], [0], color=INK, ls="--", lw=1.1, label="pooled over 450 pairs"),
        ],
        fontsize=SMALL_SIZE,
        loc="lower right",
        framealpha=0.92,
    )


def panel_b(ax: plt.Axes) -> None:
    rows = _read(RESULTS / "finaudit_multilabel/primary_violation_drop.csv")
    rows.sort(key=lambda r: float(r["recall_drop"]))

    labels = [f"{r['model'].split(':')[-1]}\n{SOURCE_LABEL[r['cell']]}" for r in rows]
    drops = [float(r["recall_drop"]) for r in rows]
    lows = [float(r["ci95_low"]) for r in rows]
    highs = [float(r["ci95_high"]) for r in rows]
    holm = [float(r["holm_p"]) for r in rows]

    positions = list(range(len(rows)))
    for y, (drop, low, high, p) in enumerate(zip(drops, lows, highs, holm)):
        significant = p < 0.05
        colour = HOT if significant else MUTED
        ax.plot([low, high], [y, y], color=colour, linewidth=1.6, solid_capstyle="butt")
        ax.plot(
            [drop],
            [y],
            marker="D" if significant else "o",
            color=colour,
            markersize=5.5 if significant else 4.5,
        )
        ax.text(
            high + 0.02,
            y,
            f"Holm $p$={p:.3g}" if p >= 1e-4 else f"Holm $p$={p:.0e}",
            va="center",
            fontsize=SMALL_SIZE,
            color=INK if significant else MUTED,
        )

    median = sorted(drops)[len(drops) // 2 - 1 : len(drops) // 2 + 1]
    median_value = sum(median) / 2
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.axvline(median_value, color=INK, linestyle=":", linewidth=1.1)
    ax.text(
        median_value - 0.014,
        0.90,
        f"median {median_value:.3f}",
        transform=mtransforms.blended_transform_factory(ax.transData, ax.transAxes),
        ha="right",
        fontsize=SMALL_SIZE,
        style="italic",
        color=INK,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=SMALL_SIZE)
    ax.set_xlim(-0.3, 0.92)
    ax.set_xlabel("Violation-recall drop: one defect $\\rightarrow$ two", fontsize=LABEL_SIZE)
    ax.set_title("(b) Study B: six cells, three survive Holm", fontsize=TITLE_SIZE,
                 loc="left", pad=9)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 6.0))
    panel_a(ax_a)
    panel_b(ax_b)
    fig.tight_layout(pad=0.8, h_pad=1.8)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = OUT_DIR / f"{STEM}.{suffix}"
        metadata = {"Creator": "Matplotlib", "CreationDate": None, "ModDate": None} if path.suffix == ".pdf" else None
        fig.savefig(path, dpi=200, metadata=metadata)
        print(f"wrote {path.relative_to(ROOT)}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
