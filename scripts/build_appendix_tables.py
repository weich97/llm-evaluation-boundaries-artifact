"""Emit the report's appendix tables as a LaTeX fragment, straight from the data.

The appendix exists to show per-cell numbers rather than the summaries the body
quotes. Hand-copying them into the manuscript would reintroduce exactly the
drift this repository keeps finding, so the fragment is generated and the
manuscript ``\\input``s it. Nothing in the appendix is typed by a person.

Writes paper/tech_report/appendix_tables.tex.
"""

from __future__ import annotations

import collections
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/results"
OUT = ROOT / "paper/tech_report/appendix_tables.tex"

#: The outcomes worth tabulating per cell. The replay records nine; the four
#: here are the ones any claim in the body rests on. The rest are in the CSV.
METRICS = ("total_return", "sharpe", "max_drawdown", "execution_fill_rate")

#: Source names as printed in the body tables; the CSV keeps producer ids.
SOURCE_LABEL = {
    "tooluse": "tool-use",
    "trading:deepseek_v4_pro": "trading A",
    "trading:glm_5_direct": "trading B",
}

#: Estimand names follow the manuscript's Section 5.2 notation and Figure 1.
ESTIMAND_LABEL = {
    "execution_within_I0": r"$D_{\mathrm{E0}}$: execution $\mid$ tape from E0",
    "execution_within_I1": r"$D_{\mathrm{E1}}$: execution $\mid$ tape from E1",
    "execution_shapley": r"$\bar D$: balanced destination",
    "response_origin_within_X0": r"$O_{\mathrm{E0}}$: response origin $\mid$ replay at E0",
    "response_origin_within_X1": r"$O_{\mathrm{E1}}$: response origin $\mid$ replay at E1",
    "response_origin_shapley": r"$\bar O$: balanced tape origin",
    "interaction": r"$D_{\mathrm{E1}}-D_{\mathrm{E0}}$: interaction",
    "observed_diagonal": r"$\bar D+\bar O$: observed diagonal",
}
ESTIMAND_ORDER = list(ESTIMAND_LABEL)

#: Outcome names as printed in the body; the CSV keeps its column identifiers.
METRIC_LABEL = {
    "total_return": "total return",
    "sharpe": "Sharpe",
    "max_drawdown": "max drawdown",
    "execution_fill_rate": "fill rate",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tex(value: str) -> str:
    return value.replace("_", r"\_").replace("&", r"\&")


def study_a_table() -> str:
    rows = _read(RESULTS / "v0_3_fixed_intent_replay/factorial_estimands.csv")
    by_metric: dict[str, dict[str, dict[str, str]]] = collections.defaultdict(dict)
    for row in rows:
        by_metric[row["metric"]][row["estimand"]] = row

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Study A, every estimand of the fixed-tape $2\times2$ replay, on the four "
        r"outcomes the body draws on. Intervals are seed-cluster bootstrap ($10{,}000$ "
        r"draws, $10$ clusters, $450$ matched pairs). The replay records five further "
        r"outcomes; they are in \texttt{factorial\_estimands.csv}.}",
        r"\label{tab:app-factorial}",
        r"\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Outcome & Estimand & Estimate & 95\% CI \\",
        r"\midrule",
    ]
    for index, metric in enumerate(METRICS):
        cells = by_metric.get(metric, {})
        if not cells:
            continue
        if index:
            lines.append(r"\midrule")
        first = True
        for estimand in ESTIMAND_ORDER:
            row = cells.get(estimand)
            if not row:
                continue
            label = METRIC_LABEL[metric] if first else ""
            first = False
            lines.append(
                f"{label} & {ESTIMAND_LABEL[estimand]} & "
                f"{float(row['estimate']):+.4f} & "
                f"$[{float(row['ci95_low']):+.4f}, {float(row['ci95_high']):+.4f}]$ \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def study_b_table() -> str:
    rows = _read(RESULTS / "finaudit_multilabel/primary_violation_drop.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Study B, the six auditor $\times$ source cells recorded in the "
        r"internally hash-bound multi-label plan. Recall is on the target violation, within matched "
        r"triplets; the discordant counts are the McNemar pairs. Intervals are unadjusted "
        r"95\% paired percentile-bootstrap confidence intervals based on $10{,}000$ "
        r"resamples. $p$-values are from two-sided exact McNemar tests with Holm "
        r"correction across the six primary comparisons.}",
        r"\label{tab:app-cells}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Auditor & Source & $n$ & Single & Dual & Drop & 95\% CI & $b/c$ & Holm $p$ \\",
        r"\midrule",
    ]
    for row in rows:
        auditor = row["model"].split(":")[-1]
        cell = SOURCE_LABEL[row["cell"]]
        holm = float(row["holm_p"])
        holm_text = f"{holm:.3g}" if holm >= 1e-4 else f"{holm:.1e}"
        bold = r"\bf " if holm < 0.05 else ""
        lines.append(
            f"{bold}{_tex(auditor)} & {bold}{_tex(cell)} & {row['triples']} & "
            f"{float(row['single_violation_recall']):.3f} & "
            f"{float(row['dual_violation_recall']):.3f} & "
            f"{bold}{float(row['recall_drop']):+.3f} & "
            f"$[{float(row['ci95_low']):+.3f}, {float(row['ci95_high']):+.3f}]$ & "
            f"{row['single_only']}/{row['dual_only']} & {bold}{holm_text} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> int:
    fragment = "\n\n".join(
        [
            "% Generated by scripts/build_appendix_tables.py -- do not edit by hand.",
            study_a_table(),
            study_b_table(),
        ]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(fragment + "\n", encoding="utf-8", newline="")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(fragment.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
