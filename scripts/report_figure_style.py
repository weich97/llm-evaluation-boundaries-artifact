"""Print-size typography shared by the three financial-agent report figures.

The manuscript text block is 6.5 inches wide. Export at that width so 9-point
labels remain 9-point labels in the paper, rather than shrinking an 11-inch plot.
STIX fonts ship with Matplotlib; no system font or LaTeX installation is needed.
"""
from __future__ import annotations

import matplotlib as mpl

FIGURE_WIDTH = 6.5
LABEL_SIZE = 10
SMALL_SIZE = 9
TITLE_SIZE = 11


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "font.size": LABEL_SIZE,
        "mathtext.fontset": "stix",
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
    })
