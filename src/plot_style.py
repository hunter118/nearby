"""Shared vector-figure style for the paper and research diagnostics."""

from __future__ import annotations

import matplotlib as mpl


def configure_paper_plots() -> None:
    """Match the manuscript's Computer Modern typography in vector PDFs."""

    mpl.rcParams.update(
        {
            "font.family": "cmr10",
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
