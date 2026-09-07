"""Plot helpers used across the course notebooks.

Deliberately small. Anything you would write once in a notebook stays in the
notebook; this module holds only the things five chapters would otherwise
copy-paste -- above all the plate map, which is how you check that a
well-level annotation is right before trusting any analysis built on it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#: All 16 rows and 24 columns of a 384-well plate, used so that unused wells
#: are drawn as empty rather than silently omitted.
PLATE_ROWS = tuple("ABCDEFGHIJKLMNOP")
PLATE_COLUMNS = tuple(range(1, 25))


def set_style() -> None:
    """House style: no top/right spines, readable defaults, editable PDF text."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,   # keep text editable in Illustrator
            "ps.fonttype": 42,
        }
    )


def plate_map(
    wells: pd.DataFrame,
    value: str,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = "tab20",
    annotate: bool = False,
    empty_color: str = "#f2f2f2",
) -> Axes:
    """Draw a 384-well plate coloured by one column of ``wells``.

    Works for both categorical values (condition) and numeric ones (timepoint,
    cell count). Unused wells are drawn in ``empty_color`` so the plate's shape
    -- rows A and P empty, four column blocks -- is visible rather than implied.

    Parameters
    ----------
    wells
        One row per used well, with ``row``, ``column`` and ``value`` columns.
        Typically :func:`mcs2026.layout.read_wells`, or that joined to counts.
    value
        Column to colour by.
    annotate
        Write the value into each well. Useful for 18 conditions, useless for
        cell counts.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))

    series = wells.set_index(["row", "column"])[value]
    numeric = pd.api.types.is_numeric_dtype(series)

    if numeric:
        codes = series.astype(float)
        levels = None
    else:
        levels = sorted(series.dropna().astype(str).unique(), key=_natural_key)
        lookup = {level: i for i, level in enumerate(levels)}
        codes = series.astype(str).map(lookup)

    grid = np.full((len(PLATE_ROWS), len(PLATE_COLUMNS)), np.nan)
    for (row, column), code in codes.items():
        if row in PLATE_ROWS and column in PLATE_COLUMNS:
            grid[PLATE_ROWS.index(row), PLATE_COLUMNS.index(column)] = code

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(empty_color)
    image = ax.imshow(np.ma.masked_invalid(grid), cmap=colormap, aspect="equal")

    ax.set_xticks(range(len(PLATE_COLUMNS)), [str(c) for c in PLATE_COLUMNS], fontsize=7)
    ax.set_yticks(range(len(PLATE_ROWS)), PLATE_ROWS, fontsize=7)
    ax.set_xticks(np.arange(-0.5, len(PLATE_COLUMNS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PLATE_ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title or value)

    if annotate:
        for (row, column), code in codes.items():
            if row in PLATE_ROWS and column in PLATE_COLUMNS and not pd.isna(code):
                ax.text(
                    PLATE_COLUMNS.index(column),
                    PLATE_ROWS.index(row),
                    str(series.loc[(row, column)]),
                    ha="center", va="center", fontsize=5,
                )

    if numeric:
        ax.figure.colorbar(image, ax=ax, shrink=0.7, label=value)
    elif levels is not None and len(levels) <= 24:
        handles = [
            plt.Line2D([], [], marker="s", linestyle="", markersize=7,
                       color=colormap(i / max(len(levels) - 1, 1)), label=level)
            for i, level in enumerate(levels)
        ]
        ax.legend(
            handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
            fontsize=7, title=value, title_fontsize=8, ncols=1 + len(levels) // 13,
        )
    return ax


def panel_grid(n: int, *, ncols: int = 3, size: tuple[float, float] = (3.6, 3.0)):
    """A figure with ``n`` axes on a grid, with the unused ones removed."""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False
    )
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.remove()
    return fig, flat[:n]


def _natural_key(text: str):
    """Sort ``Cnd2`` before ``Cnd10``."""
    import re

    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(text))]


def savefig(fig: Figure, path, **kwargs) -> None:
    """Save and close, so a long notebook does not accumulate open figures."""
    fig.savefig(path, **kwargs)
    plt.close(fig)
