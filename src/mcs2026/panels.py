"""The four biological themes, and how to select features for one.

The course groups the antibody panel into four themes. They are **sets, not a
partition**: Calreticulin is an ER protein *and* a metabolic readout, and p-S6
sits on the PI3K-mTOR axis, so it belongs to both signaling and metabolism.
A marker appearing twice is deliberate.

Two further groups are not themes but are needed for the analysis:
``identity`` supplies the lineage markers used to call cell types (chapter 07),
and ``cell_cycle`` is a covariate you want to be able to regress out or check.
"""

from __future__ import annotations

import pandas as pd

#: Markers per theme, using the short names from :func:`mcs2026.layout.short_marker_name`.
PANELS: dict[str, tuple[str, ...]] = {
    "signaling": (
        "Foxo3a", "Foxo1", "FGFR1", "PDGFRa", "beta-Catenin",
        "YAP1", "p-S6", "p-AKT", "c-Myc", "RNAPII-pS5",
    ),
    "mechanics": (
        "p-MyosinIIa", "a-Tubulin", "ZO-1", "E-cadherin",
        "Fibronectin", "LAMA4", "LaminA", "LaminB1",
    ),
    "metabolism": (
        "Mitochondria", "Pmp70", "GRP78", "HSP90", "Calreticulin",
        "p-S6", "p-AKT",          # the PI3K-mTOR axis, shared with signaling
    ),
    "organelles": (
        "EEA1", "GM130", "Giantin", "LAMP1", "DDX6",
        "Calreticulin", "GRP78",  # ER, shared with metabolism
        "Pmp70", "Mitochondria",  # peroxisome, mitochondrion
        "LaminA", "LaminB1",      # nuclear envelope, shared with mechanics
    ),
    "identity": (
        "Oct4", "Nanog", "Sox2", "GATA3", "GATA4", "GATA6", "SOX17",
    ),
    "cell_cycle": ("CyclinA2", "p21", "Ki67"),
}

#: The four themes the course is organised around, in teaching order.
THEMES: tuple[str, ...] = ("signaling", "mechanics", "metabolism", "organelles")

#: Stains that failed QC and must never enter an analysis.
#: They are still present as live columns in the feature table.
FAILED_MARKERS: tuple[str, ...] = ("PDGFRa/round0", "Integrin-b1")


def _first_theme() -> dict[str, str]:
    """Marker -> its first listed theme, for a one-label-per-feature ``var`` column."""
    mapping: dict[str, str] = {}
    for theme in ("identity", "cell_cycle", *THEMES):
        for marker in PANELS[theme]:
            mapping.setdefault(marker, theme)
    return mapping


#: Marker -> single theme label, used to fill ``var['theme']``.
#: Markers in more than one panel get their first listed theme here; use
#: :func:`resolve_panel` when you want the true overlapping membership.
MARKER_THEME: dict[str, str] = _first_theme()


def markers(theme: str) -> tuple[str, ...]:
    """Marker names in one panel."""
    try:
        return PANELS[theme]
    except KeyError:
        raise KeyError(f"unknown panel {theme!r}; choose from {sorted(PANELS)}") from None


def resolve_panel(
    var: pd.DataFrame,
    theme: str,
    *,
    family: str | tuple[str, ...] | None = "Intensity",
    statistic: str | tuple[str, ...] | None = "mean_intensity",
    drop_failed: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Feature columns belonging to one theme.

    Parameters
    ----------
    var
        Annotated ``var`` table from :func:`mcs2026.decode.annotate`.
    theme
        A key of :data:`PANELS`.
    family
        Restrict to these feature families. The default, ``"Intensity"``,
        gives one number per marker per cell -- the right starting point.
        Pass ``None`` for every family, or ``("Intensity", "Texture")`` to
        include the Haralick and Laws texture features as well.
    statistic
        Restrict to these statistics. The default keeps ``mean_intensity``
        only. Pass ``None`` to keep all five intensity statistics.
    drop_failed
        Exclude stains marked ``failed`` in the staining sheet.
    verbose
        Report which requested markers had no matching column. Silence here
        is how a typo becomes a missing marker nobody notices.

    Returns the matching column names, ordered as the panel is declared.
    """
    wanted = markers(theme)
    selection = var[var["marker"].isin(wanted)]

    if drop_failed:
        selection = selection[~selection["failed"]]
    if family is not None:
        families = (family,) if isinstance(family, str) else tuple(family)
        selection = selection[selection["family"].isin(families)]
    if statistic is not None:
        statistics = (statistic,) if isinstance(statistic, str) else tuple(statistic)
        selection = selection[selection["statistic"].isin(statistics)]

    found = set(selection["marker"])
    missing = [m for m in wanted if m not in found]
    if verbose and missing:
        print(f"[{theme}] no columns for: {', '.join(missing)}")

    order = {marker: i for i, marker in enumerate(wanted)}
    selection = selection.assign(_order=selection["marker"].map(order)).sort_values(
        ["_order", "round", "statistic"]
    )
    return selection.index.tolist()


def panel_table(var: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """One row per panel: markers requested, markers found, columns selected.

    Print this once per notebook. It is the fastest way to see that a panel
    silently lost a marker.
    """
    rows = []
    for theme in PANELS:
        columns = resolve_panel(var, theme, verbose=False, **kwargs)
        found = var.loc[columns, "marker"].nunique() if columns else 0
        rows.append(
            {
                "panel": theme,
                "markers_requested": len(set(PANELS[theme])),
                "markers_found": found,
                "columns": len(columns),
            }
        )
    out = pd.DataFrame(rows).set_index("panel")
    out["complete"] = out["markers_found"] == out["markers_requested"]
    return out
