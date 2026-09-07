"""Shared machinery for the four theme chapters.

Chapter 02 builds the well-level, control-normalised table by hand, so you see
every step once. Chapters 03-06 then ask the *same* four questions of four
different panels, and repeating that code four times would hide the only thing
that changes -- the panel.

Everything here works on the well-level table: one row per well, one column per
marker, in units of control-well standard deviations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

CONTROL = "DMSO"
#: The condition that shifts every marker by 4-11 control SDs (see chapter 02).
OUTLIER_CONDITION = "Phorbol 12-myristate 13-acetate (PMA)"


def panel_markers(var: pd.DataFrame, theme: str, available) -> list[str]:
    """Marker names in ``theme`` that are actually present in the well table."""
    from mcs2026 import panels

    return [m for m in panels.markers(theme) if m in set(available)]


def effect_table(
    wells: pd.DataFrame,
    markers: list[str],
    *,
    control: str = CONTROL,
    by_timepoint: bool = True,
) -> pd.DataFrame:
    """Mean shift from the control, per condition and marker.

    Values are already in control-SD units, so this is a mean of z-scores: the
    number says "this many control-well standard deviations from the control".
    """
    group = ["condition", "timepoint"] if by_timepoint else ["condition"]
    table = wells.groupby(group, observed=True)[markers].mean()
    # `level=0` is only valid on the MultiIndex that grouping by two keys makes.
    if isinstance(table.index, pd.MultiIndex):
        return table.drop(index=control, level=0, errors="ignore")
    return table.drop(index=control, errors="ignore")


def rank_effects(
    wells: pd.DataFrame,
    markers: list[str],
    *,
    control: str = CONTROL,
    drop_outlier: bool = True,
) -> pd.DataFrame:
    """Every condition x marker pair, ranked by absolute effect.

    Includes a Mann-Whitney p-value **and** the smallest p-value the design can
    produce, so a saturated p-value is visible rather than mistaken for a
    strong one. See chapter 02 on why that floor exists.
    """
    frame = wells if not drop_outlier else wells[wells.condition != OUTLIER_CONDITION]
    reference = frame[frame.condition == control]

    rows = []
    for condition, block in frame.groupby("condition", observed=True):
        if condition == control:
            continue
        for marker in markers:
            treated, base = block[marker].dropna(), reference[marker].dropna()
            if len(treated) < 3 or len(base) < 3:
                continue
            rows.append({
                "condition": condition,
                "marker": marker,
                "shift": treated.mean() - base.mean(),
                "n_wells": len(treated),
                "p": stats.mannwhitneyu(treated, base).pvalue,
                "p_floor": minimum_p(len(treated), len(base)),
            })
    out = pd.DataFrame(rows)
    out["abs_shift"] = out["shift"].abs()
    out["at_p_floor"] = np.isclose(out["p"], out["p_floor"])
    return out.sort_values("abs_shift", ascending=False).reset_index(drop=True)


def minimum_p(n_treated: int, n_control: int) -> float:
    """Smallest two-sided Mann-Whitney p-value achievable with these group sizes.

    With 3 treated and 5 control wells there are C(8,3) = 56 rank orderings, so
    no effect -- however large -- can give a p below 2/56 = 0.0357. Reporting
    this next to the p-value is the honest way to show that the test has hit the
    limit of the experiment's replication, not of the biology.
    """
    from math import comb

    return 2.0 / comb(n_treated + n_control, n_treated)


def compare_to_control(
    wells: pd.DataFrame,
    marker: str,
    condition: str,
    *,
    control: str = CONTROL,
) -> pd.DataFrame:
    """One marker, one condition, per timepoint, against the control wells."""
    rows = []
    for timepoint, block in wells.groupby("timepoint", observed=True):
        treated = block.loc[block.condition == condition, marker].dropna()
        base = block.loc[block.condition == control, marker].dropna()
        if len(treated) < 2 or len(base) < 2:
            continue
        rows.append({
            "timepoint": timepoint,
            "n_treated": len(treated),
            "n_control": len(base),
            "shift": round(treated.mean() - base.mean(), 2),
            "p": round(stats.mannwhitneyu(treated, base).pvalue, 4),
            "p_floor": round(minimum_p(len(treated), len(base)), 4),
        })
    return pd.DataFrame(rows)


def theme_summary(wells: pd.DataFrame, markers: list[str], *, control: str = CONTROL) -> pd.Series:
    """One number per condition: mean |shift| across the panel.

    This is what chapter 08 stacks across the four themes to ask whether each
    perturbation moves its own theme most.
    """
    effects = effect_table(wells, markers, control=control, by_timepoint=False)
    return effects.abs().mean(axis=1).sort_values(ascending=False)
