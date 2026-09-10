"""Shared machinery for Part 3 and Part 4.

Two kinds of function live here.

**Cell level** -- ``normalise_cells``, ``stratified``, ``sketch`` and
``transfer_labels`` operate on the single-cell ``AnnData``. Stage 1 calls them
to build ``mcs2026_clean.h5ad`` and its sketch; Stage 2 loads the result.

**Well level** -- everything else works on the table of one row per well, one
column per marker, in units of control-well standard deviations. The well is
the replicate unit, so that is what every statistical test runs on.

Each of these was written out longhand in a notebook once, where you can see
every step. It lives here so that the *next* chapter does not have to repeat it,
and so that a change is made in one place rather than six.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

CONTROL = "DMSO"
#: The condition that shifts every marker by 4-11 control SDs (see chapter 02).
OUTLIER_CONDITION = "Phorbol 12-myristate 13-acetate (PMA)"


def well_means(data, columns, *, name_by="marker") -> "pd.DataFrame":
    """Collapse a cell-level AnnData to one row per well.

    The well is the replicate unit for every test in Part 3, so this is the
    aggregation almost everything else starts from. Columns are renamed to their
    marker, and the condition/timepoint/well keys are carried along.

    ``timepoint_h`` is an ordered categorical -- right for plotting, wrong for
    arithmetic, since pandas refuses to subtract categoricals. It is cast to int
    here so callers do not have to remember.
    """
    import numpy as np

    frame = pd.DataFrame(
        np.asarray(data[:, list(columns)].X),
        columns=[data.var.loc[c, name_by] for c in columns],
    )
    frame["well"] = data.obs.well.astype(str).values
    frame["condition"] = data.obs.condition.astype(str).values
    frame["timepoint"] = data.obs.timepoint_h.astype(int).values
    return (frame.groupby(["condition", "timepoint", "well"], observed=True)
            .mean().reset_index())


def marker_columns(var: pd.DataFrame, *, statistic: str = "mean_intensity") -> list[str]:
    """The usable marker columns: one statistic, no DAPI, no failed stains."""
    keep = (
        (var["family"] == "Intensity")
        & (var["statistic"] == statistic)
        & var["marker"].notna()
        & (var["marker"] != "DAPI")
        & (~var["failed"])
    )
    return var.index[keep].tolist()


def normalise_cells(
    data,
    columns,
    *,
    control: str = CONTROL,
    by: str = "timepoint_h",
    condition_key: str = "condition",
) -> np.ndarray:
    """``log2``, then centre and scale on the control cells of the same group.

    This is Step 17 applied to single cells rather than to well means: every
    value comes out as "this many control-*cell* standard deviations from the
    controls of its own timepoint". Normalising within ``by`` matters because
    the controls themselves drift across 36-84 h; pooling them would put early
    cells below zero and late cells above it by construction.

    A column whose control cells have no spread at all carries no information,
    and is returned as zeros rather than as infinities.

    Returns a dense ``float32`` array of ``(n_obs, len(columns))``.
    """
    raw = np.asarray(data[:, list(columns)].X, dtype="float64")
    logged = np.log2(raw + 1.0)

    groups = np.asarray(data.obs[by].astype(str))
    conditions = np.asarray(data.obs[condition_key].astype(str))
    out = np.zeros_like(logged)

    for value in pd.unique(groups):
        rows = groups == value
        reference = logged[rows & (conditions == control)]
        if reference.shape[0] < 2:
            raise ValueError(
                f"{by}={value!r} has {reference.shape[0]} {control} cells; "
                "cannot scale a group without controls"
            )
        spread = reference.std(axis=0, ddof=1)
        usable = spread > 0
        block = np.zeros((int(rows.sum()), logged.shape[1]))
        block[:, usable] = (
            (logged[rows][:, usable] - reference.mean(axis=0)[usable]) / spread[usable]
        )
        out[rows] = block

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def stratified(
    obs: pd.DataFrame,
    n: int,
    *,
    by=("condition", "timepoint_h"),
    seed: int = 0,
) -> np.ndarray:
    """Positions of ``~n`` cells, evenly spread over every ``by`` combination.

    The alternative students reach for is ``adata[:n]``, which on a file written
    well by well hands you one corner of the plate. This keeps every condition
    and timepoint represented, in equal numbers rather than in proportion.
    """
    rng = np.random.default_rng(seed)
    groups = obs.groupby(list(by), observed=True).indices
    per_group = max(1, n // max(len(groups), 1))
    picked = [
        rng.choice(rows, size=min(per_group, len(rows)), replace=False)
        for rows in groups.values()
    ]
    return np.sort(np.concatenate(picked))


def sketch(matrix: np.ndarray, n: int, *, seed: int = 0) -> np.ndarray:
    """Positions of ``n`` cells covering the *geometry* of ``matrix``.

    Geometric sketching (Hie et al. 2019) lays a grid over the data space and
    takes roughly one cell per occupied box, so a region holding 40 cells and a
    region holding 40,000 contribute the same. Rare states survive; uniform
    sampling reproduces the common ones and loses the rest.

    ``matrix`` should already be low-dimensional -- the 38 normalised markers,
    or a PCA of something wider. See chapter 04 on why.
    """
    from geosketch import gs

    return np.sort(np.asarray(gs(matrix, n, seed=seed, replace=False), dtype=int))


def transfer_labels(
    source: np.ndarray,
    labels,
    target: np.ndarray,
    *,
    k: int = 15,
) -> np.ndarray:
    """Carry labels from the sketched cells out to every cell.

    You annotated 30,000 cells; you have 650,000. A k-nearest-neighbour vote in
    the same space each was embedded in gives the rest their label. This is what
    makes sketching worth doing -- analyse a tractable sample, then put the
    answer back on the whole dataset.

    ``source`` and ``target`` must have the same number of columns.
    """
    from sklearn.neighbors import KNeighborsClassifier

    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"source has {source.shape[1]} columns, target has {target.shape[1]}"
        )
    model = KNeighborsClassifier(n_neighbors=min(k, len(source)))
    model.fit(source, np.asarray(labels))
    return model.predict(target)


def neighbour_purity(
    data,
    labels,
    *,
    within=None,
    exclude=None,
    neighbors_key: str | None = None,
) -> dict:
    """How often does a cell's graph neighbour share its label, against chance?

    An embedding "looks mixed" or "looks separated" according to the reader.
    This is the same question as a number: the fraction of neighbour pairs whose
    two cells carry the same label, divided by the fraction you would get if the
    labels were scattered at random over the same cells.

    A ratio near **1** means the graph knows nothing about that label -- the
    *reassuring* answer for a batch variable, the disappointing one for a
    treatment. Above 1 means cells sharing that label sit together.

    Two arguments exist because these labels nest inside each other, and a nested
    label inherits its parent's signal:

    ``within``
        Count only neighbour pairs from the same block, and rescale chance to
        match. Separates a label from something it sits inside -- well from
        timepoint, since every well has exactly one.
    ``exclude``
        Count only neighbour pairs from *different* groups. Separates a label
        from something that sits inside **it** -- plate row from well, since a
        row is only ever a bag of wells.

    ``neighbors_key`` picks which graph, for objects carrying more than one.

    Self-pairs never count. Scanpy's ``distances`` matrix sometimes stores each
    cell as its own nearest neighbour and sometimes does not, depending on which
    backend built the graph -- and a cell trivially shares its own label, so
    leaving those in would pull every ratio toward 1 by an amount that varies
    with the data.

    Returns ``{"observed", "expected", "ratio", "pairs"}``.
    """
    key = data.uns[neighbors_key or "neighbors"]["distances_key"]
    graph = data.obsp[key]
    labels = np.asarray(labels)
    n = len(labels)
    blocks = np.asarray(within) if within is not None else np.zeros(n, int)
    # Each cell is its own group by default, so "different group" also means
    # "not itself" -- see the note about self-pairs above.
    groups = np.asarray(exclude) if exclude is not None else np.arange(n)

    # Every stored edge at once: `rows` is the cell, `cols` its neighbour.
    rows = np.repeat(np.arange(n), np.diff(graph.indptr))
    cols = graph.indices
    keep = (blocks[rows] == blocks[cols]) & (groups[rows] != groups[cols])
    same = int((labels[rows[keep]] == labels[cols[keep]]).sum())
    pairs = int(keep.sum())

    # Chance level, as an explicit count of eligible pairs: draw two cells from
    # the same block, reject them if `exclude` puts them in the same group, and
    # ask how often their labels match.
    matching = eligible = 0.0
    frame = pd.DataFrame({"block": blocks, "label": labels, "group": groups})
    for _, chunk in frame.groupby("block", observed=True):
        by_label = chunk.groupby("label", observed=True).size()
        by_group = chunk.groupby("group", observed=True).size()
        by_both = chunk.groupby(["label", "group"], observed=True).size()
        matching += float((by_label ** 2).sum() - (by_both ** 2).sum())
        eligible += float(len(chunk) ** 2 - (by_group ** 2).sum())

    expected = matching / eligible if eligible else float("nan")
    return {"observed": same / pairs if pairs else float("nan"),
            "expected": expected,
            "ratio": (same / pairs) / expected if pairs and expected else float("nan"),
            "pairs": pairs}


def transfer_values(
    source: np.ndarray,
    values,
    target: np.ndarray,
    *,
    k: int = 15,
) -> np.ndarray:
    """The continuous twin of :func:`transfer_labels`.

    Same idea, same space, same caveats -- a k-nearest-neighbour average rather
    than a vote. Used to carry a pseudotime computed on the sketch out to every
    cell, so that per-well means are computed on a real sample of each well
    rather than on a sketch that took different numbers of cells from each.
    """
    from sklearn.neighbors import KNeighborsRegressor

    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"source has {source.shape[1]} columns, target has {target.shape[1]}"
        )
    model = KNeighborsRegressor(n_neighbors=min(k, len(source)))
    model.fit(source, np.asarray(values, dtype="float64"))
    return model.predict(target)


def panel_markers(theme: str, available) -> list[str]:
    """Marker names in ``theme`` that are actually present in ``available``.

    ``available`` is usually ``wells.columns``. Panels are declared by hand and
    the plate does not always carry every marker in one, so resolving against
    what is really there beats indexing with a name that is not.
    """
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
