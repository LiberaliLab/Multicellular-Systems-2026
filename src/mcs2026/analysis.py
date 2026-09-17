"""Shared machinery for Part 3 and Part 4.

Two kinds of function live here.

**Cell level** -- ``normalise_cells``, ``stratified``, ``sketch`` and
``transfer_labels`` operate on the single-cell ``AnnData``. Stage 1 calls them
to build ``mcs2026_intensity.h5ad`` and its sketch; Stage 2 loads the result.

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
#: Both vehicles. ``normalise_cells`` references them together, because between
#: them they cover seven plate rows rather than four. The effect tables further
#: down still compare against ``CONTROL`` alone.
CONTROLS = ("DMSO", "PBS")
#: The condition that shifts every marker by 4-11 control SDs (see chapter 02).
OUTLIER_CONDITION = "Phorbol 12-myristate 13-acetate (PMA)"


def by_well(data, columns=None, *, name_by=None) -> "pd.DataFrame":
    """Average per well: one row per condition x timepoint x well.

    The well is the replicate unit for every test in Part 3, so this is the
    aggregation almost everything else starts from. It does exactly one thing --
    take the mean of what is in ``X`` -- because by the time Stage 2 opens the
    intensity object ``X`` already holds normalised values and there is nothing left
    to decide. Normalisation happens once, to cells, in ``normalise_cells``.

    ``columns`` defaults to every column. ``name_by`` renames them through
    ``var``: chapter 02 passes ``"marker"``, because on the wide table the column
    names are still channel-and-round strings.

    ``timepoint_h`` is an ordered categorical -- right for plotting, wrong for
    arithmetic, since pandas refuses to subtract categoricals. It is cast to int
    here so callers do not have to remember.

    It averages one well at a time rather than building one dense table. The
    answer is 223 rows whatever you pass, but the input need not be small: the
    wide plate is roughly 650,000 x 2,587, which is 6.8 GB dense before pandas
    makes a copy of its own to group on.
    """
    names = list(data.var_names) if columns is None else list(columns)
    positions = data.var_names.get_indexer(names)
    if (positions < 0).any():
        missing = [n for n, p in zip(names, positions) if p < 0]
        raise KeyError(f"not in var: {missing[:5]}")
    if name_by is not None:
        names = [data.var.loc[c, name_by] for c in names]

    keys = pd.DataFrame({
        "condition": data.obs.condition.astype(str).values,
        "timepoint": data.obs.timepoint_h.astype(int).values,
        "well": data.obs.well.astype(str).values,
    })
    labels, blocks = [], []
    for key, index in keys.groupby(list(keys.columns), observed=True).indices.items():
        labels.append(key)
        blocks.append(np.asarray(data.X[index])[:, positions].mean(axis=0))

    frame = pd.DataFrame(np.vstack(blocks), columns=names)
    for level, name in enumerate(keys.columns):
        frame.insert(level, name, [k[level] for k in labels])
    return frame.sort_values(list(keys.columns)).reset_index(drop=True)


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
    controls=CONTROLS,
    by: str = "timepoint_h",
    condition_key: str = "condition",
    log2: bool = True,
) -> np.ndarray:
    """``log2``, put the origin at the first timepoint, take the unit from the plate.

    The origin and the unit answer different questions, and they come from
    different cells.

    **Origin -- what counts as zero.** The median of the control cells at the
    *first* timepoint. One fixed point, so zero means "an untreated cell at the
    start of the experiment" and the controls' own development across 36-84 h
    stays visible in the numbers. Centring each timepoint on its own controls
    would instead define the control as zero at every timepoint: a moving origin
    cannot show movement, and in a time course that erases the experiment.

    One origin is enough because all four timepoints sit on the same plate and
    are imaged in the same rounds. The round-to-round staining drift of Step 15
    is therefore a single number per marker, shared by every timepoint, and
    subtracting one constant removes it everywhere. A per-timepoint origin
    removes that drift *and* the biology; this removes only the drift.

    **Unit -- what counts as one.** The standard deviation of every control cell
    on the plate, taken about the median of its own vehicle x timepoint group.
    Using the whole plate makes the estimate stable; taking deviations about each
    group's own centre stops the drift, and the gap between the two vehicles,
    from inflating it. That gap is real -- 11 of the 38 markers separate DMSO
    from PBS by more than a control SD -- so pooling the two naively would widen
    the unit and quietly shrink every effect measured in it.

    **Why the centre is a median and the unit is not.** They are different
    estimation problems. The centre asks *where is the middle*, and a handful of
    very bright cells -- debris, doublets, a dying cell full of autofluorescence
    -- drag a mean off the bulk while leaving a median where it was. So the
    origin is a median.

    The unit asks *how wide is this*, and there the median absolute deviation has
    a failure mode the standard deviation does not: it is decided entirely by the
    middle half of the data, so if more than half the cells share a value it is
    exactly zero however the rest behave. That is not hypothetical here. A marker
    that is not expressed reads at background in most cells, and on this plate
    GATA4 sits at exactly zero in 69% of control cells and p21 in 53% -- so their
    MADs come out 0.000 and 0.006 against standard deviations of 2.25 and 1.58.
    The first would be deleted by the guard below and the second inflated about
    260-fold. Measured across all 38 markers, a MAD unit gives the control block
    a spread of 46 and a worst z-score of 1934; a standard-deviation unit gives
    1.13 and 14.

    The MAD is the more robust estimator and the wrong one for this data: what it
    is designed to ignore is exactly where a switching marker keeps its signal.

    A column whose control cells have no spread at all carries no information,
    and comes back as zeros rather than as infinities. With a standard deviation
    that now means genuinely constant, rather than merely sparse.

    ``log2=False`` skips the transform and centres and scales the values as they
    are. Intensities want the log -- they are multiplicative and right-skewed --
    but a *bounded ratio* does not: eccentricity, solidity and extent live on
    roughly 0 to 1 and are already near-symmetric, so logging them squashes one
    end for nothing and in fact makes them more skewed. The origin and the unit
    are computed the same way either way; this flag changes one line.

    Returns a dense ``float32`` array of ``(n_obs, len(columns))``.
    """
    raw = np.asarray(data[:, list(columns)].X, dtype="float64")
    logged = np.log2(raw + 1.0) if log2 else raw

    groups = np.asarray(data.obs[by].astype(int))
    conditions = np.asarray(data.obs[condition_key].astype(str))
    present = [c for c in controls if (conditions == c).any()]
    if not present:
        raise ValueError(
            f"none of {list(controls)} appear in obs[{condition_key!r}]; "
            "cannot normalise without control cells"
        )
    first = groups.min()

    centres, deviations = [], []
    for vehicle in present:
        is_vehicle = conditions == vehicle
        origin = logged[is_vehicle & (groups == first)]
        if origin.shape[0] < 2:
            raise ValueError(
                f"{vehicle!r} has {origin.shape[0]} cells at {by}={first}; "
                "cannot place the origin without them"
            )
        centres.append(np.median(origin, axis=0))
        # Deviations about each group's own median, so neither the drift across
        # timepoints nor the DMSO-PBS gap counts as spread. The centre is still a
        # median here; it is only the width that is a standard deviation.
        for value in np.unique(groups[is_vehicle]):
            block = logged[is_vehicle & (groups == value)]
            deviations.append(block - np.median(block, axis=0))

    centre = np.mean(centres, axis=0)
    scale = np.vstack(deviations).std(axis=0, ddof=1)

    out = np.zeros_like(logged)
    usable = scale > 0
    out[:, usable] = (logged[:, usable] - centre[usable]) / scale[usable]
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

    Values arrive in control-SD units, so this is a difference of z-scores: the
    number says "this many control-well standard deviations from the control".

    The control is subtracted **explicitly**, and that matters. It is tempting to
    skip it -- the normalisation puts the controls near zero, so a plain group
    mean already looks like a shift -- but that only holds at the one timepoint
    the origin was taken from. Everywhere else the controls have moved, and a
    group mean would report their own development as though it were a treatment
    effect. See Step 17 of chapter 03 for why the origin is fixed rather than
    following the controls around.
    """
    if by_timepoint:
        table = wells.groupby(["condition", "timepoint"], observed=True)[markers].mean()
        reference = (wells[wells.condition == control]
                     .groupby("timepoint", observed=True)[markers].mean())
        matched = reference.reindex(table.index.get_level_values("timepoint"))
        table = table - matched.to_numpy()
        # `level=0` is only valid on the MultiIndex that grouping by two keys makes.
        return table.drop(index=control, level=0, errors="ignore")

    table = wells.groupby("condition", observed=True)[markers].mean()
    table = table - wells.loc[wells.condition == control, markers].mean()
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
