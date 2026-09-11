"""Tests for the cell-level helpers in ``mcs2026.analysis``.

Stage 1 builds ``mcs2026_clean.h5ad`` out of these four functions, so a silent
change here would corrupt every chapter downstream without any notebook
failing. They run on a small synthetic plate rather than on the real 13 GB
table, so they are fast and need no cluster data.

    pytest tests/
"""

import numpy as np
import pandas as pd
import pytest

from mcs2026 import analysis

ad = pytest.importorskip("anndata")

CONDITIONS = ["DMSO", "PBS", "DrugA", "DrugB"]
TIMEPOINTS = [36, 48, 60, 84]
MARKERS = [f"M{i}" for i in range(6)]

#: A per-marker staining offset, identical at every timepoint. All four
#: timepoints sit on one plate and are imaged in the same rounds, so this is
#: what round-to-round drift actually looks like: a constant, not a trend.
DRIFT = np.array([0.0, 60.0, -30.0, 20.0, 0.0, 10.0])
#: M1 develops over 36-84 h in *every* condition, controls included. This is the
#: time course, and normalisation must not remove it.
TIME_COURSE = 1.5
#: The two vehicles are not interchangeable -- on the real plate 11 of 38 markers
#: separate them by more than a control SD. Here M2 does.
VEHICLE_GAP = 12.0
#: DrugA moves M0 and nothing else.
DRUG_EFFECT = 30.0


@pytest.fixture(scope="module")
def plate():
    """A miniature plate: 4 conditions x 4 timepoints x 3 wells x 40 cells.

    It carries the three things the normalisation has to tell apart: a staining
    drift that is an artefact, a time course that is the experiment, and a gap
    between the two vehicles that must not become part of the unit.
    """
    rng = np.random.default_rng(0)
    rows, values = [], []
    for timepoint in TIMEPOINTS:
        for condition in CONDITIONS:
            for replicate in range(3):
                well = f"{condition}-{timepoint}-{replicate}"
                centre = 100.0 + DRIFT
                centre[1] += TIME_COURSE * (timepoint - TIMEPOINTS[0])
                block = rng.normal(loc=centre, scale=4.0, size=(40, len(MARKERS)))
                if condition == "PBS":
                    block[:, 2] += VEHICLE_GAP
                if condition == "DrugA":
                    block[:, 0] += DRUG_EFFECT
                values.append(block)
                rows.extend(
                    {"well": well, "condition": condition, "timepoint_h": timepoint}
                    for _ in range(40)
                )

    obs = pd.DataFrame(rows)
    obs["timepoint_h"] = pd.Categorical(obs.timepoint_h, categories=TIMEPOINTS,
                                        ordered=True)
    obs.index = obs.index.astype(str)
    data = ad.AnnData(X=np.vstack(values).astype("float32"), obs=obs)
    data.var_names = MARKERS
    data.var["marker"] = [f"marker_{m}" for m in MARKERS]
    return data


def controls_of(data):
    return np.isin(data.obs.condition.values, ["DMSO", "PBS"])


def timepoints_of(data):
    return data.obs.timepoint_h.astype(int).values


# --------------------------------------------------------------------------
# normalise_cells
# --------------------------------------------------------------------------

def test_the_origin_is_the_first_timepoint(plate):
    """Zero means "an untreated cell at the start", not "an untreated cell"."""
    normalised = analysis.normalise_cells(plate, MARKERS)
    early = normalised[controls_of(plate) & (timepoints_of(plate) == TIMEPOINTS[0])]
    assert np.abs(np.median(early, axis=0)).max() < 0.4


def test_the_time_course_survives(plate):
    """The whole point. Centring each timepoint on its own controls would put
    every one of these at zero by construction, which in a time course is the
    experiment thrown away."""
    normalised = analysis.normalise_cells(plate, MARKERS)
    controls, timepoint = controls_of(plate), timepoints_of(plate)

    trace = [float(np.median(normalised[controls & (timepoint == v)], axis=0)[1])
             for v in TIMEPOINTS]
    assert trace == sorted(trace), trace
    assert trace[-1] > 3.0, trace


def test_staining_drift_does_not_survive(plate):
    """A per-marker offset shared by every timepoint is an artefact, and goes.

    DRIFT moves five markers by up to 60 units; only M1 is allowed to move with
    time, so every other marker must sit flat across the whole course.
    """
    normalised = analysis.normalise_cells(plate, MARKERS)
    controls, timepoint = controls_of(plate), timepoints_of(plate)
    steady = [0, 2, 3, 4, 5]

    for value in TIMEPOINTS:
        median = np.median(normalised[controls & (timepoint == value)], axis=0)
        assert np.abs(median[steady]).max() < 1.0, (value, median)


def test_the_planted_effect_is_recovered(plate):
    normalised = analysis.normalise_cells(plate, MARKERS)
    treated = plate.obs.condition.values == "DrugA"

    shifts = normalised[treated].mean(axis=0)
    assert shifts[0] > 4              # 30 units against a control spread of ~4
    # M1 carries the time course and M2 the vehicle gap -- a DMSO-vehicle drug
    # sits half a gap below zero on M2 by construction. M3-M5 carry nothing.
    assert np.abs(shifts[[3, 4, 5]]).max() < 0.6


def test_both_vehicles_place_the_origin(plate):
    """Neither vehicle alone is zero; the origin sits between the two."""
    normalised = analysis.normalise_cells(plate, MARKERS)
    condition, timepoint = plate.obs.condition.values, timepoints_of(plate)
    start = timepoint == TIMEPOINTS[0]

    dmso = float(np.median(normalised[(condition == "DMSO") & start], axis=0)[2])
    pbs = float(np.median(normalised[(condition == "PBS") & start], axis=0)[2])
    assert dmso < -0.5 and pbs > 0.5, (dmso, pbs)
    assert abs(dmso + pbs) < 0.4, (dmso, pbs)


def test_the_vehicle_gap_does_not_inflate_the_unit(plate):
    """Pooling two references that differ must not widen what counts as one.

    The unit is the spread *within* a vehicle, so within either vehicle the
    normalised values must still have a spread of about 1. Had the DMSO-PBS gap
    been counted as spread, the unit would be wider and both would come out well
    below 1 -- quietly shrinking every effect measured in them.
    """
    normalised = analysis.normalise_cells(plate, MARKERS)
    dmso = plate.obs.condition.values == "DMSO"
    pbs = plate.obs.condition.values == "PBS"

    assert 0.8 < normalised[dmso][:, 2].std() < 1.3
    assert 0.8 < normalised[pbs][:, 2].std() < 1.3
    # and the gap itself survives, as a difference of several units
    gap = np.median(normalised[pbs][:, 2]) - np.median(normalised[dmso][:, 2])
    assert gap > 2.0, gap


def test_output_is_finite_float32(plate):
    normalised = analysis.normalise_cells(plate, MARKERS)
    assert normalised.dtype == np.float32
    assert np.isfinite(normalised).all()


def test_a_dead_column_becomes_zeros_not_infinities(plate):
    """A marker with no spread in the controls carries no information."""
    flat = plate.copy()
    flat.X[:, 4] = 7.0
    normalised = analysis.normalise_cells(flat, MARKERS)
    assert np.isfinite(normalised).all()
    assert np.allclose(normalised[:, 4], 0)


def test_one_vehicle_is_enough(plate):
    """A plate with only DMSO still normalises, against DMSO alone."""
    single = plate[plate.obs.condition != "PBS"].copy()
    normalised = analysis.normalise_cells(single, MARKERS)
    start = timepoints_of(single) == TIMEPOINTS[0]
    controls = single.obs.condition.values == "DMSO"
    assert np.abs(np.median(normalised[controls & start], axis=0)).max() < 0.4


def test_no_control_cells_at_all_is_an_error(plate):
    orphan = plate[~np.isin(plate.obs.condition, ["DMSO", "PBS"])].copy()
    with pytest.raises(ValueError, match="cannot normalise without control cells"):
        analysis.normalise_cells(orphan, MARKERS)


def test_controls_missing_from_the_first_timepoint_is_an_error(plate):
    """The origin has to come from somewhere, and silence would be worse."""
    late = plate[~(controls_of(plate) & (timepoints_of(plate) == TIMEPOINTS[0]))].copy()
    with pytest.raises(ValueError, match="cannot place the origin"):
        analysis.normalise_cells(late, MARKERS)


# --------------------------------------------------------------------------
# by_well
# --------------------------------------------------------------------------

def test_by_well_gives_one_row_per_well(plate):
    wells = analysis.by_well(plate)
    assert len(wells) == plate.obs.well.nunique()
    assert list(wells.columns[:3]) == ["condition", "timepoint", "well"]
    assert list(wells.columns[3:]) == MARKERS


def test_by_well_matches_a_plain_groupby(plate):
    """It reads one well at a time to stay small; the answer must be identical."""
    frame = pd.DataFrame(np.asarray(plate.X), columns=MARKERS)
    frame["condition"] = plate.obs.condition.astype(str).values
    frame["timepoint"] = timepoints_of(plate)
    frame["well"] = plate.obs.well.astype(str).values
    expected = (frame.groupby(["condition", "timepoint", "well"], observed=True)
                .mean().reset_index())

    got = analysis.by_well(plate)
    pd.testing.assert_frame_equal(got, expected, check_dtype=False, atol=1e-5)


def test_by_well_can_rename_through_var(plate):
    wells = analysis.by_well(plate, MARKERS[:2], name_by="marker")
    assert list(wells.columns) == ["condition", "timepoint", "well",
                                   "marker_M0", "marker_M1"]


def test_by_well_rejects_a_column_that_is_not_there(plate):
    with pytest.raises(KeyError, match="not in var"):
        analysis.by_well(plate, ["M0", "nonexistent"])


# --------------------------------------------------------------------------
# stratified
# --------------------------------------------------------------------------

def test_stratified_keeps_every_condition_and_timepoint(plate):
    picked = analysis.stratified(plate.obs, 160)
    obs = plate.obs.iloc[picked]
    assert set(obs.condition) == set(CONDITIONS)
    assert set(obs.timepoint_h.astype(int)) == set(TIMEPOINTS)
    assert len(picked) == len(set(picked.tolist()))


def test_stratified_beats_slicing_the_front_of_the_file(plate):
    """``adata[:n]`` is the habit this function exists to replace."""
    n = 160
    sliced = plate.obs.iloc[:n]
    picked = plate.obs.iloc[analysis.stratified(plate.obs, n)]
    assert sliced.condition.nunique() < len(CONDITIONS)
    assert picked.condition.nunique() == len(CONDITIONS)


def test_stratified_is_reproducible(plate):
    first = analysis.stratified(plate.obs, 80, seed=7)
    assert np.array_equal(first, analysis.stratified(plate.obs, 80, seed=7))
    assert not np.array_equal(first, analysis.stratified(plate.obs, 80, seed=8))


# --------------------------------------------------------------------------
# sketch
# --------------------------------------------------------------------------

def test_sketch_returns_the_requested_number_of_distinct_cells(plate):
    pytest.importorskip("geosketch")
    normalised = analysis.normalise_cells(plate, MARKERS)
    picked = analysis.sketch(normalised, 200)
    assert len(picked) == 200
    assert len(set(picked.tolist())) == 200
    assert picked.max() < plate.n_obs


def test_sketch_keeps_a_rare_population_that_uniform_sampling_loses():
    """The reason to sketch at all: 1% of cells, kept in proportion to *space*."""
    pytest.importorskip("geosketch")
    rng = np.random.default_rng(0)
    common = rng.normal(size=(9_900, 8))
    rare = rng.normal(loc=9.0, size=(100, 8))
    matrix = np.vstack([common, rare])
    is_rare = np.arange(len(matrix)) >= 9_900

    sketched = is_rare[analysis.sketch(matrix, 500)].sum()
    uniform = is_rare[rng.choice(len(matrix), 500, replace=False)].sum()
    assert sketched > 4 * uniform


# --------------------------------------------------------------------------
# transfer_labels
# --------------------------------------------------------------------------

def test_transfer_labels_recovers_held_out_labels():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    labels = rng.integers(0, 3, size=3_000)
    matrix = centres[labels] + rng.normal(scale=0.7, size=(3_000, 2))

    train, test = slice(0, 2_000), slice(2_000, None)
    predicted = analysis.transfer_labels(matrix[train], labels[train], matrix[test])

    accuracy = (predicted == labels[test]).mean()
    assert accuracy > 0.95, accuracy
    assert len(predicted) == 1_000


def test_transfer_labels_refuses_mismatched_spaces():
    with pytest.raises(ValueError, match="source has 3 columns, target has 2"):
        analysis.transfer_labels(np.zeros((10, 3)), ["a"] * 10, np.zeros((5, 2)))


# --------------------------------------------------------------------------
# neighbour_purity
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graphed():
    """Two well-separated blobs, each split by a label that means nothing."""
    sc = pytest.importorskip("scanpy")
    rng = np.random.default_rng(0)
    real = np.repeat(["left", "right"], 400)
    matrix = np.where(real[:, None] == "left", -4.0, 4.0) + rng.normal(size=(800, 5))

    data = ad.AnnData(X=matrix.astype("float32"))
    data.obs["real"] = real
    data.obs["meaningless"] = rng.permutation(np.repeat(["a", "b"], 400))
    sc.pp.neighbors(data, n_neighbors=15, use_rep="X", random_state=0)
    return data


def test_purity_is_high_for_a_label_the_graph_knows(graphed):
    result = analysis.neighbour_purity(graphed, graphed.obs.real)
    assert result["observed"] > 0.99
    assert result["ratio"] > 1.9          # chance is 0.5 for two equal groups


def test_purity_is_one_for_a_label_the_graph_does_not_know(graphed):
    result = analysis.neighbour_purity(graphed, graphed.obs.meaningless)
    assert 0.9 < result["ratio"] < 1.1, result


def test_within_restricts_to_blocks_and_rescales_chance(graphed):
    """Inside one blob the real label is constant, so it carries no information."""
    result = analysis.neighbour_purity(
        graphed, graphed.obs.real, within=graphed.obs.real)
    assert result["observed"] == pytest.approx(1.0)
    assert result["expected"] == pytest.approx(1.0)
    assert result["ratio"] == pytest.approx(1.0)


def test_purity_reads_the_named_graph(graphed):
    sc = pytest.importorskip("scanpy")
    data = graphed.copy()
    sc.pp.neighbors(data, n_neighbors=5, use_rep="X", key_added="tight", random_state=0)
    default = analysis.neighbour_purity(data, data.obs.real)
    tight = analysis.neighbour_purity(data, data.obs.real, neighbors_key="tight")
    assert tight["pairs"] < default["pairs"]


def test_self_pairs_never_count(graphed):
    """Some scanpy backends store each cell as its own neighbour; some do not.

    Whether they do must not change the answer, so inject the self-loops and
    check the result is identical.
    """
    import scipy.sparse as sp

    key = graphed.uns["neighbors"]["distances_key"]
    clean = analysis.neighbour_purity(graphed, graphed.obs.meaningless)

    looped = graphed.copy()
    graph = looped.obsp[key].tolil()
    graph.setdiag(1e-12)
    looped.obsp[key] = sp.csr_matrix(graph)
    assert looped.obsp[key].nnz == graphed.obsp[key].nnz + graphed.n_obs

    with_loops = analysis.neighbour_purity(looped, looped.obs.meaningless)
    assert with_loops == clean


def test_exclude_removes_a_nested_labels_borrowed_signal():
    """A label that only groups because it contains something that groups."""
    sc = pytest.importorskip("scanpy")
    rng = np.random.default_rng(1)
    # four clumps, loose enough that neighbourhoods cross between them;
    # "pair" bundles them two at a time and means nothing on its own
    clump = np.repeat(np.arange(4), 250)
    matrix = np.eye(4)[clump] * 3.0 + rng.normal(size=(1000, 4))
    data = ad.AnnData(X=matrix.astype("float32"))
    data.obs["clump"] = clump.astype(str)
    data.obs["pair"] = (clump // 2).astype(str)
    sc.pp.neighbors(data, n_neighbors=15, use_rep="X", random_state=0)

    naive = analysis.neighbour_purity(data, data.obs.pair)
    adjusted = analysis.neighbour_purity(data, data.obs.pair, exclude=data.obs.clump)

    assert adjusted["pairs"] > 0, "clumps too tight -- no cross-clump neighbours"
    assert naive["ratio"] > 1.3, naive              # looks like real structure
    assert adjusted["ratio"] < naive["ratio"] - 0.2, (naive, adjusted)


# --------------------------------------------------------------------------
# effect_table
# --------------------------------------------------------------------------

@pytest.fixture
def drifting_wells():
    """Well means where the controls develop over time and DrugA adds 2.0 on top.

    This is what the clean object now looks like: the origin is fixed at the
    first timepoint, so the controls are only at zero there.
    """
    rows = []
    for timepoint in TIMEPOINTS:
        drift = 0.5 * (timepoint - TIMEPOINTS[0]) / 12
        for condition, extra in [("DMSO", 0.0), ("DrugA", 2.0)]:
            for replicate in range(3):
                rows.append({"condition": condition, "timepoint": timepoint,
                             "well": f"{condition}{timepoint}{replicate}",
                             "M0": drift + extra + 0.01 * replicate})
    return pd.DataFrame(rows)


def test_effect_table_subtracts_the_controls_own_drift(drifting_wells):
    """A group mean would report the controls' development as a drug effect.

    The planted effect is 2.0 at every timepoint. Without an explicit
    subtraction the late timepoints would read 2.5, 3.0 and 4.0.
    """
    effects = analysis.effect_table(drifting_wells, ["M0"])
    assert np.allclose(effects["M0"].values, 2.0, atol=1e-6), effects


def test_effect_table_pooled_over_time_also_subtracts(drifting_wells):
    effects = analysis.effect_table(drifting_wells, ["M0"], by_timepoint=False)
    assert effects.loc["DrugA", "M0"] == pytest.approx(2.0, abs=1e-6)


def test_effect_table_drops_the_control_row(drifting_wells):
    for by_timepoint in (True, False):
        effects = analysis.effect_table(drifting_wells, ["M0"], by_timepoint=by_timepoint)
        assert "DMSO" not in effects.index.get_level_values(0)
