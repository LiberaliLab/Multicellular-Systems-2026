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

CONDITIONS = ["DMSO", "DrugA", "DrugB", "DrugC"]
TIMEPOINTS = [36, 48, 60, 84]
MARKERS = [f"M{i}" for i in range(6)]


@pytest.fixture(scope="module")
def plate():
    """A miniature plate: 4 conditions x 4 timepoints x 3 wells x 40 cells.

    Timepoints carry a large offset so that anything failing to normalise
    *within* timepoint shows up immediately; ``DrugA`` shifts M0 only.
    """
    rng = np.random.default_rng(0)
    rows, values = [], []
    for timepoint in TIMEPOINTS:
        for condition in CONDITIONS:
            for replicate in range(3):
                well = f"{condition[:3]}{timepoint}{replicate}"
                block = rng.normal(loc=100.0 + 40 * timepoint, scale=5.0,
                                   size=(40, len(MARKERS)))
                if condition == "DrugA":
                    block[:, 0] += 60.0
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
    return data


# --------------------------------------------------------------------------
# normalise_cells
# --------------------------------------------------------------------------

def test_controls_centre_at_zero_within_each_timepoint(plate):
    normalised = analysis.normalise_cells(plate, MARKERS)
    controls = plate.obs.condition.values == "DMSO"
    timepoint = plate.obs.timepoint_h.astype(int).values

    for value in TIMEPOINTS:
        block = normalised[controls & (timepoint == value)]
        assert np.allclose(block.mean(axis=0), 0, atol=1e-5)
        assert np.allclose(block.std(axis=0, ddof=1), 1, atol=1e-5)


def test_timepoint_offset_does_not_survive(plate):
    """The raw values differ 15-fold across timepoints; the normalised must not."""
    normalised = analysis.normalise_cells(plate, MARKERS)
    timepoint = plate.obs.timepoint_h.astype(int).values

    raw_spread = np.ptp([plate.X[timepoint == v].mean() for v in TIMEPOINTS])
    normalised_spread = np.ptp([normalised[timepoint == v].mean() for v in TIMEPOINTS])
    assert raw_spread > 1000
    assert normalised_spread < 0.2


def test_the_planted_effect_is_recovered(plate):
    normalised = analysis.normalise_cells(plate, MARKERS)
    treated = plate.obs.condition.values == "DrugA"

    shifts = normalised[treated].mean(axis=0)
    assert shifts[0] > 8           # 60 units against a control SD of ~5
    assert np.abs(shifts[1:]).max() < 0.5


def test_output_is_finite_float32(plate):
    normalised = analysis.normalise_cells(plate, MARKERS)
    assert normalised.dtype == np.float32
    assert np.isfinite(normalised).all()


def test_a_dead_column_becomes_zeros_not_infinities(plate):
    """A marker with no spread in the controls carries no information."""
    flat = plate.copy()
    flat.X[:, 2] = 7.0
    normalised = analysis.normalise_cells(flat, MARKERS)
    assert np.isfinite(normalised).all()
    assert np.allclose(normalised[:, 2], 0)


def test_a_group_without_controls_is_an_error(plate):
    orphan = plate[plate.obs.condition != "DMSO"].copy()
    with pytest.raises(ValueError, match="cannot scale a group without controls"):
        analysis.normalise_cells(orphan, MARKERS)


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
