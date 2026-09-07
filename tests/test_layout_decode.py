"""Regression tests for the plate-layout parser and the feature-name decoder.

These lock in the exact structure of the course dataset. They need only the
layout workbook and the committed list of column names -- not the 13 GB
feature table -- so they run anywhere in about a second.

    pytest tests/
"""

from pathlib import Path

import pandas as pd
import pytest

from mcs2026 import decode, layout, panels

REPO = Path(__file__).resolve().parents[1]
XLSX = REPO / "metadata" / "L_ayout_384_Haralick_Thresholds.xlsx"
VAR_NAMES = REPO / "metadata" / "var_names.txt"

N_OBS = 733_556  # cells in 1_FE_pooled1.h5ad, for the memory arithmetic


@pytest.fixture(scope="module")
def stainings() -> pd.DataFrame:
    return layout.read_stainings(XLSX)


@pytest.fixture(scope="module")
def wells() -> pd.DataFrame:
    return layout.read_wells(XLSX)


@pytest.fixture(scope="module")
def var(stainings) -> pd.DataFrame:
    names = VAR_NAMES.read_text().splitlines()
    return decode.annotate(names, stainings)


# --------------------------------------------------------------------------
# the plate
# --------------------------------------------------------------------------
def test_eighteen_conditions_with_two_controls():
    conditions = layout.read_conditions(XLSX)
    assert len(conditions) == 18
    assert not conditions["condition_code"].duplicated().any()
    assert set(conditions.loc[conditions["is_control"], "compound"]) == {"DMSO", "PBS"}


def test_224_used_wells_over_four_timepoints(wells):
    assert len(wells) == 224
    assert not wells["well"].duplicated().any()
    assert sorted(wells["timepoint_h"].unique()) == [36, 48, 60, 84]
    assert set(wells["row"]) == set("BCDEFGHIJKLMNO")
    assert wells.groupby("timepoint_h", observed=True).size().eq(56).all()


def test_dmso_is_replicated_more_than_the_rest(wells):
    """The design is deliberately unbalanced: 5 DMSO wells per timepoint, 3 of
    everything else. Every later comparison normalises to DMSO, so it gets the
    extra replicates. 17 * 3 + 5 == 56."""
    per_group = wells.groupby(["condition_code", "timepoint_h"], observed=True).size()
    assert per_group.loc["Cnd18"].eq(5).all()
    assert per_group.drop("Cnd18").eq(3).all()


# --------------------------------------------------------------------------
# the staining sheet
# --------------------------------------------------------------------------
def test_eighteen_rounds_forty_stains_two_failed(stainings):
    assert stainings["round"].nunique() == 18
    assert sorted(stainings["round"].unique()) == [
        0, 1, 2, 3, 4, 8, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 27, 28
    ]
    non_dapi = stainings[stainings["marker"] != "DAPI"]
    assert len(non_dapi) == 40
    failed = stainings[stainings["failed"]]
    assert len(failed) == 2
    assert set(zip(failed["round"], failed["marker"])) == {
        (0, "PDGFRa"), (28, "Integrin-b1")
    }


@pytest.mark.parametrize(
    ("channel", "round_", "marker"),
    [
        ("Texas Red", 0, "Foxo3a"),
        ("FITC", 0, "beta-Catenin"),
        ("FITC", 1, "Nanog"),
        ("Cy5", 24, "p-AKT"),
        ("FITC", 28, "Calreticulin"),
        ("Cy5", 28, "Integrin-b1"),
    ],
)
def test_channel_round_decodes_to_the_right_marker(stainings, channel, round_, marker):
    match = stainings[
        (stainings["channel"] == channel) & (stainings["round"] == round_)
    ]
    assert list(match["marker"]) == [marker]


# --------------------------------------------------------------------------
# the feature names
# --------------------------------------------------------------------------
def test_every_feature_name_parses(var):
    assert len(var) == 4464
    assert var["family"].notna().all()


def test_feature_block_sizes(var):
    counts = var["family"].value_counts().to_dict()
    assert counts == {
        "Texture": 3364,     # 58 statistics x 58 channel-rounds
        "Population": 414,   # 23 descriptors x 18 rounds
        "Morphology": 396,   # 22 descriptors x 18 rounds
        "Intensity": 290,    # 5 statistics  x 58 channel-rounds
    }
    assert sum(counts.values()) == 4464


def test_decode_is_total_in_both_directions(var, stainings):
    """Nothing in the data the sheet cannot name, nothing in the sheet without
    data. This is the assertion that catches a changed panel."""
    report = decode.check(var, stainings)
    assert report["channel_round_combinations"] == 58
    assert report["combinations_in_sheet_without_data"] == []
    assert report["features_with_a_marker"] == 3654
    assert report["failed_stain_features"] == 126
    assert report["reference_dapi_features"] == 1134
    assert report["structural_features"] == 810


# --------------------------------------------------------------------------
# the panels
# --------------------------------------------------------------------------
def test_every_panel_resolves_completely(var):
    table = panels.panel_table(var)
    incomplete = table.index[~table["complete"]].tolist()
    assert not incomplete, f"panels missing markers: {incomplete}"


def test_panels_deliberately_overlap():
    """Calreticulin is ER and metabolism; p-S6 is signaling and metabolism."""
    assert "Calreticulin" in panels.PANELS["metabolism"]
    assert "Calreticulin" in panels.PANELS["organelles"]
    assert "p-S6" in panels.PANELS["signaling"]
    assert "p-S6" in panels.PANELS["metabolism"]


def test_failed_stains_never_reach_a_panel(var):
    for theme in panels.PANELS:
        columns = panels.resolve_panel(var, theme, verbose=False)
        assert not var.loc[columns, "failed"].any()


# --------------------------------------------------------------------------
# slimming
# --------------------------------------------------------------------------
def test_slimming_keeps_the_dapi_drift_record(var):
    """DAPI intensity is kept for every round on purpose.

    Morphology and Population are byte-identical across rounds, so one round is
    lossless. DAPI is not: it is genuinely re-imaged each round and its level
    moves a lot, which is exactly what makes it the record of staining drift.
    So DAPI *intensity* survives for all 18 rounds while DAPI *texture* does not.
    """
    keep = decode.slim_mask(var)
    dapi = var["is_reference"]
    texture = var["family"] == "Texture"

    kept_dapi_intensity = var.loc[keep & dapi & ~texture, "round"]
    assert kept_dapi_intensity.nunique() == 18

    kept_dapi_texture = var.loc[keep & dapi & texture, "round"]
    assert kept_dapi_texture.unique().tolist() == [0]


def test_slimming_drops_only_what_it_should(var):
    keep = decode.slim_mask(var)
    assert not var.loc[keep, "failed"].any()
    # the duplicated structural blocks collapse to one round
    assert var.loc[keep & var["is_structural"], "round"].unique().tolist() == [0]
    assert var.loc[keep & var["is_structural"]].shape[0] == 45  # 22 morphology + 23 population

    dropped = var[~keep]
    assert dropped["failed"].sum() == 126
    assert dropped["is_structural"].sum() == 765          # 17 of 18 duplicate rounds
    assert len(dropped) == 126 + 765 + 986

    report = decode.slim_report(var, keep, N_OBS)
    assert report.loc[0, "GB_float32"] > 13
    assert report.loc[1, "GB_float32"] < 8


def test_dropping_texture_is_the_escape_hatch(var):
    """For a session that cannot hold 8 GB, texture is the thing to sacrifice."""
    lean = decode.slim_mask(var, drop_texture=True)
    assert (var.loc[lean, "family"] != "Texture").all()
    assert lean.sum() < 400
