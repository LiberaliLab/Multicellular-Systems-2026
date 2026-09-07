"""Turn bare feature-column names into an annotated ``var`` table.

The single-cell table arrives with ``var`` completely empty: 4,464 strings and
nothing else. The strings follow a grammar, but crucially **the marker is not
in the name** -- only the channel and the imaging round are::

    cells_Intensity_mean_intensity_Texas Red_0
          |-family-||--statistic--||-channel-||round|

    cells_Texture_Haralick-Mean-contrast-2_Cy5_24
    cells_Morphology_area_18                     <- no channel
    cells_Population_n_neighbours_radius_300_4   <- no channel

Recovering the marker means joining ``(channel, round)`` against the staining
sheet -- see :mod:`mcs2026.layout`. That join is what this module does, and it
is the single most important step in Part 3.
"""

from __future__ import annotations

import re

import pandas as pd

from mcs2026.layout import CHANNELS

#: Feature families that carry a channel (one antibody per channel per round).
CHANNEL_FAMILIES = ("Intensity", "Texture")
#: Feature families measured once per round on the segmentation itself.
STRUCTURAL_FAMILIES = ("Morphology", "Population")

_CHANNEL_ALT = "|".join(re.escape(c) for c in sorted(CHANNELS, key=len, reverse=True))

_CHANNEL_RE = re.compile(
    rf"^cells_(?P<family>[^_]+)_(?P<statistic>.+)_(?P<channel>{_CHANNEL_ALT})_(?P<round>\d+)$"
)
_STRUCTURAL_RE = re.compile(
    r"^cells_(?P<family>Morphology|Population)_(?P<statistic>.+)_(?P<round>\d+)$"
)


def parse_var_names(var_names) -> pd.DataFrame:
    """Split every feature name into ``family``, ``statistic``, ``channel``, ``round``.

    ``channel`` is ``None`` for Morphology and Population features, which are
    measured on the segmentation mask and so have no channel.

    Anything that matches neither pattern comes back with ``family = None``.
    Callers should assert that this never happens -- an unparsed name means the
    panel changed and the decoder needs updating, and it is far better to fail
    loudly here than to silently analyse 4,000 anonymous numbers.
    """
    records = []
    for name in var_names:
        match = _CHANNEL_RE.match(name)
        if match is not None:
            group = match.groupdict()
            records.append(
                {
                    "feature": name,
                    "family": group["family"],
                    "statistic": group["statistic"],
                    "channel": group["channel"],
                    "round": int(group["round"]),
                }
            )
            continue

        match = _STRUCTURAL_RE.match(name)
        if match is not None:
            group = match.groupdict()
            records.append(
                {
                    "feature": name,
                    "family": group["family"],
                    "statistic": group["statistic"],
                    "channel": None,
                    "round": int(group["round"]),
                }
            )
            continue

        records.append(
            {"feature": name, "family": None, "statistic": None,
             "channel": None, "round": None}
        )

    out = pd.DataFrame.from_records(records)
    out["round"] = out["round"].astype("Int64")
    return out.set_index("feature")


def annotate(var_names, stainings: pd.DataFrame, themes: dict | None = None) -> pd.DataFrame:
    """Full ``var`` annotation: parse the names, then join the staining sheet.

    Parameters
    ----------
    var_names
        ``adata.var_names``.
    stainings
        Output of :func:`mcs2026.layout.read_stainings`.
    themes
        Marker -> theme mapping. Defaults to :data:`mcs2026.panels.MARKER_THEME`.

    Returns a table indexed by feature name with, on top of the parsed parts:
    ``marker``, ``theme``, ``species``, ``intensity_threshold``,
    ``haralick_threshold``, ``failed`` and ``is_reference`` (the DAPI channel).
    """
    if themes is None:
        from mcs2026.panels import MARKER_THEME

        themes = MARKER_THEME

    var = parse_var_names(var_names)

    key = stainings.set_index(["channel", "round"])
    columns = ["marker", "species", "intensity_threshold", "haralick_threshold", "failed"]
    joined = var.join(key[columns], on=["channel", "round"])

    joined["failed"] = joined["failed"].notna() & joined["failed"].astype("boolean").fillna(False)
    joined["failed"] = joined["failed"].astype(bool)
    joined["is_reference"] = joined["channel"].eq("DAPI")
    joined["is_structural"] = joined["family"].isin(STRUCTURAL_FAMILIES)
    joined["theme"] = joined["marker"].map(themes)
    # Morphology and Population describe cell shape and neighbourhood, not a
    # marker, so they get their own theme rather than being left unlabelled.
    joined.loc[joined["family"] == "Morphology", "theme"] = "morphology"
    joined.loc[joined["family"] == "Population", "theme"] = "population"
    return joined


def structure_report(var: pd.DataFrame) -> pd.DataFrame:
    """Counts per family -- the table to print when checking a decode."""
    out = (
        var.groupby("family", dropna=False)
        .agg(
            columns=("statistic", "size"),
            statistics=("statistic", "nunique"),
            rounds=("round", "nunique"),
            channels=("channel", "nunique"),
        )
        .sort_values("columns", ascending=False)
    )
    out["share"] = (out["columns"] / len(var)).map("{:.1%}".format)
    return out


def check(var: pd.DataFrame, stainings: pd.DataFrame) -> dict:
    """Assert the decode is complete. Returns the numbers worth printing.

    Raises :class:`AssertionError` if any feature name failed to parse, or if
    any ``(channel, round)`` in the data has no antibody in the sheet.
    """
    unparsed = var.index[var["family"].isna()].tolist()
    assert not unparsed, f"{len(unparsed)} feature names did not parse: {unparsed[:5]}"

    in_data = set(
        map(tuple, var.loc[var["channel"].notna(), ["channel", "round"]].to_numpy())
    )
    in_sheet = set(map(tuple, stainings[["channel", "round"]].to_numpy()))
    missing = in_data - in_sheet
    unused = in_sheet - in_data
    assert not missing, f"channel/round combinations with no antibody in the sheet: {sorted(missing)}"

    named = var["marker"].notna()
    return {
        "features": len(var),
        "channel_round_combinations": len(in_data),
        "combinations_in_sheet_without_data": sorted(unused),
        "features_with_a_marker": int(named.sum()),
        "failed_stain_features": int(var["failed"].sum()),
        "reference_dapi_features": int(var["is_reference"].sum()),
        "structural_features": int(var["is_structural"].sum()),
    }


# --------------------------------------------------------------------------
# deciding what to drop
# --------------------------------------------------------------------------
def slim_mask(
    var: pd.DataFrame,
    *,
    keep_structural_round: int | None = 0,
    keep_dapi_intensity: bool = True,
    keep_dapi_texture_round: int | None = 0,
    drop_failed: bool = True,
    drop_texture: bool = False,
) -> pd.Series:
    """Boolean mask of features to *keep*, with every choice explicit and justified.

    The defaults are not guesses -- each one was checked against the data:

    ``keep_structural_round=0``
        Morphology and Population are **byte-identical in all 18 rounds**. They
        are measured once on the segmentation and copied alongside every round,
        so keeping one round is provably lossless and drops 765 columns.
        Verify it yourself rather than trusting this docstring::

            cols = var.index[(var.family == "Morphology") & (var.statistic == "area")]
            block = adata[:, list(cols)].to_memory().X
            (block == block[:, :1]).all()

    ``keep_dapi_intensity=True``
        DAPI, unlike the above, genuinely **is** re-imaged every round, and its
        level moves a long way across the experiment (the median drops roughly
        sixteen-fold between rounds 8 and 14, then partly recovers). That makes
        the per-round DAPI intensity the record of staining and imaging drift,
        and the reason intensities from different rounds are not directly
        comparable. It is 90 columns. Keep them.

    ``keep_dapi_texture_round=0``
        DAPI *texture* is a different matter -- 1,044 columns describing
        chromatin arrangement in the reference channel. One round is enough.

    ``drop_failed=True``
        Two stains are marked ``failed`` in the staining sheet and are still
        present as 126 live columns. They are noise with a name.

    ``drop_texture=False``
        Texture is 75% of the matrix. Dropping it makes everything fast at the
        cost of the sub-cellular structure the organelle panel is mostly about.
        Set ``True`` if memory is tight.

    Pass ``None`` to either ``keep_*_round`` to keep every round.
    """
    keep = pd.Series(True, index=var.index)
    is_dapi = var["is_reference"]
    is_texture = var["family"] == "Texture"

    if drop_failed:
        keep &= ~var["failed"]
    if drop_texture:
        keep &= ~is_texture
    if keep_structural_round is not None:
        keep &= ~(var["is_structural"] & (var["round"] != keep_structural_round))

    # DAPI is split in two: intensity is the drift record, texture is bulk.
    if keep_dapi_texture_round is not None:
        keep &= ~(is_dapi & is_texture & (var["round"] != keep_dapi_texture_round))
    if not keep_dapi_intensity and keep_dapi_texture_round is not None:
        keep &= ~(is_dapi & ~is_texture & (var["round"] != keep_dapi_texture_round))
    return keep


def slim_report(var: pd.DataFrame, keep: pd.Series, n_obs: int) -> pd.DataFrame:
    """Before/after table, including the float32 memory each version needs."""

    def gigabytes(n_features: int) -> float:
        return n_obs * n_features * 4 / 1e9

    rows = [
        {"stage": "as loaded", "features": len(var), "GB_float32": gigabytes(len(var))},
        {"stage": "after slimming", "features": int(keep.sum()),
         "GB_float32": gigabytes(int(keep.sum()))},
    ]
    out = pd.DataFrame(rows)
    out["GB_float32"] = out["GB_float32"].round(2)
    return out
