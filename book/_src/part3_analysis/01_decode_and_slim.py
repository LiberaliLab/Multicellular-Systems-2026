# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (MCS 2026)
#     language: python
#     name: mcs2026
# ---

# %% [markdown]
# # 01 · Decode and slim
#
# In this notebook you will:
#
# - open a 13 GB table without loading it into memory
# - parse all 4,464 feature names, and **assert** that none were missed
# - join the chapter-00 decoder so every column knows its marker and theme
# - tidy the cell metadata into something you can plot without surprises
# - decide what to throw away, and write down why
# - save a slim table that every later chapter opens in seconds
#
# :::{important}
# Run this notebook **once**. Everything after it works on the file it writes.
#
# It is also the notebook people expect to be the expensive one, and it is not: the 13 GB
# file is opened `backed`, so it is never in memory. The peak here is about 8 GB, when the
# slim table is materialised at the end. Chapter 02 costs twice that.
# :::
#
# ## The problem
#
# The feature table arrives as raw as it gets:
#
# ```text
# AnnData object with n_obs × n_vars = 733556 × 4464
#     obs: 'label', 'well_name', 'ROI', 'is_border_internal', 'is_border_external',
#          'Barcode', 'Path', 'ABs', 'Medium', 'Concentration', 'Cell_line', 'Organoid_ID'
# ```
#
# No layers, no embeddings, and `var` has **no columns at all** — 4,464 strings and
# nothing else. The 13 GB is a dense `float32` array, so simply loading it and calling
# one scanpy function that copies would need 26 GB.
#
# By the end of this notebook it is annotated, a third of the size, and ready to work on.

# %%
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path.cwd().parents[1] / "src"))

from mcs2026 import decode, layout, panels, plotting
from mcs2026.config import H5AD_FULL, H5AD_SLIM, LAYOUT_XLSX

plotting.set_style()
pd.set_option("display.width", 140)

# %% [markdown]
# ## 1. Open it backed
#
# `backed="r"` reads the metadata and leaves the matrix on disk. `obs`, `var` and the
# shape come back immediately; `X` is a lazy handle you only pay for when you slice it.
#
# This is the habit worth taking away: **look before you load.**

# %%
adata = sc.read_h5ad(H5AD_FULL, backed="r")
adata

# %%
print(f"{adata.n_obs:,} cells x {adata.n_vars:,} features")
print(f"dense float32 in memory: {adata.n_obs * adata.n_vars * 4 / 1e9:.1f} GB")
print(f"var columns: {list(adata.var.columns)}   <- empty")

# %% [markdown]
# ## 2. Parse the feature names
#
# The names follow a grammar:
#
# ```text
# cells_Intensity_mean_intensity_Texas Red_0
#       |-family-||--statistic--||-channel-||round|
#
# cells_Texture_Haralick-Mean-contrast-2_Cy5_24
# cells_Morphology_area_18                      <- no channel
# cells_Population_n_neighbours_radius_300_4    <- no channel
# ```
#
# Two families carry a channel, because they measure a stain. Two do not, because they
# measure the segmentation mask itself.
#
# :::{warning}
# `Texas Red` contains a **space**. A naive `name.split("_")` looks like it works — until
# 630 columns silently come out wrong. `mcs2026.decode` matches the channel names
# explicitly instead.
# :::

# %%
var = decode.parse_var_names(adata.var_names)
var.head()

# %%
unparsed = var.index[var.family.isna()]
print(f"names that did not parse: {len(unparsed)}")
assert len(unparsed) == 0, f"decoder does not cover: {list(unparsed[:5])}"

# %% [markdown]
# That assertion is the most valuable line in the notebook. If this dataset is ever
# replaced by another plate with a different panel, it fails here — loudly, on the first
# cell — rather than 200 cells later in a plot that looks plausible and is wrong.
#
# ### What the matrix is actually made of

# %%
decode.structure_report(var)

# %% [markdown]
# **Texture is three-quarters of the table.** 52 Haralick features plus 6 Laws texture
# energies, per channel, per round. Haralick features describe how intensity is
# *arranged* within a cell — whether a signal is smooth or punctate, clustered or
# uniform. For organelle markers that is exactly the interesting part: a lysosome marker
# is not informative because it is bright, but because it is *speckled*.
#
# **Intensity is only 6.5%** — the five familiar summary statistics per channel-round.
#
# **Morphology and Population appear once per round** — 18 copies each, describing the
# same cells. Whether they were re-measured or merely copied is a question we settle with
# one line of code below, and the answer decides whether they are a quality signal or
# dead weight.

# %% [markdown]
# ## 3. Join the decoder
#
# Now the important step. `(channel, round)` from the column names, joined against the
# staining sheet from chapter 00.

# %%
stainings = layout.read_stainings(LAYOUT_XLSX)
var = decode.annotate(adata.var_names, stainings)
var.sample(6, random_state=0)[["family", "statistic", "channel", "round", "marker", "theme"]]

# %% [markdown]
# ### Check the join is total — in both directions

# %%
report = decode.check(var, stainings)
for key, value in report.items():
    print(f"  {key:38s} {value}")

# %% [markdown]
# `decode.check` asserts two things, and both matter:
#
# 1. **Nothing in the data that the sheet cannot name.** A `(channel, round)` with no
#    antibody would mean the sheet is incomplete, and any marker-based analysis would
#    have a hole in it.
# 2. **Nothing in the sheet without data.** An antibody listed but never measured would
#    mean a marker silently missing from every panel.
#
# Both sets are empty. 58 channel-round combinations, all accounted for.

# %% [markdown]
# ## 4. Tidy the cell metadata
#
# `obs` needs work too. Two of its columns are actively misleading.

# %%
adata.obs.head(3)

# %%
for column in ["Medium", "Concentration", "Cell_line", "Barcode", "ABs"]:
    print(f"{column:15s} {adata.obs[column].nunique():>3} levels  "
          f"{list(adata.obs[column].unique()[:5])}")

# %% [markdown]
# :::{warning}
# **`Concentration` is not a concentration. It holds `'36h'`, `'48h'`, `'60h'`, `'84h'` —
# it is the timepoint.**
#
# A column whose name says one thing and whose contents say another is a bug waiting to
# happen: someone will eventually treat it as a dose. Rename it now, while you are
# looking at it.
# :::

# %%
obs = adata.obs.copy()

# Timepoint: strip the 'h', and make it an ORDERED categorical so that plots and
# groupbys put 36 before 84 rather than sorting it as text.
obs["timepoint_h"] = pd.Categorical(
    obs["Concentration"].astype(str).str.rstrip("h").astype(int),
    categories=[36, 48, 60, 84], ordered=True,
)

# Condition: swap the Cnd codes for compound names, and make DMSO the FIRST level so
# every default comparison is against the vehicle control.
compounds = layout.read_conditions(LAYOUT_XLSX).set_index("condition_code")["compound"]
ordered = ["DMSO", "PBS"] + sorted(set(compounds) - {"DMSO", "PBS"})
obs["condition"] = pd.Categorical(
    obs["Medium"].map(compounds), categories=ordered, ordered=False
)
obs["condition_code"] = obs["Medium"].astype(str)

# Well position, for plate maps and for testing at the right unit.
obs["well"] = obs["well_name"].astype(str)
obs["row"] = obs["well"].str[0]
obs["column"] = obs["well"].str[1:].astype(int)

obs[["well", "row", "column", "condition", "timepoint_h"]].head()

# %% [markdown]
# Four columns hold exactly one value each. They are provenance — true of the whole
# experiment, not of any particular cell — so they belong in `uns`, not in a
# 733,556-row column.

# %%
provenance = {c: adata.obs[c].unique()[0] for c in ["Barcode", "Cell_line", "ABs", "Path"]}
provenance

# %%
obs = obs.drop(columns=["Barcode", "Cell_line", "ABs", "Path", "Medium", "Concentration"])
print("obs columns now:", list(obs.columns))

# %% [markdown]
# ### Sanity check against the plate layout
#
# The layout workbook says which condition is in which well. `obs` says so too, from a
# completely separate route. They must agree — and if they do not, something is wrong
# with the plate map or with the image-to-well assignment, and nothing downstream is
# trustworthy.

# %%
wells = layout.read_wells(LAYOUT_XLSX)
from_obs = obs.groupby("well", observed=True)["condition_code"].first()
from_sheet = wells.set_index("well")["condition_code"]

shared = from_obs.index.intersection(from_sheet.index)
mismatched = shared[from_obs[shared].values != from_sheet[shared].values]
print(f"wells in obs: {len(from_obs)} | in sheet: {len(from_sheet)} | shared: {len(shared)}")
print(f"conditions disagreeing: {len(mismatched)}")
assert len(mismatched) == 0, f"layout and obs disagree on {list(mismatched[:5])}"

# %% [markdown]
# ## 5. Decide what to drop
#
# This is the part that actually makes the file smaller, and the part where you have to
# think. Every drop below is a decision with a reason, not a default.

# %% [markdown]
# ### The two failed stains
#
# From chapter 00: PDGFRα in round 0 and Integrin β1 in round 28 are marked `failed`.
# Their columns are still here, still full of numbers.

# %%
var[var.failed].groupby(["marker", "round", "channel"], observed=True).size().rename("columns")

# %% [markdown]
# ### Morphology and Population, repeated 18 times
#
# These describe the segmentation mask — area, eccentricity, how many neighbours a cell
# has. They do not depend on which antibody was used, so there is no obvious reason to
# measure them 18 times.
#
# But "no obvious reason" is not evidence. Before dropping 765 columns, **check the
# assumption**: are these re-measured each round, or copied?

# %%
area_columns = var.loc[
    var.index[(var.family == "Morphology") & (var.statistic == "area")]
].sort_values("round").index
area = adata[:, list(area_columns)].to_memory().X

identical = (area == area[:, :1]).all(axis=0)
pd.DataFrame({
    "round": var.loc[area_columns, "round"].astype(int).values,
    "median_area": np.median(area, axis=0),
    "identical_to_round_0": identical,
}).set_index("round").T

# %% [markdown]
# **Byte-identical in all 18 rounds.** Not similar — the same numbers.
#
# So the cells were segmented once, and the resulting measurements were joined alongside
# every round when the table was assembled. Keeping one round is provably lossless, and
# drops 765 columns.
#
# :::{note}
# It was worth one cell to check. *Measured* per round and *copied* per round look
# identical in a column name and mean completely different things: the first is a quality
# signal you would want to plot, the second is dead weight. Had they differed, that
# drift would have been a finding — cells changing shape across elution cycles is a real
# failure mode of 4i.
# :::

# %% [markdown]
# ### DAPI is a different story
#
# DAPI is also in every round, and it would be easy to wave it away the same way. Do the
# same check before deciding.

# %%
dapi_columns = var.loc[
    var.index[(var.family == "Intensity")
              & (var.statistic == "mean_intensity")
              & (var.channel == "DAPI")]
].sort_values("round").index
dapi = adata[:, list(dapi_columns)].to_memory().X

dapi_by_round = pd.Series(
    np.median(dapi, axis=0),
    index=var.loc[dapi_columns, "round"].astype(int).values,
    name="median DAPI intensity",
)
print("identical to round 0:", int((dapi == dapi[:, :1]).all(axis=0).sum()), "of", dapi.shape[1])
dapi_by_round

# %%
ax = dapi_by_round.plot(marker="o", figsize=(6.5, 3.4))
ax.set(xlabel="imaging round", ylabel="median DAPI intensity (a.u.)",
       title="DAPI is re-imaged every round — and it drifts")
ax.set_ylim(0, dapi_by_round.max() * 1.15)
ax.set_xticks(dapi_by_round.index)
ax.tick_params(axis="x", labelsize=7)

# %%
print(f"rounds 0-8:    median {dapi_by_round.loc[:8].median():.0f}")
print(f"rounds 14-28:  median {dapi_by_round.loc[14:].median():.0f}")
print(f"largest step:  round 8 -> 14, {dapi_by_round.loc[8] / dapi_by_round.loc[14]:.0f}x drop")

# %% [markdown]
# DAPI is genuinely re-measured, and it moves a *long* way: the median falls about
# fifteen-fold between round 8 and round 14, then settles at roughly a quarter of where
# it started. Note where that step is — exactly the gap where rounds 5 to 13 are missing
# from the panel. Something changed about the imaging between those sessions.
#
# The nuclei did not change. The measurement did.
#
# Two consequences, and they shape everything after this notebook:
#
# **1. Keep the per-round DAPI intensity.** It is 90 columns, and it is the record of
# how staining and imaging drifted. Throwing it away would discard the only evidence of
# this problem. (DAPI *texture* is a different matter — 1,044 columns of chromatin
# arrangement in a reference channel. One round is plenty.)
#
# **2. Intensities from different rounds are not comparable.** A marker imaged in round
# 15 and one imaged in round 0 sit on different scales for reasons that have nothing to
# do with biology. Any analysis that puts them side by side — a heatmap across markers,
# a PCA over the whole panel — is partly measuring the round.
#
# :::{important}
# This is why chapter 02 normalises **within round, against the DMSO controls**, rather
# than z-scoring the whole matrix at once. The design puts control wells in every round,
# which is what makes the correction possible.
# :::

# %% [markdown]
# ### Apply the decisions

# %%
keep = decode.slim_mask(
    var,
    keep_structural_round=0,        # verified identical above: lossless
    keep_dapi_intensity=True,       # the drift record, 90 columns
    keep_dapi_texture_round=0,      # chromatin texture in the reference channel
    drop_failed=True,               # the two stains that did not work
    drop_texture=False,             # keep Haralick: the organelle panel needs it
)
decode.slim_report(var, keep, adata.n_obs)

# %%
dropped = var[~keep]
pd.Series({
    "failed stains": int(dropped.failed.sum()),
    "duplicate morphology / population": int(dropped.is_structural.sum()),
    "DAPI texture beyond round 0": int((dropped.is_reference & (dropped.family == "Texture")).sum()),
    "total dropped": len(dropped),
}, name="columns")

# %% [markdown]
# :::{admonition} If your session is still too small
# :class: tip
#
# `drop_texture=True` removes the 3,364 texture columns and takes the table under 1 GB.
# You lose the sub-cellular structure that the organelle panel is largely about, so it is
# a real cost — but a working notebook beats a killed kernel.
# :::

# %% [markdown]
# ## 6. Materialise and write
#
# Only now do we read pixels off disk — and only the columns we decided to keep.

# %%
slim = adata[:, keep.values].to_memory()
slim.obs = obs
slim.var = var[keep].copy()
slim.uns["provenance"] = provenance
slim.uns["decode_report"] = {k: str(v) for k, v in report.items()}
slim

# %%
slim.write_h5ad(H5AD_SLIM, compression="gzip")
print(f"wrote {H5AD_SLIM}")
print(f"  {slim.n_obs:,} cells x {slim.n_vars:,} features")
print(f"  {slim.n_obs * slim.n_vars * 4 / 1e9:.2f} GB in memory "
      f"(was {adata.n_obs * adata.n_vars * 4 / 1e9:.1f} GB)")

# %% [markdown]
# ## What we gained
#
# `var` is no longer a list of strings. It is a table you can ask questions of:

# %%
slim.var.groupby("theme", observed=True).agg(
    columns=("marker", "size"), markers=("marker", "nunique")
).sort_values("columns", ascending=False)

# %%
panels.panel_table(slim.var)

# %% [markdown]
# Every panel resolves completely. `resolve_panel` is how the theme chapters select
# their features:

# %%
signaling = panels.resolve_panel(slim.var, "signaling")
slim.var.loc[signaling, ["marker", "round", "channel", "statistic"]]

# %% [markdown]
# ---
#
# ## Exercises
#
# ### 1. What did keeping Haralick cost?
#
# Compute the slim size with `drop_texture=True` and compare. What fraction of the
# remaining table is texture? Would you keep it?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# lean = decode.slim_mask(var, drop_texture=True)
# print(decode.slim_report(var, lean, adata.n_obs))
# print(var[keep].family.value_counts(normalize=True))
# ```
#
# Texture is around 90% of the *slim* table — dropping it takes ~2,500 features down to
# ~240, and the file from ~7 GB to under 1 GB.
#
# Whether to keep it depends on the question. For the signaling panel, mean intensity is
# most of the signal and texture adds little. For organelles, texture *is* the signal:
# whether Golgi is compact or dispersed is a texture measurement, not a brightness one.
# The course keeps it for that reason.
# :::

# %% [markdown]
# ### 2. Find a marker that moved rounds
#
# Using `slim.var`, find every marker measured in more than one round. Why is it there
# twice, and which copy should an analysis use?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# per_marker = slim.var.groupby("marker", observed=True)["round"].nunique()
# per_marker[per_marker > 1]
# ```
#
# Only `PDGFRa`, in round 18. Round 0's copy failed and was dropped by `slim_mask`, so
# the surviving one is automatically the good one — which is the point of encoding
# `failed` in the sheet rather than remembering it.
# :::

# %% [markdown]
# ### 3. Cells per well
#
# Count cells per well and draw it as a plate map (`plotting.plate_map`). Is the
# variation random, or does it have structure? What would a row-wise gradient mean, given
# that each condition sits in fixed rows?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# counts = (slim.obs.groupby("well", observed=True)
#           .size().rename("n_cells").reset_index())
# counts["row"] = counts.well.str[0]
# counts["column"] = counts.well.str[1:].astype(int)
# plotting.plate_map(counts, "n_cells", cmap="magma", title="Cells per well")
# ```
#
# Cell number is itself a phenotype — a compound that kills cells or blocks division
# gives fewer. So structure here is expected and interesting.
#
# The trap is that condition and row are confounded by design. A smooth gradient down the
# plate could be an edge or pipetting artefact rather than biology, and you cannot
# separate the two from this plate alone. Chapter 02 checks the controls specifically,
# because DMSO wells appear in every row and so give a position readout at fixed
# treatment.
# :::

# %% [markdown]
# ---
#
# **Next:** [02 · QC and normalisation](02_qc_and_normalisation.ipynb) — border cells,
# staining thresholds, plate effects, and measuring everything against DMSO.
