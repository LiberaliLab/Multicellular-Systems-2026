# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python (MCS 2026)
#     language: python
#     name: mcs2026
# ---

# %% [markdown]
# # 03 · Normalisation
#
# Steps 15 to 18. Chapter 02 decided what to throw away; this one
# makes what is left **comparable**, and then packs it into the single object.
#
# | | |
# |---|---|
# | **Step 15** | Are intensities comparable across rounds? |
# | **Step 16** | Is any condition an outlier? |
# | **Step 17** | Normalise to the controls |
# | **Step 18** | Assemble the three objects |
#

# %%
import sys
from datetime import date
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

sys.path.insert(0, str(Path.cwd().parents[2] / "src"))

from mcs2026 import analysis, panels, plotting
from mcs2026.config import H5AD_SLIM, LAYOUT_XLSX

plotting.set_style()
pd.set_option("display.width", 140)


### full_adata is the filtered anndata from the previous notebook

full_adata = sc.read_h5ad(H5AD_SLIM.with_name("mcs2026_qc.h5ad"))


print(f"{full_adata.n_obs:,} cells x {full_adata.n_vars:,} features, "
      f"{full_adata.obs.well.nunique()} wells")

# %% [markdown]
# ## Step 15 · Are intensities comparable across rounds?
#
# [Step 9](01_columns_to_markers.ipynb) found that DAPI — the same stain, on the same
# nuclei, every round — moves by more than an order of magnitude across the experiment.
# Here is what that means for the markers.

# %%
dapi_columns = full_adata.var.loc[
    full_adata.var.index[(full_adata.var.channel == "DAPI") & (full_adata.var.statistic == "mean_intensity")]
].sort_values("round").index

dapi = pd.Series(
    np.median(np.asarray(full_adata[:, list(dapi_columns)].X), axis=0),
    index=full_adata.var.loc[dapi_columns, "round"].astype(int).values, name="median DAPI",
)

marker_cols = analysis.marker_columns(full_adata.var)
marker_median = pd.Series(
    np.median(np.asarray(full_adata[:, marker_cols].X), axis=0),
    index=full_adata.var.loc[marker_cols, "round"].astype(int).values,
).groupby(level=0).median()

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(dapi.index, dapi.values, marker="o", label="DAPI (reference stain)")
ax.plot(marker_median.index, marker_median.values, marker="s", label="markers (median)")
ax.set(xlabel="imaging round", ylabel="median intensity (a.u.)", yscale="log",
       title="Signal level depends on the round, not only on the biology")
ax.set_xticks(dapi.index)
ax.tick_params(axis="x", labelsize=7)
ax.legend()

# %% [markdown]
# The nuclei did not change between rounds. The measurement did.
#
# So a heatmap that puts a round-2 marker next to a round-27 marker, or a PCA over the
# whole panel, is partly ranking markers by *when they were imaged*. Two things follow:
#
# - **compare a marker to itself across conditions**, never one marker's absolute level to
#   another's;
# - **centre every marker on its own controls**, which is Step 17.

# %% [markdown]
# ## Step 16 · Is any condition an outlier?
#
# Before normalising, look at the conditions as a whole. Unsupervised methods are
# obedient: hand them one sample that differs from everything else and they will spend
# their first component describing it.

# %%
wells = analysis.by_well(full_adata, marker_cols, name_by="marker")
names = [full_adata.var.loc[c, "marker"] for c in marker_cols]

logged = np.log2(wells[names] + 1)
centred = logged - logged[wells.condition == "DMSO"].mean()
spread = centred.groupby(wells.condition).mean().abs().mean(axis=1).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(7, 4))
colours = ["firebrick" if v == spread.index[0] else "0.5" for v in spread.index]
ax.barh(range(len(spread)), spread.values, color=colours)
ax.set(yticks=range(len(spread)), yticklabels=spread.index,
       xlabel="mean |log2 shift vs DMSO|", title="How far each condition sits from the control")
ax.invert_yaxis()

# %% [markdown]
# Here we show the mean marker intensity (all markers) deviation by condition, with DMSO as the zero.
#

# %% [markdown]
# **PMA moves every marker, by a lot** — several times further than anything else. Phorbol esters drive naive human ES
# cells out of pluripotency, so could be biologically reasonable; however it is still
# an outlier statistically.
#
# :::{important}
# Thus we need to make a decision :
#
# - **keep PMA** and it dominates the first component of every embedding, so the plot
#   describes PMA-vs-everything rather than the structure among the other seventeen;
# - **drop PMA** and you lose a real, strong biological effect;
# - **flag it and analyse it knowingly**, which is what this course does.
#
# What you must not do is leave it in silently and then interpret component 1 as if it
# were about subtle signaling differences.
# :::

# %% [markdown]
# #### Marking the cells coming from the PMA condition
# We can mark these cells in the anndata object, in order to better disntinguish it during analysis.

# %%
full_adata.obs["is_outlier_condition"] = full_adata.obs.condition.astype(str).eq(
    "Phorbol 12-myristate 13-acetate (PMA)"
)
print(f"  flagged {full_adata.obs.is_outlier_condition.sum():,} cells in "
      f"{full_adata.obs.loc[full_adata.obs.is_outlier_condition, 'well'].nunique()} wells")

# %% [markdown]
# ## Step 17 · Normalization: Set the origin and the unit
#
# Normalising requires **two** steps. The **origin** says what counts as zero. The **unit** says what counts
# as one. They answer different questions, they can be estimated from different cells, and
# here they have to be.
#
# ### The formula
#
# $$
# z \;=\; \frac{\log_2(x + 1) \;-\; B_{36}}{S}
# $$
#
# **In one sentence:** how far this cell sits from an untreated cell at the *start* of the
# experiment, counted in units of how much untreated cells normally differ from each other.
#
# | symbol | what it is | Fibronectin |
# |---|---|---|
# | $x$ | the measured mean intensity of one marker in one cell | 120 |
# | $B_{36}$ | **the origin** — median of that marker in control cells at 36 h | 5.88 |
# | $S$ | **the unit** — spread of control cells about their own group, pooled over the plate | 0.71 |
# | $z$ | the answer, in **control SDs** | **+1.47** |
#
# ```{image} ../../images/normalisation_formula.png
# :alt: The three steps of the normalisation, shown on Fibronectin control cells at 36 h.
# :width: 100%
# ```
#
# **Why each piece is there**
#
# - **$\log_2(x+1)$** — intensities multiply rather than add, so a doubling should look the
#   same whether it is 50→100 or 500→1000. The $+1$ is there because a marker that is switched
#   off reads exactly 0 and $\log_2 0$ is undefined; $\log_2(0+1) = 0$ sends "no signal"
#   cleanly to zero. To go back: $x = 2^{z S + B_{36}} - 1$.
#   
# - **$-\,B_{36}$** — one fixed starting line, so the controls' own development across 36–84 h
#   stays visible. Subtract each timepoint's *own* controls instead and every timepoint reads
#   zero by construction — you cannot measure a change against a reference that moves with it.
#
#   
# - **$\div\,S$** — puts every marker in the same units, so a bright marker and a dim one can
#   sit on the same axis. One unit is one typical cell-to-cell difference among untreated cells.
#
#
# :::{note}
# **Two refinements the code makes.** There are two vehicle controls, so $B_{36}$ is the
# *average of the DMSO and PBS medians* rather than one pooled median — the two are not
# interchangeable. And $S$ is built from each control cell's deviation from **its own vehicle
# at its own timepoint** before pooling, so neither the drift across time nor the gap between
# the vehicles is counted as spread.
# :::
#
# :::{tip}
# **One ruler, everywhere.** $B_{36}$ and $S$ are each a single number per marker, used for
# every cell at every timepoint. That is what lets all four timepoints share one UMAP: the same
# measurement always gives the same number, so a GATA4-positive cell is a GATA4-positive cell
# whether it was fixed at 36 h or 84 h.
# :::
#
#
# ### The origin — the controls at the *first* timepoint
#
# The obvious move is to centre each timepoint on its own control wells. It is also the one
# mistake this step exists to prevent.
#
# **A moving origin cannot show movement.** Centre every timepoint on its own controls and
# the control sits at zero at 36 h, zero at 60 h and zero at 84 h — not because nothing
# happened to it, but because you defined it that way. What survives is "different from the
# control *at the same moment*", which is a fair question and not the one a time course
# asks. The controls' own development — the thing that 36 to 84 hours was for — would be
# gone before the first plot.
#
# So the origin is fixed **once** (**$-\,B_{36}$**), from the control cells at the first timepoint. Zero means
# *an untreated cell at the start of the experiment*, and every other cell — control or
# treated, early or late — is measured from that one point.
#
# **Does one origin still remove the round-to-round drift from Step 15?** It does, and the
# plate layout is the reason. All four timepoints sit on the **same plate** and go through
# the same eighteen rounds: the cells were treated at staggered times and fixed together, so
# p-S6 in a 36 h well and p-S6 in an 84 h well were stained in the same round, in the same
# buffer, on the same day.
#
# The drift is therefore **one number per marker, shared by every timepoint**, and a single
# subtraction removes it everywhere.
#
# ### The unit — the spread of the controls, across the whole plate
#
# The unit wants as many control cells as it can get, so that it is stable. But "the spread
# of all control cells pooled" is too much: pooled naively it would also contain the drift
# across time and the gap between the two vehicles, and both would be counted as noise.
# Anything later measured in a unit that wide is quietly shrunk.
#
# So take each control cell's deviation **from the median of its own group at its own
# timepoint** (**$\div\,S$**), and pool those instead. The unit becomes the spread of a control cell about
# its own group: plate-wide and stable, without the two things that are not spread.
#
# :::{extra}
# :class{dropdown}
# ### Why median and not mean, and why SD and not the MAD
#
# Single-cell intensity distributions have long right tails, and debris, doublets and dying
# cells live in them. A mean follows them; a median does not. So the **origin** is a median,
# and it should be.
#
# The obvious next step is to make the **unit** robust in the same way, with the median
# absolute deviation (MAD).
#
# The MAD is decided entirely by the middle half of the data. If more than half the cells
# share a value, it is **exactly zero**, however the rest behave. That is not a hypothetical
# in 4i: a marker that is not expressed reads at background in most cells. On this plate
# **GATA4 sits at exactly zero in 69% of control cells** and **p21 in 53%**, so their MADs
# came out `0.000` and `0.006` against standard deviations of `2.25` and `1.58`. The guard
# further down deletes a column with no spread, so GATA4 — the most dynamic marker in the
# whole panel — was silently replaced by zeros, and p21's values were inflated about
# 260-fold.
#
# | unit | spread of the control block | dead columns | worst \|z\| |
# |---|---|---|---|
# | 1.4826 × MAD | **45.7** | **1** | **1934** |
# | standard deviation | **1.13** | 0 | 14 |
#
# So the unit is a **standard deviation** of those deviations. It is the less robust
# estimator and the right one here, because the two failure modes are not comparable: a few
# bright cells make an SD somewhat too wide, which understates effects gradually and in the
# conservative direction, while a MAD on a switching marker goes to zero, which destroys it.
# **What the MAD is built to ignore is exactly where a marker that switches on keeps its
# signal.**
#
# :::
#
#
# ### Controls
#
# DMSO and PBS are both controls, and between them they cover **seven plate rows** rather
# than four. They are not interchangeable, though:
# [chapter 07](../2_controls/07_umap.ipynb) finds 11 of the 38 markers separating them by
# more than a control SD. So each vehicle contributes its own median, the origin is the
# average of the two, and the gap between them never gets into the unit.
#

# %%
# Here we apply the normalisation 
normalised_intensities = analysis.normalise_cells(full_adata, marker_cols)
print(f"  {normalised_intensities.shape[0]:,} cells x {normalised_intensities.shape[1]} markers, {normalised_intensities.dtype}")
print(f"  origin: {', '.join(analysis.CONTROLS)} cells at {min(full_adata.obs.timepoint_h.astype(int))} h")

# %% [markdown]
# The code behind `normalise_cells` lives in
# [`src/mcs2026/analysis.py`](https://github.com/Maaraujo-nv/Multicellular-Systems-2026/blob/main/src/mcs2026/analysis.py)
# if you want to read what it actually does.

# %% [markdown]
# ### Watch it work, on three markers that should move
#
# **GATA4**, **Fibronectin** and **Calreticulin** are all expected to rise as the system develops — endoderm specification, matrix deposition, and the secretory load that comes with it. They were also stained in rounds **2, 22 and 28**, spread right across the run, which is the second reason to pick them: if all three still trace a clean course in time, round order is not what you are looking at.
#
# Both rows are summarised **per well**, because the well is the replicate unit and because a cell median is the wrong summary for a marker that is simply absent from most cells. GATA4 is negative in more than half the control cells at every timepoint, so its cell median sits at zero however many cells have switched on. A well mean counts them.

# %%
controls = full_adata.obs.condition.astype(str).isin(analysis.CONTROLS).values
timepoint = full_adata.obs.timepoint_h.astype(int).values

WATCH = ["GATA4", "Fibronectin", "Calreticulin"]
watch_cols = [c for c in marker_cols if full_adata.var.loc[c, "marker"] in WATCH]
names = [full_adata.var.loc[c, "marker"] for c in marker_cols]
hours = sorted(set(timepoint))

# The recipe this chapter argues against, without reimplementing it: normalising one
# timepoint at a time *is* centring on that timepoint's own controls, because within a
# single timepoint "the first timepoint" is that timepoint. `full_adata[rows]` is a view, so
# this costs one 38-column block per timepoint rather than a copy of the wide table.
per_timepoint = np.zeros_like(normalised_intensities)
for value in hours:
    rows = timepoint == value
    per_timepoint[rows] = analysis.normalise_cells(full_adata[rows], marker_cols)


# Everything below is summarised per well, the replicate unit. A cell median would be the
# wrong summary here: GATA4 is simply absent from more than half the control cells at
# every timepoint, so its cell median stays at zero no matter how many cells switch on.
# A well mean counts them.
def control_wells(matrix):
    block = ad.AnnData(X=matrix, obs=full_adata.obs.copy(), var=pd.DataFrame(index=names))
    table = analysis.by_well(block)
    return table[table.condition.isin(analysis.CONTROLS)]

fixed_origin, moving_origin = control_wells(normalised_intensities), control_wells(per_timepoint)
raw = analysis.by_well(full_adata, watch_cols, name_by="marker")
raw = raw[raw.condition.isin(analysis.CONTROLS)]

fig, axes = plotting.panel_grid(6, ncols=3, size=(4.0, 3.1))
for k, column in enumerate(watch_cols):
    name = full_adata.var.loc[column, "marker"]

    ax = axes[k]
    logged = np.log2(raw[name] + 1)
    ax.plot(raw.timepoint, logged, "o", color="0.6", ms=4, alpha=0.8)
    ax.plot(hours, [logged[raw.timepoint == h].median() for h in hours],
            "-", color="black", lw=1.6)
    ax.set(title=f"{name}  ·  round {int(full_adata.var.loc[column, 'round'])}",
           xlabel="hours", ylabel="log2(intensity + 1)", xticks=hours)

    ax = axes[k + 3]
    for label, table, colour in [
        ("origin fixed at 36 h", fixed_origin, "firebrick"),
        ("origin re-set each timepoint", moving_origin, "steelblue"),
    ]:
        ax.plot(table.timepoint, table[name], "o", color=colour, ms=3.5, alpha=0.4)
        ax.plot(hours, [table.loc[table.timepoint == h, name].median() for h in hours],
                "o-", color=colour, lw=1.8, ms=5, label=label)
    ax.axhline(0, color="0.7", lw=1, ls="--")
    ax.set(xlabel="hours", ylabel="control-cell SDs", xticks=hours)
    if k == 0:
        ax.legend(fontsize=7.5, loc="best")
fig.tight_layout()

# %% [markdown]
# **Top row — the measurement.** Absolute level in log2 intensity, before anything has been
# done to it. The three markers sit at quite different heights, and that is the round they
# were stained in rather than the biology. It is exactly what subtracting one origin per
# marker is for.
#
# **Bottom row — the same three markers, the two recipes.** For **Fibronectin** and
# **Calreticulin** the blue trace is flat while the red one climbs. Here we see the difference that makes centering on the first timepoint's controls versus on each timepoint's own controls.
#
# **GATA4 is the interesting exception, and worth the paragraph.** There both traces rise.
# The reason is that GATA4 is *off* in most control cells, so the control median sits on the
# floor at every timepoint — and a moving origin can only erase what the origin is able to
# move to. When the whole population shifts, re-centring takes the shift with it; when a
# minority switches on from zero, the median never notices, so there is nothing for the
# re-centring to remove.
#
# That is the useful version of the lesson. The per-timepoint recipe does not flatten
# everything uniformly — it flattens exactly the changes you are most likely to care about
# in a time course, and leaves a misleading impression of safety on the ones it misses.
#
#

# %% [markdown]
# ## Step 18 · Assemble the three objects
#
# Everything Stage 1 established is currently scattered: the decoder is in a `var` table, the
# dropped wells in a `uns` entry, and the normalised values in `normalised_intensities`, a bare array that
# exists only in this notebook's memory.
#
# This step puts them together, so that every chapter after this one begins with a single
# `read_h5ad` and no set-up.
#
# ```{image} ../../images/clean_object_light.svg
# :class: only-light
# :alt: The analysis AnnData: X holding normalised values, a raw layer, an annotated var table, a tidy obs table, and provenance in uns.
# ```
# ```{image} ../../images/clean_object_dark.svg
# :class: only-dark
# :alt: The analysis AnnData: X holding normalised values, a raw layer, an annotated var table, a tidy obs table, and provenance in uns.
# ```

# %% [markdown]
# ### Which columns go in it?
#
# 2,587 features survived Step 9. The object carries **38** of them — one mean intensity per
# antibody per cell. That is a large cut and it needs an argument, not a preference.

# %%
counts = full_adata.var.family.value_counts()
pd.DataFrame({"columns": counts, "% of the table": (100 * counts / counts.sum()).round(1)})

# %% [markdown]
#
# :::{important}
# **For this course, we drop texture.** Texture is not noise. It measures something the
# mean cannot — *how* a protein is arranged rather than how much of it there is — and for
# several markers on this plate that is where the effect lives: the mean barely moves while
# the texture of the same marker shifts by several control SDs. A compound that redistributes
# a protein without changing its abundance is invisible to every number in this object.
#
# This course asks the first question. "How much of each protein is in this cell" is
# answerable with 38 interpretable numbers, every one of which reads back to an antibody, and
# a component you cannot name is a component you cannot report.
#
# :::

# %%
marker_cols = analysis.marker_columns(full_adata.var)
marker_names = full_adata.var.loc[marker_cols, "marker"].tolist()
print(f"{len(marker_cols)} markers, one column each")
print(f"  {', '.join(marker_names[:10])} …")

# %%
full_adata

# %% [markdown]
# ### Build it
#
# **You already have the object.** `full_adata` is an `AnnData` — every surviving cell × every
# surviving feature — so nothing here is built from scratch. This step does two things to it:
# writes down the per-cell measurements that are currently buried among the columns, then keeps
# the 38 markers and swaps in the normalised numbers.
#
# An `AnnData` has five places to put things, and knowing which is which is most of what there
# is to know about it:
#
# | slot | one row per | what goes in it |
# |---|---|---|
# | `X` | cell × marker | the numbers you analyse — normalised, in control SDs |
# | `layers["raw"]` | cell × marker | the same table, as measured |
# | `var` | **marker** | the decoder from Step 7 — channel, round, family, theme, thresholds |
# | `obs` | **cell** | well, condition, timepoint, and the measurements that are not stains |
# | `uns` | the whole object | what was dropped, why, and how the numbers were made |
#
# **The one rule: `obs` has to line up with `X`'s rows, and `var` with its columns.** Subsetting
# keeps that true for you — `full_adata[:, marker_cols]` takes the columns *and* their matching `var`
# rows in a single move. Assembling the pieces by hand is where alignment quietly goes wrong.
#
# #### Three files, because choosing 38 markers is a decision
#
# They are written in this order, and each one is a subset of the one before it:
#
# | file | shape | `X` | reach for it when |
# |---|---|---|---|
# | `mcs2026_full.h5ad` | cells × 2,587 | as measured | you want texture, a shape feature dropped below, or the population columns |
# | `mcs2026_intensity.h5ad` | cells × 38 | normalised | doing anything in Stage 2 or Part 4 |
# | `mcs2026_intensity_shape.h5ad` | cells × 43 | normalised | you want shape and intensity in the same space |
#
# The 38 markers are the right set for the question this course asks. They are not the only set,
# and an archive is what makes that choice reversible.

# %% [markdown]
# ### Before doing anything to the current object
#
# Nothing here needs a backup yet. `full_adata.X` is exactly what the microscope measured, and
# this chapter never writes over it — the archive saved in a moment **is** the raw data. A
# `layers["raw"]` snapshot appears later, on the object whose `X` really does get replaced by
# the normalised numbers.
#

# %% [markdown]
# #### The per-cell measurements: out of `X`, into `obs`
#
# They are measurements of the *cell* rather than of a stain, so they do not belong in a table
# of antibodies — but they are worth having on every row.
#

# %% [markdown]
# First we add morphology features, this all come from DAPI measurements and since they are not intensities they rarely depend on round.

# %%
PER_CELL = {"area": "area", "eccentricity": "eccentricity", "solidity": "solidity",
            "extent": "extent", "roundness": "roundness",
            "well_centroid-0": "x_in_well", "well_centroid-1": "y_in_well"}

# Build columns in obs with morphology
for statistic, name in PER_CELL.items():
    found = full_adata.var.index[(full_adata.var.family == "Morphology")
                                 & (full_adata.var.statistic == statistic)]
    if len(found) == 0:                     # fail loudly rather than silently skip a column
        raise KeyError(f"no Morphology column for {statistic!r}")

    full_adata.obs[name] = np.asarray(full_adata[:, found[0]].X).ravel()

# %% [markdown]
# Then we add DAPI mean intensity at round 0, even though it changes with round 

# %%
dapi_column = full_adata.var.index[(full_adata.var.channel == "DAPI")
                                   & (full_adata.var.statistic == "mean_intensity")
                                   & (full_adata.var["round"] == 0)][0]

full_adata.obs["dapi"] = np.asarray(full_adata[:, dapi_column].X).ravel()

# %% [markdown]
# We add some metadata to uns and save the file

# %%

full_adata.uns["provenance"] = {
    **full_adata.uns.get("provenance", {}),
    "built_by": "Multicellular Systems 2026, Part 3 Stage 1, chapter 03",
    "built_on": date.today().isoformat(),
    "layout_workbook": LAYOUT_XLSX.name,
    "cells_removed": "border cells — is_border_external or is_border_internal",
    "wells_removed": "; ".join(f"{w}: {why}" for w, why in full_adata.uns["dropped_wells"].items()),
}


# The archive: every column, as measured, with the obs and provenance above.
full_adata.uns["provenance"]["features"] = f"{full_adata.n_vars:,} columns, as measured"


full_adata.write_h5ad(H5AD_SLIM.with_name("mcs2026_full.h5ad"), compression="gzip")


# %% [markdown]
# **Why the `obs` measurements are not normalised.** `normalise_cells` exists to remove
# round-to-round staining drift, and these were not stained — the shape features come from the
# segmentation mask, measured **once per cell**, which is exactly what Step 9 found when the 396
# morphology columns turned out byte-identical across all 18 rounds. There is no artefact to
# remove.
#
# They do change over the course — median area more than halves from 36 h to 84 h as cells
# divide and crowd, while every shape ratio stays flat — but that is biology. So they go in as
# measured: `area` and the positions in pixels, the four ratios on their natural 0–1 scale.
# "This cell is 2,000 px" means something; "+0.3 control SDs of area" does not.
#
# Keep it in mind when you plot: **`X` is in control SDs, these `obs` columns are in measured
# units.** If you ever want to cluster on shape, z-score it at that point, where the choice is
# visible.

# %% [markdown]
# ### Now we can create a new anndata with only the intensity features and normalized

# %% [markdown]
# We duplicate our data but only including intensities in the X 

# %%
intensity_adata = full_adata[:, marker_cols].copy()

# %%
intensity_adata


# %% [markdown]
# Now the three edits that turn the copy into the analysis object: snapshot the measured
# numbers in `layers["raw"]`, swap the normalised ones into `X`, and rename the columns from
# channel-and-round codes to marker names. The panel definitions go into `uns` at the same
# time, so no chapter after this one has to import the course package to know which markers
# belong to which theme.
#
# ```{note}
# The snapshot uses `.astype("float32")`, which always returns a new array.
# `np.asarray(x, dtype="float32")` would **not**: given an array that is already `float32` it
# hands back the very same object, and the "snapshot" would be a second name for `X`.
# ```
#

# %%
intensity_adata.layers["raw"] = intensity_adata.X.astype("float32")  # a real copy, not a view
intensity_adata.X = normalised_intensities                           # swap in the normalised numbers
intensity_adata.var["column"] = marker_cols                          # keep the channel-and-round name
intensity_adata.var_names = marker_names                             # and show the marker instead

# The panel definitions travel with the object, so no chapter after this one has to import
# the package to know which markers belong to which theme. `var["theme"]` labels each marker;
# this keeps the *declared* panels too, including markers that are not in these 38.
intensity_adata.uns["panels"] = {name: list(members) for name, members in panels.PANELS.items()}
intensity_adata.uns["themes"] = list(panels.THEMES)

intensity_adata.uns["provenance"] = {
    **intensity_adata.uns["provenance"],
    "features": f"{len(marker_cols)} marker mean intensities, decoded from 4,464 raw columns",
    "X": ("log2(x + 1); origin = median of the DMSO+PBS cells at the first timepoint, "
          "unit = SD of control cells about their own vehicle x timepoint median"),
    "layers_raw": "mean intensity as measured",
    "units": "X in control-cell SDs; obs measurements in pixels and ratios",
}
intensity_adata

# %%
intensity_adata.write_h5ad(H5AD_SLIM.with_name("mcs2026_intensity.h5ad"), compression="gzip")
print(f"  mcs2026_intensity.h5ad  {intensity_adata.n_obs:,} cells x {intensity_adata.n_vars} markers, normalised")

# Every statistical test in Part 4 works on well means rather than cells, because the well is
# what was independently treated. That table is not a fourth file -- it is one groupby on the
# object you just wrote, and this is the line the later chapters use, where the course package
# is no longer imported:
wells = (intensity_adata.to_df()
         .groupby([intensity_adata.obs.condition.astype(str),
                   intensity_adata.obs.timepoint_h.astype(int),
                   intensity_adata.obs.well.astype(str)], observed=True).mean()
         .rename_axis(["condition", "timepoint", "well"]).reset_index())
wells.iloc[:4, :6].round(2)

# %% [markdown]
# #### A third object, if you want shape in the analysis
#
# `X` holds 38 antibodies. The shape measurements sit in `obs`, so they can colour a plot but
# never reach a PCA or a neighbour graph. To get them **into** the analysis they have to go into
# `X` — and then they have to arrive on the same scale as the markers, or they will not join the
# analysis so much as replace it.
#
# **Why, in one number.** Join the raw columns and `area` is **99.9997%** of the total variance
# across all 43; the 38 markers together are **0.0003%**.
#
# | | variance, unscaled | share of the 43 columns |
# |---|---|---|
# | `area` — pixels, up to ~35,000 | 12,591,660 | **99.9997%** |
# | all 38 markers together — already ~1 SD | 41.1 | 0.0003% |
# | the four ratios — bounded near 0–1 | ≈ 0 | ≈ 0 |
#
# PCA maximises variance and a neighbour graph measures Euclidean distance. Both are therefore
# decided by whichever column happens to carry the biggest numbers — and on this data that means
# **PC1 explains 100.00% of the variance with a loading that is 99.6% `area`**. Thirty-eight
# antibodies, contributing nothing you could see. Not a subtle bias: the analysis is simply gone.
#
# So every column gets the same treatment the markers got in Step 17 — centred on the control
# cells at the first timepoint, scaled by the control spread — so that **one unit means the same
# thing in all 43**.
#
# **One column is treated differently, and the data picks which.** `area` is a pixel count
# spanning two orders of magnitude and strongly right-skewed, exactly like an intensity, so it
# wants `log2` first: that takes its skew from **+2.37 to +0.12**, and the logged and unlogged
# versions correlate at only **0.892**, so the choice genuinely changes the answer. The four
# ratios are bounded and already near-symmetric — the two versions agree to three decimals, and
# logging makes three of the four *more* skewed. Taking the log of a bounded ratio squashes one
# end for nothing, which is what `log2=False` is for.
#
# **Position stays out of `X`.** `x_in_well` and `y_in_well` say where a cell sat, not what it
# is. In `X` they would push the embedding to cluster cells by their position in the well — the
# exact artefact [chapter 07](../2_controls/07_umap.ipynb) spends its purity table hunting for.
# They stay in `obs`, where "does this effect depend on position?" is a good question to ask.

# %%
# 38 markers and 5 shape features in one X, every column in control-cell SDs.
SHAPE = ["area", "eccentricity", "solidity", "extent", "roundness"]
shape_cols = [full_adata.var.index[(full_adata.var.family == "Morphology")
                                   & (full_adata.var.statistic == s)][0] for s in SHAPE]

shape_z = np.hstack([
    analysis.normalise_cells(full_adata, shape_cols[:1]),                # a count: log2 first
    analysis.normalise_cells(full_adata, shape_cols[1:], log2=False),    # a ratio: do not log it
])

intensity_shape_adata = full_adata[:, marker_cols + shape_cols].copy()
intensity_shape_adata.layers["raw"] = np.asarray(intensity_shape_adata.X, dtype="float32")
intensity_shape_adata.X = np.hstack([normalised_intensities, shape_z]).astype("float32")
intensity_shape_adata.var["column"] = marker_cols + shape_cols
intensity_shape_adata.var_names = marker_names + SHAPE
intensity_shape_adata.uns["provenance"] = {
    **intensity_shape_adata.uns["provenance"],
    "features": f"{len(marker_cols)} marker mean intensities + {len(SHAPE)} shape features",
    "X": ("every column in control-cell SDs, origin at the first timepoint; log2 applied to "
          "the intensities and to area, not to the bounded ratios"),
    "units": "control-cell SDs throughout",
}
intensity_shape_adata.write_h5ad(H5AD_SLIM.with_name("mcs2026_intensity_shape.h5ad"), compression="gzip")
print(f"  mcs2026_intensity_shape.h5ad  {intensity_shape_adata.n_obs:,} cells x {intensity_shape_adata.n_vars} features")
print(f"  var['family'] splits the two blocks: "
      f"{dict(intensity_shape_adata.var.family.value_counts())}")

# %%
# The claim the whole section rests on: no column can now dominate by accident. Measured on
# every control cell, which is the population the unit was fitted on.
spread = pd.Series(intensity_shape_adata.X[controls].std(axis=0, ddof=1), index=intensity_shape_adata.var_names)
print(f"  control spread across all {len(spread)} columns: "
      f"min {spread.min():.2f}   median {spread.median():.2f}   max {spread.max():.2f}")

# Inside one timepoint a marker that has not switched on yet has little to vary -- that is
# biology, not a scaling failure. GATA4 is the clearest case.
first = min(timepoint)
early = pd.Series(intensity_shape_adata.X[controls & (timepoint == first)].std(axis=0, ddof=1),
                  index=intensity_shape_adata.var_names)
label = f"at {first} h only"
pd.DataFrame({"all timepoints": spread.round(2), label: early.round(2)}).nsmallest(4, label)

# %% [markdown]
# Every column sits at a spread of about 1, which is the entire point — `area` no longer
# outvotes the antibodies 300,000 to one, and a PCA on this object has to actually choose. Run
# one and PC1 drops from explaining 100% of the variance to about 40%, with `area` down from
# 99.6% of the loading to a couple of per cent.
#
# The second table is worth a look rather than a worry. Inside the **first timepoint alone**,
# GATA4's spread is a fraction of one unit — because at 36 h it has barely switched on, so there
# is almost nothing for it to vary by. The unit was fitted across the whole plate on purpose, and
# a marker that only becomes variable later is exactly the kind of signal that a per-timepoint
# unit would have flattened.
#
# **Nothing later in the course reads this file.** Stage 2 opens `mcs2026_intensity.h5ad`, so every
# embedding in chapters 06–10 is unchanged by its existence. It is here for questions that need
# shape and intensity in the same space — and `var["family"]` separates the two blocks, so you
# can always ask what a component is made of.

# %% [markdown]
# :::{tip}
# **Three files, three jobs.**
#
# - `mcs2026_intensity.h5ad` — one row per **cell**: embeddings, clustering, single-cell
#   distributions. This is what Stage 2 opens.
# - `mcs2026_full.h5ad` — the archive, every column as measured: for when a question needs
#   texture, a shape feature this chapter dropped, or the population columns.
# - `mcs2026_intensity_shape.h5ad` — the 38 markers **and** five shape features in one `X`, all in
#   control SDs: for asking whether shape and intensity say the same thing. Optional; nothing
#   in the course opens it.
#
# Counting cells when you should be counting **wells** is the single most common mistake in
# Part 3 — and [chapter 08](../2_controls/08_cell_type_annotation.ipynb) shows what it costs.
# The well table is not a fourth file, though: it is a `groupby` on the cell object, one line
# wherever you need it, so it can never go stale against the object it came from — and it needs
# nothing but pandas, which is why the later chapters can build it without the course package.
# :::

# %% [markdown]
# ## Stage 1 complete
#
# | | |
# |---|---|
# | **started with** | 733,556 × 4,464, `var` empty, 13.1 GB |
# | **removed** | border cells (~11%), one well, redundant DAPI and duplicated structural blocks |
# | **ended with** | `mcs2026_intensity.h5ad` — every surviving cell × 38 named markers |
# | **units** | `X` in control-cell SDs, from a fixed origin at the first timepoint; `obs` measurements as measured |
# | **and** | `mcs2026_full.h5ad` as the archive, and `mcs2026_intensity_shape.h5ad` with shape in `X` |
#
# One chapter of Stage 1 remains: [04 · Subsetting and
# sketching](04_subsetting_and_sketching.ipynb) cuts this down to something a neighbour
# graph can be built on, without throwing away the rare cells.
#
# ---
#
# ## Exercises
#
# ### 1. What does a moving origin cost you?
#
# Re-run the normalisation the way this chapter argues against — each timepoint centred on
# its own controls — and compare. What happens to the controls' own trajectory? And would a
# comparison between a treatment and its control at the *same* timepoint notice the
# difference?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# Normalising each timepoint as if it were its own experiment is exactly the old recipe,
# because within a single timepoint "the first timepoint" is that timepoint:
#
# ```python
# per_timepoint = np.zeros_like(normalised_intensities)
# for value in sorted(set(timepoint)):
#     rows = timepoint == value
#     per_timepoint[rows] = analysis.normalise_cells(full_adata[rows], marker_cols)
#
# for value in sorted(set(timepoint)):
#     block = per_timepoint[controls & (timepoint == value)]
#     print(value, round(float(np.abs(np.median(block, axis=0)).mean()), 3))
# ```
#
# **Any single-timepoint comparison survives untouched.** Part 4 compares a treatment against
# its control *within* a timepoint, and a shared offset cancels in a difference — the ranks do
# not move, so the p-values come out identical either way.
#
# What disappears is the controls' trajectory: every timepoint now prints ~0, because that
# is what you asked for. The cost is invisible in any single-timepoint comparison, which is
# precisely why it is easy to ship. It only shows up later, in
# [chapter 10](../2_controls/10_diffusion_map.ipynb), when a trajectory is supposed to run
# from 36 h to 84 h and there is nothing left for it to run along.
# :::

# %% [markdown]
# ### 2. Should both vehicles be in the reference?
#
# The origin is the average of the DMSO and PBS medians. Rebuild it from DMSO alone and
# see which markers move, and by how much.

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# dmso_only = analysis.normalise_cells(full_adata, marker_cols, controls=("DMSO",))
# moved = pd.Series(np.median(dmso_only, axis=0) - np.median(normalised_intensities, axis=0), index=names)
# print(moved.abs().sort_values(ascending=False).head(5).round(2))
# ```
#
# The markers that move are the ones where the two vehicles disagree, and the shift is
# about half the gap between them — because two references average, and one does not.
#
# Both choices are defensible. Both vehicles gives more reference cells and covers seven
# plate rows instead of four; DMSO alone ties zero to the solvent the compounds were
# actually dissolved in, which is the comparison a pharmacologist would want. What is not
# defensible is not knowing which one you did — which is why it is written into
# `uns["provenance"]` rather than left in a notebook.
# :::

# %% [markdown]
# ### 3. How large is a real effect, in this experiment?
#
# Using `analysis.rank_effects`, list the ten largest condition × marker shifts. How many
# exceed 3 control SDs? Compare that with the unit those SDs are measured in — Step 17
# sets one control SD to the standard deviation of the control cells about their own group,
# so a shift of 3 means three times the spread of a single untreated cell.

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# ranked = analysis.rank_effects(analysis.by_well(intensity_adata), names)
# print(ranked.head(10).round(2))
# print(f"beyond 3 control SDs: {(ranked.abs_shift > 3).sum()} of {len(ranked)}")
# ```
#
# A handful of pairs exceed 3 SDs and most sit well under 1. That ratio is the honest
# picture of a screen: a few strong, specific effects against a background of very little,
# and it is why ranking by effect size in control SDs is more informative than sorting by
# p-value.
# :::

# %% [markdown]
# ---
#
# **Next:** [04 · Subsetting and sketching](04_subsetting_and_sketching.ipynb).
