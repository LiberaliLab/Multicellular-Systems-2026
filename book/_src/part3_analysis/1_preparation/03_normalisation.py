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
# # 03 · Normalisation
#
# Steps 15 to 19 — the last of Stage 1. Chapter 02 decided what to throw away; this one
# makes what is left **comparable**, and then packs it into the single object every
# chapter after this one opens.
#
# | | |
# |---|---|
# | **Step 15** | Are intensities comparable across rounds? |
# | **Step 16** | Is any condition an outlier? |
# | **Step 17** | Normalise to the controls |
# | **Step 18** | Check a known answer |
# | **Step 19** | Assemble the clean object |
#
# Steps 15 and 16 are the same kind of move: **find a reason the numbers might not mean
# what they appear to**, and settle it before analysing them.

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

from mcs2026 import analysis, plotting
from mcs2026.config import H5AD_SLIM, LAYOUT_XLSX

plotting.set_style()
pd.set_option("display.width", 140)

clean = sc.read_h5ad(H5AD_SLIM.with_name("mcs2026_qc.h5ad"))
print(f"{clean.n_obs:,} cells x {clean.n_vars:,} features, "
      f"{clean.obs.well.nunique()} wells")

# %% [markdown]
# ## Step 15 · Are intensities comparable across rounds?
#
# [Step 9](01_columns_to_markers.ipynb) found that DAPI — the same stain, on the same
# nuclei, every round — moves by more than an order of magnitude across the experiment.
# Here is what that means for the markers.

# %%
dapi_columns = clean.var.loc[
    clean.var.index[(clean.var.channel == "DAPI") & (clean.var.statistic == "mean_intensity")]
].sort_values("round").index

dapi = pd.Series(
    np.median(np.asarray(clean[:, list(dapi_columns)].X), axis=0),
    index=clean.var.loc[dapi_columns, "round"].astype(int).values, name="median DAPI",
)

marker_cols = analysis.marker_columns(clean.var)
marker_median = pd.Series(
    np.median(np.asarray(clean[:, marker_cols].X), axis=0),
    index=clean.var.loc[marker_cols, "round"].astype(int).values,
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
wells = analysis.well_means(clean, marker_cols)
names = [clean.var.loc[c, "marker"] for c in marker_cols]

logged = np.log2(wells[names] + 1)
centred = logged - logged[wells.condition == "DMSO"].mean()
spread = centred.groupby(wells.condition).mean().abs().mean(axis=1).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(7, 4))
colours = ["firebrick" if v == spread.index[0] else "0.5" for v in spread.index]
ax.barh(range(len(spread)), spread.values, color=colours)
ax.set(yticks=range(len(spread)), yticklabels=spread.index,
       xlabel="mean |log2 shift vs DMSO|", title="How far each condition sits from the control")
ax.invert_yaxis()

# %%
spread.head(4).round(3)

# %% [markdown]
# **PMA moves every marker, by a lot** — several times further than anything else. That is
# not a pathway response, it is a different cell state. Phorbol esters drive naive human ES
# cells out of pluripotency, so a wholesale change is biologically reasonable; it is still
# an outlier statistically.
#
# :::{important}
# This is a decision, and it should be a conscious one:
#
# - **keep PMA** and it dominates the first component of every embedding, so the plot
#   describes PMA-vs-everything rather than the structure among the other seventeen;
# - **drop PMA** and you lose a real, strong biological effect;
# - **flag it and analyse it knowingly**, which is what this course does.
#
# What you must not do is leave it in silently and then interpret component 1 as if it
# were about subtle signaling differences.
# :::

# %%
clean.obs["is_outlier_condition"] = clean.obs.condition.astype(str).eq(
    "Phorbol 12-myristate 13-acetate (PMA)"
)
print(f"  flagged {clean.obs.is_outlier_condition.sum():,} cells in "
      f"{clean.obs.loc[clean.obs.is_outlier_condition, 'well'].nunique()} wells")

# %% [markdown]
# ## Step 17 · Normalise to the controls
#
# Three operations, in this order, each for a stated reason.
#
# **`log2`** — intensities are multiplicative and heavily right-skewed. A doubling should
# look the same whether it is 100→200 or 1000→2000.
#
# **Centre on DMSO, within timepoint** — this undoes the round-to-round drift from Step 15
# and the plate effects from Step 13 at once, because the control wells went through
# exactly the same rounds and sit in the same plate. Within *timepoint*, because the
# controls themselves change over 36–84 h and erasing that would erase the experiment.
#
# **Scale by the control SD** — puts every marker in the same units: *how many control-well
# standard deviations from the control mean*. That is what makes a signaling marker and an
# organelle marker comparable on one axis.

# %%
def normalise_to_controls(frame, feature_names, *, control="DMSO", within="timepoint"):
    """log2, centre on the control wells within each timepoint, scale by control SD.

    Units are control-well SDs: -2 means "two control standard deviations below the
    control mean at that timepoint".
    """
    logged = np.log2(frame[feature_names] + 1)
    out = logged.copy()
    is_control = frame["condition"] == control
    for _, index in frame.groupby(within, observed=True).groups.items():
        block = logged.loc[index]
        reference = block[is_control.loc[index]]
        out.loc[index] = (block - reference.mean()) / reference.std().replace(0, np.nan)
    return out

normalised = normalise_to_controls(wells, names)
for i, key in enumerate(["condition", "timepoint", "well"]):
    normalised.insert(i, key, wells[key].values)
normalised.iloc[:4, :7].round(2)

# %%
# Check it worked: within each timepoint the control wells must sit at mean 0 and SD 1,
# because that is exactly what they were scaled by.
control_check = normalised[normalised.condition == "DMSO"]
pd.DataFrame({
    "mean": control_check.groupby("timepoint", observed=True)[names].mean().mean(axis=1),
    "SD": control_check.groupby("timepoint", observed=True)[names].std().mean(axis=1),
}).round(3)

# %% [markdown]
# ## Step 18 · Check a known answer
#
# The compounds were chosen to hit specific processes, so there is a prediction to test.
# **Sapanisertib/INK128 inhibits mTOR**; mTORC1 activates S6 kinase, which phosphorylates
# ribosomal protein S6. So INK128 should lower **p-S6**.

# %%
analysis.compare_to_control(normalised, "p-S6", "Sapanisertib/INK128")

# %%
analysis.compare_to_control(normalised, "p-S6", "MK-2206")

# %% [markdown]
# Both inhibitors lower p-S6 by several control SDs, most strongly at 36 h and weakening
# steadily after — by 84 h MK-2206 has no measurable effect at all, and INK128's is a
# fraction of what it was. The compound is used up, or the cells adapt. A real result, with
# the sign the pathway predicts. **The normalisation works.**
#
# :::{warning}
# **Read the p-values, not the stars.** Every significant row reads `0.0357`, and that is
# not a coincidence: with 3 treated wells and 5 control wells there are
# $\binom{8}{3} = 56$ rank orderings, so the smallest two-sided p-value a Mann-Whitney test
# can *ever* return here is $2/56 = 0.0357$.
#
# No effect size, however enormous, produces a smaller one. The p-value has hit the floor
# set by the number of wells — the same lesson as the $\sqrt{n-1}$ ceiling in
# [Step 12](02_quality_control.ipynb), in a different disguise. `analysis.rank_effects`
# reports that floor alongside every p-value for exactly this reason.
# :::

# %% [markdown]
# ### The one that does not work
#
# The obvious readout for an AKT inhibitor is phospho-AKT itself. Try it.

# %%
analysis.compare_to_control(normalised, "p-AKT", "MK-2206")

# %% [markdown]
# Compare that with the p-S6 table above. p-S6 falls at every early timepoint, by a lot,
# in the same direction. p-AKT does nothing of the kind: it drifts down at 36 h, **up** at
# 48 h, down again at 60 h, and vanishes at 84 h. One row happens to clear the p-value
# floor, but a signal that changes sign between timepoints is not a signal.
#
# This is not a failure of the normalisation, and it is worth understanding rather than
# explaining away:
#
# - **Pathway feedback.** Inhibiting mTORC1 relieves a negative feedback loop onto the
#   receptor, which *raises* AKT phosphorylation. Inhibitor and feedback partly cancel.
# - **Where the antibody sits.** p-AKT was imaged in round 24, near the end of eighteen
#   elution cycles. Phospho-epitopes are the most fragile thing in a 4i panel.
# - **Downstream integrates.** p-S6 reflects sustained pathway output; a single
#   phospho-site is a snapshot of a fast equilibrium.
#
# The pathway *is* visible — one step sideways:

# %%
for condition in ["MK-2206", "Wortmannin", "Sapanisertib/INK128"]:
    shifts = analysis.compare_to_control(normalised, "Foxo1", condition)["shift"]
    print(f"  Foxo1 vs {condition:22s} {list(shifts)}  all negative: {bool((shifts < 0).all())}")

# %% [markdown]
# **Foxo1 falls for all three inhibitors, at every timepoint.** FoxO transcription factors
# are direct AKT substrates. The pathway is there; it is simply not where you first looked.

# %% [markdown]
# ## Step 19 · Assemble the clean object
#
# Everything Stage 1 established is currently scattered: the values are in one file, the
# decoder in a `var` table, the dropped wells in a `uns` entry, and the normalisation only
# exists as a well-level DataFrame that lives in this notebook's memory.
#
# This step puts it in one object, so that every chapter after this one begins with a
# single `read_h5ad` and no set-up.
#
# ```{image} ../../images/clean_object_light.svg
# :class: only-light
# :alt: The clean AnnData: X holding normalised values, a raw layer, an annotated var table, a tidy obs table, and provenance in uns.
# ```
# ```{image} ../../images/clean_object_dark.svg
# :class: only-dark
# :alt: The clean AnnData: X holding normalised values, a raw layer, an annotated var table, a tidy obs table, and provenance in uns.
# ```

# %% [markdown]
# ### Which columns go in it?
#
# 2,587 features survived Step 9. The object carries **38** of them — one mean intensity per
# antibody per cell. That is a large cut and it needs an argument, not a preference.

# %%
counts = clean.var.family.value_counts()
pd.DataFrame({"columns": counts, "% of the table": (100 * counts / counts.sum()).round(1)})

# %% [markdown]
# **A feature set votes by column count.** Texture is most of what survived, so it decides
# most of anything computed from all of it. That is a claim, so measure it — cheaply, at the
# well level, where the whole table is 223 rows.

# %%
from sklearn.decomposition import PCA

everything = clean.var.index.tolist()
well_wide = (pd.DataFrame(np.log2(np.abs(np.asarray(clean[:, everything].X)) + 1),
                          columns=everything)
             .assign(well=clean.obs.well.astype(str).values,
                     condition=clean.obs.condition.astype(str).values)
             .groupby(["condition", "well"], observed=True).mean().reset_index())
reference = well_wide.loc[well_wide.condition == "DMSO", everything]
# A column with no spread across the control wells carries no information here, and
# dividing by its zero would give infinities that np.fillna does not catch.
spread = reference.std().replace(0, np.nan)
scaled = (((well_wide[everything] - reference.mean()) / spread)
          .replace([np.inf, -np.inf], np.nan).fillna(0))

first = PCA(n_components=2, random_state=0).fit(scaled.values).components_[0]
weight = (pd.Series(np.abs(first), index=everything)
          .groupby(clean.var.family, observed=True).sum())
pd.DataFrame({
    "% of columns": (100 * counts / counts.sum()).round(1),
    "% of PC1's loading": (100 * weight / weight.sum()).round(1),
}).sort_values("% of columns", ascending=False)

# %% [markdown]
# **Read the two columns against each other.** Each family contributes to the first component
# in almost exactly the proportion of columns it has — not in proportion to how much it
# knows. Feed a method everything and you have not asked it an open question; you have voted
# for texture 2,262 times.
#
# **And most of those columns repeat each other.** Each marker carries 58 texture features.
# How many independent numbers is that really?

# %%
one_marker = clean.var.index[(clean.var.family == "Texture") & (clean.var.marker == "LAMP1")]
block = scaled[list(one_marker)]
correlation = block.corr().abs().values
np.fill_diagonal(correlation, np.nan)
spectrum = PCA().fit(block.values).explained_variance_ratio_
print(f"  LAMP1 has {len(one_marker)} texture columns")
print(f"  median |correlation| between them: {np.nanmedian(correlation):.2f}")
print(f"  components reaching 90% of their variance: {(np.cumsum(spectrum) < 0.9).sum() + 1}")

# %% [markdown]
# Fifty-eight columns, two real dimensions. The block is wide, not deep.
#
# :::{important}
# **What this gives up, stated plainly.** Texture is not noise. It measures something the
# mean cannot — *how* a protein is arranged rather than how much of it there is — and for
# several markers on this plate that is where the effect lives: the mean barely moves while
# the texture of the same marker shifts by several control SDs. A compound that redistributes
# a protein without changing its abundance is invisible to every number in this object.
#
# This course asks the first question. "How much of each protein is in this cell" is
# answerable with 38 interpretable numbers, every one of which reads back to an antibody, and
# a component you cannot name is a component you cannot report.
#
# The wide table stays on disk as `mcs2026_slim.h5ad` and the door is open. Three things
# will bite you when you walk through it: `Morphology_orientation` is an angle in radians, so
# it has no mean and no meaningful z-score; `Population_mean_distance_nn_50` and `_100`
# already contain NaNs; and Haralick correlations are roughly symmetric while Laws energies
# are skewed like intensities, so no single transform is right for all of them.
# :::

# %%
marker_cols = analysis.marker_columns(clean.var)
marker_names = clean.var.loc[marker_cols, "marker"].tolist()
print(f"{len(marker_cols)} markers, one column each")
print(f"  {', '.join(marker_names[:10])} …")

# %% [markdown]
# ### The same normalisation, on cells rather than wells
#
# Step 17 scaled **well means**, because that is the replicate unit for a statistical
# test. But a UMAP has one point per cell, so Stage 2 needs the same treatment applied one
# level down: `log2`, then centre and scale on the **DMSO cells of the same timepoint**.
#
# It is the identical recipe, so it lives in one function rather than being retyped:

# %%
values = analysis.normalise_cells(clean, marker_cols)
print(f"  {values.shape[0]:,} cells x {values.shape[1]} markers, {values.dtype}")

# %% [markdown]
# The check that it worked is the definition: control cells of each timepoint must come
# out centred at 0 with a spread of 1.

# %%
controls = (clean.obs.condition.astype(str) == "DMSO").values
timepoint = clean.obs.timepoint_h.astype(int).values
pd.DataFrame(
    [{"timepoint": t,
      "control cells": int((controls & (timepoint == t)).sum()),
      "mean": values[controls & (timepoint == t)].mean().round(3),
      "SD": values[controls & (timepoint == t)].std(ddof=1).round(3)}
     for t in sorted(set(timepoint))]
).set_index("timepoint")

# %% [markdown]
# ### Build it
#
# Five slots, each with a job:
#
# | slot | what goes in it |
# |---|---|
# | `X` | the normalised values — control-cell SDs, within timepoint |
# | `layers["raw"]` | the intensities as measured, so nothing is thrown away |
# | `var` | marker, channel, round, family, theme, thresholds — the decoder from Step 7 |
# | `obs` | well, row, column, condition, timepoint, replicate, area, DAPI |
# | `uns["provenance"]` | what was dropped, why, and how the numbers were made |

# %%
cleaned = ad.AnnData(
    X=values,
    obs=clean.obs.copy(),
    var=clean.var.loc[marker_cols].copy(),
)
cleaned.layers["raw"] = np.asarray(clean[:, marker_cols].X, dtype="float32")

# The long channel-and-round name was useful while decoding; from here the marker is the
# name you want, and `var["column"]` keeps the original so nothing is lost.
cleaned.var["column"] = marker_cols
cleaned.var_names = marker_names

# %% [markdown]
# **`replicate`** is the one piece of metadata the plate implies but never states: which
# of the three (or five) wells of a condition × timepoint this cell came from. Every
# statistical test in Part 4 counts wells, so the number is worth having explicitly.

# %%
well_key = (cleaned.obs[["condition", "timepoint_h", "well"]]
            .drop_duplicates()
            .sort_values(["condition", "timepoint_h", "well"]))
well_key["replicate"] = (well_key.groupby(["condition", "timepoint_h"], observed=True)
                         .cumcount() + 1)
cleaned.obs["replicate"] = (cleaned.obs.well.astype(str)
                            .map(dict(zip(well_key.well.astype(str), well_key.replicate)))
                            .astype("int8"))
well_key.groupby("condition", observed=True).replicate.max().rename("wells per timepoint")

# %% [markdown]
# Two more per-cell numbers are worth carrying across. Neither is a marker — **area** is a
# shape measurement and **DAPI** is the counterstain imaged in every round — so neither
# belongs in `X` alongside the antibodies. But both explain things the markers cannot: how
# big a cell is, and how brightly it stained overall. They go in `obs`.

# %%
area_column = clean.var.index[(clean.var.family == "Morphology")
                              & (clean.var.statistic == "area")][0]
dapi_column = clean.var.index[(clean.var.channel == "DAPI")
                              & (clean.var.statistic == "mean_intensity")
                              & (clean.var["round"] == 0)][0]
cleaned.obs["area"] = np.asarray(clean[:, area_column].X).ravel()
cleaned.obs["dapi"] = np.asarray(clean[:, dapi_column].X).ravel()
cleaned.obs[["area", "dapi"]].describe().loc[["mean", "std", "min", "max"]].round(1)

# %% [markdown]
# ### Write down what happened to it
#
# In six months you will open this file and not remember whether the border cells came
# out, or which well was dropped and why. Neither will whoever you send it to. `uns` is
# where that goes.

# %%
cleaned.uns["provenance"] = {
    **clean.uns.get("provenance", {}),
    "built_by": "Multicellular Systems 2026, Part 3 Stage 1, chapter 03",
    "built_on": date.today().isoformat(),
    "layout_workbook": LAYOUT_XLSX.name,
    "cells_removed": "border cells — is_border_external or is_border_internal",
    "wells_removed": "; ".join(f"{w}: {why}" for w, why in clean.uns["dropped_wells"].items()),
    "features": f"{len(marker_cols)} marker mean intensities, decoded from 4,464 raw columns",
    "X": "log2(x + 1), centred and scaled on the DMSO cells of the same timepoint",
    "layers_raw": "mean intensity as measured",
    "units": "control-cell standard deviations",
}
for key, value in cleaned.uns["provenance"].items():
    print(f"  {key:18s} {value}")

# %% [markdown]
# ### Save

# %%
cleaned.write_h5ad(H5AD_SLIM.with_name("mcs2026_clean.h5ad"), compression="gzip")
normalised.to_parquet(H5AD_SLIM.with_name("mcs2026_wells.parquet"))
cleaned

# %% [markdown]
# :::{tip}
# **Two files, two jobs.** `mcs2026_clean.h5ad` is one row per **cell** — embeddings,
# clustering, single-cell distributions. `mcs2026_wells.parquet` is one row per **well**,
# already averaged — every statistical test, because the well is what was independently
# treated. Reaching for the wrong one is the single most common mistake in Part 3, and
# [chapter 08](../2_controls/08_cell_type_annotation.ipynb) shows what it costs.
# :::

# %% [markdown]
# ## Stage 1 complete
#
# | | |
# |---|---|
# | **started with** | 733,556 × 4,464, `var` empty, 13.1 GB |
# | **removed** | border cells (~11%), one well, redundant DAPI and duplicated structural blocks |
# | **ended with** | `mcs2026_clean.h5ad` — every surviving cell × 38 named markers |
# | **units** | control-cell standard deviations, within timepoint |
# | **and** | `mcs2026_wells.parquet`, the same thing averaged to one row per well |
#
# One chapter of Stage 1 remains: [04 · Subsetting and
# sketching](04_subsetting_and_sketching.ipynb) cuts this down to something a neighbour
# graph can be built on, without throwing away the rare cells.
#
# ---
#
# ## Exercises
#
# ### 1. What does normalising within timepoint protect you from?
#
# Re-run the normalisation centring on **all** DMSO wells at once rather than within each
# timepoint. Does the INK128 / p-S6 result survive? What happens to the timepoint trend?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# pooled = normalise_to_controls(wells, names, within="condition")  # one group
# pooled.insert(0, "condition", wells.condition.values)
# pooled.insert(1, "timepoint", wells.timepoint.values)
# analysis.compare_to_control(pooled, "p-S6", "Sapanisertib/INK128")
# ```
#
# The drug effect largely survives, because it is big. What changes is everything that
# varies with time: the controls themselves drift across 36–84 h, so pooling them puts
# early wells below zero and late wells above it *by construction*. Any comparison across
# timepoints then measures the drift as well as the biology.
#
# Normalising within timepoint costs you the ability to compare absolute levels between
# timepoints — which you did not have anyway, for the reason in Step 15.
# :::

# %% [markdown]
# ### 2. Is PBS a second control?
#
# The plate has two controls: DMSO (the vehicle) and PBS. If they behave identically, PBS
# wells should sit near zero after normalising to DMSO. Do they?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# pbs = normalised[normalised.condition == "PBS"][names]
# print(pbs.mean().describe().round(3))
# print(f"markers beyond 2 control SDs: {(pbs.mean().abs() > 2).sum()} of {len(names)}")
# ```
#
# Mostly near zero, which is the reassurance you want: the solvent is not doing much on its
# own. Where PBS and DMSO differ, DMSO is still the right reference, because it is what the
# compounds were dissolved in — the comparison you care about is *drug versus its own
# vehicle*, not drug versus water.
# :::

# %% [markdown]
# ### 3. How large is a real effect, in this experiment?
#
# Using `analysis.rank_effects`, list the ten largest condition × marker shifts. How many
# exceed 3 control SDs? Compare that with the median control-well SD from Step 13.

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# ranked = analysis.rank_effects(normalised, names)
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
