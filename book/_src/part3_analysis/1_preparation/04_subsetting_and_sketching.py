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
# # 04 · Subsetting and sketching
#
# [Chapter 03](03_normalisation.ipynb) handed you one analysis object holding every cell that
# survived quality control. This chapter is about deliberately taking **less** of it.
#
# Two different reasons to do that, and they need different tools:
#
# - **Your question is narrower than the dataset.** You care about four organelle markers
#   and three compounds. Cut the rows and columns you do not need — *subsetting*.
# - **The dataset is larger than the method.** A neighbour graph over 653,000 cells is slow
#   enough that you stop experimenting, which hurts the analysis more than any
#   approximation. Take fewer cells, but not at random — *sketching*.
#
# By the end you will have built both, because the rest of the book needs one of each:
#
# | | | |
# |---|---|---|
# | `mcs2026_controls.h5ad` | every DMSO and PBS cell | **[Stage 2](../2_controls/intro.md)** learns the whole toolkit on it |
# | `mcs2026_sketch.h5ad` | ~30,000 cells spanning all 18 conditions | [Part 4](../../part4_final_solutions/intro.md) embeds the plate with it |
#
# The contrast between them is the point of the chapter. One is small enough to use whole;
# the other is not, and has to be sampled with care.
#
# | | |
# |---|---|
# | **1** | Subset by marker |
# | **2** | Subset by condition — and build the controls |
# | **3** | Subset by cell — and why random is not good enough |
# | **4** | Sketch the geometry instead |
# | **5** | Check what you kept, and save |

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from geosketch import gs

plt.rcParams.update({          # the house style, no package needed
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
pd.set_option("display.width", 140)

# The one path to set. Point MCS2026_DATA at the folder holding the tables, or edit this.
DATA = Path(os.environ.get("MCS2026_DATA", "/cluster/scratch/maaraujo/data_mcs_2026"))

cells = sc.read_h5ad(DATA / "mcs2026_intensity.h5ad")
print(f"{cells.n_obs:,} cells x {cells.n_vars} markers "
      f"({cells.n_obs * cells.n_vars * 4 / 1e9:.2f} GB as float32)")

def stratified(obs, n, by=("condition", "timepoint_h"), seed=0):
    """An equal draw from every condition x timepoint, rather than the front of the file."""
    rng = np.random.default_rng(seed)
    groups = obs.groupby(list(by), observed=True).indices
    per_group = max(1, n // max(len(groups), 1))
    return np.sort(np.concatenate([
        rng.choice(rows, size=min(per_group, len(rows)), replace=False)
        for rows in groups.values()]))

def sketch(matrix, n, seed=0):
    """Geometric sketching: cover the *space* evenly, so rare cells survive."""
    return np.sort(np.asarray(gs(matrix, n, seed=seed, replace=False), dtype=int))



# %%
cells

# %% [markdown]
# ## 1 · Subset by marker
#
# This is what [Step 7](01_columns_to_markers.ipynb) bought you. Before the rename, `var`
# held 4,464 strings and there was no way to ask for "the ER markers". Now there is.

# %%
cells.var.groupby("theme", observed=True).agg(
    markers=("marker", "nunique"),
    rounds=("round", lambda r: f"{r.min()}–{r.max()}"),
).sort_values("markers", ascending=False)

# %% [markdown]
# **A whole theme**, using the panel definitions in `mcs2026.panels`:
#
# Note that this returns more markers than the table above lists for `organelles`. The
# `theme` column holds each marker's *primary* theme — one label per marker, so the counts
# add up to 38 — while a **panel** is the set of markers you would want to look at for a
# question, and those overlap on purpose. Calreticulin is an ER protein and a metabolic
# readout; the lamins are organelle and mechanics both.

# %%
organelle_markers = [m for m in cells.uns["panels"]["organelles"] if m in set(cells.var_names)]

print(cells[:,organelle_markers])

cells.var.loc[organelle_markers, ["marker", "round", "channel"]]


# %% [markdown]
# **A hand-picked set**, when your question does not match a theme:

# %%
my_markers = ["LAMP1", "EEA1", "GM130", "Giantin"]

print(cells[:,my_markers])

cells.var.loc[my_markers, ["marker", "round", "channel"]]

# %% [markdown]
# ## 2 · Subset by condition
#
# ### For example we can subset control conditions 
# The plate holds two conditions that are not treatments. **DMSO** is the vehicle the
# compounds were dissolved in; **PBS** is buffer. Both were meant to leave the cells alone.
#
# That makes them the right place to learn every method in Stage 2. A technique run on
# treated cells will show you *something*, and you will not know whether the something is
# the biology or the method. Run it on two conditions where nothing should be happening and
# any structure you find is structure you made.

# %%
CONTROLS = ["DMSO", "PBS"]
controls = cells[cells.obs.condition.astype(str).isin(CONTROLS)].copy()

print(f"  {cells.n_obs:,} cells -> {controls.n_obs:,} ({controls.n_obs / cells.n_obs:.0%})")
controls

# %% [markdown]
# ### We can also subset any set of conditions

# %%
my_conditions = ["DMSO", "Sapanisertib/INK128", "MK-2206", "Wortmannin"]

subset = cells[cells.obs.condition.astype(str).isin(my_conditions)]

print(f"  {cells.n_obs:,} -> {subset.n_obs:,} cells, {subset.obs.well.nunique()} wells")

# %% [markdown]
# ### We can also subset timepoints

# %%
late = cells[cells.obs.timepoint_h.astype(int).isin([60, 84])]

print(f"  60 and 84 h only: {late.n_obs:,} cells, "
      f"{late.obs.condition.nunique()} conditions still present")

# %% [markdown]
# :::{important}
# All these different subsettings give you directly a reduced anndata object with the same characteristics as the parent
# :::
#
#
# :::{important}
# **Keep the controls.** Every value in `X` is a distance from the control cells at the
# *first* timepoint, and every comparison you will make is against controls at the matching
# timepoint — `effect_table` subtracts them for you. A subset without controls
# cannot be compared, cannot be re-normalised, and cannot be plotted on a meaningful axis.
#
# The same goes for timepoints: keep at least two, or you cannot say anything changed.
# :::

# %% [markdown]
#
# ## 3 · How to subset a specific number of cells and why
#
# Sometimes your dataset is two big, and one cannot compute everything using the full dataset. Therefore, we can subset our dataset to a smaller size to be a able to do these calculations.
#
# However, to keep the distribution, data, variability,... in our dataset we cannot do this blindly.

# %% [markdown]
# ### 3.1 The incorrect way to subset
#
# We could take the first 20k cells in the dataset, however it could perfectly be that the first 20k cells all come from the first timepoint. If we only took this cells, all our analysis would be destroyed. 

# %%
naive = cells[:20_000]
print("  cells[:20_000] gives you:")
print(f"    {naive.obs.well.nunique()} of {cells.obs.well.nunique()} wells")
print(f"    {naive.obs.condition.nunique()} of {cells.obs.condition.nunique()} conditions")
print(f"    timepoints: {sorted(naive.obs.timepoint_h.astype(int).unique().tolist())}")

# %% [markdown]
# The file is written well by well, so `cells[:20_000]` is not a sample of the experiment
# — it is whichever wells happen to sit at the top of the file. Read the counts printed
# above: wells are missing, and so are whole conditions. Any analysis of that describes
# those wells.
#

# %% [markdown]
# ### 3.2 Uniform Random Subsetting

# %%
rng = np.random.default_rng(0)
uniform = np.sort(rng.choice(cells.n_obs, 8_000, replace=False))
cells[uniform]

# %% [markdown]
# ### 3.3 Stratified Subsetting

# %%
balanced = stratified(cells.obs, 8_000)
cells[balanced]

# %% [markdown]
# **Uniform random** keeps the plate's proportions;
# **stratified** takes the same number from every condition × timepoint:
#
# :::{warning}
# A stratified subsample is **balanced, not representative**. Every condition gets the
# same number of cells, so the proportions in it no longer match the plate.
#
# Use it to *look* — embeddings, distributions, a quick heatmap. Compute proportions and
# statistics on the full table, at the well level.
# :::

# %%
for name, index in [("uniform", uniform), ("stratified", balanced)]:
    obs = cells.obs.iloc[index]
    print(f"  {name:11s} {len(index):,} cells, {obs.well.nunique()} wells, "
          f"{obs.condition.nunique()} conditions, "
          f"smallest condition {obs.condition.value_counts().min()}")

# %% [markdown]
# ### 3.4 Subsetting through sketching
#
# Both samplers above are balanced across the **plate**: conditions, timepoints, wells.
# Neither knows anything about the **measurements**. And a cell state that exists in every
# well but in only 1% of cells is rare in exactly the way that matters — it is rare in the
# data space, not on the plate.
#
# Uniform sampling keeps 1% of a 1% population. Stratifying by condition does not help,
# because the rarity is not a condition.
#
# **Geometric sketching** (Hie et al., 2019) samples the space rather than the rows. It
# lays a grid over the data and takes roughly one cell per occupied box, so a region
# holding 40 cells and a region holding 40,000 contribute about equally.

# %%
matrix = np.asarray(cells.X)
# 30,000 cells is a comfortable size for a neighbour graph. The cap keeps the sketch a
# genuine reduction when this notebook is run on a smaller extract of the plate.
TARGET = min(30_000, cells.n_obs // 8)

sketch_index = sketch(matrix, TARGET)
print(f"  {cells.n_obs:,} cells -> {len(sketch_index):,} ({len(sketch_index)/cells.n_obs:.0%})")

cells[sketch_index]

# %% [markdown]
# ### Does it actually keep more?
#
# Define "unusual" without using any labels: the 1% of cells furthest from the centre of
# the marker space. Then count how many of them each sample retains.

# %%
distance = np.linalg.norm(matrix - matrix.mean(axis=0), axis=1)
unusual = distance >= np.quantile(distance, 0.99)

comparison = pd.DataFrame([
    {"sample": name,
     "cells": len(index),
     "unusual kept": int(unusual[index].sum()),
     "% of the sample": round(100 * unusual[index].mean(), 1)}
    for name, index in [("uniform", uniform), ("stratified", balanced),
                        ("geosketch", sketch(matrix, 8_000))]
]).set_index("sample")
comparison["of all unusual cells"] = (
    100 * comparison["unusual kept"] / unusual.sum()).round(1)
comparison

# %% [markdown]
# All three samples are the same size, so this is a like-for-like comparison. Uniform and
# stratified sampling reproduce the population — roughly 1% of the sample is unusual,
# because roughly 1% of the plate is. The sketch deliberately over-represents them.
#

# %% [markdown]
# :::{warning}
# **A sketch is for looking, not for counting.** It over-represents rare cells on purpose,
# so any proportion computed on it is wrong — and wrong in a direction that flatters
# whatever is rare.
#
# Everything downstream respects this split: embeddings and clustering run on the sketch,
# every proportion and every statistic runs on the full table at the well level.
# :::

# %% [markdown]
# ## 4. How to store the outcomes of the subset

# %% [markdown]
# ### Views cost nothing; copies cost memory
#
# Slicing an `AnnData` gives you a **view**. Nothing is allocated until you copy it or write
# it out.

# %% [markdown]
# ### For Sketching
# We first can add to cells anndata - which cells were subsetted in the geosketch. Then subset using .copy() method and then add metadata and etc

# %%
in_sketch = np.zeros(cells.n_obs, dtype=bool)
in_sketch[sketch_index] = True
cells.obs["in_sketch"] = in_sketch

sketch = cells[sketch_index].copy()
sketch.uns["sketch_of"] = "mcs2026_intensity.h5ad"
sketch.uns["sketch_method"] = "geosketch.gs on the 38 normalised markers"
sketch.uns["sketch_size"] = int(len(sketch_index))
sketch.uns["source_n_obs"] = int(cells.n_obs)

sketch

# %% [markdown]
# ### For your own conditions and markers

# %%
view = cells[cells.obs.condition.astype(str).isin(my_conditions), organelle_markers]

copy = view.copy()

print(view)

print(copy)


# %% [markdown]
# Here we represent what is the difference between using the .copy() method and not using it.
#
# As you can see, in both cases you get the same output; however, the view is still part of the cells object. Meaning that it is not a separate object is just what we call a view of the parent object. Any changes made to the view will also affect the parent object, even if they have different variable names.
#
# To create a separate of an scanpy object you need to use the .copy() method. However, since this is a new object it will occupy more space in memory (which is releavant if your dataset exceeds the maximum memory of the system).
#
#

# %% [markdown]
# That is the memory lesson from [Step 11](02_quality_control.ipynb) paying off. Slice
# freely; copy deliberately.

# %% [markdown]
# ### Write down what each file is
#
# `my_subset.h5ad` on a shared cluster three weeks from now is otherwise a mystery,
# including to you. `uns` is where that goes.

# %%
controls.uns["subset_of"] = "mcs2026_intensity.h5ad"
controls.uns["subset_conditions"] = CONTROLS
controls.uns["subset_reason"] = "the two untreated conditions; Stage 2 works on all of them"

copy.uns["subset_of"] = "mcs2026_intensity.h5ad"
copy.uns["subset_conditions"] = my_conditions
copy.uns["subset_theme"] = "organelles"

controls.write_h5ad(DATA / "mcs2026_controls.h5ad", compression="gzip")
sketch.write_h5ad(DATA / "mcs2026_sketch.h5ad", compression="gzip")
cells.write_h5ad(DATA / "mcs2026_intensity.h5ad", compression="gzip")
copy.write_h5ad(DATA / "my_subset.h5ad", compression="gzip")

print(f"  mcs2026_controls.h5ad {controls.n_obs:>7,} cells   <- Stage 2 opens this")
print(f"  mcs2026_sketch.h5ad   {sketch.n_obs:>7,} cells   <- Part 4 opens this")
print(f"  mcs2026_intensity.h5ad    {cells.n_obs:>7,} cells   <- now carries obs['in_sketch']")
print(f"  my_subset.h5ad        {copy.n_obs:>7,} cells   <- yours")

# %% [markdown]
#
# ---
#
# ## Summary 
# ### Stage 1 is complete
#
# | file | one row per | what it is for |
# |---|---|---|
# | `mcs2026_intensity.h5ad` | cell | the full, normalised, annotated dataset |
# | `mcs2026_controls.h5ad` | cell | DMSO and PBS, all of them — **Stage 2** |
# | `mcs2026_sketch.h5ad` | cell | ~30,000 covering all 18 conditions — Part 4 |
# | `mcs2026_full.h5ad` | cell | the wide 2,587-column archive, for texture questions |
#
# `mcs2026_intensity.h5ad` has the full dataset with only the 38 intensity features, with the extra that now using the `obs['in_sketch']` variable you can subset to the equivalent of `mcs2026_sketch.h5ad` .
#
# You may also want the morphology features, and to do this, go through subsetting using the examples provided here but starting from `mcs2026_intensity_shape.h5ad`
#
# [Stage 2](../2_controls/intro.md) opens the controls and shows every method on them.
#
# ## What next? 
#
# ### 1. Build your group's subset
#
# Pick one of the four themes. Build a subset with that theme's markers, all conditions and
# all timepoints, and save it. How large is it, and does it pass `describe_subset`?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# theme = "signaling"          # or mechanics / metabolism / organelles
#
# mine = cells[:, [m for m in cells.uns["panels"][theme] if m in set(cells.var_names)]].copy()
#
# mine.uns["subset_theme"] = theme
#
# mine.write_h5ad(DATA / f"{theme}.h5ad", compression="gzip")
# ```
# :::

# %% [markdown]
# ## Extra: How to go back to the texture of a choosen channel

# %% [markdown]
# :::{note}
# **When you need more than 38 columns.** The intensity object carries one number per antibody
# per cell — *how much* of a protein there is. If your question is about *how it is
# arranged* — whether the Golgi is compact or dispersed, whether lysosomes are punctate —
# the texture features are the measurement, and they are back in the wide table.
#
# Reading its `var` costs nothing, because `backed="r"` opens the file without loading it
# ([Step 5](01_columns_to_markers.ipynb)):
# :::

# %%
wide= sc.read_h5ad(DATA / "mcs2026_full.h5ad", backed="r")
wide_var = wide.var
mine = wide_var[wide_var.marker.isin(my_markers)]

print(mine.family.value_counts().to_string())

wide[:,wide_var.marker.isin(my_markers)]

# %% [markdown]
# ---
#
# **Next:** [Stage 2 — The toolkit, on the controls](../2_controls/intro.md).
