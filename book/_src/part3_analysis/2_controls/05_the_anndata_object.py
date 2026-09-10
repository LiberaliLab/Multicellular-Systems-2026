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
# # 05 · The AnnData object
#
# Stage 1 built five files and this stage opens one of them. Ten minutes spent on **what is
# actually in it** saves an afternoon later, because almost every confusing error in
# single-cell analysis is really a question about which slot something lives in.
#
# | | |
# |---|---|
# | **1** | Open it, and print the slots |
# | **2** | The seven slots, and what each is for |
# | **3** | Indexing: names, masks, and the view |
# | **4** | The `obs` column type that changes your answers |
# | **5** | Which file to open for which question |

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path.cwd().parents[2] / "src"))

from mcs2026 import plotting
from mcs2026.config import H5AD_SLIM

plotting.set_style()
pd.set_option("display.width", 140)

# %% [markdown]
# ## 1 · Open it, and print the slots
#
# The first cell of every notebook you write, before you touch anything. Someone else made
# this file — possibly you, six weeks ago — and this is how you find out what they did.

# %%
cells = sc.read_h5ad(H5AD_SLIM.with_name("mcs2026_controls.h5ad"))
cells

# %% [markdown]
# That repr is the whole map. Read it as: a matrix of `n_obs × n_vars`, and then five
# tables and dictionaries hanging off its two axes.
#
# The one thing it does not print is what was *done* to the file. Stage 1 wrote that down:

# %%
for key, value in cells.uns["provenance"].items():
    print(f"  {key:18s} {value}")

# %% [markdown]
# ## 2 · The seven slots, and what each is for
#
# ```{image} ../../images/clean_object_light.svg
# :class: only-light
# :alt: The clean AnnData: X holding normalised values, a raw layer, an annotated var table, a tidy obs table, and provenance in uns.
# ```
# ```{image} ../../images/clean_object_dark.svg
# :class: only-dark
# :alt: The clean AnnData: X holding normalised values, a raw layer, an annotated var table, a tidy obs table, and provenance in uns.
# ```
#
# | slot | shape | holds |
# |---|---|---|
# | `X` | `n_obs × n_vars` | the measurements — one number per cell per marker |
# | `obs` | `n_obs` rows | what you know about each **cell** |
# | `var` | `n_vars` rows | what you know about each **marker** |
# | `layers` | `n_obs × n_vars` | alternative versions of `X` — same shape, different numbers |
# | `obsm` | `n_obs × anything` | per-cell matrices: embeddings, PCA coordinates |
# | `obsp` | `n_obs × n_obs` | per-cell-pair matrices: neighbour graphs, sparse |
# | `uns` | anything | everything else — parameters, colours, provenance |
#
# **`X` and `layers` are the same cells and the same markers.** Here `X` is normalised and
# `layers["raw"]` is what the microscope measured, so you can always get back:

# %%
print(f"  X          {cells.X.shape}  {cells.X.dtype}  "
      f"mean {cells.X.mean():+.3f}   range {cells.X.min():+.1f} to {cells.X.max():+.1f}")
raw = cells.layers["raw"]
print(f"  layers.raw {raw.shape}  {raw.dtype}  "
      f"mean {raw.mean():8.1f}   range {raw.min():.0f} to {raw.max():.0f}")

# %% [markdown]
# **`obs` and `var` are ordinary pandas DataFrames** whose row order is locked to the
# matrix. Anything you can do to a DataFrame you can do to them.

# %%
cells.obs.head(3)

# %%
cells.var.head(3)

# %% [markdown]
# **`obsm` holds matrices that share the cell axis but not the marker axis.** A PCA has one
# row per cell and 20 columns that are not markers, so it cannot go in `X` and does not
# belong in `obs`. **`obsp` holds `n_obs × n_obs` matrices** — a neighbour graph is one
# number per *pair* of cells.
#
# Chapters 06 to 09 are what fill them, one slot each. Whether they are filled *now* depends
# on how far through the course this file has been:

# %%
print(f"  obsm: {list(cells.obsm)}")
print(f"  obsp: {list(cells.obsp)}")

# %% [markdown]
# To see the shapes either way, put something there yourself. Take a slice of
# `mcs2026_clean.h5ad` — which never carries an embedding, because nothing is ever computed
# on it — and run chapters 06 and 07 in two lines. Small enough to be instant, and thrown
# away afterwards.

# %%
demo = sc.read_h5ad(H5AD_SLIM.with_name("mcs2026_clean.h5ad"))[:2_000].copy()
sc.pp.pca(demo, n_comps=10, random_state=0)
sc.pp.neighbors(demo, n_neighbors=15, n_pcs=10, random_state=0)

for key, value in demo.obsm.items():
    print(f"  obsm[{key!r:14}] {value.shape}   {value.nbytes / 1e6:6.2f} MB dense")
for key, value in demo.obsp.items():
    filled = 100 * value.nnz / value.shape[0] ** 2
    print(f"  obsp[{key!r:14}] {value.shape}   {filled:.2f}% filled, "
          f"{value.data.nbytes / 1e6:.2f} MB sparse "
          f"(vs {value.shape[0]**2 * 8 / 1e6:,.0f} MB dense)")
del demo

# %% [markdown]
# :::{important}
# **`obsp` is the slot that decides how many cells you can work with.** It is `n_obs²`, so
# the cost is quadratic: 2,000 cells is 4 million potential entries, 30,000 is 900 million,
# and the full ~653,000-cell table is 4 × 10¹¹ — about 3.4 terabytes if it were dense.
#
# Sparsity is what makes it possible at all — only the 15 nearest neighbours of each cell
# are stored, so the real cost is linear in cells. But every graph algorithm downstream
# walks that structure, and the constant is large enough that
# [chapter 04](../1_preparation/04_subsetting_and_sketching.ipynb) cut the data down before
# any of this happens — to the controls for this stage, and to a sketch for Part 4.
# :::

# %% [markdown]
# ## 3 · Indexing: names, masks, and the view
#
# `adata[cells, markers]`, and each half accepts names, positions or a boolean mask.

# %%
print(f"  by marker name        {cells[:, 'Oct4'].shape}")
print(f"  by several            {cells[:, ['Oct4', 'Nanog', 'Sox2']].shape}")
print(f"  by cell mask          {cells[cells.obs.condition == 'DMSO'].shape}")
print(f"  by both               {cells[cells.obs.timepoint_h == 84, 'GATA4'].shape}")
print(f"  by var lookup         {cells[:, cells.var.theme == 'signaling'].shape}")

# %% [markdown]
# Every one of those is a **view** — a window onto the original, costing no memory until
# you materialise it.

# %%
view = cells[cells.obs.condition == "DMSO"]
print(f"  view  is_view={view.is_view}   shares memory with the original")
copy = view.copy()
print(f"  copy  is_view={copy.is_view}   {copy.n_obs * copy.n_vars * 4 / 1e6:.1f} MB of its own")

# %% [markdown]
# :::{warning}
# **Writing to a view does not do what you expect.** Assigning into `view.obs` raises, or
# silently writes to a copy that is then discarded, depending on the version and the slot.
#
# The rule that always works: **call `.copy()` the moment you intend to modify something.**
# Read from views freely; never write to one.
# :::

# %%
# to_memory / .copy() gives you an object you own
subset = cells[cells.obs.timepoint_h.astype(int) >= 60].copy()
subset.obs["late"] = True
print(f"  {subset.n_obs:,} cells, and now an extra obs column: {'late' in subset.obs}")

# %% [markdown]
# ## 4 · The `obs` column type that changes your answers
#
# Half of `obs` is **categorical**, not string. That is not cosmetic.

# %%
print(cells.obs.dtypes.astype(str).to_string())

# %% [markdown]
# A categorical remembers every category it was declared with, whether or not any rows are
# left holding it. Subsetting an `AnnData` tidies that up for you. Subsetting `obs` as a
# plain DataFrame — which you will do constantly, because it *is* a DataFrame — does not.
#
# Same filter, two ways:

# %%
mask = cells.obs.condition.isin(["DMSO", "MK-2206"])
print(f"  AnnData subset   : {len(cells[mask].obs.condition.cat.categories)} categories")
print(f"  DataFrame subset : {len(cells.obs[mask].condition.cat.categories)} categories")

# %% [markdown]
# And that difference decides what `groupby` gives back:

# %%
frame = cells.obs[mask].assign(Oct4=np.asarray(cells[mask, "Oct4"].X).ravel())
pd.DataFrame({
    "cells (observed=False)": frame.groupby("condition", observed=False).size(),
    "mean Oct4 (observed=False)": frame.groupby("condition", observed=False).Oct4.mean().round(3),
}).head(8)

# %%
frame.groupby("condition", observed=True).agg(cells=("Oct4", "size"),
                                              mean_Oct4=("Oct4", "mean")).round(3)

# %% [markdown]
# :::{warning}
# **Sixteen of those rows are conditions that are not in the subset**, reported as 0 cells
# and a `NaN` mean. Put the first table in a bar chart and you get two real bars and sixteen
# at zero — which reads as "these compounds did nothing", the exact opposite of "these
# compounds are not in this data".
#
# **Pass `observed=True` to every `groupby` on a categorical.** Recent pandas warns and will
# eventually change the default; until then, the failure is silent and looks like a result.
#
# `timepoint_h` is also an **ordered** categorical, which is right for plotting and wrong for
# arithmetic — pandas will not subtract categories. Cast it: `obs.timepoint_h.astype(int)`.
# :::

# ## 5 · Which file to open for which question
#
# Stage 1 wrote four. Choosing the wrong one is the most common way to get a confidently
# wrong answer in Part 3.

# %%
paths = {
    "controls (this stage)":       H5AD_SLIM.with_name("mcs2026_controls.h5ad"),
    "clean    (all cells)":        H5AD_SLIM.with_name("mcs2026_clean.h5ad"),
    "sketch   (all 18, reduced)":  H5AD_SLIM.with_name("mcs2026_sketch.h5ad"),
    "wells    (statistics)":       H5AD_SLIM.with_name("mcs2026_wells.parquet"),
    "slim     (wide, 2,587 cols)": H5AD_SLIM,
}
pd.DataFrame([
    {"file": path.name, "MB on disk": round(path.stat().st_size / 1e6, 1)}
    for path in paths.values()
], index=list(paths)).rename_axis("open it for")

# %% [markdown]
# | file | one row per | open it when |
# |---|---|---|
# | `mcs2026_controls.h5ad` | cell | **you are in Stage 2.** DMSO and PBS, every cell of them |
# | `mcs2026_clean.h5ad` | cell | you need all 18 conditions — counting, proportions, projecting labels |
# | `mcs2026_sketch.h5ad` | cell | you are embedding all 18 conditions and 653,000 cells will not fit |
# | `mcs2026_wells.parquet` | well | you are running a **statistical test**. The well is the replicate |
# | `mcs2026_slim.h5ad` | cell | you need **texture**, or a marker statistic other than the mean |
#
# :::{important}
# **Stage 2 opens the controls and nothing else.** Every method in the chapters that follow
# is learned on the two conditions that received no compound — so if a technique appears to
# find something, the finding is a property of the technique.
#
# It is also the one subset small enough to use whole. 32 wells fit in memory and in
# patience; the 653,000-cell table does not, which is why Part 4 has a sketch and this stage
# does not need one.
# :::

# ## The habit
#
# One cell, at the top of every notebook, before anything else:
#
# ```python
# cells = sc.read_h5ad(path)
# print(cells)                       # the slots
# print(cells.uns["provenance"])     # what was done to it
# print(cells.obs.dtypes)            # what will need casting
# ```
#
# ---
#
# ## Exercises
#
# ### 1. Get back to the raw numbers
#
# `X` is normalised. Reconstruct the raw mean intensity of Oct4 for the DMSO cells at 36 h
# from `layers["raw"]`, and check it against `X` — what is the relationship between them?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# mask = (cells.obs.condition == "DMSO") & (cells.obs.timepoint_h.astype(int) == 36)
# block = cells[mask, "Oct4"]
# raw = np.log2(np.asarray(block.layers["raw"]).ravel() + 1)
# print("raw log2 :", raw.mean().round(3), raw.std(ddof=1).round(3))
# print("X        :", np.asarray(block.X).mean().round(3), np.asarray(block.X).std(ddof=1).round(3))
# ```
#
# The `X` values come out at 0 and 1 by construction — these *are* the control cells that
# Step 17 centred and scaled on. Any other condition at 36 h is measured against this one.
# :::

# %% [markdown]
# ### 2. Find something the repr does not tell you
#
# How many wells are in this file, how many conditions, and how many cells per well? How
# would you have found that out from the repr alone?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# print(cells.obs.well.nunique(), "wells;", cells.obs.condition.nunique(), "conditions")
# print(cells.obs.groupby("well", observed=True).size().describe())
# print(cells.uns["subset_reason"])
# ```
#
# You could not have. The repr lists the *names* of the columns, never their contents — and
# a file that lost half its wells looks exactly like one that did not. This is why the
# opening cell prints `obs` summaries and `uns`, not just the object.
# :::

# %% [markdown]
# ### 3. Break a categorical on purpose
#
# Subset to two conditions, then plot the mean of any marker per condition without passing
# `observed=True`. What does the figure show, and what would you have concluded from it?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# two = cells[cells.obs.condition.isin(["DMSO", "MK-2206"])]
# print(two.obs.groupby("condition", observed=False).size().head(8))
# ```
#
# Sixteen categories, fourteen of them empty. In a bar chart that is fourteen bars of height
# zero, which reads as "these conditions had no effect" rather than "these conditions are
# not in this subset". The two failures look identical on the page and are opposites.
# :::

# %% [markdown]
# ---
#
# **Next:** [06 · PCA](06_pca.ipynb).
