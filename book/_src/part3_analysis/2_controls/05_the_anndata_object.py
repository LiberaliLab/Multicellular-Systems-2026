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
# # 05 · The AnnData object
#
# Stage 1 built seven files and this stage opens one of them.
#
# | | |
# |---|---|
# | **1** | Open it, and print the slots |
# | **2** | The eight slots, and what each is for |
# | **3** | Indexing: names, masks, and the view |
# | **4** | The `obs` column type that changes your answers |
# | **5** | Which file to open for which question |

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

plt.rcParams.update({          # the house style, no package needed
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
pd.set_option("display.width", 140)

# The tables live here on Euler. Change this line if your copy is elsewhere.
DATA = Path("/cluster/project/mcsliberali/data_mcs_2026")


# %% [markdown]
# :::{warning}
# **Three folders, and only one of them is yours.**
#
# - `data_mcs_2026` — shared, **read-only**, and the only one you can see. This is what you
#   load at the start.
# - `file_outputs` — the course's own run folder, where the outputs printed in these pages
#   were produced. You have no access to it, but you will see it in the `DATA` line at the
#   top of chapters 06 to 10. **Replace that path with your own.**
# - `~/mcs2026` — yours. Everything you make goes here.
#
# It matters from the next chapter on, because 06 to 10 form a chain: each opens an `.h5ad`,
# adds something, and writes it back. Load from `data_mcs_2026` once, save into your own
# folder, and read from your own folder after that.
#
# [Working on Euler](../../setup/working_on_euler.md) has the pattern and the `$HOME` quota
# you need to respect.
# :::

# %% [markdown]
# ## 1 · Open it, and print the slots
#
# The first cell of every notebook you write, before you touch anything. Someone else made
# this file — possibly you, six weeks ago — and this is how you find out what they did.

# %%
cells = sc.read_h5ad(DATA / "mcs2026_controls.h5ad")
cells

# %% [markdown]
# ## 2 · The eight slots, and what each is for
#
# ```{image} ../../images/clean_object_light.svg
# :class: only-light
# :alt: The analysis AnnData: X holding normalised values, a raw layer, an annotated var table, a tidy obs table, and provenance in uns.
# ```
# ```{image} ../../images/clean_object_dark.svg
# :class: only-dark
# :alt: The analysis AnnData: X holding normalised values, a raw layer, an annotated var table, a tidy obs table, and provenance in uns.
# ```
#
# | slot | shape | holds |
# |---|---|---|
# | `X` | `n_obs × n_vars` | the measurements — one number per cell per marker |
# | `obs` | `n_obs` rows | what you know about each **cell** |
# | `var` | `n_vars` rows | what you know about each **marker** |
# | `layers` | `n_obs × n_vars` | alternative versions of `X` — same shape, different numbers |
# | `obsm` | `n_obs × anything` | per-cell matrices: embeddings, PCA coordinates |
# | `varm` | `n_vars × anything` | per-**marker** matrices: PCA loadings |
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
# on how far through the course this file has been.

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
# Keep one of the two vehicles. PBS then has no rows left -- which is the point.
mask = cells.obs.condition == "DMSO"
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
#
# ## 5 · Which file to open for which question
#
# Choosing the wrong one is the most common way to get a confidently wrong answer in Part 3.
# These are the ones you open; `mcs2026_slim.h5ad` and `mcs2026_qc.h5ad` are Stage 1's own
# intermediates and `mcs2026_full.h5ad` is the archive you go back to for texture.

# %%
paths = {
    "controls  (this stage)":       DATA / "mcs2026_controls.h5ad",
    "intensity (all cells)":        DATA / "mcs2026_intensity.h5ad",
    "sketch    (all 18, reduced)":  DATA / "mcs2026_sketch.h5ad",
    "full      (wide, 2,587 cols)": DATA / "mcs2026_full.h5ad",
}
pd.DataFrame([
    {"file": path.name, "MB on disk": round(path.stat().st_size / 1e6, 1)}
    for path in paths.values()
], index=list(paths)).rename_axis("open it for")

# %% [markdown]
# | file | one row per | open it when |
# |---|---|---|
# | `mcs2026_controls.h5ad` | cell | **you are in Stage 2.** DMSO and PBS, every cell of them |
# | `mcs2026_intensity.h5ad` | cell | you need all 18 conditions — counting, proportions, projecting labels |
# | `mcs2026_intensity_shape.h5ad` | cell | you need all 18 conditions and shape features — counting, proportions, projecting labels |
# | `mcs2026_sketch.h5ad` | cell | you are embedding all 18 conditions and 653,000 cells will not fit |
# | `mcs2026_full.h5ad` | cell | you need **texture**, or a marker statistic other than the mean |
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
#
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

# %% [markdown]
# ---
#
# ## Saving what you make
#
# Two kinds of plot in this course, and they save differently — worth knowing now rather than
# the week you need figures for a presentation.
#
# - **Figures you built yourself**, from `plt.subplots`, have a `fig`: `fig.savefig(path)`.
#   300 dpi and a tight bounding box are already set at the top of every chapter.
# - **Scanpy's plots** draw *and close* the figure themselves, so there is no `fig` to catch.
#   Pass `save="_name.png"` instead, and know that scanpy writes into `sc.settings.figdir`,
#   prepends the function name, and gives you a **PDF** if you leave the extension off.
#
# [Working on Euler](../../setup/working_on_euler.md) has both, with the gotchas spelled out.

# %% [markdown]
# ---
#
# ## Summary
#
# | slot | holds |
# |---|---|
# | `X` / `layers` | the same cells and the same markers, different numbers |
# | `obs` / `var` | ordinary DataFrames, locked to the row and column axes |
# | `obsm` / `obsp` | per-cell matrices, and per-cell-**pair** matrices |
# | `varm` | per-marker matrices — PCA loadings live here, not in `var` |
# | `uns` | everything that fits nowhere else, including `provenance` |
#
# Three habits that prevent most of the confusing errors:
#
# - **Read from views freely; call `.copy()` the moment you intend to write.**
# - **Pass `observed=True` to every `groupby` on a categorical**, or conditions that are not
#   in the subset come back as zeros that read like a result.
# - **Cast `timepoint_h` with `.astype(int)` before any arithmetic** — it is an *ordered*
#   categorical, which is right for plotting and wrong for subtraction.
#
#
#

# %% [markdown]
# ---
#
# **Next:** [06 · PCA](06_pca.ipynb).
