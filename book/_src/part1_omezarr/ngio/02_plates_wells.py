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
# # 3.2 · Plates and wells
#
# In this notebook you will:
#
# - open a whole HCS plate and ask what is in it
# - navigate rows, columns, wells, images and acquisitions
# - open images across the plate in parallel
# - **concatenate one table across every well in a single call**
#
# This is the chapter that matters most. Everything before it worked on one image;
# a screen is 384 of them, and the difference between a loop you write by hand and one
# call is the difference between an afternoon and a minute.
#
# **The 733,556-cell feature table used in Part 3 was produced exactly this way.**
#
# Follows the official
# [HCS guide](https://biovisioncenter.github.io/ngio/stable/getting_started/5_hcs/) and
# [HCS exploration tutorial](https://biovisioncenter.github.io/ngio/stable/tutorials/hcs_exploration/).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ngio import open_ome_zarr_plate, open_ome_zarr_well

# The plate lives here on Euler. Change this line if your copy is elsewhere.
PLATE_PATH = Path("/cluster/project/mcsliberali/zarr_files/dummy.zarr")

# %% [markdown]
# ## Opening a plate

# %%
plate = open_ome_zarr_plate(PLATE_PATH)
plate

# %% [markdown]
# ## What is in it
#
# Three properties give the shape of the experiment.

# %%
print("rows:        ", plate.rows)
print("columns:     ", plate.columns)
print("acquisitions:", plate.acquisition_ids)

# %% [markdown]
# ## Finding images
#
# Three methods, at three levels of granularity. All of them return **paths**, not data —
# nothing has been read.

# %%
well_paths = plate.wells_paths()
image_paths = plate.images_paths()
print(f"{len(well_paths)} wells, {len(image_paths)} images")
print("first few wells: ", well_paths[:6])
print("first few images:", image_paths[:6])

# %%
row, column = well_paths[0].split("/")
plate.well_images_paths(row=row, column=int(column))

# %% [markdown]
# A well can hold more than one image — different acquisitions of the same well, for
# instance. That is why `wells_paths()` and `images_paths()` are different methods and
# can return different counts.

# %% [markdown]
# ## Getting containers
#
# Now we actually open things. Each of these returns the `OmeZarrContainer` objects from
# [chapter 3.1](01_container.ipynb).

# %%
container = plate.get_image(row=row, column=int(column), image_path="0")
container

# %%
well_images = plate.get_well_images(row=row, column=int(column))
well_images

# %% [markdown]
# ### Opening the whole plate, in parallel
#
# :::{tip}
# Pass `max_workers="auto"` to open images concurrently. Each open is a metadata read, so
# it is round-trip bound rather than CPU bound — on a network or cloud store this is
# several times faster. The current default reads one at a time and warns; in ngio 1.2
# `"auto"` becomes the default. Pass `max_workers=1` to force serial reads silently.
# :::

# %%
all_images = plate.get_images(max_workers="auto")
print(f"{len(all_images)} containers open")
next(iter(all_images.items()))

# %% [markdown]
# ## A well on its own
#
# You can also open a single well directly, without the plate.

# %%
well = open_ome_zarr_well(PLATE_PATH / well_paths[0])
print("images in this well:", well.paths())
print("acquisition ids:    ", well.acquisition_ids)

# %% [markdown]
# ## What has been measured, plate-wide
#
# Before aggregating anything, ask which tables exist. `mode="common"` lists only the
# tables present in *every* image — the ones it is safe to concatenate.

# %%
plate.list_image_tables(mode="common")

# %%
plate.list_image_tables(mode="all")

# %% [markdown]
# The difference between the two is worth looking at. A table that exists in some wells
# but not others usually means a processing step failed somewhere, and it is much better
# to discover that here than halfway through an analysis.

# %% [markdown]
# ## The one call that matters
#
# `concatenate_image_tables` reads the named table from every image in the plate, stacks
# them, and adds three columns — `row`, `column` and `path_in_well` — so you can tell which
# image each row came from.

# %%
feature_tables = plate.list_image_tables(mode="common", filter_types="feature_table")
print("feature tables in every image:", feature_tables)

# Take a feature table if the plate has one -- one row per segmented object.
# Otherwise fall back to the first table every image shares.
table_name = (feature_tables or plate.list_image_tables(mode="common"))[0]
print("concatenating:", table_name)

# index_key gives every row a plate-wide name, built from row, column, image and label.
# Without it, rows that share a name in different wells overwrite each other.
table = plate.concatenate_image_tables(name=table_name, index_key="object", max_workers="auto")
frame = table.dataframe
print(f"{len(frame):,} rows x {frame.shape[1]} columns, from {len(image_paths)} images")
frame.head()

# %% [markdown]
# That is the whole aggregation step. Compare it with what you would otherwise write: a
# loop over rows, a loop over columns, a try/except for missing wells, a manual column
# for the well name, and a `pd.concat` at the end.
#
# :::{important}
# **This is how the Part 3 dataset was made.** A feature table per well, concatenated
# across the plate, then written out as AnnData. The 4,464 columns you decode in
# [Part 3 chapter 01](../../part3_analysis/1_preparation/01_columns_to_markers.ipynb) came out of a call like
# the one above.
#
# There is a variant, `concatenate_image_tables_as`, which returns a specific table type
# (`FeatureTable`, `RoiTable`, …) instead of the generic one — useful when you want the
# type-specific methods.
# :::

# %% [markdown]
# ### How many objects per well?

# %%
counts = frame.groupby(["row", "column"]).size().rename("objects")
print(counts.describe()[["count", "mean", "min", "max"]].round(1).to_string())
counts.head()

# %% [markdown]
# ---
#
# ## Exercises
#
# ### 1. Which wells are missing?
#
# A 384-well plate has 16 rows and 24 columns. Which of the 384 possible wells does this
# plate actually contain, and can you draw that as a plate map?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# present = pd.DataFrame(
#     [{"row": p.split("/")[0], "column": int(p.split("/")[1]), "present": 1}
#      for p in plate.wells_paths()]
# )
# print(f"{len(present)} of 384 wells used")
#
# # a 384-well plate is 16 rows by 24 columns; wells the plate does not have stay empty
# rows, columns = list("ABCDEFGHIJKLMNOP"), list(range(1, 25))
# grid = (present.pivot(index="row", columns="column", values="present")
#         .reindex(index=rows, columns=columns))
#
# fig, ax = plt.subplots(figsize=(9, 5))
# ax.imshow(grid.to_numpy(dtype=float), cmap="viridis")
# ax.set_xticks(range(24), columns)
# ax.set_yticks(range(16), rows)
# ax.set_title("Wells present")
# plt.show()
# ```
#
# A plate map is a pivot: rows down, columns across, one cell per well. `reindex` is what
# puts the missing wells back as empty cells, so the plate keeps its real shape.
#
# Empty border rows and columns are normal: edge wells evaporate faster, so they are
# often left out deliberately. Compare this map with the one in
# [Part 3 chapter 00](../../part3_analysis/1_preparation/00_what_you_are_given.ipynb).
# :::

# %% [markdown]
# ### 2. Does `max_workers` help here?
#
# Time `plate.get_images(max_workers=1)` against `plate.get_images(max_workers="auto")`.
# Is the difference large? Would you expect a different answer on a cloud store?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# import time
# for workers in [1, "auto"]:
#     start = time.time()
#     plate.get_images(max_workers=workers)
#     print(f"max_workers={workers}: {time.time() - start:.1f} s")
# ```
#
# On a fast local filesystem the gain is modest — the reads are small and the disk is
# quick. On S3 or a busy network filesystem each open costs a network round trip, and
# doing 384 of them concurrently instead of one after another is the difference between
# seconds and minutes. This is the case `max_workers` exists for.
# :::

# %% [markdown]
# ### 3. Build a per-well summary
#
# From the concatenated table, compute the mean of one numeric column per well, and draw
# it as a plate map. Does the plate show any position structure?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# measures = [c for c in frame.select_dtypes("number").columns if c != "label"]
# print(measures[:10])
# numeric = measures[0]                      # or any other name in that list
#
# summary = frame.groupby(["row", "column"])[numeric].mean().reset_index()
# summary["column"] = summary["column"].astype(int)
#
# rows, columns = list("ABCDEFGHIJKLMNOP"), list(range(1, 25))
# grid = (summary.pivot(index="row", columns="column", values=numeric)
#         .reindex(index=rows, columns=columns))
#
# fig, ax = plt.subplots(figsize=(9, 5))
# image = ax.imshow(grid.to_numpy(dtype=float), cmap="magma")
# ax.set_xticks(range(24), columns)
# ax.set_yticks(range(16), rows)
# ax.set_title(f"mean {numeric}")
# fig.colorbar(image, ax=ax, shrink=0.7, label=numeric)
# plt.show()
# ```
#
# Look for gradients rather than scatter. A smooth trend across the plate is usually
# technical — evaporation, uneven illumination, a pipetting order — and it is the reason
# Part 3 normalises against control wells spread across the plate rather than against a
# single global mean.
# :::

# %% [markdown]
# ---
#
# **Next:** [3.3 · Labels, ROIs and feature tables](03_labels_tables.ipynb).
