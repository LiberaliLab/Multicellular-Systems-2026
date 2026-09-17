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
# # 3.3 · Labels, ROIs and feature tables
#
# In this notebook you will:
#
# - use the four typed tables `ngio` understands
# - jump straight to one segmented object with a **masking ROI table**
# - read a **feature table** — the thing `ez-zarr` could not
# - get the same table as pandas, polars or AnnData
# - segment something, write the label back, and build its ROI and feature tables
#
# Tables are not part of the core OME-Zarr specification. They follow
# [ngio's table specification](https://biovisioncenter.github.io/ngio/stable/table_specs/overview/),
# which began in Fractal and is now maintained in ngio. This is where the Part 3 data
# comes from.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ngio

sys.path.insert(0, str(Path.cwd().parents[2] / "src"))
from mcs2026.config import PLATE_PATH

container = ngio.open_ome_zarr_container(PLATE_PATH / "B" / "03" / "0")
image = container.get_image()
container

# %% [markdown]
# ## Four kinds of table
#
# `ngio` recognises four typed tables plus an untyped fallback:
#
# | type | holds | indexed by |
# |---|---|---|
# | `roi_table` | arbitrary regions of interest | ROI name |
# | `masking_roi_table` | the bounding box of each segmented object | label id |
# | `feature_table` | per-object measurements | label id |
# | `condition_table` | experimental conditions | well or image |
# | `generic_table` | anything ngio cannot classify | — |

# %%
container.list_tables()

# %%
container.tables_container.table_types()

# %% [markdown]
# `list_tables` also filters by type, which is how you find a table without knowing what
# somebody called it:

# %%
for table_type in ["roi_table", "masking_roi_table", "feature_table"]:
    print(f"{table_type:20s} {container.list_tables(filter_types=table_type)}")

# %% [markdown]
# ## ROI tables
#
# The `FOV_ROI_table` records the microscope's fields of view — where each acquired tile
# sits within the well.

# %%
roi_table = container.get_roi_table("FOV_ROI_table")
rois = roi_table.rois()
print(f"{len(rois)} ROIs")
rois[0]

# %% [markdown]
# A `Roi` carries its coordinates in **world** units (micrometres) plus its name. Because
# of that it is valid at any pyramid level, and slicing with it needs no arithmetic from
# you.

# %%
roi = roi_table.get(rois[0].get_name())
patch = image.get_roi_as_numpy(roi, channel_selection=container.channel_labels[0],
                               axes_order=["y", "x"])
print(f"{roi.get_name()}: {patch.shape} pixels")

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.imshow(patch, cmap="gray", vmin=np.percentile(patch, 1), vmax=np.percentile(patch, 99.5))
ax.set_title(roi.get_name()); ax.axis("off")

# %% [markdown]
# ## Masking ROI tables
#
# A masking ROI table is indexed by **label id**, so it answers "where is object 42?"
# directly. This is the thing that made the last exercise of
# [chapter 2](../ezzarr/02_ezzarr_quicklook.ipynb) awkward.

# %%
masking_names = container.list_tables(filter_types="masking_roi_table")
masking_table = container.get_masking_roi_table(masking_names[0])
print("reference label:", masking_table.reference_label)
print("objects:", len(masking_table.rois()))

# %%
label_ids = [r.label for r in masking_table.rois() if r.label is not None]
target = label_ids[len(label_ids) // 2]

object_roi = masking_table.get_label(target)
object_roi

# %%
crop = image.get_roi_as_numpy(object_roi, channel_selection=container.channel_labels[0],
                              axes_order=["y", "x"])
label = container.get_label(masking_table.reference_label)
mask = label.get_roi_as_numpy(object_roi, axes_order=["y", "x"])

fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
axes[0].imshow(crop, cmap="gray"); axes[0].set_title(f"object {target}")
axes[1].imshow(mask == target, cmap="gray"); axes[1].set_title("its mask")
axes[2].imshow(crop, cmap="gray")
axes[2].contour(mask == target, levels=[0.5], colors="yellow", linewidths=1.5)
axes[2].set_title("outline")
for ax in axes:
    ax.axis("off")
fig.tight_layout()

# %% [markdown]
# Two lines to go from an object id to its pixels, at the right place and the right size.
# `Roi.zoom()` pads the box if you want context around the object:

# %%
zoomed = masking_table.get_label(target).zoom(2.0)
context = image.get_roi_as_numpy(zoomed, channel_selection=container.channel_labels[0],
                                 axes_order=["y", "x"])
print(f"1x: {crop.shape}   2x: {context.shape}")

# %% [markdown]
# ## Feature tables
#
# The one `ez-zarr` could not read. A feature table is per-object measurements, indexed
# by label id.

# %%
feature_names = container.list_tables(filter_types="feature_table")
feature_names

# %%
feature_table = container.get_feature_table(feature_names[0])
print("reference label:", feature_table.reference_label)
feature_table.dataframe.head()

# %% [markdown]
# ### Three views of the same table
#
# Every table exposes all three representations regardless of how it is stored on disk.
# Pick whichever suits the next step.

# %%
print("pandas: ", type(feature_table.dataframe).__name__, feature_table.dataframe.shape)
print("polars: ", type(feature_table.lazy_frame).__name__)
print("anndata:", type(feature_table.anndata).__name__, feature_table.anndata.shape)

# %% [markdown]
# :::{note}
# The **AnnData** view is the bridge to Part 3. `X` holds the measurements, `obs` the
# per-object metadata — the same structure the 733,556-cell table uses. Aggregate one of
# these per well across a plate with `concatenate_image_tables`
# ([chapter 3.2](02_plates_wells.ipynb)) and you have built the Part 3 dataset.
# :::

# %% [markdown]
# ### Joining features to positions
#
# Both tables are indexed by label id, so they join directly.

# %%
features = feature_table.dataframe
positions = pd.DataFrame([
    {"label": r.label,
     "x_um": r.get("x").start if r.get("x") else np.nan,
     "y_um": r.get("y").start if r.get("y") else np.nan}
    for r in masking_table.rois() if r.label is not None
]).set_index("label")

joined = features.join(positions, how="inner") if features.index.name else \
    features.set_index(features.columns[0]).join(positions, how="inner")
joined.head()

# %%
numeric = joined.select_dtypes("number").columns
colour_by = next((c for c in numeric if c not in ("x_um", "y_um")), numeric[0])

fig, ax = plt.subplots(figsize=(6, 5.5))
scatter = ax.scatter(joined.x_um, joined.y_um, c=joined[colour_by], s=14, cmap="viridis")
ax.set(xlabel="x (um)", ylabel="y (um)", title=f"objects coloured by {colour_by}")
ax.invert_yaxis(); ax.set_aspect("equal")
fig.colorbar(scatter, ax=ax, shrink=0.8, label=colour_by)

# %% [markdown]
# ## Making your own
#
# Tables can be pure in-memory objects — they do not have to come off disk.

# %%
from ngio import Roi
from ngio.tables import RoiTable

my_roi = Roi.from_values(slices={"x": (0.0, 128.0), "y": (0.0, 128.0)}, name="corner")
my_table = RoiTable(rois=[my_roi])
my_table

# %%
whole_image = container.build_image_roi_table("whole_image")
whole_image.rois()[0]

# %% [markdown]
# ### Segment, then write it back
#
# The full round trip: read pixels, segment, `derive_label`, `set_array`, `consolidate`,
# then build the ROI and feature tables for what you made.

# %%
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table
from skimage.morphology import remove_small_objects

small = container.get_image(path="2")
plane = small.get_as_numpy(channel_selection=container.channel_labels[0], axes_order=["y", "x"])

smoothed = ndimage.gaussian_filter(plane.astype(float), sigma=2)
binary = smoothed > threshold_otsu(smoothed)
binary = remove_small_objects(binary, min_size=50)      # min_size, not max_size
segmentation, n_objects = ndimage.label(binary)
print(f"{n_objects} objects")

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
axes[0].imshow(plane, cmap="gray", vmin=np.percentile(plane, 1), vmax=np.percentile(plane, 99.5))
axes[0].set_title("image")
axes[1].imshow(np.ma.masked_where(segmentation == 0, segmentation % 20 + 1),
               cmap="tab20", interpolation="nearest")
axes[1].set_title(f"{n_objects} objects")
for ax in axes:
    ax.axis("off")

# %% [markdown]
# :::{warning}
# Writing requires the store to be writable. On the shared course data this will fail —
# correctly. Point `PLATE_PATH` at your own copy first, or read the cells below without
# running them.
# :::

# %%
# new_label = container.derive_label("my_segmentation", ref_image=small, overwrite=True)
# new_label.set_array(segmentation.astype(np.uint16), axes_order=["y", "x"])
# new_label.consolidate()          # rebuild the coarser pyramid levels
#
# roi_table = container.build_masking_roi_table("my_segmentation")
# container.add_table("my_segmentation_ROI_table", roi_table, backend="csv", overwrite=True)
print("derive_label -> set_array -> consolidate -> build_masking_roi_table -> add_table")

# %% [markdown]
# `consolidate()` is not automatic. Write level 0 and skip it, and the coarser levels
# still hold the old data — the image looks right zoomed in and wrong zoomed out.

# %%
from ngio.tables import FeatureTable

measurements = pd.DataFrame(regionprops_table(
    segmentation, intensity_image=plane,
    properties=("label", "area", "mean_intensity", "eccentricity", "solidity"),
))
measurements.head()

# %%
my_features = FeatureTable(measurements, reference_label="my_segmentation")
# container.add_table("my_features", my_features, backend="parquet", overwrite=True)
my_features

# %% [markdown]
# The `backend` argument decides the on-disk format — `anndata` (the default), `parquet`,
# `csv` or `json`. It does not change how you read the table back: `.dataframe`,
# `.lazy_frame` and `.anndata` all work whatever you chose.

# %% [markdown]
# ---
#
# ## Exercises
#
# ### 1. Which object is the biggest?
#
# Using the feature table and the masking ROI table, find the largest object in the well
# and display it with its outline.

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# area_column = next(c for c in features.columns if "area" in c.lower())
# biggest = features[area_column].idxmax()
#
# roi = masking_table.get_label(int(biggest)).zoom(1.3)
# crop = image.get_roi_as_numpy(roi, channel_selection=container.channel_labels[0],
#                               axes_order=["y", "x"])
# mask = label.get_roi_as_numpy(roi, axes_order=["y", "x"])
# plt.imshow(crop, cmap="gray")
# plt.contour(mask == int(biggest), levels=[0.5], colors="yellow")
# plt.axis("off")
# ```
#
# This is the quality-control loop in miniature: a number in a table looks odd, and two
# lines later you are looking at the object it came from. Being able to do that quickly
# is what stops a segmentation artefact becoming a finding.
# :::

# %% [markdown]
# ### 2. Does the backend change anything?
#
# Write the same feature table with `backend="csv"` and `backend="parquet"`. Compare the
# file sizes on disk, and check that `.dataframe` returns the same thing either way.

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# Parquet is typically several times smaller than CSV and much faster to read, because it
# is columnar and typed. CSV is readable in a text editor and by anything, which is
# occasionally worth more than speed.
#
# The point is that `.dataframe` is identical either way: the backend is a storage
# decision, not an API decision, so you can change your mind later without touching the
# analysis code.
# :::

# %% [markdown]
# ### 3. Compare your segmentation with the one that shipped
#
# You segmented at pyramid level 2 with a crude Otsu threshold. The plate already
# contains a segmentation. How many objects does each find, and how do the area
# distributions compare? What would you change?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# theirs = label.get_as_numpy(axes_order=["y", "x"])
# print("shipped:", len(np.unique(theirs)) - 1, "objects")
# print("mine:   ", n_objects, "objects")
# ```
#
# Expect yours to find fewer and larger objects: a plain threshold merges touching cells
# into one blob, where the shipped segmentation almost certainly used a watershed or a
# trained model to split them. Working at level 2 also costs you small objects entirely.
#
# The honest conclusion is that thresholding is a starting point, not a method — and the
# reason feature tables are worth QC-ing before you trust them.
# :::

# %% [markdown]
# ---
#
# That is Part 1. You can now open an OME-Zarr, navigate a plate, read and write labels
# and tables, and aggregate measurements across a screen.
#
# **Next:** [Part 2 — Napari](../../part2_napari/intro.md), to look at the same data with
# your own eyes. Or jump to
# [Part 3](../../part3_analysis/intro.md), which starts from a table built exactly the way
# [chapter 3.2](02_plates_wells.ipynb) built one.
