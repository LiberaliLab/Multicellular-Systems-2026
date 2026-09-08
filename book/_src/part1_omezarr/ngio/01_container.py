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
# # 3.1 · Containers, images and labels
#
# In this notebook you will:
#
# - open an OME-Zarr with `ngio` and ask it what it contains
# - get an image at a chosen pyramid level, or at a chosen physical resolution
# - read pixels as NumPy or as Dask, and know when to use which
# - work in **micrometres** instead of pixels
# - load a segmentation and crop to a region of interest
#
# `ez-zarr` was for looking. **`ngio`** is for working: it reads and writes, it
# understands labels and tables, and it knows about plates. This chapter covers one
# image; [chapter 3.2](02_plates_wells.ipynb) scales it to 384 wells.
#
# The material follows the official
# [ngio getting-started guide](https://biovisioncenter.github.io/ngio/stable/getting_started/1_ome_zarr_containers/).
# We use **ngio 1.1.0** — code written for 0.5.x differs in a few places, listed in the
# [cheat sheet](../cheatsheet.md).

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ngio

sys.path.insert(0, str(Path.cwd().parents[2] / "src"))
from mcs2026.config import PLATE_PATH

image_path = PLATE_PATH / "B" / "03" / "0"
print("ngio", ngio.__version__)

# %% [markdown]
# ## The container
#
# `open_ome_zarr_container` gives you an **`OmeZarrContainer`**. It is the whole image
# group: the pyramid, the labels, the tables and the metadata — but *not* the pixels.
# Nothing has been read yet.

# %%
container = ngio.open_ome_zarr_container(image_path)
container

# %% [markdown]
# The repr is deliberately terse: how many pyramid levels, how many labels, how many
# tables. Ask for detail when you want it.

# %%
print("levels:         ", container.levels)
print("level paths:    ", container.level_paths)
print("channels:       ", container.channel_labels)
print("3D?             ", container.is_3d)
print("time series?    ", container.is_time_series)
print("labels:         ", container.list_labels())
print("tables:         ", container.list_tables())

# %% [markdown]
# :::{note}
# Opening is cheap because it only reads JSON. This matters when you open 384 of them in
# [the next chapter](02_plates_wells.ipynb) — the cost there is one metadata read
# per well, not one image read.
# :::

# %% [markdown]
# ## Getting an image
#
# `get_image()` returns an **`Image`**: a handle on one pyramid level. Still no pixels.
#
# There are three ways to choose the level, and the third is the one to prefer.

# %%
image = container.get_image()                    # full resolution (default)
coarse = container.get_image(path="2")           # a specific pyramid level
print("level 0:", image.shape, image.pixel_size)
print("level 2:", coarse.shape, coarse.pixel_size)

# %% [markdown]
# Asking for a **physical resolution** rather than a level number is more robust: your
# code then does not depend on how many levels this particular plate happens to have.

# %%
from ngio import PixelSize

target = PixelSize(x=2.0, y=2.0, z=1.0)
matched = container.get_image(pixel_size=target, strict=False)
print(f"asked for 2.0 um/px, got {matched.pixel_size.x:.3f} um/px at level {matched.path}")

# %% [markdown]
# `strict=False` means "give me the nearest level"; `strict=True` raises unless a level
# matches exactly.

# %% [markdown]
# ## Reading pixels
#
# Two methods, and the choice matters.
#
# **`get_as_numpy`** reads into memory. Use it when the result fits.
#
# **`get_as_dask`** returns a lazy array. Nothing is read until you compute, and
# operations are applied chunk by chunk. Use it when the result does not fit, or when you
# are about to reduce it anyway.

# %%
print("axes:", image.axes, " shape:", image.shape, " dtype:", image.dtype)

# %%
small = container.get_image(path="3")
data = small.get_as_numpy(channel_selection=container.channel_labels[0], axes_order=["y", "x"])
print("numpy:", data.shape, data.dtype, f"{data.nbytes / 1e6:.1f} MB")

# %%
lazy = image.get_as_dask()
print("dask: ", lazy.shape, "  nothing read yet")
print("a maximum-intensity projection is also free until you ask for it:")
projected = lazy.max(axis=1) if image.is_3d else lazy
projected

# %% [markdown]
# :::{tip}
# `channel_selection` takes a channel **name**, not an index. Names survive a change in
# acquisition order; indices do not. `axes_order` says which axes you want back and in
# what order, so you can ask for `["y", "x"]` and get a 2-D array rather than remembering
# whether to write `arr[0, 0]` or `arr[:, 0]`.
# :::

# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(data, cmap="gray", vmin=np.percentile(data, 1), vmax=np.percentile(data, 99.5))
ax.set_title(f"{container.channel_labels[0]} — level {small.path}")
ax.axis("off")

# %% [markdown]
# ## Physical units
#
# Every image knows how big a pixel is, so you never have to carry a conversion factor
# around by hand.

# %%
pixel_size = image.pixel_size
print("pixel size:", pixel_size)
print(f"  x = {pixel_size.x:.4f} um")
print(f"  in-plane area of one pixel = {pixel_size.xy_plane_area:.4f} um^2")

# %% [markdown]
# This is why every measurement in Part 3 is in micrometres and square micrometres
# rather than pixels: an area in pixels means nothing without the objective it was taken
# with.

# %% [markdown]
# ## Regions of interest
#
# A **`Roi`** is a named box in *world* coordinates — micrometres, not pixels. That means
# the same ROI is valid at every pyramid level, which is the point.

# %%
from ngio import Roi

roi = Roi.from_values(
    slices={"x": (0.0, 200.0), "y": (0.0, 200.0)},
    name="top_left_corner",
)
roi

# %%
patch = image.get_roi_as_numpy(roi, channel_selection=container.channel_labels[0],
                               axes_order=["y", "x"])
print(f"200 x 200 um at level 0 -> {patch.shape} pixels")

coarse_patch = coarse.get_roi_as_numpy(roi, channel_selection=container.channel_labels[0],
                                       axes_order=["y", "x"])
print(f"the same ROI at level 2  -> {coarse_patch.shape} pixels")

# %% [markdown]
# The same physical region, fewer pixels — because the ROI is defined in micrometres and
# each level resolves it differently. You did no arithmetic.

# %% [markdown]
# ## Labels
#
# A **label** is a segmentation stored as its own multiscale image, where each pixel
# holds the id of the object it belongs to. It shares the coordinate system with the
# image.

# %%
container.list_labels()

# %%
label_name = container.list_labels()[0]
label = container.get_label(label_name, pixel_size=small.pixel_size, strict=False)
mask = label.get_as_numpy(axes_order=["y", "x"])
print(f"label '{label_name}': {mask.shape}, {len(np.unique(mask)) - 1} objects")

# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(data, cmap="gray", vmin=np.percentile(data, 1), vmax=np.percentile(data, 99.5))
overlay = np.ma.masked_where(mask == 0, mask % 20 + 1)
ax.imshow(overlay, cmap="tab20", alpha=0.45, interpolation="nearest")
ax.set_title(f"{label_name} over {container.channel_labels[0]}")
ax.axis("off")

# %% [markdown]
# :::{important}
# Note `pixel_size=small.pixel_size` when fetching the label. Image and label must be at
# the **same resolution** before you overlay them, and asking by pixel size rather than
# by level number is how you guarantee that even if the two pyramids have different
# numbers of levels.
# :::

# %% [markdown]
# ---
#
# ## Exercises
#
# ### 1. NumPy or Dask?
#
# How much memory would `image.get_as_numpy()` need at level 0, for all channels? Compute
# it from `shape` and `dtype` before you try it.

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# import numpy as np
# nbytes = np.prod(image.shape) * np.dtype(image.dtype).itemsize
# print(f"{nbytes / 1e9:.2f} GB")
# ```
#
# For one well of a high-content plate this is often several gigabytes — fine on Euler
# with a large session, fatal on a laptop. The habit worth forming is to compute the
# number before the read, not after the crash.
# :::

# %% [markdown]
# ### 2. Walk the pyramid
#
# Print the shape and pixel size of every level. By what factor does each step change
# them, and does `z` change too?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# for path in container.level_paths:
#     level = container.get_image(path=path)
#     print(f"level {path}: shape {level.shape}, "
#           f"x = {level.pixel_size.x:.4f} um, z = {level.pixel_size.z:.4f} um")
# ```
#
# `x` and `y` double each step; `z` normally does not, because the pyramid is usually
# built only in the imaging plane. That asymmetry is why `PixelSize` keeps the three
# axes separate instead of storing one number.
# :::

# %% [markdown]
# ### 3. Crop one object
#
# Pick an object id from the label image, find its bounding box, and show just that
# object. *(Hint: chapter 3.3 has a much better way — try it the hard way first so you can
# appreciate the difference.)*

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# from scipy import ndimage
#
# ids = np.unique(mask); ids = ids[ids > 0]
# target_id = ids[len(ids) // 2]
# ys, xs = np.where(mask == target_id)
# crop = data[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
# plt.imshow(crop, cmap="gray"); plt.title(f"object {target_id}"); plt.axis("off")
# ```
#
# This works but you did the bookkeeping. In chapter 3.3 the masking ROI table hands you
# the box directly — `masking_table.get_label(target_id)` — already in micrometres and
# already valid at any pyramid level.
# :::

# %% [markdown]
# ---
#
# **Next:** [3.2 · Plates and wells](02_plates_wells.ipynb) — the same ideas, times 384.
