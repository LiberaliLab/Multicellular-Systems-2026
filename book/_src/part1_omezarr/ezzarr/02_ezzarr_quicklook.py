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
# # 2 · A quick look with ez-zarr
#
# In this notebook you will:
#
# - open a single image and a whole plate with `ez-zarr`
# - plot a plate overview and inspect its layout
# - pull one well out of a plate, two different ways
# - plot a well with `matplotlib` and with `ez-zarr`'s own plotting
# - read a table — and find the limitation that motivates `ngio`
#
# Chapter 01 walked the directory tree by hand. `ez-zarr` does that for you. It is small,
# it is quick, and it is the right tool when what you want is to **look** at the data.

# %%
# Loading the necessary packages
from pathlib import Path

import numpy                                        # array operations
import matplotlib.pyplot as plt                     # for finer-grained plotting

from ez_zarr import ome_zarr, plotting, utils       # simple interaction with zarr files

# The plate lives here on Euler. Change this line if your copy is elsewhere.
PLATE_PATH = Path("/cluster/project/mcsliberali/zarr_files/dummy.zarr")

# %% [markdown]
# ## Paths
#
# `PLATE_PATH` is the line you set at the top of this chapter — see [The data](../../setup/data.md). Everything
# else is built from it, so this is the only line you would change to point at a
# different plate.

# %%
image_path_prefix = "B/03/0"                        # row / column / image

zarr_path = PLATE_PATH                              # the plate
image_path = Path(zarr_path, image_path_prefix)     # one image inside it
image_path

# %% [markdown]
# ## Loading your first image
#
# A single well image, straight from its path.

# %%
imageA = ome_zarr.Image(image_path)

# %% [markdown]
# ## Loading your first plate
#
# Plates are made of wells. An OME-Zarr normally represents a single plate, and within it
# we have all the wells. They can be accessed individually, as above, but you can also
# load the whole plate at once.
#
# :::{note}
# This takes longer — loading may take a minute or two, because ez-zarr opens every
# image in the plate to build the list.
# :::

# %%
# returns the plate list -- a list of images.
# NOTE: the input has to be a string, not a Path.
plateL = ome_zarr.import_plate(str(zarr_path))

# %% [markdown]
# ## Dealing with plates
#
# With a plate in hand, the next step is to see the layout and pull out a well of
# interest.
#
# ### 1. Plotting

# %%
plateL.plot()

# %% [markdown]
# ### 2. Layout

# %%
plateL.get_layout()

# %% [markdown]
# ### 3. Extracting one well
#
# Either by index in the list, or explicitly by well name.

# %%
well_single = plateL[0]
# or
well_B02 = plateL["B02"]

# %% [markdown]
# ## Dealing with wells
#
# We have loaded a well directly and extracted one from a plate. Now we can explore what
# a single well object offers.
#
# ### 1. What a well object contains

# %%
well_B02

# %%
imageA

# %% [markdown]
# You will notice that depending on whether it was accessed from the plate or loaded
# directly from a path, some plate-localisation information is displayed differently.
#
# The repr is a good summary: number of channels and their names, how many pyramid
# levels, the scale factor between levels, the physical spacing of a voxel, and which
# segmentations and tables exist.

# %% [markdown]
# ### 2. Plotting
#
# For single wells there are two routes: through `matplotlib.pyplot`, or through
# `ez-zarr`'s own plotting.
#
# **The matplotlib route** gives you full control, and is worth doing once so you can see
# that the array is just an array.

# %%
arr = imageA.get_array_by_coordinate()
print("array shape:", arr.shape, "  (channel, z, y, x)")

with plt.style.context("dark_background"):
    fig = plt.figure(figsize=(4, 4))
    fig.set_dpi(150)
    # arr[2, 0] establishes the channel used for plotting, currently channel 2
    plt.imshow(arr[2, 0], cmap="gray", vmin=100, vmax=600)
    plt.title(imageA.name)
    plt.show()
    plt.close()

# %% [markdown]
# **The ez-zarr route** does the tedious parts — channel colours, display ranges, and a
# scale bar in real units — in one call.

# %%
imageA.plot(
    pyramid_level=0,
    channels=[1],
    channel_colors=["white"],
    channel_ranges=[[100, 1000]],
    title=imageA.name,
    scalebar_micrometer=150,
    scalebar_color="yellow",
    scalebar_position="topleft",
    scalebar_label=True,
    fig_width_inch=15,
    fig_height_inch=10,
    fig_dpi=200,
)

# %% [markdown]
# :::{tip}
# `pyramid_level=0` is full resolution. For a quick look at a whole well, a higher level
# is much faster and looks identical on screen — this is the pyramid from chapter 01
# doing its job. Try `pyramid_level=2`.
# :::

# %% [markdown]
# ### 3. Segmentations and tables
#
# The repr listed the segmentations and tables that live alongside the image. Ask for the
# names as lists, because a table is read **by name** — and the names differ from plate to
# plate.

# %%
labels = imageA.get_label_names()
tables = imageA.get_table_names()
print("segmentations:", labels)
print("tables:       ", tables)

# %%
# A table named after a segmentation holds one row per object. If this plate has
# none, fall back to the first table it lists.
table_name = next((t for t in tables
                   if any(name in t for name in labels) and ("ROI" in t or "masking" in t)),
                  tables[0])
print("reading table:", table_name)

# %%
df = imageA.get_table(table_name=table_name)
df

# %% [markdown]
# Each row is one segmented object, with its bounding box in micrometres — where it is
# and how big it is, in real units rather than pixels.
#
# :::{important}
# This package is very user friendly, but it has limitations — which you meet as soon as
# you try to load a **feature table**. Those table types are not in the format ez-zarr
# expects, so it cannot read them.
#
# That is one of the main reasons to also work with the next package, **`ngio`**. Feature
# tables are exactly where the Part 3 dataset comes from, so this is not a corner case:
# it is the thing you will need most.
# :::

# %% [markdown]
# ---
#
# ## Exercises
#
# ### 1. Colour the channels
#
# The plot above shows one channel in white. Show two or three channels at once, each in
# its own colour, with sensible display ranges. What are the channel names on this plate,
# and which index is which?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# for index, channel in enumerate(imageA.get_channels()):
#     print(index, channel["label"])
#
# imageA.plot(
#     pyramid_level=1,
#     channels=[0, 1, 2],
#     channel_colors=["blue", "green", "red"],
#     channel_ranges=[[100, 1000], [100, 1000], [100, 600]],
#     title="three channels",
#     scalebar_micrometer=150,
#     scalebar_label=True,
# )
# ```
#
# `get_channels()` lists them in order, so the first one is index 0. Getting the
# ranges right matters more than the colours: too wide and everything looks black, too
# narrow and everything saturates.
# :::

# %% [markdown]
# ### 2. How much faster is a higher pyramid level?
#
# Time `get_array_by_coordinate()` at level 0 and at level 2, and compare the array
# shapes. Was the picture on screen any worse?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# import time
# for level in [0, 2]:
#     start = time.time()
#     a = imageA.get_array_by_coordinate(pyramid_level=level, as_NumPy=True)
#     print(f"level {level}: shape {a.shape}, {time.time() - start:.2f} s")
# ```
#
# `as_NumPy=True` is what makes this a fair test: without it ez-zarr hands back a lazy
# array, nothing has been read yet, and both timings come out as zero.
#
# Level 2 is a quarter the width and a quarter the height, so a sixteenth of the pixels
# and roughly a sixteenth of the read. On a screen that is a few hundred pixels wide,
# nothing is lost — you cannot display detail you have already thrown away by zooming out.
# :::

# %% [markdown]
# ### 3. Segmentation masks on top of the image
#
# Overlay a segmentation on the image, and then plot only the region covered by a single
# mask.
#
# *(Read the arguments of `imageA.plot` before you reach for the array. Then look at how
# [chapter 3.3](../ngio/03_labels_tables.ipynb) gets the same object as data, with `ngio`.)*

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# print(imageA.get_label_names())
# label = imageA.get_label_names()[0]        # or any other name in that list
#
# # 1. the segmentation, drawn semi-transparently over the image
# imageA.plot(
#     label_name=label,
#     pyramid_level=1,
#     channels=[1],
#     channel_colors=["white"],
#     channel_ranges=[[100, 1000]],
#     scalebar_micrometer=150,
# )
#
# # 2. one object: read the label image, take one of its ids, let plot() crop to it
# lab = imageA.get_array_by_coordinate(label_name=label, as_NumPy=True)
# ids = numpy.unique(lab[lab > 0])
# value = int(ids[len(ids) // 2])            # one from the middle of the list
# print(f"{len(ids)} objects, plotting label value {value}")
#
# imageA.plot(
#     label_name=label,
#     label_value=value,
#     extend_pixels=20,
#     pyramid_level=0,
#     channels=[1],
#     channel_colors=["white"],
#     channel_ranges=[[100, 1000]],
# )
# ```
#
# Neither step touches a pixel coordinate. `label_name` overlays the mask; `label_value`
# finds that object's bounding box and crops to it; `extend_pixels` leaves a margin around
# it.
#
# Doing the crop by hand is where it gets fiddly. A table gives positions in micrometres,
# the array is indexed in pixels, the pixel size is different at every pyramid level — and
# `get_array_by_coordinate()` gives you the *lowest* resolution unless you ask for another,
# while `get_scale()` returns the channel axis too unless you pass
# `spatial_axes_only=True`. Get any one of those wrong and you have a crop of the wrong
# place, with no error to tell you.
#
# What ez-zarr gives you here is a *picture*. [Chapter 3.3](../ngio/03_labels_tables.ipynb)
# gets the same object as data — `image.get_roi_as_numpy(roi)` returns the pixels for one
# row of a table, ready to measure. That is the difference between the two libraries in a
# nutshell.
# :::

# %% [markdown]
# ---
#
# **Next:** [3.1 · Containers, images and labels](../ngio/01_container.ipynb).
