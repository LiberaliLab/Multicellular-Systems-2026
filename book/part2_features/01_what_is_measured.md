# 1 · What gets measured

A measurement needs two images:

- the **intensity image** — how bright each pixel is, in one channel;
- the **label image** — which object each pixel belongs to. `0` is background, `7` is cell 7.

The label image is what a segmentation produces, and it is the thing that makes "the mean
intensity **of a cell**" a sensible phrase at all. Without it you have a picture; with it
you have objects.

With those two images, four kinds of question can be asked.

## How much — intensity

Take the pixels of one cell in one channel and summarise them.

| measure | what it is for |
|---|---|
| `mean_intensity` | the usual one — concentration, and it does not grow with cell size |
| `sum_intensity` | the total amount in the cell, which does grow with size |
| `max_intensity` | the brightest pixel — a small bright speck survives here and is diluted in the mean |
| `min_intensity` | usually just the background inside the mask |
| `std_intensity` | how uneven the signal is |

Five numbers per cell per channel. Here that is **290 columns**.

## What shape — morphology

These use only the label image, so there is one set per cell however many channels you
stained.

| measure | in words |
|---|---|
| `area` | how many pixels |
| `perimeter` | how long the outline is |
| `eccentricity` | `0` is a circle, `1` is a line |
| `solidity` | area ÷ area of its convex hull — `1` is smooth, less is ragged |
| `extent` | area ÷ area of its bounding box |
| `centroid-0`, `centroid-1` | where it is |

**Those are not our names.** They are the output names of `regionprops`, the standard
scikit-image function, so every one of them can be looked up in its documentation. The
dataset has **396 morphology columns** — 22 measures, written out once per imaging round.

## How it is arranged — texture

Two cells can hold the same amount of a protein and look nothing alike: one smooth, one in
bright puncta. A mean cannot tell them apart. Texture can.

This dataset uses **Haralick** features (52 of them) and **Laws texture energies** (6). Both
are long-established, and both are honestly hard to interpret one at a time — a number
called `Haralick-Mean-contrast-2` means something precise, but not something you can
picture.

Texture is **3,364 of the 4,464 columns**, three-quarters of the table. That is why Part 3
keeps it in the archive rather than in the object you analyse.

## What is around it — neighbourhood

Not about the cell at all, but about its surroundings: how many neighbours within 100
pixels, the mean distance to the 10 nearest, the local density. **414 columns.** For a
tissue question this is often the interesting part.

## Where 4,464 columns come from

Nothing here is clever; it is multiplication.

- **Intensity and texture** are measured per channel per round. This screen has 58
  channel-rounds, so 5 × 58 = 290 intensity columns and 58 × 58 = 3,364 texture columns.
- **Shape and neighbourhood** come from the mask, so they have no channel — but they were
  still written out once for each of the 18 rounds. 22 × 18 = 396 and 23 × 18 = 414.

Those four numbers add up to 4,464. Writing shape out 18 times is pure repetition, and
[Part 3](../part3_analysis/1_preparation/01_columns_to_markers.ipynb) proves it is
repetition before throwing 17 copies away.
