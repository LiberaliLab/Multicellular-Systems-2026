# 2 · How, and with what

Two steps, always in this order.

## Step 1 — segmentation

Turn pixels into objects. The output is a **label image**: the same size as the picture,
`0` where there is background and an integer where there is a cell.

- **Cellpose** and **StarDist** are the two deep-learning tools most labs reach for now.
  They handle nuclei and whole cells out of the box and are usually good enough without
  training anything.
- **Thresholding plus watershed** is the classical route. It is fast, it needs no GPU, and
  it is easier to explain when it goes wrong.

This step decides everything after it. A mask that merges two nuclei gives you one cell with
twice the area and a diluted mean, and no later analysis can undo that or even notice.

## Step 2 — measurement

Now the label image and the intensity image go in together, and a table comes out.

**`regionprops`** — from scikit-image — is the workhorse. You give it a label image, an
intensity image and a list of property names; it gives you one row per object.
[Chapter 3.3](../part1_omezarr/ngio/03_labels_tables.ipynb) of Part 1 does exactly this on
one well, in about five lines. It is worth rereading now that you know what the columns are.

**CellProfiler** is the same ideas without code. You assemble a pipeline of modules in a
window, point it at a folder of images and it writes a CSV. It is very widely used, it is
reproducible as long as you keep the pipeline file, and its measurements have different
names but the same meanings.

**Fractal** is what produced *this* dataset. It runs those steps across a whole plate stored
as OME-Zarr and writes the result back into the container as a table — which is why Part 1
could read it with `ngio` without converting anything first.

## What you end up with

One row per cell, one column per measure-channel-round. What the cell *is* — which well,
which condition, which timepoint — is not in there. That comes from the plate layout and
gets joined on afterwards, which is the first thing
[Part 3](../part3_analysis/1_preparation/00_what_you_are_given.ipynb) does.

:::{important}
**What to measure is a decision, not a default.** Every tool above will happily hand you
thousands of columns. Many will be near-copies of each other, and a few will be wrong in
ways that only surface much later. Part 3 opens by taking this table apart for exactly that
reason — and it ends up keeping 2,587 of the 4,464 columns, because nearly half the
table turns out to be repetition.
:::
