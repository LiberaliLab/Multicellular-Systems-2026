# 02 · Opening an OME-Zarr

You can read an OME-Zarr with `ez-zarr` and `ngio`. Now open one in the viewer.

```{important}
Laptop, not Euler — see [Napari setup](../setup/napari_local.md). You will need a copy of
one well or a small plate locally; your instructor gives you the path in the first
session.
```

## Opening it

An OME-Zarr is a **directory**, not a file — this is the thing from
[Part 1 chapter 01](../part1_omezarr/01_what_is_omezarr.ipynb) that trips people up here.

1. Drag the `.zarr` **folder** onto the napari window.
2. Napari asks which reader to use. Choose **napari-ome-zarr**.

Or from a terminal:

```bash
napari --plugin napari-ome-zarr /path/to/plate.zarr
```

If no dialogue appears and you get an error about an unreadable file, you have probably
dragged something *inside* the `.zarr` rather than the directory itself.

## What you get

The plugin reads the metadata you looked at by hand in Part 1, so several things are set
up for you:

- **one layer per channel**, named from the OME metadata rather than `Image [0]`
- **the multiscale pyramid**, so zooming out loads a coarse level and zooming in loads a
  fine one — you are never waiting for the full-resolution array
- **the pixel size**, so the scale bar is in micrometres
- **the display windows** stored in the file, as starting contrast limits

Watch the pyramid working: zoom out fully, then in. Napari fetches only the chunks and
the level it needs. This is the payoff for the format's complexity.

## Channels

Each channel is its own layer, so set each one separately:

1. select the channel in the layer list
2. set its **colormap**
3. set its **contrast limits**
4. set **blending** to `additive` on every layer above the bottom one

Now toggle each layer's eye icon in turn. Which structures are in which channel? Look up
the round each marker was imaged in — [Part 3 chapter 00](../part3_analysis/00_experiment_and_layout.ipynb)
built exactly that table.

## Segmentations

The labels live under `labels/` inside the image directory. The plugin usually offers
them alongside the channels; if not, drag the specific label directory in separately.

A segmentation **must** load as a **Labels** layer, not an Image layer. If it appears as
a grey blur, right-click the layer and choose **Convert to Labels**.

With a labels layer selected:

- **contour** (in layer controls) draws outlines instead of filled regions — much better
  for judging whether a boundary follows the cell
- **opacity** blends the filled version over the image
- hovering shows the object id in the status bar
- the **eye** toggles it, which is the fastest way to compare mask against image

### Judging a segmentation

This is the main reason to open the viewer at all. Look for:

- **merged objects** — one mask covering two cells that touch
- **split objects** — one cell broken into pieces
- **missed objects** — dim cells with no mask
- **boundaries that are too tight or too loose**, which biases every intensity
  measurement made inside them

```{tip}
Note the id of anything odd. In
[Part 1 chapter 05](../part1_omezarr/05_labels_rois_tables.ipynb) you can go straight from
an id to its measurements with `masking_table.get_label(id)`, and decide whether the
feature table's outliers are biology or segmentation.
```

## Finding a well in a plate

Drag a whole plate in and the plugin shows the wells it finds. For a 384-well plate this
is slow and unwieldy — the viewer is not a plate browser.

In practice: **open one well at a time.** Use `ngio` to decide which well is interesting
(Part 1 chapter 04 aggregates a table across the plate in one call), then open that well
here to look at it. Analysis narrows down; the viewer inspects.

```{note}
There is a plugin for browsing plates, **napari-ome-zarr-navigator**. It pins `ngio<0.6`
and would break the `ngio` 1.1.0 used everywhere else in this course, so if you want it,
put it in an environment of its own. Nothing here needs it.
```

## Saving what you see

**File → Save Screenshot** captures the canvas. For a figure, set the contrast
deliberately, turn the scale bar on, and say in the caption what the contrast limits
were — a screenshot without them is not a measurement.

---

**Next:** [Exercises](exercises.md).
