# 01 · A tour of napari

Everything so far produced numbers and static plots. Sometimes you need to *look*: scrub
through channels, turn a segmentation on and off, zoom into one cell and decide whether
the mask is right.

**Napari** is a multi-dimensional image viewer for exactly that.

```{important}
This chapter runs on **your own laptop**. See [Napari setup](../setup/napari_local.md)
for why, and for the install.
```

## Starting it

With your napari environment active:

```bash
napari
```

An empty grey window appears. Nothing is loaded yet.

## The window

Four regions, and it is worth knowing what each is for before you load anything.

| region | what it does |
|---|---|
| **canvas** (centre) | the image |
| **layer list** (upper left) | one entry per loaded thing, drawn bottom-to-top |
| **layer controls** (above the list) | settings for the *selected* layer |
| **sliders** (below the canvas) | one per dimension beyond the two on screen — z, time, sometimes channel |

The thing that catches everyone out: **layer controls apply to the selected layer only**.
If a slider seems to do nothing, you have almost certainly selected a different layer
than the one you are looking at.

## Layers

Napari is built around layers, and the type matters because each is rendered differently.

**Image layers** hold intensities and are drawn with a colormap and a contrast range.

**Labels layers** hold integer object ids. Each id gets its own colour, `0` is
transparent background, and hovering shows the id under the cursor. Loading a
segmentation as an image layer instead of a labels layer is a common mistake — you get a
grey blur, because the ids are being treated as brightness.

**Points** and **Shapes** layers hold coordinates rather than pixels — useful for marking
positions or drawing regions.

Layers stack. Drag them in the list to reorder; the eye icon toggles visibility; the
opacity slider blends them.

## Contrast limits

The single most important control, and the one most likely to mislead you.

Microscopy images are usually 16-bit — values from 0 to 65,535 — while your screen shows
256 grey levels. The **contrast limits** decide which range of values maps to that
visible range. Everything below the lower limit is black; everything above the upper is
white.

- Drag the two handles of the **contrast limits** slider to set the range.
- Click **auto-contrast** for a quick reasonable guess.
- Right-click the slider to type exact numbers.

```{warning}
Changing contrast changes **what you see**, never what the data is. Two channels that
look equally bright may differ tenfold. Never compare brightness across channels or
across images by eye — that is what the analysis in Part 3 is for. Use the viewer to
judge *shape*, *position* and *segmentation quality*, and the numbers to judge intensity.
```

## Colormaps and blending

For a single channel, `gray` is honest and `magma` shows faint structure better.

For several channels at once, give each a colour (`blue`, `green`, `red`, `magenta`) and
set the **blending** of the upper layers to `additive`. Without additive blending the top
layer simply hides the ones underneath.

```{tip}
Green-and-magenta is a better two-colour pair than green-and-red: it stays readable for
the majority of people with red-green colour blindness, and overlap comes out white
either way.
```

## Navigating

| action | how |
|---|---|
| pan | click and drag |
| zoom | scroll |
| reset view | the home button, bottom left |
| move through z or time | the sliders under the canvas |
| 2D / 3D | the cube button, bottom left |

3D mode renders volumetrically. It is striking and it is slow — switch back to 2D for
anything you are inspecting carefully.

## The scale bar

By default the scale bar counts *pixels*, which is nearly useless. If the layer knows its
pixel size, the bar can show micrometres:

```python
viewer.scale_bar.visible = True
viewer.scale_bar.unit = "um"
```

Or through the menu: **View → Scale Bar → Scale Bar Visible**.

When you open an OME-Zarr with the plugin in the [next chapter](02_omezarr_in_napari.md),
the pixel size comes along with the file, so the bar is correct automatically. That is one
of the practical arguments for the format.

## The console

The button at the bottom left opens a Python console **inside** the viewer, with the
viewer available as `viewer`:

```python
viewer.layers
viewer.layers[0].data.shape
viewer.layers[0].contrast_limits = [100, 2000]
```

Useful when a setting is easier to type than to find.

---

**Next:** [02 · Opening an OME-Zarr](02_omezarr_in_napari.md).
