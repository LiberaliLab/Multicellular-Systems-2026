# Napari

Everything so far produced numbers and static plots. Sometimes you just need to *look* at
the data: scrub through channels, turn a segmentation on and off, zoom into one cell and
decide whether the mask is right.

**Napari** is an image viewer for exactly that. This part is **optional**, it is short,
and it has one goal:
**be able to open an OME-Zarr and look at it comfortably.**

```{important}
This part runs on **your own laptop**, not on Euler. Napari needs OpenGL 3.2+, and
forwarding graphics from a cluster caps OpenGL at 1.4 — it cannot work.
[Chapter 00](00_setup.md) is the install, and it is not part of the Euler setup you did
before the course.
```

## Chapters

| | |
|---|---|
| [00](00_setup.md) | **Install it first** — a separate environment, on your laptop |
| [01](01_napari_tour.md) | The viewer: layers, contrast, channels, scale bars |
| [02](02_omezarr_in_napari.md) | Opening an OME-Zarr: pyramids, channels, label overlays |
| [ex](exercises.md) | Exercises |

## What this part is not

There is no scripting of napari here, and no annotating masks and feeding them back into
the analysis. Both are worth learning; neither is the point of this course. If you want
them afterwards, the napari documentation is genuinely good.
