# Part 1 — OME-Zarr

A high-content screen is not a folder of images. A single 384-well plate can be hundreds
of gigabytes, in thousands of chunks, at several resolutions at once, with segmentation
masks and measurement tables stored alongside the pixels. **OME-Zarr** is the format the
field has settled on for exactly this, and this part is about reading it fluently.

You will use two libraries, and the order is deliberate:

**`ez-zarr`** is small and immediate. Point it at a plate, get a picture. It is the right
tool for *looking*, and you will reach for it constantly.

**`ngio`** is the fuller library. It knows about wells and acquisitions, reads and writes
labels and tables, and can pull one table out of all 384 wells in a single call. That
last capability is how the Part 3 dataset was built.

## Three sections

### [Understanding OME-Zarr](concept/intro.md)

What the format is, before any library hides it.

| | |
|---|---|
| [1](concept/01_what_is_omezarr.ipynb) | What OME-Zarr actually is — on disk, by hand |

### [Looking with ez-zarr](ezzarr/intro.md)

The small library, for when you just want to see the data.

| | |
|---|---|
| [2](ezzarr/02_ezzarr_quicklook.ipynb) | Plates, wells and a first look at the images |

### [Working with ngio](ngio/intro.md)

The fuller library — and the one that built the Part 3 dataset.

| | |
|---|---|
| [3.1](ngio/01_container.ipynb) | Containers, images, labels, pixel sizes, ROIs |
| [3.2](ngio/02_plates_wells.ipynb) | Plates and wells: from 384 wells to one table |
| [3.3](ngio/03_labels_tables.ipynb) | Labels, ROI tables and feature tables |

And a [cheat sheet](cheatsheet.md) covering both libraries.

```{note}
Runs on Euler, in the **Python (MCS 2026)** kernel. Set `DATA_ROOT` in
`src/mcs2026/config.py` first — see [The data](../setup/data.md).
```
