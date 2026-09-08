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

## Chapters

| | |
|---|---|
| [1](01_what_is_omezarr.ipynb) | What OME-Zarr actually is — on disk, by hand |
| [2](02_ezzarr_quicklook.ipynb) | `ez-zarr`: plates, wells and a first look at the images |
| [3.1](03_ngio/01_container.ipynb) | `ngio`: containers, images, labels, pixel sizes, ROIs |
| [3.2](03_ngio/02_plates_wells.ipynb) | Plates and wells: from 384 wells to one table |
| [3.3](03_ngio/03_labels_tables.ipynb) | Labels, ROI tables and feature tables |
| [cheat](cheatsheet.md) | Cheat sheet |

```{note}
Runs on Euler, in the **Python (MCS 2026)** kernel. Set `DATA_ROOT` in
`src/mcs2026/config.py` first — see [The data](../setup/data.md).
```
