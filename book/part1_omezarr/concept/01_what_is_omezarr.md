# 1 · What OME-Zarr is

In this chapter you will:

- look at an OME-Zarr **as directories on disk**
- read the metadata that makes it an image rather than a pile of files
- understand chunks, and why a 100 GB image opens instantly
- understand the multiscale pyramid, and how to choose a level
- find where labels and tables live

:::{important}
**Nothing on this page runs, and nothing on it is meant to.** The code is here so you can
see *how* each thing is read, not so you can execute it — every snippet is an illustration.

The chapters after this one are where you open the data yourself.
:::

## The problem it solves

A high-content screen produces a lot of pixels. One 384-well plate, imaged in several
channels over several rounds, runs to hundreds of gigabytes. Two things then become
awkward with a conventional image file:

**You cannot open part of it.** A TIFF is a single stream. To read the top-left corner
of one well you must, in the general case, read your way there.

**You cannot open it at low resolution.** Drawing a 10,000 × 10,000 pixel well on a
1,000-pixel-wide screen means reading 100 million pixels to display one million.

**OME-Zarr** solves both by storing the image as many small files instead of one large
one: a directory tree of **chunks**, at several **resolutions**, with **metadata** in
plain JSON. The format is a community specification, **OME-NGFF**, and the data you use
in this course was written by [Fractal](https://fractal-analytics-platform.github.io/).

```python
import json
from pathlib import Path

# The plate lives here on Euler. Change this line if your copy is elsewhere.
PLATE_PATH = Path("/cluster/project/mcsliberali/zarr_files/dummy.zarr")

print(PLATE_PATH)
print("exists:", PLATE_PATH.exists())
```

## An OME-Zarr is a directory

The first surprising thing: it is not a file. Everything below is ordinary filesystem
navigation — no imaging library involved.

```python
def show_tree(path: Path, prefix: str = "", depth: int = 0, max_depth: int = 2, max_entries: int = 6):
    """Print the first few entries of a directory tree."""
    if depth > max_depth:
        return
    entries = sorted(p for p in path.iterdir() if not p.name.startswith("."))
    for i, entry in enumerate(entries[:max_entries]):
        marker = "└── " if i == len(entries[:max_entries]) - 1 else "├── "
        suffix = "/" if entry.is_dir() else ""
        print(f"{prefix}{marker}{entry.name}{suffix}")
        if entry.is_dir():
            show_tree(entry, prefix + "    ", depth + 1, max_depth, max_entries)
    if len(entries) > max_entries:
        print(f"{prefix}    ... and {len(entries) - max_entries} more")

show_tree(PLATE_PATH)
```

## The plate hierarchy

For a high-content screen the tree follows the plate:

```text
plate.zarr/
├── .zattrs           <- plate metadata: which rows, columns, acquisitions
├── B/                <- row
│   ├── 03/           <- column
│   │   └── 0/        <- the image (there can be several per well)
│   │       ├── .zattrs
│   │       ├── 0/    <- pyramid level 0, full resolution
│   │       ├── 1/    <- level 1, half the size in x and y
│   │       ├── ...
│   │       ├── labels/    <- segmentation masks
│   │       └── tables/    <- ROIs and measurements
```

So the path `B/03/0` means **row B, column 3, image 0** — the addressing you will use
for the rest of Part 1.

```{image} ../../images/zarr_hierarchy_light.svg
:class: only-light
:alt: A 384-well plate map with well B/03 highlighted, next to the matching directory tree plate.zarr/B/03/0 containing pyramid levels, labels and tables.
```

```{image} ../../images/zarr_hierarchy_dark.svg
:class: only-dark
:alt: A 384-well plate map with well B/03 highlighted, next to the matching directory tree plate.zarr/B/03/0 containing pyramid levels, labels and tables.
```

```python
plate_metadata = json.loads((PLATE_PATH / ".zattrs").read_text())
plate_meta = plate_metadata["plate"]
print("rows:       ", [r["name"] for r in plate_meta["rows"]][:8], "...")
print("columns:    ", [c["name"] for c in plate_meta["columns"]][:8], "...")
print("wells:      ", len(plate_meta["wells"]))
print("acquisitions:", plate_meta.get("acquisitions", "none declared"))
```

## Inside one image

Pick a well and look at its metadata. Two keys matter: `multiscales` describes the
pyramid, and `omero` describes how the channels should be displayed.

```python
image_path = PLATE_PATH / plate_meta["wells"][0]["path"] / "0"
print("image:", image_path.relative_to(PLATE_PATH.parent))

image_metadata = json.loads((image_path / ".zattrs").read_text())
print("keys:", list(image_metadata))
```

```python
multiscale = image_metadata["multiscales"][0]
print("axes:", [axis["name"] for axis in multiscale["axes"]])
for dataset in multiscale["datasets"]:
    scale = dataset["coordinateTransformations"][0]["scale"]
    print(f"  level {dataset['path']}: scale {[round(s, 3) for s in scale]}")
```

Each level is a separate array, and the `scale` says how big one pixel is in **physical
units** at that level. Level 0 is full resolution; each step usually halves x and y.

This is what makes the format usable: to draw a thumbnail you read the smallest level,
not the largest one downsampled.

```{image} ../../images/zarr_pyramid_light.svg
:class: only-light
:alt: Four nested squares showing pyramid levels 0 to 3, each labelled with its pixel dimensions and micrometres per pixel, halving at every step.
```

```{image} ../../images/zarr_pyramid_dark.svg
:class: only-dark
:alt: Four nested squares showing pyramid levels 0 to 3, each labelled with its pixel dimensions and micrometres per pixel, halving at every step.
```

```python
channels = image_metadata.get("omero", {}).get("channels", [])
for channel in channels:
    window = channel.get("window", {})
    print(f"  {channel.get('label', '?'):16s} colour #{channel.get('color', '------')}  "
          f"display {window.get('start', '?')}-{window.get('end', '?')}")
```

## Chunks

Now look inside one pyramid level. The array is not one file — it is many.

```python
level_zero = image_path / "0"
array_metadata_path = level_zero / "zarr.json" if (level_zero / "zarr.json").exists() else level_zero / ".zarray"
array_metadata = json.loads(array_metadata_path.read_text())

shape = array_metadata.get("shape")
chunks = (array_metadata.get("chunks")
          or array_metadata.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape"))
print("full array shape:", shape)
print("chunk shape:     ", chunks)
print("data type:       ", array_metadata.get("dtype") or array_metadata.get("data_type"))
```

```python
chunk_files = [p for p in level_zero.rglob("*") if p.is_file() and not p.name.startswith((".", "zarr.json"))]
print(f"{len(chunk_files)} chunk files at level 0")
if chunk_files:
    sizes = sorted(p.stat().st_size for p in chunk_files)
    print(f"median chunk on disk: {sizes[len(sizes) // 2] / 1e3:.0f} kB")
```

**This is the whole trick.** To read a region you compute which chunks it touches and
read only those. Reading a 512 × 512 window from a 10,000 × 10,000 image touches a
handful of small files, not the whole array — whether the store is on this disk, on a
network filesystem, or in an S3 bucket.


```{image} ../../images/zarr_chunks_light.svg
:class: only-light
:alt: A pyramid level drawn as a grid of chunk files, with a dashed read window covering four of the forty chunks.
```

```{image} ../../images/zarr_chunks_dark.svg
:class: only-dark
:alt: A pyramid level drawn as a grid of chunk files, with a dashed read window covering four of the forty chunks.
```

Chunk shape is therefore a real design decision. Chunks that are too small mean a lot
of file overhead; too large and every small read pulls in data you do not want.

## Labels and tables

Two things sit alongside the pixels. Neither is part of the core image.

```python
labels_path = image_path / "labels"
if labels_path.exists():
    print("labels:", json.loads((labels_path / ".zattrs").read_text()).get("labels", []))

tables_path = image_path / "tables"
if tables_path.exists():
    print("tables:", json.loads((tables_path / ".zattrs").read_text()).get("tables", []))
```

**Labels** are segmentation masks stored as their own multiscale image, where the pixel
value is the object id. They share the coordinate system with the image, so label 42 is
at the same physical position in both.

**Tables** hold regions of interest and per-object measurements. They are not in the
OME-NGFF core specification — they follow
[ngio's table specification](https://biovisioncenter.github.io/ngio/stable/table_specs/overview/),
which originated in Fractal. This is where the feature table behind Part 3 comes from.

:::{note}
You have now seen everything the libraries in the next four chapters are doing. They
read these same JSON files and these same chunks — they just save you from writing the
path arithmetic yourself, and they handle the parts of the specification that are
fiddlier than they look.
:::

---

**Next:** [2 · A quick look with ez-zarr](../ezzarr/02_ezzarr_quicklook.ipynb).
