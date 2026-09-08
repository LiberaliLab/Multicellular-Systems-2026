# Cheat sheet — ez-zarr and ngio

Quick reference for Part 1. Versions: **`ngio` 1.1.0**, **`ez-zarr` 0.4.2**.

## ez-zarr — for looking

```python
from ez_zarr import ome_zarr

plate = ome_zarr.import_plate(str(plate_path))   # note: needs a str, not a Path
plate.plot()                                     # the whole plate
plate.get_layout()                               # which wells exist

image = plate["B02"]                             # one well
image                                            # prints channels, levels, tables
array = image.get_array_by_coordinate(pyramid_level=1, as_NumPy=True)
image.get_scale(pyramid_level=1)

image.plot(
    pyramid_level=0, channels=[1], channel_colors=["white"],
    channel_ranges=[[100, 1000]],
    scalebar_micrometer=150, scalebar_color="yellow", scalebar_label=True,
)

image.get_table(table_name="nuclei_ROI_table")   # ROI tables only
```

## ngio — for working

### Open

```python
import ngio

container = ngio.open_ome_zarr_container(image_path)   # one image
plate     = ngio.open_ome_zarr_plate(plate_path)       # a whole HCS plate
well      = ngio.open_ome_zarr_well(well_path)         # one well
```

### Inspect

```python
container.levels          container.level_paths
container.channel_labels  container.num_channels
container.is_3d           container.is_time_series
container.list_labels()   container.list_tables()
```

### Read pixels

```python
image = container.get_image()                      # full resolution
image = container.get_image(path="1")              # a pyramid level
image = container.get_image(pixel_size=ps, strict=False)

image.shape   image.axes   image.dtype   image.pixel_size

data = image.get_as_numpy(channel_selection="DAPI", axes_order=["y", "x"])
lazy = image.get_as_dask()                         # for larger-than-memory
patch = image.get_roi_as_numpy(roi, channel_selection="DAPI")
```

### Labels

```python
label = container.get_label("nuclei")
mask  = label.get_as_numpy(axes_order=["y", "x"])

new = container.derive_label("my_segmentation", overwrite=True)
new.set_array(mask, axes_order=["y", "x"])
new.consolidate()                                  # rebuild the pyramid
```

### Tables

```python
container.list_tables()
container.list_tables(filter_types="feature_table")

roi     = container.get_roi_table("FOV_ROI_table")
masking = container.get_masking_roi_table("nuclei_ROI_table")
feature = container.get_feature_table("nuclei")

feature.dataframe        # pandas
feature.lazy_frame       # polars
feature.anndata          # AnnData -- all three, whatever the backend

from ngio.tables import FeatureTable
container.add_table("my_features",
                    FeatureTable(df, reference_label="nuclei"),
                    backend="parquet", overwrite=True)
```

### Plates and wells

```python
plate.rows       plate.columns
plate.wells_paths()
plate.images_paths()

well      = plate.get_well("B", 3)
container = plate.get_image("B", 3, "0")

# the one that matters: every well's table, stacked
table = plate.concatenate_image_tables("nuclei_features")
df = table.dataframe
```

### ROIs

```python
table = container.get_roi_table("FOV_ROI_table")
for roi in table.rois():
    print(roi.name)
    pixels = roi.to_pixel(pixel_size=image.pixel_size)
    xs = pixels.get("x")
    xs.start, xs.length

roi = masking.get_label(42)      # the ROI around object 42
roi.zoom(1.5)                    # pad it by 50%
```

## Gotchas

| | |
|---|---|
| `import_plate` needs a **`str`**, not a `Path` | `ome_zarr.import_plate(str(p))` |
| An OME-Zarr is a **directory** | copy with `rsync -a` or `scp -r`, never a plain `cp file` |
| `Roi` objects are **frozen** in ngio 1.1.0 | build a new one, do not mutate |
| ez-zarr cannot read feature tables | use `ngio.get_feature_table` |
| `consolidate()` is not automatic | call it after `set_array`, or coarse levels go stale |

## Moving from older ngio

Code written for `ngio` 0.5.x — including the BioVisionCenter workshop — differs:

| 0.5.x | 1.1.0 |
|---|---|
| `from ngio.experimental.iterators import ...` | `from ngio.iterators import ...` |
| `set_as_numpy_transform(array, slicing_ops, axes_ops)` | `on_set(array, ctx)` |
| `set_axes_units(...)` | `set_space_unit(...)` / `set_time_unit(...)` |
