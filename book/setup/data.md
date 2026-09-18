# The data

Nothing is downloaded. Both datasets already sit on Euler, and every chapter states the
path to them in its own first cell — there is no config file and no environment variable.
What you read at the top of a notebook is exactly what it opens.

## Where it is

```text
/cluster/project/mcsliberali/
├── zarr_files/
│   └── dummy.zarr            <- the OME-Zarr plate, Part 1
└── data_mcs_2026/
    ├── 1_FE_pooled1.h5ad     <- the raw feature table, 13 GB
    ├── mcs2026_slim.h5ad     <- and the objects Stage 1 builds from it
    └── ...
```

So the two lines you will meet are:

```python
PLATE_PATH = Path("/cluster/project/mcsliberali/zarr_files/dummy.zarr")   # Part 1
DATA       = Path("/cluster/project/mcsliberali/data_mcs_2026")           # Part 3
```

If your copy of the data is somewhere else, change that one line in the chapter you are
working on. Nothing else reads it.

## Overview

**A high-content screening plate**, in OME-Zarr — used in Parts 1 and 2.

**A single-cell feature table**, as AnnData `.h5ad` — used in Part 3. One 384-well plate
of HNES1 human naive embryonic stem cells, 4i multiplexed immunofluorescence:

| | |
|---|---|
| Cells | 733,556 |
| Features | 4,464 |
| Conditions | 18 (16 perturbations + DMSO and PBS controls) |
| Timepoints | 36, 48, 60 and 84 hours |
| Used wells | 224 (rows B–O) |
| Imaging rounds | 18 |
| Size in memory | **13.1 GB** as dense float32 |


## Do not copy the data

The feature table is 13 GB and your `$HOME` quota is around 16 GB. Read it where it is.
Part 3 chapter 01 writes a much smaller version, and *that* one you keep.

Chapters 01 to 04 of Part 3 have already been run for you, and their outputs are in
`data_mcs_2026` alongside the raw table. You read those chapters to understand what they
did; you do not have to run them.
