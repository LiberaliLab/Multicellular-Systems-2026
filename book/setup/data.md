# The data

Nothing is downloaded. Both datasets already sit on Euler in a read-only directory, and
you point the notebooks at them.

## What there is

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

**The plate layout workbook** is small enough to live in the repository, at
`metadata/L_ayout_384_Haralick_Thresholds.xlsx`. You already have it. It is the only
place that records which antibody was in which channel in which round, which makes it
the decoder for the entire feature table.

## Point the notebooks at it

Copy the example config and edit one line:

```bash
cp src/mcs2026/config.example.py src/mcs2026/config.py
```

Open `src/mcs2026/config.py` and set `DATA_ROOT` to the path you were given in the first
session:

```python
DATA_ROOT = Path("/cluster/work/.../mcs2026")   # <- the path from the first session
```

That is the only path you ever have to set. `config.py` is listed in `.gitignore`, so
your local paths never end up in a commit — which is the point: a repository full of
other people's absolute paths is a repository nobody else can run.

```{note}
`config.example.py` **is** committed and `config.py` is **not**. If you clone the repo
fresh, you will not have a `config.py` until you make one.
```

## Check it worked

In a notebook running the **Python (MCS 2026)** kernel:

```python
from mcs2026.config import DATA_ROOT, H5AD_FULL, PLATE_PATH
print(DATA_ROOT.exists(), PLATE_PATH.exists(), H5AD_FULL.exists())
```

Three `True`s and you are ready.

## Do not copy the data

The feature table is 13 GB and your `$HOME` quota is around 16 GB. Read it where it is.
Part 3 chapter 01 writes a much smaller version, and *that* one you keep.
