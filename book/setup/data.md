# The data

Nothing is downloaded. Both datasets already sit on Euler in a read-only directory, and
you point the notebooks at them.

Data can be found in the project folder.

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
