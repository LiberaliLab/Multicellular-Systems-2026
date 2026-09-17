# Stage 1 — Preparing the data

You will not be handed a tidy table.

What arrives is one `AnnData` object of 733,556 cells × 4,464 features — and a `var` table
holding 4,464 column names and **no columns at all**. A feature is called
`cells_Intensity_mean_intensity_Texas Red_0`, which says the imaging channel and the round.
It does not say the antibody. Nothing in the file does.

```{image} ../../images/feature_table_anatomy_light.svg
:class: only-light
:alt: The AnnData object: a 733,556 by 4,464 matrix X, an obs table of 12 per-cell columns, and a var table holding 4,464 names and no columns at all.
```
```{image} ../../images/feature_table_anatomy_dark.svg
:class: only-dark
:alt: The AnnData object: a 733,556 by 4,464 matrix X, an obs table of 12 per-cell columns, and a var table holding 4,464 names and no columns at all.
```

The antibody is recorded in one place only: a separate Excel workbook, filled in by hand at
the bench. Making those two meet is what Stage 1 is for.

## The nineteen steps

Five notebooks, one continuous sequence. Nothing is assumed that an earlier step did not
establish.

### [00 · What you are given](00_what_you_are_given.ipynb)

| | | |
|---|---|---|
| **1** | The data you will analyse | what is in the object, and what is missing from it |
| **2** | The plate | 18 conditions, 4 timepoints, 224 wells |
| **3** | The staining sheet | which antibody, which channel, which round |
| **4** | Save the tidy metadata tables | the decoder, written out as CSV |

Runs on the workbook alone — you can do this chapter on a laptop, before the 13 GB file is
anywhere near you.

### [01 · From columns to markers](01_columns_to_markers.ipynb)

| | | |
|---|---|---|
| **5** | Open the table, and look at its structure | 13 GB, without loading it |
| **6** | Parse the column names | into family, statistic, channel, round |
| **7** | **Rename the features with their markers** | the join — the point of the whole stage |
| **8** | Tidy the cell metadata | conditions, timepoints, wells |
| **9** | Decide what to drop | and write down why |
| **10** | Save the slim table | 4,464 columns become 2,587 |

### [02 · Quality control](02_quality_control.ipynb)

| | | |
|---|---|---|
| **11** | Remove cells you cannot trust | border cells — *Part 3's memory peak* |
| **12** | Should any wells be dropped? | one should; three that look worse should not |
| **13** | Does plate position matter? | the controls answer it |
| **14** | Save the cleaned table | |

### [03 · Normalisation](03_normalisation.ipynb)

| | | |
|---|---|---|
| **15** | Are intensities comparable across rounds? | they are not, and this shows why |
| **16** | Is any condition an outlier? | one is, badly |
| **17** | Normalise to the controls | within timepoint, in control-SD units |
| **18** | Check a known answer | a drug whose effect you can predict |
| **19** | **Assemble the clean object** | one file, every slot filled |

### [04 · Subsetting and sketching](04_subsetting_and_sketching.ipynb)

Cutting it down two ways: to the markers and conditions your question needs, and to a
number of cells a neighbour graph can be built on — without throwing away the rare cells,
which is the part that takes thought.

## What you end up with

```{image} ../../images/clean_object_light.svg
:class: only-light
:alt: The clean AnnData: X holding normalised values, a raw layer, an annotated var table, a tidy obs table, and provenance in uns.
```
```{image} ../../images/clean_object_dark.svg
:class: only-dark
:alt: The clean AnnData: X holding normalised values, a raw layer, an annotated var table, a tidy obs table, and provenance in uns.
```

| file | one row per | what it is for |
|---|---|---|
| `mcs2026_clean.h5ad` | cell | the full, normalised, annotated dataset — **this is the one you open** |
| `mcs2026_sketch.h5ad` | cell | ~30,000 of them, covering the space — embeddings |
| `mcs2026_full.h5ad` | cell | the wide 2,587-column archive, for texture questions |

Every chapter after this one opens one of those with `sc.read_h5ad` and starts working — and
**nothing after this chapter imports the course package**. Everything a later chapter needs to
know about the data travels inside the object: `var` names the markers and their themes, `obs`
carries the well, condition, timepoint and the per-cell measurements, and `uns` records how the
numbers were made. A well-level table, when a statistical test needs one, is a `groupby` away.

`mcs2026_slim.h5ad` and `mcs2026_qc.h5ad` also exist. They are the intermediates this stage
passes between its own chapters, and once Stage 1 has run you can delete them.

```{important}
Ask for at least **32 GB** on JupyterHub. The peak is Step 11, not the step that touches
the 13 GB file — see [Euler and JupyterHub](../../setup/euler_jupyterhub.md).
```
