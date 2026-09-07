# Further reading

## OME-Zarr and NGFF

- **[The OME-NGFF specification](https://ngff.openmicroscopy.org/latest/)** — the actual
  spec. Shorter and more readable than you expect; worth skimming once.
- **[Moore et al. 2021, *Nature Methods*](https://www.nature.com/articles/s41592-021-01326-w)**
  — OME-NGFF, and why bioimaging needed a cloud-friendly format.
- **[`ngio` documentation](https://biovisioncenter.github.io/ngio/)** — API reference and
  tutorials. This course uses **1.1.0**; anything written for 0.5.x will differ.
- **[`ngio` workshop](https://biovisioncenter.github.io/ngio-workshop/)** — a good
  complement, though it targets `ngio` 0.5.x, so a few calls have moved.
- **[`ez-zarr` documentation](https://fmicompbio.github.io/ez_zarr/)**

## Fractal

- **[Fractal](https://fractal-analytics-platform.github.io/)** — the pipeline framework
  that converted the raw microscope output into the OME-Zarr you read in Part 1.

## Napari

- **[napari tutorials](https://napari.org/stable/tutorials/index.html)**
- **[napari-ome-zarr](https://github.com/ome/napari-ome-zarr)**

## Multiplexed imaging

- **[Gut, Herrmann & Pelkmans 2018, *Science*](https://www.science.org/doi/10.1126/science.aar7042)**
  — iterative indirect immunofluorescence imaging (4i), the method behind the Part 3 data.
- **Haralick, Shanmugam & Dinstein 1973** — the texture features that make up 75% of the
  feature table. Worth reading once, to know what those columns actually measure.

## Single-cell analysis

- **[scanpy](https://scanpy.readthedocs.io)** and
  **[AnnData](https://anndata.readthedocs.io)** — the analysis stack used in Part 3.
- **[Current best practices in single-cell analysis](https://www.sc-best-practices.org/)**
  — written for transcriptomics, but the chapters on normalisation, batch effects and
  statistics apply directly to image-derived features.

## Statistics for imaging screens

- **[Lord, Velle, Mullins & Fritz-Laylin 2020, *J Cell Biol*](https://rupress.org/jcb/article/219/6/e202001064/151717)**
  — "SuperPlots: communicating reproducibility and variability in cell biology". Short,
  and directly about the replicate-unit problem this course keeps returning to.

## Image ANalysis Basics
- [Methods in Cell Analysis and Laboratory Automation: Image Analysis](https://m-albert.github.io/scu_lab_course_ia)
by Marvin Albert and Andreas Cuny (Single Cell Unit, D-BSSE, ETH Zürich).
