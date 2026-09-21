# Multicellular Systems 2026 — Hands-On Image Analysis

Course material for the hands-on image-analysis sessions of **Multicellular Systems**
(ETH Zürich, D-BSSE / Liberali Lab).

**Website: <https://liberalilab.github.io/Multicellular-Systems-2026/>** — start there.
This repository is the source; the site is the readable version.

## What the course covers

| Part | Topic | Where you run it |
|------|-------|------------------|
| **1** | OME-Zarr: reading high-content screening data with `ez-zarr` and `ngio` | Euler |
| **2** | From images to numbers: what a feature is, and how it is measured | — (no code) |
| **3** | Multi-condition single-cell analysis: signaling, cell mechanics, metabolism, organelles | Euler |
| *optional* | Viewing OME-Zarr in Napari | **your own laptop** |

Part 3 works on a 4i multiplexed immunofluorescence screen of human naive embryonic stem
cells — one 384-well plate, 18 perturbations × 4 timepoints, ~734,000 cells profiled over
4,464 features.

## Getting started

Is your choice if to clone the git repository and have it locally as a guide or if to copy and paste directly from the website.

```bash
git clone https://github.com/LiberaliLab/Multicellular-Systems-2026.git
cd Multicellular-Systems-2026
```


```{note}
What is written above is not mandatory clone, if you wanna proceed with cloning you will have to also take that into account during installation. 
And you will have to read through the git repo and website, as many questions are solved there.
```


Then follow **Setup** on the website, in order:

1. **Euler and JupyterHub** — getting an account and a running notebook server
2. **The Python environment** — modules, virtual environment, and the `mcs2026` kernel
3. **The data** — where the plate and the feature table live

Napari needs a separate, laptop-only install. Its setup page lives with the optional
napari part rather than in Setup, because nothing else depends on it.

Short version, on Euler:

```bash
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda eth_proxy
python -m venv $HOME/venvs/mcs2026
source $HOME/venvs/mcs2026/bin/activate
pip install -r environment/requirements.txt
python -m ipykernel install --user --name mcs2026 --display-name "Python (MCS 2026)"
```

There is no config file to create. Every chapter states the path to the data in its own
first cell, so what you read at the top of a notebook is exactly what it opens:

```python
PLATE_PATH = Path("/cluster/project/mcsliberali/zarr_files/dummy.zarr")   # Part 1
DATA       = Path("/cluster/project/mcsliberali/data_mcs_2026")           # Part 3
```

If your copy of the data is elsewhere, change that one line in the chapter you are
working on.

## Repository layout

```
book/          course chapters (the website is built from here)
book/_src/     jupytext sources for the notebooks -- edit these, not the .ipynb
src/mcs2026/   small helper package used by the notebooks
metadata/      plate layout and marker tables, generated from the layout workbook
environment/   requirements files and the Euler JupyterHub config
tests/         regression tests for the layout parser and the feature decoder
tools/         build the notebooks from _src, and execute them against real data
```

## Working on the course material

Notebooks are authored as [jupytext](https://jupytext.readthedocs.io) `.py` files in
`book/_src/` and built into the `.ipynb` the book ships. This keeps diffs readable.

```bash
./tools/build_notebooks.sh                      # rebuild all, keeping stored outputs
./tools/build_notebooks.sh part3_analysis/2_controls/06_pca       # just one
python tools/execute_notebooks.py book/part3_analysis    # run against real data
pytest tests/                                   # 56 checks on layout, decoding + helpers
```

### Building the website locally

Optional — GitHub Actions builds and publishes the site on every push to `main`. Do this
only when you want to preview a change before pushing it.

Use a **separate** environment from the course one: students never build the book, and
this way a documentation dependency can never disturb the analysis environment.

```bash
micromamba create -n mcs2026-book python=3.11 -y
micromamba activate mcs2026-book
pip install -r environment/requirements-book.txt
```

Python 3.11 matches the version the GitHub runner uses, so a build that works here works
in CI. If you use `venv` instead of micromamba, `python -m venv ~/venvs/mcs2026-book`
does the same job — but do not install into the system Python.

```bash
jupyter-book build book/
python -m http.server 8000 --directory book/_build/html
```

Then open <http://localhost:8000>. Serve it rather than opening the HTML directly, or
search and some links will not work.

To rebuild from scratch, with broken cross-references treated as errors (what CI does):

```bash
rm -rf book/_build && jupyter-book build book/ --warningiserror
```

### Publishing

The site is built and deployed by `.github/workflows/book.yml` on every push to `main`.
For this to work, the repository's **Settings → Pages → Source** must be set to
**GitHub Actions** — not "Deploy from a branch", which makes GitHub render `README.md`
with Jekyll and ignore the book entirely.

## Authors

Silvia Barbiero and Manuel I. Araujo Novoa

## Acknowledgements

Silvia Barbiero and Simon Suppinger, whose 4i multiplexing screen is the dataset used in
Part 3. This course builds on
[ETH_MulticellularCourse_LiberaliLab_HandsOn](https://github.com/LiberaliLab/ETH_MulticellularCourse_LiberaliLab_HandsOn)
(2025 edition).

`ngio` is developed by the [BioVisionCenter](https://github.com/BioVisionCenter/ngio) (UZH);
`ez-zarr` by the Friedrich Miescher Institute. The data was processed with
[Fractal](https://fractal-analytics-platform.github.io/).

## License

MIT — see [LICENSE](LICENSE).
