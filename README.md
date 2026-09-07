# Multicellular Systems 2026 — Hands-On Image Analysis

Course material for the hands-on image-analysis sessions of **Multicellular Systems**
(ETH Zürich, D-BSSE / Liberali Lab).

**Website: <https://maaraujo-nv.github.io/Multicellular-Systems-2026/>** — start there.
This repository is the source; the site is the readable version.

## What the course covers

| Part | Topic | Where you run it |
|------|-------|------------------|
| **1** | OME-Zarr: reading high-content screening data with `ez-zarr` and `ngio` | Euler |
| **2** | Viewing OME-Zarr in Napari | **your own laptop** |
| **3** | Multi-condition single-cell analysis: signaling, cell mechanics, metabolism, organelles | Euler |

Part 3 works on a 4i multiplexed immunofluorescence screen of human naive embryonic stem
cells — one 384-well plate, 18 perturbations × 4 timepoints, ~734,000 cells profiled over
4,464 features.

## Getting started

```bash
git clone https://github.com/Maaraujo-nv/Multicellular-Systems-2026.git
cd Multicellular-Systems-2026
```

Then follow **Setup** on the website, in order:

1. **Euler and JupyterHub** — getting an account and a running notebook server
2. **The Python environment** — modules, virtual environment, and the `mcs2026` kernel
3. **Napari** — laptop-only, for Part 2
4. **The data** — where the plate and the feature table live, and how to point at them

Short version, on Euler:

```bash
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda eth_proxy
python -m venv $HOME/venvs/mcs2026
source $HOME/venvs/mcs2026/bin/activate
pip install -r environment/requirements.txt
python -m ipykernel install --user --name mcs2026 --display-name "Python (MCS 2026)"
```

Then copy `src/mcs2026/config.example.py` to `src/mcs2026/config.py` and set `DATA_ROOT`
to the path you were given. `config.py` is gitignored, so your paths stay yours.

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
./tools/build_notebooks.sh part3_analysis/03_signaling   # just one
python tools/execute_notebooks.py book/part3_analysis    # run against real data
pytest tests/                                   # 19 checks on layout + decoding
```

Build the website locally:

```bash
pip install -r environment/requirements-book.txt
jupyter-book build book/
```

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
