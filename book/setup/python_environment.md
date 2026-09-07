# The Python environment

One virtual environment, one Jupyter kernel, used by Part 1 and Part 3.

Do this **once**, in a terminal on Euler ([how to get there](euler_jupyterhub.md)).

## 1. Load the software stack

```bash
module load stack/2024-05 gcc/13.2.0 python/3.11.6_cuda eth_proxy
```

Three of those four matter to you:

- `python/3.11.6_cuda` gives Python **3.11.6**. `ngio` requires 3.11 or newer.
- `eth_proxy` gives the node internet access, which `pip` needs. Without it, the
  install below hangs and then fails with a timeout.
- `stack/2024-05` selects the module collection the other two come from.

```{tip}
`module load` is not permanent — it lasts for the current shell. You re-run it every
time you open a new terminal. That is why the same line also goes into
`jupyterlabrc`.
```

## 2. Get the course repository

```bash
cd $HOME
git clone https://github.com/Maaraujo-nv/Multicellular-Systems-2026.git
cd Multicellular-Systems-2026
```

## 3. Create the virtual environment

A **virtual environment** is a private copy of Python with its own packages. It keeps
this course's package versions from colliding with anything else you install, and it
means you can delete the whole thing and start again without consequences.

```bash
python -m venv $HOME/venvs/mcs2026
source $HOME/venvs/mcs2026/bin/activate
```

Your prompt now starts with `(mcs2026)`. That is how you know the environment is active.

```bash
pip install --upgrade pip
pip install -r environment/requirements.txt
```

This takes a few minutes. It installs `ngio` 1.1.0 and `ez-zarr` 0.4.2 — the two OME-Zarr
libraries the course uses — along with `scanpy`, `pandas` and the rest.

```{note}
`ez-zarr` and `ngio` coexist happily: `ez-zarr` needs `zarr>=3.0`, `ngio` needs
`zarr>=3.1.6`, and the second satisfies the first. You do **not** need separate
environments for Part 1's two libraries.
```

## 4. Register the kernel

The environment exists, but JupyterLab does not know about it yet:

```bash
python -m ipykernel install --user --name mcs2026 --display-name "Python (MCS 2026)"
```

Restart your JupyterHub server (**File → Hub Control Panel → Stop My Server**, then start
it again). **Python (MCS 2026)** now appears in the launcher and in the kernel picker at
the top right of every notebook.

```{important}
Every notebook in this course must run in the **Python (MCS 2026)** kernel. If you open
a notebook and `import ngio` fails, look at the kernel name first — it is the cause
about nine times out of ten.
```

## 5. Check it worked

```bash
python -c "import ngio, ez_zarr, scanpy; print('ngio', ngio.__version__)"
```

Expected output:

```
ngio 1.1.0
```

## Starting again

If the environment gets into a state you do not understand, delete it and repeat from
step 3. Nothing of yours lives inside it:

```bash
rm -rf $HOME/venvs/mcs2026
```
