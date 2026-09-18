# Setup

Do these four things, in this order, **before the first session**.

1. **[Euler and JupyterHub](euler_jupyterhub.md)** — get onto the cluster and start a
   notebook server in your browser.
2. **[The Python environment](python_environment.md)** — one virtual environment, one
   kernel, used by Parts 1 and 3.
3. **[The data](data.md)** — point the notebooks at the plate and the feature table.
4. **[Napari](napari_local.md)** — a *separate* install on your own laptop, for the
   optional napari part. Skip it unless you want that part.

```{warning}
Steps 2 and 4 are deliberately separate environments, and step 4 is deliberately not on
Euler. The reasons are on those pages. Installing napari into the Euler environment will
not work, and will break the Euler environment while it fails.
```

## If something goes wrong

Ask. But first, two places worth checking:

- **JupyterHub will not start** → the reason is in `~/jupyterhub-logs/` on Euler.
- **`import ngio` fails in a notebook** → you are almost certainly in the wrong kernel.
  Check the kernel name in the top right of JupyterLab; it should say
  **Python (MCS 2026)**.
