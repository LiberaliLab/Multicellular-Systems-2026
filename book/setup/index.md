# Setup

Do these three things, in this order, **before 2 October** — that is the first session
where we work on Euler. Try it on your own; whatever did not work, we sort out together
in the room that morning.

1. **[Euler and JupyterHub](euler_jupyterhub.md)** — get onto the cluster and start a
   notebook server in your browser.
2. **[The Python environment](python_environment.md)** — one virtual environment, one
   kernel, used by Parts 1 and 3.
3. **[The data](data.md)** — where the plate and the feature table live on Euler.

```{note}
The optional [napari part](../part2_napari/intro.md) needs a *separate* install on your own
laptop, and its setup page lives with it rather than here. You only need it if you do that
part. Installing napari into the Euler environment will not work, and will break the Euler
environment while it fails.
```

## If something goes wrong

Ask. But first, two places worth checking:

- **JupyterHub will not start** → the reason is in `~/jupyterhub-logs/` on Euler.
- **`import ngio` fails in a notebook** → you are almost certainly in the wrong kernel.
  Check the kernel name in the top right of JupyterLab; it should say
  **Python (MCS 2026)**.
