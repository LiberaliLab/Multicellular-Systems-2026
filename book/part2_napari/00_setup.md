# Napari (on your laptop)

This part runs on **your own machine**, not on
Euler. This is the one place the course
leaves the cluster, and the reason is worth understanding rather than just working
around.

## Why not on Euler

Napari renders with **OpenGL 3.2 or newer**. When you forward a graphical application
from a cluster over SSH (X11 forwarding), indirect rendering caps OpenGL at **1.4**.
The versions are not close, and no amount of configuration bridges them: napari starts,
fails to create a canvas, and exits.

So: the taught parts on Euler, napari on your laptop, deliberately.

## Install

Napari needs its **own** virtual environment, separate from the Euler one. It is a large
desktop application with its own Qt dependencies, and mixing it into the analysis
environment causes conflicts in both directions.

```bash
python -m venv ~/venvs/mcs2026-napari
source ~/venvs/mcs2026-napari/bin/activate      # Windows: ~\venvs\mcs2026-napari\Scripts\activate
pip install -r environment/requirements-napari.txt
```

That gives you napari plus **`napari-ome-zarr`**, the plugin that lets napari open an
OME-Zarr directly.

Start it with:

```bash
napari
```

An empty napari window should appear. That is all the check you need.

```{admonition} Prefer a click-to-install app?
:class: tip

Napari also ships as a standalone installer from
<https://napari.org/stable/tutorials/fundamentals/installation.html>. It is easier to
install, but you then have to add the `napari-ome-zarr` plugin through
**Plugins → Install/Uninstall Plugins**, and searching for `napari-ome-zarr`.
Either route works for this course.
```

## Getting the data onto your laptop

You do not need the whole plate — one well is plenty and small enough to copy. From a
terminal **on your laptop** (not on Euler):

```bash
mkdir -p ~/mcs2026-data
rsync -avh --progress \
  <nethz>@euler.ethz.ch:<course-data-path>/plate_subset.zarr \
  ~/mcs2026-data/
```

Your instructor gives you `<course-data-path>` and the name of the subset in the first
session.

```{warning}
An OME-Zarr is a **directory**, not a file — often with thousands of small chunk files
inside it. Copy it with `rsync -a` or `scp -r`, and do not try to open it by
double-clicking.
```

## One plugin to avoid

You may come across **`napari-ome-zarr-navigator`**, a nice plugin for browsing HCS
plates. It pins `ngio<0.6`, and this course uses `ngio` 1.1.0. Installing it into either
of your environments will downgrade `ngio` and break Parts 1 and 3.

If you want it, give it a third environment of its own. Nothing in the napari part needs it.
