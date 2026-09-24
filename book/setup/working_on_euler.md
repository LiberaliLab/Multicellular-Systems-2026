# Working on Euler

Two things that are not about analysis, and will each cost you an afternoon if nobody
says them first.

## The data folder is read-only

`/cluster/project/mcsliberali/data_mcs_2026` holds the raw feature table and everything
Stage 1 built. **You can read it. You cannot write to it.**

That matters because the chapters from 05 onwards form a chain: each one opens an `.h5ad`,
adds something, and writes it back. Chapter 06 adds the PCA, 07 the UMAP and the neighbour
graph, 08 the cell states, 10 the pseudotime. Those writes have to land somewhere you own.

### The pattern

Open the shared file once, then work out of your own folder:

```python
from pathlib import Path

DATA = Path("/cluster/project/mcsliberali/data_mcs_2026")   # shared, read-only
MINE = Path.home() / "mcs2026"                              # yours
MINE.mkdir(exist_ok=True)
```

From chapter 05 on, read from `MINE` if you have already been through the chapter before
it, and fall back to the shared copy the first time:

```python
path = MINE / "mcs2026_controls_downstream.h5ad"
cells = sc.read_h5ad(path if path.exists() else DATA / "mcs2026_controls.h5ad")

# ... your analysis ...

cells.write_h5ad(MINE / "mcs2026_controls_downstream.h5ad", compression="gzip")
```

The chapters are written with a single `DATA` because one path reads more clearly than two.
**You have to change the write to `MINE`**, and then the read in every chapter after it.

:::{warning}
Your `$HOME` quota is around 16 GB. The controls object is small and fine to keep there.
**Do not copy the 13 GB feature table into your home folder** — read that one where it
lives.
:::

## Saving a figure

There are two kinds of plot in this course and they save differently. Worth knowing before
the week you need figures for a presentation.

### Figures you built yourself

Anything from `plt.subplots` gives you a `fig`, which has `savefig`:

```python
fig.savefig(MINE / "oct4_by_well.png")
```

The house settings at the top of every chapter already set 300 dpi and a tight bounding
box, so there is nothing else to pass. Use `.pdf` instead of `.png` if it has to scale on
a slide.

### Scanpy's plots

`sc.pl.*` draws *and closes* the figure itself, so there is no `fig` to catch afterwards.
Pass `save=` instead:

```python
sc.settings.figdir = MINE                                   # the default is ./figures
sc.pl.umap(cells, color="cell_state", save="_states.png")   # -> MINE/umap_states.png
```

Three things about `save=` that catch people out:

- it writes into `sc.settings.figdir`, **not** your working directory — and creates that
  folder for you;
- it **prepends the function name**, so the call above produces `umap_states.png`, not
  `_states.png`;
- **without an extension you get a PDF**, because scanpy's default figure format is PDF.

```{note}
Scanpy reads the same `savefig.dpi` that the chapters already configure, so both routes
give you 300 dpi with tight margins without passing anything.
```
