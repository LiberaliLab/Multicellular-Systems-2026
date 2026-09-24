# Working on Euler

Two things that are not about analysis, and will each cost you an afternoon if nobody
says them first.

## Three folders, and only one of them is yours

| | |
|---|---|
| `/cluster/project/mcsliberali/data_mcs_2026` | Shared, **read-only**, and the only one you can see. The raw feature table and everything Stage 1 built. |
| `/cluster/project/mcsliberali/file_outputs` | The **course's own** run folder — where the outputs printed in these pages were produced. You have no access to it. |
| `~/mcs2026` | **Yours.** Everything you make goes here. |

:::{important}
**The `DATA` line at the top of chapters 06 to 10 points at `file_outputs`. That path is
ours, not yours — replace it with your own folder.**

Those chapters were run to produce the figures you see on these pages, and the path they
carry is the one that run used. Left as it is, it will fail for you, because you cannot
read or write there.
:::

### The pattern

Chapters 06 to 10 form a chain: each one opens an `.h5ad`, adds something, and writes it
back. Chapter 06 adds the PCA, 07 the UMAP and the neighbour graph, 08 the cell states, 10
the pseudotime. So you need somewhere to put them.

```python
from pathlib import Path

DATA = Path("/cluster/project/mcsliberali/data_mcs_2026")   # shared, read-only
MINE = Path.home() / "mcs2026"                              # yours
MINE.mkdir(exist_ok=True)
```

**Load from `DATA` once, at the start**, to make your own copy:

```python
cells = sc.read_h5ad(DATA / "mcs2026_controls.h5ad")        # chapter 06
# ... your analysis ...
cells.write_h5ad(MINE / "mcs2026_controls_downstream.h5ad", compression="gzip")
```

**From chapter 07 onwards, read and write `MINE`:**

```python
cells = sc.read_h5ad(MINE / "mcs2026_controls_downstream.h5ad")
# ... your analysis ...
cells.write_h5ad(MINE / "mcs2026_controls_downstream.h5ad", compression="gzip")
```

The two files that stay in `DATA` all the way through are the shared Stage 1 outputs that
nothing downstream rewrites — `mcs2026_intensity.h5ad` (chapter 08) and
`mcs2026_sketch.h5ad` (chapter 10). Keep reading those from `DATA`.

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
