# Stage 2 — The toolkit, on the controls

Stage 1 left you with a clean, named, normalised object and one deliberate cut through it:
**every DMSO and PBS cell, and nothing else.**

That subset is where you learn every method in this stage. The reason is not convenience.

```{important}
**Neither control is a treatment.** DMSO is the vehicle the compounds were dissolved in;
PBS is buffer. Both were meant to leave the cells alone.

So when a method run on these cells appears to find something, you have learned something
about the method. That is the only setting in which you can calibrate a technique at all —
run it on treated cells and any structure it produces is a claim you have no way to check.
```

The subset is also the one size that fits. Thirty-two wells is about 97,000 cells, so
**every chapter here uses all of them** — no subsampling, no sketch, nothing to caveat. The
full plate is 653,000 and does not go through a neighbour graph, which is why
[Part 4](../../part4_final_solutions/intro.md) works on a sketch and this stage does not
have to.

```{tableofcontents}
```

## The six chapters

| | | |
|---|---|---|
| **05** | [The AnnData object](05_the_anndata_object.ipynb) | what is in the file, and how to open it without copying it |
| **06** | [PCA](06_pca.ipynb) | the axes of largest variation — and what the first one is made of |
| **07** | [UMAP](07_umap.ipynb) | choosing the eight markers everything is built on, then a picture — and what it says when you stop looking and start counting |
| **08** | [Cell-type annotation](08_cell_type_annotation.ipynb) | naming states, then projecting them onto the whole plate |
| **09** | [PAGA](09_paga.ipynb) | how the states connect — the shape, not the counts |
| **10** | [Diffusion maps](10_diffusion_map.ipynb) | an ordering rather than groups, and a null result you can trust |

## What the controls turn out to say

Three of these chapters end by measuring something the algorithm never saw, and the answers
are worth having in advance:

| | |
|---|---|
| the embedding does not invent a difference between DMSO and PBS | neighbour purity 1.1× chance |
| there is no plate-position effect | row purity 1.0× chance, once same-well pairs are excluded |
| but **cells from the same well sit together** | 2.2× chance, within one condition and timepoint |
| the differentiation axis is not made by the compounds | it is already there in untreated cells, and transfers |

The third of those is the one that changes how you work. Two untreated conditions, one
timepoint, and a cell's own well still predicts its neighbours better than anything except
which day it was fixed. Cells within a well are not independent, and that is a measurement
rather than a principle.

## Three ideas that run through all six

**The space you compute in decides the answer.** Chapter 06 finds the largest axis of
variation is how brightly a cell stained. Chapter 07 then declines to build on it, cutting to
**eight lineage markers** — and measures what that buys: the same biological signal, and
*half* the well-to-well batch signal. Chapter 10 runs one algorithm on the two graphs and
gets two different meanings. None of that is a property of the method; all of it is a
property of what you fed it.

That cut is the third and last of the course's reductions — 4,464 columns to 2,587 in
[Step 9](../1_preparation/01_columns_to_markers.ipynb), to 38 in
[Step 19](../1_preparation/03_normalisation.ipynb), to 8 here. Each one is argued rather than
assumed, and only the first is lossless.

**Learn it somewhere safe, then check that it transfers.** Chapters 08 and 10 both end by
carrying something off the controls and onto the whole plate — the state labels, and the
pseudotime axis. Each time, whether it transfers is a question with a number attached.

**Check the thing the algorithm never saw.** A UMAP coloured by plate row, a pseudotime
compared against the clock, a cluster profile you have to be able to name, a PAGA edge that
has to survive its threshold. Every chapter ends with one, because a method that cannot fail
has not told you anything.
