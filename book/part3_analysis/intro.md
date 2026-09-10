# Part 3 — Multi-condition analysis

This is the long part, and the one closest to real work.

## The experiment

One 384-well plate of **HNES1 human naive embryonic stem cells**, perturbed with 16
compounds plus two controls, fixed at four timepoints, and imaged over 18 rounds of 4i
multiplexed immunofluorescence. Segmentation and feature extraction give you:

$$733{,}556 \text{ cells} \times 4{,}464 \text{ features}$$

with a `var` table that holds the 4,464 column names and nothing else. Which antibody each
column measures is recorded only in an Excel workbook filled in at the bench.

## Two stages, then your turn

### [Stage 1 — Preparing the data](1_preparation/intro.md) · Steps 1–19

One continuous pipeline. Structure → rename the columns → drop what is redundant → drop the
one well that failed → normalise to the controls → assemble it all into a single object,
and reduce it to something you can build a neighbour graph on.

It ends with **one file that every later chapter opens**: every surviving cell, 38 named
markers, normalised values in `X`, the raw ones in a layer, the decoder in `var`, and a
written record in `uns` of what was thrown away and why.

### [Stage 2 — The toolkit, on the controls](2_controls/intro.md) · chapters 05–10

Every method the analysis needs — the AnnData object, PCA, UMAP, cell-type annotation, PAGA
and diffusion maps — learned on **DMSO and PBS alone**.

```{important}
**Stage 2 contains no treatments, so it can draw no conclusions about them.** Neither
control is a compound, which makes them the one place a technique can be calibrated: if a
method appears to find something here, what you have learned is about the method.

It is also the one subset small enough to use whole. Every chapter works on all ~97,000
control cells, with no subsampling to caveat.
```

Two things travel out of this stage and into the rest of the course: the **cell-state
labels**, projected from the controls onto all 653,000 cells, and the **pseudotime axis**,
which chapter 10 shows means the same thing across all eighteen conditions.

### [11 · Your turn](11_your_turn.ipynb)

The skeleton for your group's own analysis: state the question, subset to your panel, look,
test at the well level, make one figure someone else can read. Its worked example runs on a
panel none of the groups is assigned, so the mechanics are demonstrated without the answer
being given.

The worked answers are in [Part 4](../part4_final_solutions/intro.md).

## The four themes

Rather than throwing 4,464 features at a clustering algorithm and hoping, the group work is
organised around four biological questions, each with its own antibody panel:

| Theme | Question |
|---|---|
| **Signaling** | Which pathways are on, and does inhibiting one show up where you expect? |
| **Cell mechanics** | How are cells shaped, attached and mechanically coupled? |
| **Metabolism** | What is the state of the mitochondria, the ER, the stress machinery? |
| **Organelles** | How are Golgi, lysosomes, endosomes and peroxisomes organised? |

The panels **overlap on purpose**. Calreticulin is an ER protein and a metabolic readout;
p-S6 sits on the PI3K–mTOR axis and belongs to both signaling and metabolism. Forcing a
marker into exactly one box would be tidier and less true.

The perturbations line up with the themes, which is what makes the whole thing testable:
MK-2206, Wortmannin, INK128 and PF-4708671 all hit the signaling axis; Glucose, LipidMix
and MEM AA are metabolic. So there is a real question with an expected answer —
**does each perturbation move its own theme most?**

```{important}
Part 3 needs a JupyterHub session with **at least 32 GB of memory** (4 cores × 8 GB).
The peak is Step 11, not the step that touches the 13 GB file — see
[the setup page](../setup/euler_jupyterhub.md) for why.
```
