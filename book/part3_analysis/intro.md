# Part 3 — Multi-condition analysis

This is the long part, and the one closest to real work.

## The experiment

One 384-well plate of **HNES1 human naive embryonic stem cells**, perturbed with 16
compounds plus two controls, fixed at four timepoints, and imaged over 18 rounds of 4i
multiplexed immunofluorescence. Segmentation and feature extraction give you:

$$733{,}556 \text{ cells} \times 4{,}464 \text{ features}$$

## The four themes

Rather than throwing 4,464 features at a clustering algorithm and hoping, the analysis is
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
**does each perturbation move its own theme most?** — and chapter 08 asks it.

## Three stages

### [Stage 1 — Preparing the data](1_preparation/intro.md)

From 4,464 anonymous columns to an annotated, normalised table.

| | |
|---|---|
| [00](1_preparation/00_experiment_and_layout.ipynb) | The plate layout, and building the staining decoder |
| [01](1_preparation/01_decode_and_slim.ipynb) | 4,464 unnamed columns → a named, loadable table |
| [02](1_preparation/02_qc_and_normalisation.ipynb) | Quality control and normalising to the controls |

### [Stage 2 — The four themes](2_themes/intro.md)

One panel at a time, each chapter the same five steps.

| | |
|---|---|
| [03](2_themes/03_signaling.ipynb) | Signaling |
| [04](2_themes/04_mechanics.ipynb) | Cell-specific mechanics |
| [05](2_themes/05_metabolism.ipynb) | Metabolism |
| [06](2_themes/06_organelles.ipynb) | Organelles |

### [Stage 3 — Going further](3_going_further/intro.md)

Cell identity, the replicate unit, and whether the four themes hold together.

| | |
|---|---|
| [07](3_going_further/07_celltypes_and_proportions.ipynb) | Cell types and how their proportions shift |
| [08](3_going_further/08_integration.ipynb) | Putting the four themes back together |

```{important}
Part 3 needs a JupyterHub session with **at least 32 GB of memory** (4 cores × 8 GB).
The peak is in chapter 02, not in chapter 01 — see
[the setup page](../setup/euler_jupyterhub.md) for why.
```
