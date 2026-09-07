# Group projects

The last session is yours. In groups, you take one theme further than the notebooks do,
and present what you found in ten minutes.

## How it works

1. **You will be assigned a group** .
2. **Write down the question first**, in one sentence, before any code. If you cannot
   state it in a sentence, the analysis will not have a conclusion either.
3. **Sketch the steps** as bullet points, then make your notebook's sections match them.
4. **Present**

## The groups

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 1 — Signaling
Four compounds target the PI3K–AKT–mTOR axis (MK-2206, Wortmannin, INK128, PF-4708671)
and one activates it (IGF). Do their effects on p-AKT and p-S6 order the way the pathway
says they should? Does the ordering hold at every timepoint?
:::

:::{grid-item-card} 2 — Cell mechanics
Cell shape, junctions and the nuclear lamina. Which perturbations change *shape* without
changing *markers*, or the other way round? Is population density a confounder — do
crowded wells simply have smaller cells?
:::

:::{grid-item-card} 3 — Metabolism
Glucose, LipidMix, MEM AA and UCL-TRO-1938 are metabolic perturbations. Do they move the
metabolic panel more than the organelle panel? Where do the mTOR readouts sit — with
signaling, or with metabolism?
:::

:::{grid-item-card} 4 — Organelles
Golgi, lysosome, endosome, peroxisome, ER. Which organelle markers move together across
perturbations, and does that grouping match cell biology? Geldanamycin (HSP90) and
Cycloheximide (translation) are the interesting ones here.
:::

:::{grid-item-card} 5 — Cross-theme
Take the condition × theme summary from chapter 08 and interrogate it. Does each
perturbation really move its own theme most? Which ones do not, and is that biology or
an artefact of how the panels were defined?
:::
::::

## What makes a good project

- **A stated replicate unit.** There are 733,556 cells but only **224 wells**, and 3
  wells per condition per timepoint (5 for DMSO). A p-value computed over cells is
  measuring how many cells you imaged. Say which unit you tested at, and why.
- **A control comparison.** "Higher" only means something next to DMSO or PBS.
- **A negative result, stated plainly.** "We expected X and did not see it" is a finding.
  Quietly switching to a different analysis until something is significant is not.
- **A figure someone else can read** — axis labels, units, and the number of replicates.

