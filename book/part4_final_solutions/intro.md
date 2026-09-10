# Part 4 — Solutions

Part 3 stopped short on purpose. It taught the pipeline and the toolkit, ran everything on
the whole dataset, and never said what any compound did.

This part says.

```{warning}
**These are the answers.** If your group has an analysis to present, do it first. Reading a
worked solution before attempting the question is the fastest way to learn nothing — you
will find every step obvious and be unable to reproduce any of it.
```

```{tableofcontents}
```

## What is here

**[00 · The whole dataset, one analysis](00_full_dataset_analysis.ipynb)** is the complete
worked answer, using only techniques from Part 3:

| | |
|---|---|
| **1** | check what Stage 1 handed over |
| **2–3** | every condition and every marker in one heatmap, then which markers move together |
| **4–5** | ranked effects, the p-value floor read honestly, and how they change over time |
| **6** | the embeddings, coloured by treatment — and why you must not read them by eye |
| **7** | trajectories: pseudotime as a readout, and the statistic that decides whether you see anything |
| **8** | proportions, and the replicate unit settled with a number |

Then one chapter per theme, each asking the same five questions of a different panel:

| theme | question | panel |
|---|---|---|
| **[01 · Signaling](01_signaling.ipynb)** | which pathways are on, and does inhibiting one show up where you expect? | Foxo3a, Foxo1, FGFR1, PDGFRα, β-Catenin, YAP1, p-S6, p-AKT, c-Myc, RNAPII-pS5 |
| **[02 · Mechanics](02_mechanics.ipynb)** | how are cells shaped, attached and mechanically coupled? | p-Myosin IIa, α-Tubulin, ZO-1, E-cadherin, Fibronectin, LAMA4, Lamin A/B1 |
| **[03 · Metabolism](03_metabolism.ipynb)** | what state are the mitochondria, ER and stress machinery in? | Mitochondria, Pmp70, GRP78, HSP90, Calreticulin, p-S6, p-AKT |
| **[04 · Organelles](04_organelles.ipynb)** | how are Golgi, lysosomes, endosomes and peroxisomes organised? | EEA1, GM130, Giantin, LAMP1, DDX6, Calreticulin, GRP78, Pmp70, Mitochondria, Lamin A/B1 |

And **[05 · Integration](05_integration.ipynb)** puts the four back together to ask whether
each perturbation really moves its own theme most.

```{note}
The panels **overlap on purpose**. Calreticulin is an ER protein and a readout of metabolic
stress; p-S6 sits on the PI3K–mTOR axis and belongs to both signaling and metabolism.
Forcing each marker into one box would be tidier and less true.
```

## How to read a solution

Not straight through. For each section, cover the output, predict what it will say, then
look. Where you were wrong is the only part worth your time — and the disagreements are
often about *how much* evidence there is rather than about the direction of an effect,
which is the harder thing to learn.

Three places these chapters deliberately go against the obvious reading, and they are the
ones to slow down on:

- **p-AKT is a poor readout of AKT inhibition**, and chapter 01 explains why rather than
  explaining it away.
- **PC1 at the cell level is brightness**; at the well level it is not. Chapter 00 section 6
  shows the same data giving two different first components.
- **A pseudotime median can show nothing while the population has changed completely.**
  Chapter 00 section 7 is entirely about choosing the summary statistic from the shape of
  the distribution.

The [group projects](../projects.md) page has the presentation format and the starting
questions.
