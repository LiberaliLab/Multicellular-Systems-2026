# Stage 2 — The four themes

Rather than throwing 4,464 features at a clustering algorithm and hoping, the analysis is
organised around four biological questions, each with its own antibody panel.

| theme | question |
|---|---|
| **Signaling** | Which pathways are on, and does inhibiting one show up where you expect? |
| **Cell mechanics** | How are cells shaped, attached and mechanically coupled? |
| **Metabolism** | What state are the mitochondria, the ER and the stress machinery in? |
| **Organelles** | How are Golgi, lysosomes, endosomes and peroxisomes organised? |

```{tableofcontents}
```

All four chapters follow the same five steps, so once you have read one carefully the
others are quick: resolve the panel, compare each condition to the DMSO control, rank the
effects honestly, build an embedding from that theme alone, and look at how single cells
shift over the four timepoints.

```{note}
The panels **overlap on purpose**. Calreticulin is an ER protein and a readout of
metabolic stress; p-S6 sits on the PI3K–mTOR axis and belongs to both signaling and
metabolism. Forcing each marker into exactly one box would be tidier and less true.
```

This is the stage the group projects build on — one theme per group.
