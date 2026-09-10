# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (MCS 2026)
#     language: python
#     name: mcs2026
# ---

# %% [markdown]
# # 08 · Cell-type annotation
#
# Chapters 06 and 07 asked *how* the control cells vary. This one asks a different question:
# **what kind of cell is each one?** — and then carries the answer off the controls and onto
# the whole plate.
#
# The answer is not something the algorithm gives you. Clustering produces groups with
# numbers on them; turning a number into a name is a judgement you make from the markers,
# and defending that judgement is most of the work.
#
# | | |
# |---|---|
# | **1** | The graph you already have |
# | **2** | Cluster, and choose a resolution |
# | **3** | Read each cluster's profile, and check for the ones that are not cell types |
# | **4** | Name them, and show them on the UMAP |
# | **5** | Project the labels onto all 653,000 cells |

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path.cwd().parents[2] / "src"))

from mcs2026 import analysis, panels, plotting
from mcs2026.config import H5AD_SLIM

plotting.set_style()
pd.set_option("display.width", 140)

cells = sc.read_h5ad(H5AD_SLIM.with_name("mcs2026_controls.h5ad"))
print(f"{cells.n_obs:,} control cells x {cells.n_vars} markers")
print(f"  obsm: {list(cells.obsm)}   obsp: {list(cells.obsp)}")

# %% [markdown]
# ## 1 · The graph you already have
#
# [Chapter 07](07_umap.ipynb) chose the features and built the graph: the eight identity
# markers, and a neighbour graph over them. It also showed why — the same graph built on all
# 38 markers carried nearly twice as much well-to-well batch signal for no gain in
# biological signal.
#
# Nothing more is needed here. Clustering is a question asked of that graph, and so are
# chapters 09 and 10.

# %%
identity = panels.resolve_panel(cells.var, "identity", verbose=False)
print(f"  {len(identity)} markers: {', '.join(cells.var.loc[identity, 'marker'])}")
print(f"  graph: {cells.obsp['connectivities'].nnz:,} edges over {cells.n_obs:,} cells")
print(f"  obsm : {list(cells.obsm)}")

# %% [markdown]
# :::{tip}
# **The space you cluster in and the space you plot in do not have to be the same.** Here
# they are — clusters and UMAP both come from the eight markers — but it is normal, and
# often clearer, to cluster on a focused panel and show the result on an embedding built
# from something wider. Say which is which in the caption, because a reader will assume
# they match.
# :::

# %% [markdown]
# ## 2 · Cluster, and choose a resolution
#
# Leiden finds communities in the graph. `resolution` controls how finely it cuts, and it
# is a **choice**, not a result: there is no value the data prefers.

# %%
for resolution in [0.05, 0.1, 0.3, 0.6]:
    sc.tl.leiden(cells, resolution=resolution, key_added=f"leiden_{resolution}",
                 flavor="igraph", n_iterations=2, random_state=0)
    counts = cells.obs[f"leiden_{resolution}"].value_counts()
    print(f"  resolution {resolution:<5} {len(counts):2d} clusters, "
          f"smallest {counts.min():>5,} cells ({counts.min()/cells.n_obs:.2%})")

# %% [markdown]
# More clusters is not more biology. The test to apply is not a score — it is a question:
# **can you say what each cluster is positive for?** Anything you cannot name, you cannot
# report.
#
# Read the sweep above for **stability** rather than for a best value. The three-cluster
# solution survives a two-fold change in resolution; the eight- and fifteen-cluster ones
# appear and then immediately fragment further. A structure that holds while you vary the
# parameter is worth reporting, and you report the parameter with it.

# %%
cells.obs["cluster"] = cells.obs["leiden_0.1"]
cells.obs.cluster.value_counts().sort_index().rename("cells")

# %% [markdown]
# ## 3 · Read each cluster's profile
#
# A cluster number means nothing. The profile — the mean of each marker within the cluster
# — is what you name it from.

# %%
profile = (pd.DataFrame(cells.obsm["X_identity"], columns=identity)
           .groupby(cells.obs.cluster.astype(str).values, observed=True).mean())
profile.index.name = "cluster"
profile.round(2)

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.2))
im = ax.imshow(profile.values, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
ax.set(yticks=range(len(profile)), yticklabels=profile.index, ylabel="cluster")
ax.set_xticks(range(len(identity)))
ax.set_xticklabels(identity, rotation=45, ha="right")
ax.set_title("Mean marker level per cluster (control-cell SDs)")
fig.colorbar(im, ax=ax, shrink=0.85, label="SDs")
fig.tight_layout()

# %% [markdown]
# ### Check for clusters that are not cell types
#
# Before naming anything, run the check that would catch a bad cluster. Two symptoms, both
# cheap to test:
#
# - it is **tiny** — a fraction of a percent of the cells;
# - it is defined by a marker being extremely **low** rather than by any marker being high.
#
# Together those describe a technical failure, not a lineage: a field where that staining
# round did not work, or a registration slip that put the wrong pixels under the mask.

# %%
sizes = cells.obs.cluster.value_counts()
check = pd.DataFrame({
    "cells": sizes,
    "share": (sizes / cells.n_obs).round(4),
    "highest marker": profile.max(axis=1).round(2),
    "lowest marker": profile.min(axis=1).round(2),
})
check["suspicious"] = (check.share < 0.005) & (check["lowest marker"] < -3)
check.sort_index()

# %% [markdown]
# Nothing is flagged here: every cluster is a substantial share of the cells, and every one
# is defined by markers being **high**. The check passing is a result worth stating, not a
# step to skip once you have seen it pass once.
#
# :::{warning}
# **If a cluster is flagged, do not name it.** A cluster is a cell type only if you can say
# what it is *positive* for. Drop it, and say in writing that you did.
#
# One caution before you generalise the rule. Dropping a small artefact cluster is safe only
# when it is spread across the plate; had a staining round failed in one *block* of wells,
# the same artefact would sit in one timepoint and would be indistinguishable from a
# treatment effect. You cannot tell which case you are in without looking, which is the
# reason to look.
# :::

# %% [markdown]
# ## 4 · Name them, and show them on the UMAP
#
# Name from the profile, with the thresholds written down. Someone reading this should be
# able to disagree with a specific number rather than with your judgement in general.

# %%
def name_cluster(row, size_fraction):
    if size_fraction < 0.005:
        return "artefact"
    if row[["Oct4", "Nanog", "Sox2"]].mean() > 0.3:
        return "Pluripotent"
    if row[["GATA4", "SOX17"]].mean() > 0.5:
        return "Hypoblast"
    return "TE-like"

STATES = ["Pluripotent", "TE-like", "Hypoblast", "artefact"]

names = {cluster: name_cluster(profile.loc[cluster], sizes[cluster] / cells.n_obs)
         for cluster in profile.index}
assigned = [s for s in STATES if s in set(names.values())]
cells.obs["cell_state"] = pd.Categorical(cells.obs.cluster.map(names).astype(str),
                                         categories=assigned)
pd.DataFrame({"cluster": list(names), "name": list(names.values()),
              "cells": [sizes[c] for c in names]}).set_index("cluster")

# %% [markdown]
# The canonical annotation figure is a **dot plot**: colour is the mean level, dot size is
# the fraction of cells expressing it. Both together say more than either alone — a marker
# can be high on average because every cell has a little, or because a few cells have a
# lot, and those are different claims.

# %%
sc.pl.dotplot(cells, identity, groupby="cell_state", standard_scale="var",
              swap_axes=False, figsize=(6.5, 2.6))

# %% [markdown]
# And on the embedding from chapter 07 — remembering that this UMAP was and the clusters come from the same eight markers, so this
# is a picture of the clustering rather than independent evidence for it:

# %%
sc.pl.umap(cells, color=["cell_state", "leiden_0.6"], ncols=2, s=6, frameon=False)

# %% [markdown]
# ## 5 · Project the labels onto all 653,000 cells
#
# You have annotated **32 untreated wells**. The plate has 224, and every count,
# proportion and statistical test in [Part 4](../../part4_final_solutions/intro.md) has to
# run on all of them.
#
# So push the labels outward: a k-nearest-neighbour vote in the same seven-marker space
# gives every cell on the plate the label of the control cells it sits among.
#
# This is a **reference mapping**, and it is worth naming as such because it is how
# annotation is usually done in practice — you label a clean reference once, carefully, and
# project it onto everything else rather than re-clustering each new sample and hoping the
# clusters correspond. The controls are the reference here, and the fact that they saw no
# compound is exactly what makes them one.
#
# :::{warning}
# **A projection can only return labels it was given.** If a compound drives cells into a
# state that does not exist in DMSO or PBS, every one of those cells will still be assigned
# one of these three names — the nearest, not the right one.
#
# That is a real limitation and not a fixable one; it is the price of a fixed reference.
# What you can do is *notice*: keep the distance to the nearest labelled neighbour, and treat
# cells that are far from everything as unclassified rather than as members. Exercise 3.
# :::
#
# First, does the vote work at all? Hold out a fifth of the annotated cells and predict them:

# %%
rng = np.random.default_rng(0)
order = rng.permutation(cells.n_obs)
train, test = order[: int(0.8 * cells.n_obs)], order[int(0.8 * cells.n_obs):]
space = cells.obsm["X_identity"]
labels = cells.obs.cell_state.astype(str).values

predicted = analysis.transfer_labels(space[train], labels[train], space[test])
accuracy = (predicted == labels[test]).mean()
print(f"  held-out accuracy: {accuracy:.1%} on {len(test):,} control cells")
print(pd.crosstab(pd.Series(labels[test], name="annotated"),
                  pd.Series(predicted, name="predicted")).to_string())

# %% [markdown]
# Read the confusion table, not only the accuracy. A high overall score can hide a state
# that is never recovered — and the states you most want to count are usually the rare
# ones, which contribute least to the average.

# %%
full = sc.read_h5ad(H5AD_SLIM.with_name("mcs2026_clean.h5ad"))
full.obs["cell_state"] = pd.Categorical(
    analysis.transfer_labels(space, labels, np.asarray(full[:, identity].X)),
    categories=assigned,
)

pd.DataFrame({
    "controls %": 100 * cells.obs.cell_state.value_counts(normalize=True),
    "all 18 conditions %": 100 * full.obs.cell_state.value_counts(normalize=True),
}).round(2)

# %% [markdown]
# :::{important}
# **The two columns are close, and the gap is the whole of Part 4.**
#
# Pluripotent and TE-like barely move. Hypoblast is a couple of points lower across the
# plate than in the controls — which says that, taken together, the sixteen compounds
# suppress it slightly more often than they promote it. *Which* compounds, and by how much,
# is not visible here and is not supposed to be.
#
# Two rules that follow, and both are load-bearing:
#
# - **Proportions are computed on `mcs2026_clean.h5ad`, never on the controls.** The
#   controls are 32 wells chosen for being untreated; their composition is not the plate's.
# - **A proportion over cells is still not evidence about a treatment.** The plate has three
#   wells per condition per timepoint, and that — not the cell count — is the amount of
#   independent evidence. Count at the well level. Exercise 2 shows what ignoring this
#   costs.
# :::

# %% [markdown]
# ## 6 · Save

# %%
cells.write_h5ad(H5AD_SLIM.with_name("mcs2026_controls.h5ad"), compression="gzip")
full.write_h5ad(H5AD_SLIM.with_name("mcs2026_clean.h5ad"), compression="gzip")
print(f"  controls : {cells.n_obs:,} cells, obs['cell_state'] + clusters + embeddings")
print(f"  clean    : {full.n_obs:,} cells, obs['cell_state'] projected from the controls")

# %% [markdown]
# ---
#
# ## Exercises
#
# ### 1. Resolution is a choice — make it and defend it
#
# Look at the profiles at resolution 0.3 and 0.6. How many of those clusters can you name?
# Which merge back together at 0.1, and were they worth separating?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# for resolution in [0.1, 0.3, 0.6]:
#     labels = cells.obs[f"leiden_{resolution}"].values
#     print(f"\nresolution {resolution}")
#     print(pd.DataFrame(cells.obsm["X_identity"], columns=identity)
#           .groupby(labels).mean().round(2))
# ```
#
# At the higher resolutions several clusters differ by *degree* rather than by which
# markers are on — a slightly brighter version of the pluripotent cluster is not a lineage.
# That is the sign you have gone past what seven markers can support.
#
# The honest report names the resolution you used, and says that the conclusion holds at
# the neighbouring values too. If it does not hold, that is the finding.
# :::

# %% [markdown]
# ### 2. Does the artefact cluster matter?
#
# Recompute the state proportions per condition with the artefact cluster kept in. Does
# anything move? Was dropping it important, or merely tidy?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# It barely moves — the cluster is a few dozen cells among tens of thousands.
#
# That is worth knowing, and it is not an argument for leaving it in. It is harmless here
# because it is small and spread across conditions; the same artefact concentrated in one
# block of the plate would be indistinguishable from a treatment effect. You cannot tell
# which case you are in without looking, which is the reason to look.
# :::

# %% [markdown]
# ### 3. Count the same thing two ways
#
# Pick any condition and any timepoint. Compare its hypoblast fraction with DMSO twice:
# once treating every **cell** as an observation, once treating every **well** as one. How
# far apart are the two p-values?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# from scipy import stats
#
# frame = full.obs[["condition", "timepoint_h", "well", "cell_state"]].copy()
# late = frame[frame.timepoint_h.astype(int) == 84]
#
# # per cell
# a = (late[late.condition == "MK-2206"].cell_state == "Hypoblast").astype(int)
# b = (late[late.condition == "DMSO"].cell_state == "Hypoblast").astype(int)
# table = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
# print("cells:", stats.chi2_contingency(table)[1])
#
# # per well
# per_well = (late.groupby(["condition", "well"]).cell_state
#             .apply(lambda s: (s == "Hypoblast").mean()).reset_index())
# t = per_well.loc[per_well.condition == "MK-2206", "cell_state"]
# c = per_well.loc[per_well.condition == "DMSO", "cell_state"]
# print("wells:", stats.mannwhitneyu(t, c).pvalue,
#       " floor:", analysis.minimum_p(len(t), len(c)))
# ```
#
# The gap is enormous — many orders of magnitude — and only one of the two numbers is
# about the treatment.
#
# Cells within a well were pipetted together, dosed from the same drop, incubated in the
# same corner of the same plate and imaged in the same session. If that well went wrong,
# all of its cells went wrong together. Counting them as independent measures how many
# cells you imaged, not what the compound did.
#
# The well-level p-value will sit on the floor set by having three wells. That is not a
# weaker result; it is the honest one.
# :::

# %% [markdown]
# ---
#
# **Next:** [09 · PAGA](09_paga.ipynb).

# %% [markdown]
# ### 4. What does the projection do with a cell it has never seen?
#
# Keep the distance to the nearest labelled neighbour for every cell on the plate. Are the
# treated cells further from the reference than the control cells are? Which conditions sit
# furthest out?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# from sklearn.neighbors import NearestNeighbors
# tree = NearestNeighbors(n_neighbors=1).fit(space)
# distance, _ = tree.kneighbors(np.asarray(full[:, identity].X))
# out = pd.DataFrame({"d": distance.ravel(),
#                     "condition": full.obs.condition.astype(str).values})
# print(out.groupby("condition").d.median().sort_values(ascending=False).head(6).round(3))
# ```
#
# A condition sitting far from the reference is telling you the projection is extrapolating
# for it, and its labels should be read as "nearest available name" rather than as an
# identification.
#
# There is no threshold that is correct here. What matters is reporting the distance
# alongside the labels, so a reader can see which assignments the reference actually
# supports — the same habit as reporting the p-value floor next to a p-value.
# :::
