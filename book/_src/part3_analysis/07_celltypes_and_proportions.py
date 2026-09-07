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
# # 07 · Cell types and proportions
#
# In this notebook you will:
#
# - cluster cells on the **identity** panel to find cell states
# - read a cluster's identity off its marker profile, and spot one that is an artefact
# - watch a differentiated state emerge over 36–84 hours
# - ask which perturbations block or promote it
# - and settle, with a number, **what the replicate unit is**
#
# The four theme chapters asked how much of a marker a cell has. This one asks a
# different question: **what kind of cell is it, and how many of each kind are there?**

# %%
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

sys.path.insert(0, str(Path.cwd().parents[1] / "src"))

from mcs2026 import analysis, panels, plotting
from mcs2026.config import H5AD_SLIM

plotting.set_style()
pd.set_option("display.width", 140)

cells = sc.read_h5ad(H5AD_SLIM.with_name("mcs2026_qc.h5ad"))
print(f"{cells.n_obs:,} cells")

# %% [markdown]
# ## 1. The identity panel
#
# Seven transcription factors, chosen because they mark the lineages a naive human
# embryonic stem cell can become:
#
# | markers | lineage |
# |---|---|
# | Oct4, Nanog, Sox2 | **pluripotent** — the starting state |
# | GATA3 | **trophectoderm** |
# | GATA4, GATA6, SOX17 | **hypoblast** (primitive endoderm) |

# %%
identity = panels.markers("identity")
columns = [
    cells.var.index[(cells.var.marker == marker)
                    & (cells.var.statistic == "mean_intensity")
                    & (cells.var.family == "Intensity")][0]
    for marker in identity
]
pd.DataFrame({"marker": identity, "column": columns,
              "round": cells.var.loc[columns, "round"].values})

# %% [markdown]
# ## 2. Scale within timepoint
#
# Chapter 02 established that intensities are not comparable across imaging rounds. Here
# there is a second reason to be careful: we want clusters that describe **cell state**,
# not **timepoint**. If the four timepoints sit at different absolute levels, clustering
# will happily hand back four clusters that are really the four timepoints.
#
# So each marker is z-scored *within* each timepoint.

# %%
raw = pd.DataFrame(np.log2(np.asarray(cells[:, columns].X) + 1), columns=identity)
timepoint = cells.obs.timepoint_h.astype(int).values

scaled = raw.copy()
for value in [36, 48, 60, 84]:
    mask = timepoint == value
    block = raw.loc[mask]
    scaled.loc[mask] = (block - block.mean()) / block.std()

scaled.describe().loc[["mean", "std"]].round(2)

# %% [markdown]
# ## 3. Cluster
#
# A neighbour graph on seven markers, then Leiden. The resolution controls how many
# clusters you get, and it is a **choice**, not a result — see the exercises.

# %%
subset = ad.AnnData(scaled.values.astype("float32"))
subset.var_names = identity
sc.pp.neighbors(subset, n_neighbors=15, use_rep="X", random_state=0)
sc.tl.leiden(subset, resolution=0.1, flavor="igraph", n_iterations=2, random_state=0)

clusters = subset.obs.leiden.astype(str).values
pd.Series(clusters).value_counts().sort_index().rename("cells")

# %% [markdown]
# ## 4. What is each cluster?
#
# A cluster number means nothing. Read the marker profile and give it a name.

# %%
profile = scaled.groupby(clusters).mean()
profile.index.name = "cluster"
profile.round(2)

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.2))
im = ax.imshow(profile.values, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
ax.set(yticks=range(len(profile)), yticklabels=profile.index, ylabel="cluster")
ax.set_xticks(range(len(identity)))
ax.set_xticklabels(identity, rotation=45, ha="right")
ax.set_title("Mean marker z-score per cluster")
fig.colorbar(im, ax=ax, shrink=0.85, label="z")
fig.tight_layout()

# %% [markdown]
# Reading the rows:
#
# - one cluster is **high on Oct4, Nanog and Sox2** — still pluripotent
# - one is **high on GATA3** with pluripotency low — trophectoderm-like
# - one is **high on GATA4 and SOX17** — hypoblast
# - and one is tiny, with a single marker at roughly −4.6
#
# That last one deserves attention.

# %%
sizes = pd.Series(clusters).value_counts()
suspicious = profile.abs().max(axis=1).idxmax()
print(f"cluster {suspicious}: {sizes[suspicious]} cells "
      f"({sizes[suspicious] / len(clusters):.3%} of the data)")
profile.loc[[suspicious]].round(2)

# %% [markdown]
# :::{warning}
# **This is not a cell state.** A few dozen cells, defined by one marker being extremely
# *low* rather than any marker being high, is the signature of a technical failure — a
# field where that staining round did not work, or a registration slip.
#
# A cluster is only a cell type if you can say what it is *positive* for. Drop it, and
# say that you did.
# :::

# %%
state_names = {}
for cluster in profile.index:
    row = profile.loc[cluster]
    if sizes[cluster] < 0.005 * len(clusters):
        state_names[cluster] = "artefact"
    elif row[["Oct4", "Nanog", "Sox2"]].mean() > 0.3:
        state_names[cluster] = "Pluripotent"
    elif row[["GATA4", "SOX17"]].mean() > 0.5:
        state_names[cluster] = "Hypoblast"
    else:
        state_names[cluster] = "TE-like"

state = pd.Series(clusters).map(state_names).values
keep = state != "artefact"
print(f"dropped {(~keep).sum()} artefact cells")
pd.Series(state[keep]).value_counts().rename("cells")

# %% [markdown]
# ## 5. The hypoblast emerges

# %%
by_time = pd.crosstab(timepoint[keep], state[keep], normalize="index")
by_time.index.name = "hours"
(100 * by_time).round(1)

# %%
fig, ax = plt.subplots(figsize=(6, 3.6))
for column in by_time.columns:
    ax.plot(by_time.index, 100 * by_time[column], marker="o", label=column)
ax.set(xlabel="hours", ylabel="% of cells", title="Cell-state composition over time")
ax.set_xticks(by_time.index)
ax.legend()

# %% [markdown]
# The hypoblast fraction climbs steadily while the culture ages. That is the experiment
# working: naive cells left in these conditions specify primitive endoderm, and by 84
# hours a substantial minority have done so.
#
# Everything that follows asks **which perturbations change that**.

# %% [markdown]
# ## 6. Proportions per condition — at the well level
#
# Here is the step that matters. We do **not** pool all cells of a condition together.
# We compute the composition **of each well**, because the well is what was
# independently treated.

# %%
frame = pd.DataFrame({
    "well": cells.obs.well.astype(str).values,
    "condition": cells.obs.condition.astype(str).values,
    "timepoint": timepoint,
    "state": state,
})[keep]

proportions = (
    frame.groupby(["condition", "timepoint", "well"])["state"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)
print(f"{len(proportions)} wells")
proportions.head()

# %%
ranking = proportions.groupby("condition")["Hypoblast"].agg(["mean", "count"])
ranking["mean"] = (100 * ranking["mean"]).round(2)
ranking = ranking.sort_values("mean").rename(columns={"mean": "hypoblast %", "count": "wells"})
ranking

# %%
fig, ax = plt.subplots(figsize=(7, 5))
control = ranking.loc["DMSO", "hypoblast %"]
colors = ["firebrick" if v < control else "seagreen" for v in ranking["hypoblast %"]]
ax.barh(range(len(ranking)), ranking["hypoblast %"], color=colors)
ax.axvline(control, color="black", linestyle="--", linewidth=1, label="DMSO")
ax.set(yticks=range(len(ranking)), yticklabels=ranking.index,
       xlabel="hypoblast cells (%)", title="Which conditions block or promote hypoblast?")
ax.invert_yaxis(); ax.legend()

# %% [markdown]
# A clear, ordered result:
#
# - **MK-2206 (AKT inhibitor) has the lowest hypoblast fraction of all** — well below
#   DMSO. INK128 (mTOR) and PF-4708671 (S6K) sit with it.
# - **Retinoic acid has the highest** — well above DMSO.
#
# So the PI3K–AKT–mTOR axis that chapters 03 and 05 found to be pharmacologically
# engaged is the same axis that decides whether these cells specify hypoblast. The
# marker-level result and the composition-level result are the same story told twice.

# %%
for condition in ["MK-2206", "Sapanisertib/INK128", "RA"]:
    print(f"\n{condition} vs DMSO — hypoblast %, per timepoint")
    rows = []
    for value in [36, 48, 60, 84]:
        block = proportions[proportions.timepoint == value]
        treated = block.loc[block.condition == condition, "Hypoblast"]
        reference = block.loc[block.condition == "DMSO", "Hypoblast"]
        rows.append({
            "timepoint": value,
            "treated %": round(100 * treated.mean(), 2),
            "DMSO %": round(100 * reference.mean(), 2),
            "n_wells": len(treated),
            "p": round(stats.mannwhitneyu(treated, reference).pvalue, 4),
            "p_floor": round(analysis.minimum_p(len(treated), len(reference)), 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# At 84 hours the effect is large: DMSO reaches about 19% hypoblast, MK-2206 under 3%,
# retinoic acid over 30%.

# %% [markdown]
# ## 7. The replicate unit, settled
#
# Take the single clearest comparison — MK-2206 against DMSO at 84 hours — and test it
# two ways.

# %%
late = frame[frame.timepoint == 84]
treated_cells = (late[late.condition == "MK-2206"].state == "Hypoblast").astype(int)
control_cells = (late[late.condition == "DMSO"].state == "Hypoblast").astype(int)

table = np.array([
    [treated_cells.sum(), len(treated_cells) - treated_cells.sum()],
    [control_cells.sum(), len(control_cells) - control_cells.sum()],
])
cell_p = stats.chi2_contingency(table)[1]

block = proportions[proportions.timepoint == 84]
treated_wells = block.loc[block.condition == "MK-2206", "Hypoblast"]
control_wells = block.loc[block.condition == "DMSO", "Hypoblast"]
well_p = stats.mannwhitneyu(treated_wells, control_wells).pvalue

print(f"counting CELLS:  {len(treated_cells)} vs {len(control_cells)}   p = {cell_p:.2e}")
print(f"counting WELLS:  {len(treated_wells)} vs {len(control_wells)}     p = {well_p:.4f}")
print(f"\nsmallest p the well-level design can produce: "
      f"{analysis.minimum_p(len(treated_wells), len(control_wells)):.4f}")

# %% [markdown]
# :::{important}
# **The same comparison, off by sixteen orders of magnitude.**
#
# The cell-level p-value treats every cell as an independent experiment. They are not:
# cells in one well were pipetted together, treated with the same drop of compound,
# incubated in the same corner of the same plate and imaged in the same session. If that
# well was mis-pipetted, all several hundred of its cells are wrong together.
#
# The experiment has **3 treated wells**. That is the amount of independent evidence
# available, and no amount of imaging can increase it. The well-level p of 0.0357 is not
# a weaker result — it is the honest one, and it is sitting exactly on the floor set by
# having three wells.
#
# Cell-level statistics are not useless: they describe the *distribution within* a
# condition, which is the thing imaging is uniquely good at. But they cannot tell you
# that a **treatment** did something. Only replicated treatments can do that.
# :::

# %%
fig, ax = plt.subplots(figsize=(6.5, 4))
conditions = ["DMSO", "MK-2206", "Sapanisertib/INK128", "RA"]
for i, condition in enumerate(conditions):
    values = 100 * block.loc[block.condition == condition, "Hypoblast"]
    ax.scatter(np.full(len(values), i) + np.random.default_rng(0).normal(0, 0.04, len(values)),
               values, s=60, zorder=3, edgecolor="black", linewidth=0.5)
    ax.hlines(values.mean(), i - 0.22, i + 0.22, color="black", linewidth=2, zorder=4)
ax.set(xticks=range(len(conditions)),
       xticklabels=[c.split("/")[0][:12] for c in conditions],
       ylabel="hypoblast cells (%)", title="84 h — every point is one well")
ax.set_ylim(bottom=0)

# %% [markdown]
# **Plot the wells.** Three points and a mean say honestly how much evidence there is.
# A bar chart with an error bar computed over 40,000 cells would look far more
# convincing and would be describing something else entirely.

# %% [markdown]
# ## 8. Save

# %%
cells.obs["cell_state"] = pd.Categorical(
    np.where(keep, state, "artefact"),
    categories=["Pluripotent", "TE-like", "Hypoblast", "artefact"],
)
cells.write_h5ad(H5AD_SLIM.with_name("mcs2026_states.h5ad"), compression="gzip")
proportions.to_parquet(H5AD_SLIM.with_name("mcs2026_proportions.parquet"))
print("saved cell states and well-level proportions")

# %% [markdown]
# ---
#
# ## Exercises
#
# ### 1. Resolution is a choice
#
# Re-run the clustering at resolution 0.05, 0.1 and 0.3. How many clusters do you get,
# and do the extra ones at 0.3 have interpretable marker profiles? How would you decide?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# for resolution in [0.05, 0.1, 0.3]:
#     sc.tl.leiden(subset, resolution=resolution, key_added=f"r{resolution}",
#                  flavor="igraph", n_iterations=2, random_state=0)
#     labels = subset.obs[f"r{resolution}"].astype(str).values
#     print(f"\nresolution {resolution}: {len(set(labels))} clusters")
#     print(scaled.groupby(labels).mean().round(2))
# ```
#
# At 0.3 you get around ten clusters, and several are hard to name — they differ by
# degree rather than by which markers are on. That is the sign you have gone past what
# seven markers can support.
#
# The honest test is not a metric but a question: **can you say what each cluster is
# positive for?** If not, either lower the resolution or measure more markers. Do not
# report clusters you cannot name.
# :::

# %% [markdown]
# ### 2. Does the artefact cluster change any conclusion?
#
# Repeat the MK-2206 comparison keeping the artefact cluster. Does the answer move? Was
# dropping it important, or merely tidy?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# It barely moves — the cluster is a few dozen cells out of tens of thousands.
#
# That is worth knowing, and it is not an argument for leaving it in. It happens to be
# harmless here because it is tiny and spread across conditions. Had that staining round
# failed in one *block* of the plate, the same artefact would sit in one timepoint and
# would look exactly like a treatment effect. You cannot tell which case you are in
# without looking, which is the reason to look.
# :::

# %% [markdown]
# ### 3. Does composition explain the marker results?
#
# Chapter 03 found that MK-2206 lowers p-S6. This chapter finds it lowers the hypoblast
# fraction. Are these the same fact? Compute the mean p-S6 **within each cell state
# separately** and see whether MK-2206 still lowers it inside the pluripotent cells.

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# ps6 = cells.var.index[(cells.var.marker == "p-S6")
#                       & (cells.var.statistic == "mean_intensity")][0]
# by_state = pd.DataFrame({
#     "value": np.log2(np.asarray(cells[:, ps6].X).ravel() + 1),
#     "state": cells.obs.cell_state.astype(str).values,
#     "condition": cells.obs.condition.astype(str).values,
#     "well": cells.obs.well.astype(str).values,
# })
# per_well = by_state.groupby(["state", "condition", "well"])["value"].mean().reset_index()
# print(per_well[per_well.condition.isin(["DMSO", "MK-2206"])]
#       .groupby(["state", "condition"])["value"].mean().unstack().round(2))
# ```
#
# This is a **composition versus per-cell** question, and it comes up constantly. If a
# treatment changes the mixture of cell types, any whole-population average changes too
# — without any individual cell having changed.
#
# Here p-S6 stays lower under MK-2206 *within* the pluripotent cells, so the drug is
# doing something to cells directly, not only shifting the mixture. Had the effect
# vanished inside every state, the correct conclusion would have been the opposite.
# :::

# %% [markdown]
# ---
#
# **Next:** [08 · Integration](08_integration.ipynb) — do the four themes agree?
