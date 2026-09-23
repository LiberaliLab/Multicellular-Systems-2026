# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python (MCS 2026)
#     language: python
#     name: mcs2026
# ---

# %% [markdown]
# # 00 · The whole dataset, one analysis
#
# Part 3 taught the techniques and stopped before using them on a question. This chapter
# uses all of them, on all of the data, to answer the one the experiment was designed for:
# **what did each compound do?**
#
# | | |
# |---|---|
# | **1** | Check what you were handed |
# | **2** | The whole experiment in one heatmap |
# | **3** | Which markers move together? |
# | **4** | Rank the effects — and read the p-values honestly |
# | **5** | Markers across the four timepoints |
# | **6** | The embeddings, coloured by treatment |
# | **7** | Trajectories — does a treatment move cells along the axis? |
# | **7b** | Topology — does a treatment change the *shape*, not just the counts? |
# | **8** | Proportions, and the replicate unit settled |
#
# Nothing here needs a step that Part 3 did not cover. What is new is that the answers are
# stated.

# %%
from math import comb
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats

plt.rcParams.update({          # the house style, no package needed
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

def panel_grid(n, *, ncols=3, size=(3.6, 3.0)):
    """A figure with `n` axes on a grid, the unused ones removed."""
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(size[0] * ncols, size[1] * nrows))
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.remove()
    return fig, flat[:n]
pd.set_option("display.width", 150)

# The tables live here on Euler. Change this line if your copy is elsewhere.
DATA = Path("/cluster/project/mcsliberali/data_mcs_2026")

cells = sc.read_h5ad(DATA / "mcs2026_intensity_downstream.h5ad")
wells = (cells.to_df()
         .groupby([cells.obs.condition.astype(str), cells.obs.timepoint_h.astype(int),
                   cells.obs.well.astype(str)], observed=True).mean()
         .rename_axis(["condition", "timepoint", "well"]).reset_index())
sketch = sc.read_h5ad(DATA / "mcs2026_sketch.h5ad")
names = [c for c in wells.columns if c not in ("condition", "timepoint", "well")]

print(f"  wells : {len(wells)} x {len(names)} markers, in control-SD units")
print(f"  cells : {cells.n_obs:,} x {cells.n_vars}")
print(f"  sketch: {sketch.n_obs:,} cells, {list(sketch.obsm)}")

def effect_table(wells, markers, control="DMSO", by_timepoint=True):
    """Mean shift from the control, per condition and marker, in control SDs.

    The control is subtracted explicitly. Skipping it looks safe -- the
    normalisation puts the controls near zero -- but that only holds at the one
    timepoint the origin came from; everywhere else a plain group mean would
    report the controls' own development as a treatment effect.
    """
    if by_timepoint:
        table = wells.groupby(["condition", "timepoint"], observed=True)[markers].mean()
        reference = (wells[wells.condition == control]
                     .groupby("timepoint", observed=True)[markers].mean())
        matched = reference.reindex(table.index.get_level_values("timepoint"))
        return (table - matched.to_numpy()).drop(index=control, level=0, errors="ignore")
    table = wells.groupby("condition", observed=True)[markers].mean()
    return (table - wells.loc[wells.condition == control, markers].mean()
            ).drop(index=control, errors="ignore")

def rank_effects(wells, markers, control="DMSO", drop=None):
    """Every condition x marker pair, ranked by absolute shift.

    `p_floor` is the smallest p-value this design can produce: with n treated
    and m control wells there are C(n+m, n) orderings, so a two-sided
    Mann-Whitney can never go below 2/C(n+m, n). A row sitting at the floor is
    not more significant than another row at the floor, whatever its effect size.
    """
    frame = wells if drop is None else wells[wells.condition != drop]
    reference = frame[frame.condition == control]
    rows = []
    for condition, block in frame.groupby("condition", observed=True):
        if condition == control:
            continue
        for marker in markers:
            treated, base = block[marker].dropna(), reference[marker].dropna()
            if len(treated) < 3 or len(base) < 3:
                continue
            rows.append({"condition": condition, "marker": marker,
                         "shift": treated.mean() - base.mean(), "n_wells": len(treated),
                         "p": stats.mannwhitneyu(treated, base).pvalue,
                         "p_floor": 2 / comb(len(treated) + len(base), len(treated))})
    out = pd.DataFrame(rows)
    out["abs_shift"] = out["shift"].abs()
    out["at_p_floor"] = np.isclose(out["p"], out["p_floor"])
    return out.sort_values("abs_shift", ascending=False).reset_index(drop=True)

def compare_to_control(wells, marker, condition, control="DMSO"):
    """One marker, one condition, per timepoint, against the control wells."""
    rows = []
    for timepoint, block in wells.groupby("timepoint", observed=True):
        treated = block.loc[block.condition == condition, marker].dropna()
        base = block.loc[block.condition == control, marker].dropna()
        if len(treated) < 2 or len(base) < 2:
            continue
        rows.append({"timepoint": timepoint, "n_treated": len(treated),
                     "n_control": len(base),
                     "shift": round(treated.mean() - base.mean(), 2),
                     "p": round(stats.mannwhitneyu(treated, base).pvalue, 4),
                     "p_floor": round(2 / comb(len(treated) + len(base), len(treated)), 4)})
    return pd.DataFrame(rows)

OUTLIER = cells.obs.condition[cells.obs.is_outlier_condition].astype(str).iloc[0]

def transfer_values(source, values, target, k=15):
    """Average a continuous value over the k nearest source cells."""
    model = KNeighborsRegressor(n_neighbors=min(k, len(source)))
    model.fit(source, np.asarray(values, dtype="float64"))
    return model.predict(target)



# %% [markdown]
# ## 1 · Check what you were handed
#
# Five cells, before anything else. Not a repeat of Stage 1 — a check that the file in
# front of you is the one Stage 1 described, which is the cheapest way to catch an analysis
# built on the wrong version of a table.

# %%
for key, value in cells.uns["provenance"].items():
    print(f"  {key:18s} {value}")

# %%
pd.DataFrame({
    "wells": [cells.obs.well.nunique(), wells.well.nunique()],
    "conditions": [cells.obs.condition.nunique(), wells.condition.nunique()],
    "timepoints": [cells.obs.timepoint_h.nunique(), wells.timepoint.nunique()],
    "markers": [cells.n_vars, len(names)],
    "DMSO present": ["DMSO" in set(cells.obs.condition.astype(str)),
                     "DMSO" in set(wells.condition)],
}, index=["cells", "wells"])

# %% [markdown]
# And the one numerical check that the normalisation is still what it claims: control cells
# must sit at 0 with a spread of 1, within every timepoint.

# %%
controls = (cells.obs.condition.astype(str) == "DMSO").values
timepoint = cells.obs.timepoint_h.astype(int).values
X = np.asarray(cells.X)
pd.DataFrame([
    {"timepoint": value,
     "control cells": int((controls & (timepoint == value)).sum()),
     "mean": round(float(X[controls & (timepoint == value)].mean()), 3),
     "SD": round(float(X[controls & (timepoint == value)].std(ddof=1)), 3)}
    for value in sorted(set(timepoint))
]).set_index("timepoint")

# %% [markdown]
# Everything below works at the **well** level unless it says otherwise. Section 8
# establishes why with a number: 65,000 cells are not 65,000 experiments.

# %% [markdown]
# ## 2 · The whole experiment in one heatmap
#
# 17 conditions × 38 markers, each cell of the grid a mean shift from DMSO in control-well
# standard deviations. PMA is excluded — [Step 16](../part3_analysis/1_preparation/03_normalisation.ipynb)
# showed it moves everything, so leaving it in would flatten the colour scale for the rest.

# %%
effects = effect_table(wells, names, by_timepoint=False)
effects = effects.drop(index=OUTLIER, errors="ignore")
order_rows = effects.abs().mean(axis=1).sort_values(ascending=False).index
order_cols = effects.abs().mean(axis=0).sort_values(ascending=False).index
matrix = effects.loc[order_rows, order_cols]

fig, ax = plt.subplots(figsize=(13, 6))
limit = np.nanpercentile(np.abs(matrix.values), 98)
im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
ax.set(yticks=range(len(order_rows)), yticklabels=order_rows)
ax.set_xticks(range(len(order_cols)))
ax.set_xticklabels(order_cols, rotation=90, fontsize=8)
ax.set_title("Condition × marker — shift from DMSO (control SDs)")
fig.colorbar(im, ax=ax, shrink=0.75, label="control SDs")
fig.tight_layout()

# %% [markdown]
# Rows and columns are both sorted by how much they move, which does most of the work of
# reading a heatmap: the interesting conditions rise to the top, the responsive markers to
# the left, and the large blank region tells you honestly how much of a screen does nothing.

# %%
pd.DataFrame({
    "mean |shift|": effects.abs().mean(axis=1).sort_values(ascending=False).head(6).round(2),
}).join(pd.DataFrame({
    "strongest marker": effects.abs().idxmax(axis=1),
    "its shift": effects.apply(lambda r: r[r.abs().idxmax()], axis=1).round(2),
}))

# %% [markdown]
# ## 3 · Which markers move together?
#
# If two markers respond the same way to all 17 conditions, they are reporting on the same
# thing — whether or not the panel calls them different themes.

# %%
correlation = effects.corr(method="spearman")

from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

distance = 1 - correlation.fillna(0)
np.fill_diagonal(distance.values, 0)
tree = linkage(squareform(distance.values, checks=False), method="average")
order = [correlation.index[i] for i in leaves_list(tree)]

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(correlation.loc[order, order].values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
ax.set_title("Markers clustered by how they respond across conditions")
fig.colorbar(im, ax=ax, shrink=0.6, label="Spearman r")
fig.tight_layout()

# %% [markdown]
# Blocks along the diagonal are groups of markers that rise and fall together. Compare them
# with the panel definitions — where the data and the biology agree, you have a real
# module; where they disagree, ask whether the markers share an imaging round, a protein
# half-life or a dependence on growth rate rather than a pathway.

# %% [markdown]
# ## 4 · Rank the effects — and read the p-values honestly

# %%
ranked = rank_effects(wells, names, drop=OUTLIER)
ranked.head(15).round(3)

# %%
fig, ax = plt.subplots(figsize=(7.5, 5))
top = ranked.head(18)[::-1]
labels = [f"{r.condition.split('/')[0][:16]} · {r.marker}" for r in top.itertuples()]
# NOTE: `top.shift` is the DataFrame method, not the column. With a column named
# "shift" you must index it explicitly.
values = top["shift"].values
ax.barh(range(len(top)), values,
        color=["steelblue" if v > 0 else "firebrick" for v in values])
ax.set(yticks=range(len(top)), yticklabels=labels, xlabel="shift from DMSO (control SDs)",
       title="The 18 largest effects")
ax.axvline(0, color="0.3", lw=0.8)
ax.tick_params(axis="y", labelsize=8)

# %%
print(f"  {int(ranked.at_p_floor.sum())} of {len(ranked)} comparisons sit at the p-value floor")
print(f"  the floor here is {ranked.p_floor.min():.2e}  "
      f"(n = {ranked.n_wells.max()} treated vs 20 control)")

# %% [markdown]
# **None of them.** `rank_effects` pools the four timepoints, so each comparison has 12
# treated wells against 20 control ones — $\binom{32}{12}$ rank orderings, and a floor
# around $10^{-8}$. There is plenty of room, and the p-values mean something.
#
# Now do the same comparison one timepoint at a time.

# %%
per_timepoint = pd.concat([
    compare_to_control(wells, marker, condition).assign(
        condition=condition, marker=marker)
    for condition, marker in [("Sapanisertib/INK128", "p-S6"), ("MK-2206", "p-S6"),
                              ("RA", "SOX17"), ("Cycloheximide", "beta-Catenin")]
])
per_timepoint[["condition", "marker", "timepoint", "n_treated", "shift", "p", "p_floor"]]

# %% [markdown]
# :::{warning}
# **Every significant row reads 0.0357, and none can ever read less.** With 3 treated wells
# against 5 control ones there are $\binom{8}{3} = 56$ orderings, so the smallest
# achievable two-sided p-value is $2/56 = 0.0357$ — regardless of whether the shift is
# −1.3 or −7.8 control SDs.
#
# The floor is not a property of the biology. It is a property of **how you chose to
# group**: pool the timepoints and it vanishes, split them and it binds. This is the same
# ceiling as the √(n−1) limit in
# [Step 12](../part3_analysis/1_preparation/02_quality_control.ipynb), wearing different clothes.
#
# So: **sort by `shift`, report `p` beside it, and always say how many wells it came from.**
# And when you pool to gain power, say that too — pooling four timepoints averages a real
# early effect together with a real late non-effect, and reports something that happened at
# neither time.
# :::

# %% [markdown]
# ## 5 · Markers across the four timepoints
#
# The heatmap above averaged over time. Some effects only exist early, and averaging hides
# them.

# %%
by_time = effect_table(wells, names, by_timepoint=True)
by_time = by_time.drop(index=OUTLIER, level=0, errors="ignore")

focus = ["Sapanisertib/INK128", "MK-2206", "RA", "Cycloheximide"]
show = [m for m in ["p-S6", "Foxo1", "SOX17", "beta-Catenin", "LaminB1", "HSP90"] if m in names]

fig, axes = panel_grid(len(focus), ncols=2, size=(5.2, 3.2))
for ax, condition in zip(axes, focus):
    block = by_time.loc[condition]
    for marker in show:
        ax.plot(block.index, block[marker], marker="o", ms=4, label=marker)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set(title=condition.split("/")[0][:24], xlabel="hours",
           ylabel="shift (control SDs)")
    ax.set_xticks(block.index)
axes[0].legend(fontsize=7, ncols=2)
fig.tight_layout()

# %% [markdown]
# Read the shapes, not just the heights. An effect that grows with time is a different
# claim from one that peaks and fades — the second usually means the compound was consumed
# or the cells adapted, which is worth knowing before you design the next experiment.

# %% [markdown]
# ## 6 · The embeddings, coloured by treatment
#
# Stage 2 built its embeddings on the **controls**, so there are none for the plate. Build
# them here, with exactly the calls chapters 07 and 10 used — the point of learning a method
# somewhere safe is that it then transfers unchanged.
#
# On the **sketch**, not on all 653,000 cells: a neighbour graph over the full table is the
# one thing that genuinely does not fit, which is why
# [chapter 04](../part3_analysis/1_preparation/04_subsetting_and_sketching.ipynb) made one.

# %%

identity = [m for m in sketch.uns["panels"]["identity"] if m in set(sketch.var_names)]
sketch.obsm["X_identity"] = np.asarray(sketch[:, identity].X)

sc.pp.pca(sketch, n_comps=20, random_state=0)
sc.pp.neighbors(sketch, n_neighbors=15, n_pcs=15, random_state=0)
sc.tl.umap(sketch, random_state=0)

sc.pp.neighbors(sketch, n_neighbors=15, use_rep="X_identity",
                key_added="identity", random_state=0)
sc.tl.diffmap(sketch, n_comps=10, neighbors_key="identity")

print(f"  {sketch.n_obs:,} sketched cells from {sketch.obs.condition.nunique()} conditions")
print(f"  obsm: {list(sketch.obsm)}")

# %% [markdown]
# One panel per compound, each highlighted against every other cell in grey.

# %%
SHORT = {"Phorbol 12-myristate 13-acetate (PMA)": "PMA",
         "Sapanisertib/INK128": "INK128"}

umap = sketch.obsm["X_umap"]
condition = sketch.obs.condition.astype(str).values
focus = ["DMSO", "MK-2206", "Sapanisertib/INK128", "RA", "IL6",
         "Phorbol 12-myristate 13-acetate (PMA)"]

fig, axes = panel_grid(len(focus), ncols=3, size=(4.0, 3.6))
for ax, name in zip(axes, focus):
    mask = condition == name
    ax.scatter(umap[~mask, 0], umap[~mask, 1], c="0.88", s=2)
    ax.scatter(umap[mask, 0], umap[mask, 1], c="firebrick", s=2.5)
    ax.set(title=f"{SHORT.get(name, name)}  ({mask.sum():,} cells)", xticks=[], yticks=[])
fig.tight_layout()

# %% [markdown]
# :::{warning}
# **Do not read occupancy off this figure**, however much it invites you to. The panels
# hold different numbers of cells — 877 for DMSO against 237 for PMA — so "denser" and
# "more cells" are confounded in every comparison you might make by eye. And a few hundred
# red points scattered on a grey map will look different from another few hundred whatever
# the truth is.
# :::
#
# What it is good for is generating candidates. PMA occupies a band the others do not, and
# MK-2206 looks thin along the bottom edge — the region
# [chapter 08](../part3_analysis/2_controls/08_cell_type_annotation.ipynb) named
# hypoblast. Both are worth checking. Section 8 checks them, by counting, at the well
# level, which is the only form in which either would count.
#
# The same view on the diffusion map, which orders rather than groups:

# %%
diffmap = sketch.obsm["X_diffmap"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
for ax, name in zip(axes, ["DMSO", "MK-2206", "RA"]):
    mask = condition == name
    ax.scatter(diffmap[~mask, 1], diffmap[~mask, 2], c="0.88", s=2)
    s = ax.scatter(diffmap[mask, 1], diffmap[mask, 2],
                   c=np.asarray(sketch[:, "Oct4"].X).ravel()[mask], cmap="magma", s=3)
    ax.set(title=SHORT.get(name, name), xlabel="DC1", ylabel="DC2",
           xticks=[], yticks=[])
fig.colorbar(s, ax=axes, shrink=0.8, label="Oct4")

# %% [markdown]
# ### And the PCA
#
# [Chapter 06](../part3_analysis/2_controls/06_pca.ipynb) found PC1 to be overall
# brightness and left the question of what the components *separate* open. Answer it at the
# well level, where a point is an experiment rather than a cell:

# %%
space = ad.AnnData(wells[names].fillna(0).to_numpy(dtype="float32"))
sc.pp.pca(space, n_comps=4, random_state=0)
frame = pd.DataFrame(space.obsm["X_pca"][:, :2], columns=["PC1", "PC2"])
frame["condition"] = wells.condition.values
frame["timepoint"] = wells.timepoint.values

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
s = axes[0].scatter(frame.PC1, frame.PC2, c=frame.timepoint, cmap="viridis", s=45,
                    edgecolor="black", linewidth=0.3)
axes[0].set(title="wells, coloured by timepoint")
fig.colorbar(s, ax=axes[0], shrink=0.8, label="hours")

for name, colour in [("DMSO", "steelblue"),
                     ("Phorbol 12-myristate 13-acetate (PMA)", "firebrick")]:
    mask = frame.condition == name
    axes[1].scatter(frame.PC1[~mask], frame.PC2[~mask], c="0.85", s=35)
    axes[1].scatter(frame.PC1[mask], frame.PC2[mask], c=colour, s=55,
                    edgecolor="black", linewidth=0.3, label=SHORT.get(name, name))
axes[1].legend(fontsize=8); axes[1].set(title="the control and the outlier")

share = space.uns["pca"]["variance_ratio"]
for ax in axes:
    ax.set(xlabel=f"PC1 ({100 * share[0]:.0f}%)", ylabel=f"PC2 ({100 * share[1]:.0f}%)")
fig.tight_layout()

# %% [markdown]
# At the well level PC1 is no longer brightness — the normalisation removed that when it
# expressed every well as a distance from its own controls. What is left is the axis that
# separates **how far a well moved from DMSO**, and PMA sits at the end of it alone. That
# is the same conclusion Step 16 reached from a single marker, arrived at from the other
# direction.

# %% [markdown]
# ## 7 · Trajectories — does a treatment move cells along the axis?
#
# [Chapter 10](../part3_analysis/2_controls/10_diffusion_map.ipynb) built a pseudotime on
# the **controls** and showed that the axis it sits on means the same thing across all
# eighteen conditions. That is what makes it usable here: an untreated coordinate, against
# which treated cells can be placed.
#
# **Carry it to every cell**, the same k-nearest-neighbour way the labels were carried —
# and for the same reason. Measuring the treatments against a reference built from the
# controls is the whole design; recomputing a pseudotime *within* each condition would give
# every condition its own axis and make them incomparable by construction.

# %%
controls = sc.read_h5ad(DATA / "mcs2026_controls_downstream.h5ad")
pseudotime = transfer_values(
    controls.obsm["X_identity"],
    controls.obs.dpt_pseudotime.values,
    np.asarray(cells[:, identity].X),
)
cells.obs["pseudotime"] = pseudotime
print(f"  reference: {controls.n_obs:,} control cells")
print(f"  transferred to {len(pseudotime):,} cells, "
      f"range {pseudotime.min():.3f}–{pseudotime.max():.3f}")

track = pd.DataFrame({
    "pseudotime": pseudotime,
    "condition": cells.obs.condition.astype(str).values,
    "timepoint": cells.obs.timepoint_h.astype(int).values,
    "well": cells.obs.well.astype(str).values,
    "state": cells.obs.cell_state.astype(str).values,
})
track.groupby("state").pseudotime.median().round(3).rename("median")

# %% [markdown]
# **Then choose how to summarise it — and this is where the analysis is won or lost.**
# Look at the distributions at 84 hours before picking a statistic:

# %%
late = track[track.timepoint == 84]
threshold = track.loc[track.state == "Hypoblast", "pseudotime"].quantile(0.10)

fig, ax = plt.subplots(figsize=(7.5, 4))
for name, colour in [("MK-2206", "firebrick"), ("DMSO", "0.35"), ("RA", "seagreen")]:
    values = late.loc[late.condition == name, "pseudotime"]
    ax.hist(values, bins=60, range=(0, 0.7), histtype="step", lw=1.8,
            density=True, color=colour, label=f"{name} (n={len(values):,})")
ax.axvline(threshold, color="black", ls="--", lw=1)
ax.text(threshold, ax.get_ylim()[1] * 0.95, "  hypoblast threshold", fontsize=8, va="top")
ax.set(xlabel="pseudotime", ylabel="density", title="84 hours — the same axis, three conditions")
ax.legend()
fig.tight_layout()

# %%
summary = late.groupby("condition").pseudotime.agg(
    mean="mean", median="median",
    p90=lambda s: s.quantile(0.90),
    past_threshold=lambda s: (s > threshold).mean())
summary.loc[["DMSO", "MK-2206", "RA"]].round(3)

# %% [markdown]
# :::{important}
# **The median does not move; the tail does.** Between DMSO and MK-2206 the median
# pseudotime changes by less than 0.01, while the fraction of cells past the hypoblast
# threshold falls almost four-fold.
#
# That is not a contradiction — it is what a **mixture** looks like. The compound is not
# pushing every cell backwards along the axis. It is stopping a minority of cells from ever
# reaching the far end, and leaving the majority exactly where they were.
#
# So the summary statistic decides whether you see anything:
#
# | statistic | what it reports |
# |---|---|
# | median | nothing — the bulk of cells did not move |
# | mean | a shift, but only because the tail drags it |
# | fraction past a threshold | the actual effect, at its actual size |
#
# **Choose the statistic from the shape of the distribution, not from habit.** And having
# arrived at "count the cells past a threshold", notice that this is just counting cell
# states with extra steps — which is section 8, done properly.
# :::

# %%
per_well = late.groupby(["condition", "well"]).pseudotime.apply(
    lambda s: (s > threshold).mean()).reset_index(name="past_threshold")
control_wells = per_well.loc[per_well.condition == "DMSO", "past_threshold"]

pd.DataFrame([
    {"condition": name,
     "% past threshold": round(100 * block.past_threshold.mean(), 1),
     "DMSO %": round(100 * control_wells.mean(), 1),
     "n_wells": len(block),
     "p": round(stats.mannwhitneyu(block.past_threshold, control_wells).pvalue, 4),
     "p_floor": round(2 / comb(len(block) + len(control_wells), len(block)), 4)}
    for name, block in per_well.groupby("condition") if name != "DMSO"
]).sort_values("% past threshold").set_index("condition").round(3)

# %% [markdown]
# ## 7b · Topology — does a treatment change the *shape*?
#
# Everything so far has counted things: how much of a marker, how many cells past a
# threshold, how large a fraction. [Chapter 09](../part3_analysis/2_controls/09_paga.ipynb)
# offers a different kind of measurement.
#
# On the controls, PAGA found a path with pluripotency in the middle and two differentiated
# ends, and the hypoblast cluster joined to exactly one neighbour — the small population
# co-expressing pluripotency and GATA4/SOX17. A compound could change that picture in ways
# no proportion would reveal: **empty the bridge while leaving both endpoints populated**,
# or cut one route and leave the other.
#
# Build the same abstraction per condition and compare.

# %%
sc.tl.leiden(sketch, resolution=0.3, key_added="fine", neighbors_key="identity",
             flavor="igraph", n_iterations=2, random_state=0)

fine_profile = (pd.DataFrame(sketch.obsm["X_identity"], columns=identity)
                .groupby(sketch.obs.fine.astype(str).values, observed=True).mean())
bridge = fine_profile[(fine_profile[["Oct4", "Nanog", "Sox2"]].mean(axis=1) > 0.3)
                      & (fine_profile[["GATA4", "SOX17"]].mean(axis=1) > 0.5)].index.tolist()
print(f"  {len(fine_profile)} clusters on the plate sketch")
print(f"  co-expressing (bridge) clusters: {bridge or 'none found'}")
fine_profile.round(2).join(sketch.obs.fine.value_counts().rename("cells"))

# %% [markdown]
# ### How full is the bridge, per condition?
#
# The abstraction is the same for everyone; what differs is how many cells each condition
# puts in each node. Count at the **well** level, as always.

# %%
occupancy = pd.DataFrame({
    "condition": sketch.obs.condition.astype(str).values,
    "well": sketch.obs.well.astype(str).values,
    "in_bridge": sketch.obs.fine.astype(str).isin(bridge).values,
})
per_well = (occupancy.groupby(["condition", "well"]).in_bridge.mean()
            .reset_index(name="bridge_fraction"))
control_wells = per_well.loc[per_well.condition == "DMSO", "bridge_fraction"]

bridge_table = pd.DataFrame([
    {"condition": name,
     "% of cells in the bridge": round(100 * block.bridge_fraction.mean(), 2),
     "DMSO %": round(100 * control_wells.mean(), 2),
     "n_wells": len(block),
     "p": round(stats.mannwhitneyu(block.bridge_fraction, control_wells).pvalue, 4),
     "p_floor": round(2 / comb(len(block) + len(control_wells), len(block)), 4)}
    for name, block in per_well.groupby("condition") if name != "DMSO"
]).sort_values("% of cells in the bridge").set_index("condition")
bridge_table

# %% [markdown]
# ### And the edges themselves
#
# A connectivity is a number, so it can be compared without ever being thresholded — which
# sidesteps the choice chapter 09 warned about. Rebuild PAGA within each condition and read
# the weight on the edge into the hypoblast cluster.

# %%
hypoblast = fine_profile[["GATA4", "SOX17"]].mean(axis=1).idxmax()
rows = []
for name in ["DMSO", "PBS", "MK-2206", "Sapanisertib/INK128", "RA", "IL6"]:
    block = sketch[sketch.obs.condition.astype(str) == name].copy()
    if block.obs.fine.nunique() < 3 or block.n_obs < 200:
        continue
    sc.pp.neighbors(block, n_neighbors=15, use_rep="X_identity", random_state=0)
    sc.tl.paga(block, groups="fine")
    weights = pd.DataFrame(block.uns["paga"]["connectivities"].toarray(),
                           index=block.obs.fine.cat.categories,
                           columns=block.obs.fine.cat.categories)
    rows.append({
        "condition": SHORT.get(name, name),
        "cells": block.n_obs,
        "bridge %": round(100 * block.obs.fine.astype(str).isin(bridge).mean(), 2),
        "hypoblast %": round(100 * (block.obs.fine.astype(str) == hypoblast).mean(), 2),
        "strongest edge into hypoblast": round(float(weights[hypoblast].max()), 3),
    })
pd.DataFrame(rows).set_index("condition")

# %% [markdown]
# :::{important}
# **The first two columns agree; the third is broken, and it is worth seeing why.**
#
# Bridge occupancy and hypoblast occupancy move together — the compounds with an empty
# bridge are the compounds with an empty destination. That is coherent, and the well-level
# table above puts a p-value on it.
#
# The **edge weight** does the opposite of what it should: MK-2206 and INK128, with three
# per cent of their cells in the hypoblast cluster, report the *strongest possible*
# connectivity into it. That is not a strong route. A PAGA connectivity is a ratio, and when
# a cluster holds a handful of cells almost every edge it has is a within-neighbourhood
# edge, so the ratio saturates. The `cells` column is the tell.
#
# So the honest report is: **bridge occupancy is the measurement, and the edge weights are
# not usable at this sample size.** Reaching for a topology statistic and finding it does
# not survive the cell counts you have is a normal outcome, and saying so is better than
# quoting it because it was computed.
#
# Two structural cautions besides. A connectivity has no well-level replicate as computed
# here — one number per condition, not three — so it could not carry a p-value even if it
# were stable. And the per-condition sketches hold a few hundred cells each, which is the
# root of the saturation above.
# :::

# %% [markdown]
# ## 8 · Proportions, and the replicate unit settled
#
# The most direct question in the whole experiment: **how many cells of each kind are
# there, and does the treatment change it?**
#
# Do not pool the cells of a condition together. Compute the composition **of each well**,
# because the well is what was independently treated.

# %%
proportions = (
    track.groupby(["condition", "timepoint", "well"])["state"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reset_index()
)
print(f"  {len(proportions)} wells x {len(track.state.unique())} states")
proportions.head()

# %% [markdown]
# ### The experiment working
#
# Before any compound: does the culture do what it is supposed to?

# %%
by_time = proportions.groupby("timepoint")[["Pluripotent", "TE-like", "Hypoblast"]].mean()

fig, ax = plt.subplots(figsize=(6.4, 3.8))
for column in by_time.columns:
    ax.plot(by_time.index, 100 * by_time[column], marker="o", label=column)
ax.set(xlabel="hours", ylabel="% of cells", title="Cell-state composition over time")
ax.set_xticks(by_time.index); ax.legend()
fig.tight_layout()

(100 * by_time).round(1)

# %% [markdown]
# The hypoblast fraction climbs from under 1% to about 14% while the culture ages. That is the
# experiment working: naive cells left in these conditions specify primitive endoderm, and
# by 84 hours a substantial minority have done so.
#
# Everything below asks **which perturbations change that**.

# %%
ranking = proportions.groupby("condition")["Hypoblast"].agg(["mean", "count"])
ranking["mean"] = (100 * ranking["mean"]).round(2)
ranking = ranking.sort_values("mean").rename(
    columns={"mean": "hypoblast %", "count": "wells"})

fig, ax = plt.subplots(figsize=(7, 5))
control = ranking.loc["DMSO", "hypoblast %"]
colours = ["firebrick" if v < control else "seagreen" for v in ranking["hypoblast %"]]
ax.barh(range(len(ranking)), ranking["hypoblast %"], color=colours)
ax.axvline(control, color="black", linestyle="--", linewidth=1, label="DMSO")
ax.set(yticks=range(len(ranking)), yticklabels=ranking.index,
       xlabel="hypoblast cells (%)", title="Which conditions block or promote hypoblast?")
ax.invert_yaxis(); ax.legend()
ranking

# %% [markdown]
# A clear, ordered result:
#
# - **MK-2206, the AKT inhibitor, has the lowest hypoblast fraction of all** — a fifth of
#   DMSO. INK128 (mTOR) and PF-4708671 (S6K) sit with it.
# - **Retinoic acid has the highest**, at nearly twice DMSO.
#
# So the PI3K–AKT–mTOR axis that section 4 found to be pharmacologically engaged is the
# same axis that decides whether these cells specify hypoblast. The marker-level result and
# the composition-level result are the same story told twice, which is the strongest form
# a screen result takes.

# %%
for name in ["MK-2206", "Sapanisertib/INK128", "RA"]:
    print(f"\n{name} vs DMSO — hypoblast %, per timepoint")
    rows = []
    for value in sorted(proportions.timepoint.unique()):
        block = proportions[proportions.timepoint == value]
        treated = block.loc[block.condition == name, "Hypoblast"]
        reference = block.loc[block.condition == "DMSO", "Hypoblast"]
        rows.append({
            "timepoint": value,
            "treated %": round(100 * treated.mean(), 2),
            "DMSO %": round(100 * reference.mean(), 2),
            "n_wells": len(treated),
            "p": round(stats.mannwhitneyu(treated, reference).pvalue, 4),
            "p_floor": round(2 / comb(len(treated) + len(reference), len(treated)), 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# ### The replicate unit, settled
#
# Take the single clearest comparison — MK-2206 against DMSO at 84 hours — and test it two
# ways.

# %%
final = track[track.timepoint == 84]
treated_cells = (final[final.condition == "MK-2206"].state == "Hypoblast").astype(int)
control_cells = (final[final.condition == "DMSO"].state == "Hypoblast").astype(int)
table = np.array([
    [treated_cells.sum(), len(treated_cells) - treated_cells.sum()],
    [control_cells.sum(), len(control_cells) - control_cells.sum()],
])
cell_p = stats.chi2_contingency(table)[1]

block = proportions[proportions.timepoint == 84]
treated_wells = block.loc[block.condition == "MK-2206", "Hypoblast"]
control_wells = block.loc[block.condition == "DMSO", "Hypoblast"]
well_p = stats.mannwhitneyu(treated_wells, control_wells).pvalue

print(f"counting CELLS:  {len(treated_cells):,} vs {len(control_cells):,}   p = {cell_p:.2e}")
print(f"counting WELLS:  {len(treated_wells)} vs {len(control_wells)}         p = {well_p:.4f}")
print(f"\nsmallest p the well-level design can produce: "
      f"{2 / comb(len(treated_wells) + len(control_wells), len(treated_wells)):.4f}")
print(f"effect size: {100*treated_wells.mean():.1f}% vs {100*control_wells.mean():.1f}% hypoblast")

# %% [markdown]
# :::{important}
# **The same comparison, off by nearly forty orders of magnitude.**
#
# The cell-level p-value treats every cell as an independent experiment. They are not:
# cells in one well were pipetted together, treated with the same drop of compound,
# incubated in the same corner of the same plate and imaged in the same session. If that
# well was mis-pipetted, all several hundred of its cells are wrong together.
#
# The experiment has **3 treated wells**. That is the amount of independent evidence
# available, and no amount of imaging can increase it. The well-level p of 0.0357 is not a
# weaker result — it is the honest one, and it sits exactly on the floor set by having
# three wells.
#
# Cell-level statistics are not useless: they describe the *distribution within* a
# condition, which is what imaging is uniquely good at, and section 7 is entirely built on
# one. But they cannot tell you that a **treatment** did something. Only replicated
# treatments can do that.
# :::

# %%
fig, ax = plt.subplots(figsize=(6.5, 4))
show = ["DMSO", "MK-2206", "Sapanisertib/INK128", "RA"]
jitter = np.random.default_rng(0)
for i, name in enumerate(show):
    values = 100 * block.loc[block.condition == name, "Hypoblast"]
    ax.scatter(np.full(len(values), i) + jitter.normal(0, 0.04, len(values)), values,
               s=60, zorder=3, edgecolor="black", linewidth=0.5)
    ax.hlines(values.mean(), i - 0.22, i + 0.22, color="black", linewidth=2, zorder=4)
ax.set(xticks=range(len(show)), xticklabels=[c.split("/")[0][:12] for c in show],
       ylabel="hypoblast cells (%)", title="84 h — every point is one well")
ax.set_ylim(bottom=0)
fig.tight_layout()

# %% [markdown]
# **Plot the wells.** Three points and a mean say honestly how much evidence there is. A
# bar chart with an error bar computed over 40,000 cells would look far more convincing and
# would be describing something else entirely.

# %% [markdown]
# ### Save
#
# The four theme chapters all start from this table.

# %%
proportions.to_parquet(DATA / "mcs2026_proportions.parquet")
cells.write_h5ad(DATA / "mcs2026_intensity.h5ad", compression="gzip")
print(f"  mcs2026_proportions.parquet — {len(proportions)} wells")
print(f"  mcs2026_intensity.h5ad — now carries obs['pseudotime']")

# %% [markdown]
# ## What the whole dataset says
#
# | | |
# |---|---|
# | **the screen mostly does nothing** | and that is the normal shape of a screen, not a failure |
# | **PMA moves everything** | which is why it is excluded from the colour scales, not from the data |
# | **the PI3K–AKT–mTOR axis is engaged** | p-S6 falls under INK128 and MK-2206, most strongly early |
# | **and it decides hypoblast** | MK-2206 a fifth of DMSO, RA nearly double |
# | **the evidence is 3 wells** | every p-value that matters sits on the floor that implies |
# | **and topology is a separate question** | the compounds that empty the bridge are the ones that empty the destination |
#
# The four chapters that follow ask the same questions of one panel each, in more detail.

# %% [markdown]
# ---
#
# ## Exercises
#
# ### 1. Which markers never move?
#
# Find the markers whose largest absolute shift across all conditions is under 1 control SD.
# Are they uninformative, or just not perturbed by anything on this plate?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# quiet = effects.abs().max(axis=0).sort_values()
# print(quiet.head(10).round(2))
# ```
#
# A flat marker is not a failed marker. It may be genuinely unaffected by all 18 compounds,
# or it may be noisy enough to swamp a real effect — the per-marker control spread from
# [Step 17](../part3_analysis/1_preparation/03_normalisation.ipynb) distinguishes the two. Check that before calling an antibody useless.
# :::

# %% [markdown]
# ### 2. Does the correlation structure match the panels?
#
# Cut the marker dendrogram into four groups and compare them with the four themes defined
# in `mcs2026.panels`. Where do they disagree?

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# ```python
# from scipy.cluster.hierarchy import fcluster
# groups = pd.Series(fcluster(tree, t=4, criterion="maxclust"), index=correlation.index)
# theme = {m: t for t in cells.uns["themes"] for m in cells.uns["panels"][t]}
# print(pd.crosstab(groups, pd.Series(theme).reindex(groups.index)))
# ```
#
# Expect partial agreement at best. Data-driven groups capture *response* similarity —
# shared turnover, shared growth dependence, shared imaging round — while the themes
# capture *function*. Neither is wrong; they answer different questions.
# :::

# %% [markdown]
# ### 3. Pick a condition and describe it
#
# Choose one compound. Using the heatmap, the ranking and the timecourse, write three
# sentences: what moved, by how much, and whether the timing makes sense for its mechanism.

# %% [markdown]
# :::{admonition} Solution
# :class: dropdown
#
# There is no single answer — this is the exercise that becomes a group project.
#
# A good answer names an effect size in control SDs rather than a p-value, says how many
# wells it rests on, and states one thing that would distinguish its explanation from an
# alternative. A weak answer says "significantly increased" and stops.
# :::

# %% [markdown]
# ---
#
# **Next:** [01 · Signaling](01_signaling.ipynb).
