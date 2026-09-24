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
# # 11 · Your turn
#
# Everything up to here was demonstrated on data the whole class shares. This chapter is
# the skeleton for doing it yourself, on your group's theme, and it deliberately stops
# short of any answer.
#
# Five steps. Each is one or two cells, and the whole thing runs in a few minutes — which
# is the point. Get a complete, honest analysis end to end first, then go deeper wherever
# it turns out to be interesting.
#
# | | |
# |---|---|
# | **1** | Write the question down |
# | **2** | Subset to your panel |
# | **3** | Look at it |
# | **4** | Test it — at the well level |
# | **5** | One figure someone else can read |
#
# :::{note}
# The worked example below runs on the **cell-cycle** panel — CyclinA2, p21 and Ki67 —
# which belongs to none of the four groups. Change `THEME` on the first line to your own
# and every cell after it still works.
# :::

# %%
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

plt.rcParams.update({          # the house style, no package needed
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
pd.set_option("display.width", 140)

# The tables live here on Euler. Change this line if your copy is elsewhere.
DATA = Path("/cluster/project/mcsliberali/data_mcs_2026")

THEME = "cell_cycle"          # <- signaling / mechanics / metabolism / organelles

cells = sc.read_h5ad(DATA / "mcs2026_intensity.h5ad")
wells = (cells.to_df()
         .groupby([cells.obs.condition.astype(str), cells.obs.timepoint_h.astype(int),
                   cells.obs.well.astype(str)], observed=True).mean()
         .rename_axis(["condition", "timepoint", "well"]).reset_index())
print(f"  cells: {cells.n_obs:,} x {cells.n_vars}   wells: {wells.shape[0]} rows")

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


# %% [markdown]
# ## 1 · Write the question down
#
# In one sentence, before any code. If you cannot state it in a sentence, the analysis will
# not have a conclusion either.
#
# A usable question names **what you are measuring**, **what you are comparing it to**, and
# **what would count as an answer**:
#
# > *Do the mTOR inhibitors reduce the fraction of proliferating cells relative to DMSO,
# > and does the effect grow between 36 and 84 hours?*
#
# Not: *"we will investigate the cell cycle panel."* That has no answer, so it cannot be
# wrong, so it cannot be right either.

# %%
QUESTION = ("Do any perturbations shift the cell-cycle markers relative to DMSO, "
            "and does the shift depend on timepoint?")
print(QUESTION)

# %% [markdown]
# ## 2 · Subset to your panel
#
# `resolve_panel` prints anything it could not find. Read that line — silence is how a typo
# becomes a missing marker nobody notices.

# %%
columns = [m for m in cells.uns["panels"][THEME] if m in set(cells.var_names)]
markers = [m for m in cells.uns["panels"][THEME] if m in set(wells.columns)]

print(f"  {THEME}: {len(columns)} markers -> {markers}")
cells.var.loc[columns, ["marker", "round", "channel", "failed"]]

# %% [markdown]
# ## 3 · Look at it
#
# Before any test. You are looking for two things: whether the markers separate anything at
# all, and whether something is obviously wrong.

# %%
mine = cells[:, columns].copy()
sc.pp.pca(mine, n_comps=min(10, len(columns) - 1))
sc.pp.neighbors(mine, n_neighbors=15, random_state=0)
sc.tl.umap(mine, random_state=0)

sc.pl.umap(mine, color=[*markers, "cell_state"], ncols=2, s=4, frameon=False,
           vmin="p2", vmax="p98")

# %% [markdown]
# Colour it by `cell_state` as well as by your markers. If your whole panel simply tracks
# the cell states from [chapter 08](2_controls/08_cell_type_annotation.ipynb), then
# your theme is telling you about the mixture of cell types rather than about anything
# happening inside them — which is a real finding, and a different one.

# %% [markdown]
# ## 4 · Test it — at the well level
#
# The picture above has tens of thousands of points and **three wells per condition per
# timepoint**. The wells are the evidence.
#
# `rank_effects` does the comparison for every condition × marker pair at once, and reports
# the smallest p-value the design can produce alongside each real one.

# %%
ranked = rank_effects(wells, markers, drop=OUTLIER)
ranked.head(10).round(3)

# %% [markdown]
# Three columns to read before the `shift`:
#
# - **`n_wells`** — how much evidence there is. Three is normal here; fewer means something
#   was dropped and you should find out what.
# - **`p_floor`** — the smallest p-value achievable. If `p` equals it, the test is
#   saturated: a bigger effect could not produce a smaller number.
# - **`at_p_floor`** — that comparison, done for you.
#
# Rank by **effect size in control SDs**, and use the p-value only to say whether the
# effect is distinguishable from nothing.

# %%
print(f"  pairs at the p-value floor: {ranked.at_p_floor.sum()} of {len(ranked)}")
print(f"  pairs beyond 2 control SDs: {(ranked.abs_shift > 2).sum()}")

# %% [markdown]
# Nothing is at the floor, because `rank_effects` pools the four timepoints — 12 treated
# wells against 20 controls, where the floor is around $5\times10^{-9}$ and never binds.
#
# Split the same comparison by timepoint and it binds immediately: 3 against 5 wells, floor
# $2/\binom{8}{3} = 0.0357$. Every "significant" row below reports exactly that number,
# and the shift column is the only thing distinguishing them.

# %%
top = ranked.iloc[0]
compare_to_control(wells, top.marker, top.condition)

# %% [markdown]
# **Which grouping is right depends on your question.** Pooling asks whether the compound
# does anything at all and has the power to answer; splitting asks when it does it and has
# almost none. Run both, and say which one the claim in your talk rests on.

# %% [markdown]
# ## 5 · One figure someone else can read
#
# Axis labels, units, and **every replicate visible**. Three points and a mean say honestly
# how much evidence there is; a bar with an error bar computed over 40,000 cells would look
# far more convincing and would be describing something else.

# %%
marker, condition = top.marker, top.condition
fig, ax = plt.subplots(figsize=(6.4, 4))
rng = np.random.default_rng(0)

for i, name in enumerate(["DMSO", condition]):
    for offset, hours in zip(np.linspace(-0.18, 0.18, 4), sorted(wells.timepoint.unique())):
        values = wells.loc[(wells.condition == name) & (wells.timepoint == hours), marker]
        x = i + offset + rng.normal(0, 0.015, len(values))
        ax.scatter(x, values, s=45, zorder=3, edgecolor="black", linewidth=0.4,
                   color=plt.cm.viridis(offset / 0.36 + 0.5),
                   label=f"{hours} h" if i == 0 else None)
        ax.hlines(values.mean(), i + offset - 0.05, i + offset + 0.05, color="black", lw=1.5)

ax.axhline(0, color="0.4", ls="--", lw=1)
ax.set(xticks=[0, 1], xticklabels=["DMSO", condition[:18]],
       ylabel=f"{marker} (control-well SDs)",
       title=f"{marker}: every point is one well")
ax.legend(title="fixed at", fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## Before you present
#
# A checklist, and every item on it has already cost somebody a conclusion:
#
# | | |
# |---|---|
# | **State the replicate unit** | 733,556 cells, but 224 wells and 3 per condition per timepoint. Say which you tested at. |
# | **Compare to a control** | "Higher" means nothing without DMSO next to it. |
# | **Report the p-value floor** | If `at_p_floor` is true, say so rather than quoting the number as a result. |
# | **Say which parameters you chose** | Leiden resolution, `n_neighbors`, the root cell. All of them change the answer. |
# | **State a negative result plainly** | "We expected X and did not see it" is a finding. Trying analyses until one is significant is not. |
# | **Label the figure** | Axis labels, units, and how many replicates are on it. |
#
# ## Where to look when you are stuck
#
# | | |
# |---|---|
# | a marker is missing from your panel | [01 · From columns to markers](1_preparation/01_columns_to_markers.ipynb) |
# | you want texture, not just intensity | [04 · Subsetting and sketching](1_preparation/04_subsetting_and_sketching.ipynb) |
# | your embedding looks like nothing | [06 · PCA](2_controls/06_pca.ipynb) — check what PC1 is |
# | your clusters will not name | [08 · Cell-type annotation](2_controls/08_cell_type_annotation.ipynb) |
# | you want an ordering, not groups | [09 · Diffusion maps](2_controls/10_diffusion_map.ipynb) |
#
# The group briefs are on the [Group projects](../projects.md) page.
