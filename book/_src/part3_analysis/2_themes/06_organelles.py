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
# # 06 · Organelles
#
# **Panel:** EEA1, GM130, Giantin, LAMP1, DDX6, Calreticulin, GRP78, Pmp70, Mitochondria,
# Lamin A/B1
#
# Early endosomes (EEA1), Golgi (GM130 for *cis*-Golgi, Giantin), lysosomes (LAMP1),
# P-bodies (DDX6), ER (Calreticulin, GRP78), peroxisomes (Pmp70), mitochondria and the
# nuclear envelope.
#
# This is the panel where **texture matters most**. A lysosome marker is not informative
# because it is bright — it is informative because it is *punctate*. Two cells with
# identical mean LAMP1 can have very different lysosomal organisation, and only the
# Haralick features can tell them apart. That is why chapter 01 kept 2,262 texture
# columns rather than dropping them.
#
# The structure is the same as [chapter 03](03_signaling.ipynb); only the panel and the
# biology change.

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

THEME = "organelles"

wells = pd.read_parquet(H5AD_SLIM.with_name("mcs2026_wells.parquet"))
cells = sc.read_h5ad(H5AD_SLIM.with_name("mcs2026_qc.h5ad"))
markers = analysis.panel_markers(cells.var, THEME, wells.columns)
print(f"{len(markers)} markers: {', '.join(markers)}")

# %%
lookup = cells.var[cells.var.marker.isin(markers) & (cells.var.statistic == "mean_intensity")]
lookup[["marker", "round", "channel"]].sort_values("round").reset_index(drop=True)

# %% [markdown]
# ## Effects per condition
#
# In control-well standard deviations, against the DMSO wells of the same timepoint.
# PMA is excluded throughout — chapter 02 showed it moves every marker by 4–11 SD, so
# leaving it in would make every heatmap a picture of PMA.

# %%
effects = analysis.effect_table(wells, markers, by_timepoint=False)
effects = effects.drop(index=analysis.OUTLIER_CONDITION, errors="ignore")
order = effects.abs().mean(axis=1).sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(8.5, 5.5))
matrix = effects.loc[order]
limit = np.nanpercentile(np.abs(matrix.values), 99)
im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
ax.set(yticks=range(len(order)), yticklabels=order)
ax.set_xticks(range(len(markers)))
ax.set_xticklabels(markers, rotation=45, ha="right")
ax.set_title(f"{THEME.capitalize()} panel — shift from DMSO (control SDs)")
fig.colorbar(im, ax=ax, shrink=0.7, label="control SDs")
fig.tight_layout()

# %% [markdown]
# ## Ranked effects

# %%
ranked = analysis.rank_effects(wells, markers)
ranked.head(12).round(3)

# %% [markdown]
# ### What the data says
#
# **Pd17 is the biggest organelle mover**, raising DDX6 (+3.1) and EEA1 (+2.3). DDX6
# marks P-bodies, the cytoplasmic granules where untranslated mRNA is stored; more DDX6
# signal is consistent with translational repression and mRNA being parked rather than
# read.
#
# **The same three kinase inhibitors keep appearing** — INK128, MK-2206, Pd17 — moving
# Lamin B1 and DDX6. Organelle content is not independent of growth signaling, because
# how much Golgi or how many lysosomes a cell maintains is a function of how fast it is
# growing.
#
# **Look at what did *not* move.** GM130 and Giantin both mark the Golgi and should
# largely agree; where they do not, ask whether one antibody is simply noisier (chapter
# 02 measured a per-marker control SD for exactly this purpose) before concluding that
# the *cis*- and *medial*-Golgi diverged.

# %% [markdown]
# ## An embedding of this theme alone

# %%
from sklearn.decomposition import PCA

usable = wells[wells.condition != analysis.OUTLIER_CONDITION]
matrix = usable[markers].fillna(0).values
pca = PCA(n_components=4, random_state=0).fit(matrix)
coords = pca.transform(matrix)
print("variance explained:", pca.explained_variance_ratio_[:4].round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=usable.timepoint.values,
                          cmap="viridis", s=28)
axes[0].set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
            ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", title="by timepoint")
fig.colorbar(scatter, ax=axes[0], label="hours")

top = list(effects.abs().mean(axis=1).sort_values(ascending=False).head(3).index)
axes[1].scatter(coords[:, 0], coords[:, 1], c="0.88", s=22)
for condition, colour in zip(["DMSO"] + top, ["0.5", "firebrick", "darkorange", "seagreen"]):
    mask = (usable.condition == condition).values
    axes[1].scatter(coords[mask, 0], coords[mask, 1], c=colour, s=34,
                    label=condition.split("/")[0][:22])
axes[1].set(xlabel="PC1", ylabel="PC2", title="the three biggest movers")
axes[1].legend(fontsize=7)
fig.tight_layout()

# %%
loadings = pd.DataFrame(pca.components_[:2].T, index=markers, columns=["PC1", "PC2"])
loadings.reindex(loadings.PC1.abs().sort_values(ascending=False).index).round(2)

# %% [markdown]
# ## Single cells over time
#
# The well means say the average moved. The single-cell distributions say whether every
# cell moved a little or a subpopulation moved a lot.

# %%
# Lamin B1 tops the raw ranking, but it also sits in the mechanics panel and was
# already shown there. Pick the strongest effect on a marker that belongs to THIS
# theme alone, so the figure says something new.
shared = set().union(*(set(panels.markers(t)) for t in panels.THEMES if t != THEME))
exclusive = ranked[~ranked.marker.isin(shared)]
top_pair = exclusive.iloc[0]
marker, condition = top_pair["marker"], top_pair["condition"]
print(f"organelle-exclusive markers: {sorted(set(markers) - shared)}")
print(f"strongest such effect: {condition} on {marker} "
      f"({top_pair['shift']:+.2f} control SDs)")

column = cells.var.index[(cells.var.marker == marker)
                         & (cells.var.statistic == "mean_intensity")
                         & (cells.var.family == "Intensity")][0]
frame = pd.DataFrame({
    "value": np.log2(np.asarray(cells[:, column].X).ravel() + 1),
    "condition": cells.obs.condition.astype(str).values,
    "timepoint": cells.obs.timepoint_h.astype(int).values,
})

fig, axes = plotting.panel_grid(4, ncols=4, size=(3.3, 2.9))
for ax, timepoint in zip(axes, [36, 48, 60, 84]):
    block = frame[frame.timepoint == timepoint]
    for name, colour in [("DMSO", "0.35"), (condition, "firebrick")]:
        values = block.loc[block.condition == name, "value"]
        ax.hist(values, bins=60, histtype="step", density=True,
                color=colour, linewidth=1.4, label=name.split("/")[0][:18])
    ax.set(title=f"{timepoint} h", xlabel=f"log2 {marker}")
    if timepoint == 36:
        ax.set_ylabel("density")
        ax.legend(fontsize=6)
fig.tight_layout()

# %% [markdown]
# ## One number for the theme

# %%
summary = analysis.theme_summary(wells, markers)
summary.drop(index=analysis.OUTLIER_CONDITION, errors="ignore").head(8).round(2)

# %%
summary.to_frame(THEME).to_parquet(H5AD_SLIM.with_name(f"theme_{THEME}.parquet"))
print(f"saved theme summary for {THEME}")

# %% [markdown]
# ---
#
# **Next:** [07 · Cell types and proportions](../3_going_further/07_celltypes_and_proportions.ipynb).
