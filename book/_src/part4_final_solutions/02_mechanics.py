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
# # 02 · Cell-specific mechanics
#
# **Panel:** p-Myosin IIa, α-Tubulin, ZO-1, E-cadherin, Fibronectin, LAMA4, Lamin A, Lamin B1
#
# How cells are shaped, attached to each other and to the matrix, and how their nuclei are
# built. Cytoskeleton (myosin, tubulin), junctions (ZO-1, E-cadherin), matrix (fibronectin,
# laminin) and the nuclear lamina (Lamin A/B1).
#
# This is the panel where **morphology features matter as much as intensities** — a cell
# that changes shape has changed mechanically whether or not any marker moved.
#
# The structure is the same as [chapter 01](01_signaling.ipynb); only the panel and the
# biology change.

# %%
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path.cwd().parents[1] / "src"))

from mcs2026 import analysis, panels, plotting
from mcs2026.config import H5AD_SLIM

plotting.set_style()
pd.set_option("display.width", 140)

THEME = "mechanics"

wells = pd.read_parquet(H5AD_SLIM.with_name("mcs2026_wells.parquet"))
cells = sc.read_h5ad(H5AD_SLIM.with_name("mcs2026_clean.h5ad"))
markers = analysis.panel_markers(THEME, wells.columns)
print(f"{len(markers)} markers: {', '.join(markers)}")

# %%
lookup = cells.var[cells.var.marker.isin(markers) & (cells.var.statistic == "mean_intensity")]
lookup[["marker", "round", "channel"]].sort_values("round").reset_index(drop=True)

# %% [markdown]
# ## Effects per condition
#
# In control-well standard deviations, against the DMSO wells — pooled across
# timepoints here, so the controls' own drift over 36–84 h cancels on both sides.
# PMA is excluded throughout — [Step 16](../part3_analysis/1_preparation/03_normalisation.ipynb) showed it moves every marker by 4–11 SD, so
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
# **Cycloheximide strips the adhesion machinery.** E-cadherin falls 3.5 control SDs and
# fibronectin 2.8. Cycloheximide blocks translation, so proteins disappear at a rate set
# by their own turnover — and cell-surface adhesion proteins are replaced constantly.
# This is a protein-half-life experiment that nobody designed as one.
#
# **Lamin B1 rises under kinase inhibition** — with INK128 (+3.7), Pd17 (+2.7) and
# MK-2206 (+2.7). Lamin B1 is very long-lived, so when growth slows and dilution by cell
# division slows with it, lamin accumulates relative to everything else. The same logic,
# running the other way, that makes E-cadherin fall.
#
# Both results are about **turnover**, not about signaling — a useful reminder that an
# intensity change is a change in *abundance*, and abundance is synthesis minus
# degradation minus dilution by division.

# %% [markdown]
# ## An embedding of this theme alone

# %%
usable = wells[wells.condition != analysis.OUTLIER_CONDITION]
space = ad.AnnData(usable[markers].fillna(0).to_numpy(dtype="float32"))
sc.pp.pca(space, n_comps=4, random_state=0)
coords, variance = space.obsm["X_pca"], space.uns["pca"]["variance_ratio"]
print("variance explained:", variance[:4].round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=usable.timepoint.values,
                          cmap="viridis", s=28)
axes[0].set(xlabel=f"PC1 ({variance[0]:.0%})",
            ylabel=f"PC2 ({variance[1]:.0%})", title="by timepoint")
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
loadings = pd.DataFrame(space.varm["PCs"][:, :2], index=markers,
                        columns=["PC1", "PC2"])
loadings.reindex(loadings.PC1.abs().sort_values(ascending=False).index).round(2)

# %% [markdown]
# ## Single cells over time
#
# The well means say the average moved. The single-cell distributions say whether every
# cell moved a little or a subpopulation moved a lot.

# %%
top_pair = ranked.iloc[0]
marker, condition = top_pair["marker"], top_pair["condition"]
print(f"strongest effect in this panel: {condition} on {marker} "
      f"({top_pair['shift']:+.2f} control SDs)")

column = cells.var.index[(cells.var.marker == marker)
                         & (cells.var.statistic == "mean_intensity")
                         & (cells.var.family == "Intensity")][0]
frame = pd.DataFrame({
    "value": np.asarray(cells[:, column].X).ravel(),
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
    ax.set(title=f"{timepoint} h", xlabel=f"{marker} (control-cell SDs)")
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
# **Next:** [03 · Metabolism](03_metabolism.ipynb).
