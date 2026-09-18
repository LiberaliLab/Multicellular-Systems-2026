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
# # 04 · Organelles
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
# Haralick features can tell them apart. That is why [Step 9](../part3_analysis/1_preparation/01_columns_to_markers.ipynb) kept 2,262 texture
# columns rather than dropping them.
#
# The structure is the same as [chapter 01](01_signaling.ipynb); only the panel and the
# biology change.

# %%
from math import comb
from pathlib import Path

import anndata as ad
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

def panel_grid(n, *, ncols=3, size=(3.6, 3.0)):
    """A figure with `n` axes on a grid, the unused ones removed."""
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(size[0] * ncols, size[1] * nrows))
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.remove()
    return fig, flat[:n]
pd.set_option("display.width", 140)

# The tables live here on Euler. Change this line if your copy is elsewhere.
DATA = Path("/cluster/project/mcsliberali/data_mcs_2026")

THEME = "organelles"

cells = sc.read_h5ad(DATA / "mcs2026_intensity.h5ad")
wells = (cells.to_df()
         .groupby([cells.obs.condition.astype(str), cells.obs.timepoint_h.astype(int),
                   cells.obs.well.astype(str)], observed=True).mean()
         .rename_axis(["condition", "timepoint", "well"]).reset_index())
markers = [m for m in cells.uns["panels"][THEME] if m in set(wells.columns)]
print(f"{len(markers)} markers: {', '.join(markers)}")

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

OUTLIER = cells.obs.condition[cells.obs.is_outlier_condition].astype(str).iloc[0]


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
effects = effect_table(wells, markers, by_timepoint=False)
effects = effects.drop(index=OUTLIER, errors="ignore")
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
ranked = rank_effects(wells, markers, drop=OUTLIER)
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
usable = wells[wells.condition != OUTLIER]
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
# Lamin B1 tops the raw ranking, but it also sits in the mechanics panel and was
# already shown there. Pick the strongest effect on a marker that belongs to THIS
# theme alone, so the figure says something new.
shared = set().union(*(set(cells.uns["panels"][t]) for t in cells.uns["themes"] if t != THEME))
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
    "value": np.asarray(cells[:, column].X).ravel(),
    "condition": cells.obs.condition.astype(str).values,
    "timepoint": cells.obs.timepoint_h.astype(int).values,
})

fig, axes = panel_grid(4, ncols=4, size=(3.3, 2.9))
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
summary = effect_table(wells, markers, by_timepoint=False).abs().mean(axis=1).sort_values(ascending=False)
summary.drop(index=OUTLIER, errors="ignore").head(8).round(2)

# %%
summary.to_frame(THEME).to_parquet(DATA / f"theme_{THEME}.parquet")
print(f"saved theme summary for {THEME}")

# %% [markdown]
# ---
#
# **Next:** [05 · Integration](05_integration.ipynb).
