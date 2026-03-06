#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================================================
# ecdna_fish_pipeline_run_plot_MYC_PVT1_task1.py
# =========================================================
# Task 1:
# - For MYC_PVT1 4-color data (MYC=red, PVT1=green, CEN8=cyan, DAPI=blue),
#   run the SAME 3-color plotting workflow TWICE (no logic/param changes):
#     A) (MYC,  CEN8, DAPI)  -> treat as (red, green, DAPI)
#     B) (PVT1, CEN8, DAPI)  -> treat as (red, green, DAPI)
#
# NOTE:
# - This file keeps the original SEMA3B script workflow unchanged.
# - Only updates: group/labels/IO names + channel/column remapping to reuse the same 3-color pipeline.
# =========================================================

# =========================================================
# STEP 0:
# =========================================================
# FULL OPTIMIZED CODE (Steps   /   /   /  )
# - Step : merge CSVs + add probability columns
# - Step : colocalization probabilities -> BOX plots (4 figures) + tables (+pvalues)
# - Step : gr_per_cell g0 -> VIOLIN plots (1 axis) + stats + tables
# - Step : gr_long mean±SEM curves (valid+positive) + r_bin rule + stats + tables
# =========================================================
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass
from __future__ import annotations
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Dict, List, Tuple
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter

# -------------------------
# Optional deps
# -------------------------
try:
    from scipy.stats import gaussian_kde, ttest_1samp, ttest_rel, ttest_ind
except Exception:
    gaussian_kde = None
    ttest_1samp = None
    ttest_rel = None
    ttest_ind = None

try:
    from scipy.stats import gaussian_kde
except Exception:
    gaussian_kde = None


# =========================================================
# STEP 2:  STYLE HELPERS
# =========================================================
def apply_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 6,
        "font.weight": "normal",
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.titlesize": 6,
    })

def set_small_axes_with_right_panel(fig, ax, left=0.08, bottom=0.18, width=0.55, height=0.65):
    # 为右侧legend/文字留空
    ax.set_position([left, bottom, width, height])

def save_fig(fig, out_png: Path, out_pdf: Path = None, dpi=600):
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def add_sample_type(df: pd.DataFrame, sample_col="sample_id"):
    if df.empty:
        return df
    df["sample_type"] = df[sample_col].astype(str).str.split("_").str[0]
    return df

def add_cell_uid(df: pd.DataFrame):
    if df.empty:
        return df
    df["cell_uid"] = df["sample_id"].astype(str) + "::" + df["cell_id"].astype(str)
    return df

def _sem(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    return float(np.nanstd(x, ddof=1) / np.sqrt(x.size))

def _pval_fmt(p):
    if p is None or (not np.isfinite(p)):
        return "NA"
    # compact like script: 3 sig figs
    return f"{p:.3g}"


# =========================================================
# STEP 3: scan ROI dirs, merge CSVs, add probability columns
# =========================================================
from pandas.errors import EmptyDataError

def read_if_exists(fp: Path):
    """
    Safe CSV reader:
    - return None if file not exists / empty / only blank lines
    - avoid pandas EmptyDataError
    """
    if not fp.exists():
        return None

    # 1) fast path: truly empty file (0 bytes)
    try:
        if fp.stat().st_size == 0:
            print(f"[WARN] empty file (0B), skip: {fp}")
            return None
    except OSError:
        pass

    # 2) try read
    try:
        df = pd.read_csv(fp)
    except EmptyDataError:
        print(f"[WARN] EmptyDataError, skip: {fp}")
        return None
    except UnicodeDecodeError:
        # if some files are not utf-8, try gbk (windows common); if still fail, skip
        try:
            df = pd.read_csv(fp, encoding="gbk")
        except Exception:
            return None
    except Exception:
        # any other parsing failure -> skip this file (keeps pipeline running)
        return None

    # 3) handle "only header" or effectively empty
    if df is None or df.empty:
        return None

    if df.shape[0] == 0:
        return None

    return df

def iter_roi_dirs(root: Path):
    for d in sorted(root.iterdir()):
        if d.is_dir():
            yield d


# =========================================================
# TASK1 helper: 4-color -> 3-color remapping (only name/label mapping; computation unchanged)
# =========================================================
def remap_per_cell_4to3(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode:
      - "MYC_CEN8":   treat (red=MYC original red) and (green=CEN8 original cyan)
      - "PVT1_CEN8":  treat (red=PVT1 original green) and (green=CEN8 original cyan)

    This function ONLY renames columns to reuse the same 3-color workflow:
      - n_spots_red / n_spots_green
      - within_red_frac_with_neighbor / within_green_frac_with_neighbor
      - coloc_red_green_frac_*  (use correct pair)
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # ---------- core spot-count mapping ----------
    if mode == "MYC_CEN8":
        # green <- cyan
        if "n_spots_cyan" in out.columns:
            out["n_spots_green"] = out["n_spots_cyan"]
        if "within_cyan_frac_with_neighbor" in out.columns:
            out["within_green_frac_with_neighbor"] = out["within_cyan_frac_with_neighbor"]

        # coloc red-green <- red-cyan
        if "coloc_red_cyan_frac_red_with_cyan" in out.columns:
            out["coloc_red_green_frac_red_with_green"] = out["coloc_red_cyan_frac_red_with_cyan"]
        if "coloc_red_cyan_frac_cyan_with_red" in out.columns:
            out["coloc_red_green_frac_green_with_red"] = out["coloc_red_cyan_frac_cyan_with_red"]

    elif mode == "PVT1_CEN8":
        # red <- green (PVT1), green <- cyan (CEN8)
        if "n_spots_green" in out.columns:
            out["n_spots_red"] = out["n_spots_green"]
        if "within_green_frac_with_neighbor" in out.columns:
            out["within_red_frac_with_neighbor"] = out["within_green_frac_with_neighbor"]

        if "n_spots_cyan" in out.columns:
            out["n_spots_green"] = out["n_spots_cyan"]
        if "within_cyan_frac_with_neighbor" in out.columns:
            out["within_green_frac_with_neighbor"] = out["within_cyan_frac_with_neighbor"]

        # coloc red-green <- green-cyan (PVT1 vs CEN8)
        if "coloc_green_cyan_frac_green_with_cyan" in out.columns:
            out["coloc_red_green_frac_red_with_green"] = out["coloc_green_cyan_frac_green_with_cyan"]
        if "coloc_green_cyan_frac_cyan_with_green" in out.columns:
            out["coloc_red_green_frac_green_with_red"] = out["coloc_green_cyan_frac_cyan_with_green"]

    return out

def remap_gr_channels_4to3(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    gr_per_cell / gr_long remapping:
      - keep ONLY the two channels needed by the 3-color workflow
      - rename channel labels so that downstream logic (expects "red"/"green") stays unchanged
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if "channel" not in out.columns:
        return out

    ch = out["channel"].astype(str).str.lower()

    if mode == "MYC_CEN8":
        # keep red (MYC) and cyan (CEN8 -> green); drop original green (PVT1)
        keep = ch.isin(["red", "cyan"])
        out = out.loc[keep].copy()
        out["channel"] = out["channel"].astype(str).str.lower().replace({"cyan": "green"})

    elif mode == "PVT1_CEN8":
        # keep green (PVT1 -> red) and cyan (CEN8 -> green); drop original red (MYC)
        keep = ch.isin(["green", "cyan"])
        out = out.loc[keep].copy()
        out["channel"] = out["channel"].astype(str).str.lower().replace({"green": "red", "cyan": "green"})

    return out


# =========================================================
# =========================================================
# RUN A) MYC + CEN8 + DAPI  (treat as red+green+DAPI)
# =========================================================
# =========================================================

# =========================================================
# STEP 1: USER CONFIG
# =========================================================
PATHA = Path(r"D:\02.project\18.ecDNA\02.code\v0.1.1\autocorrelation\output_ws\MYC_PVT1")
GROUP = "MYC_CEN8"
OUT_PREFIX = "merged"

# =========================
# 自定义总输出目录（不再写到 PATHA）
# =========================
OUT_DIR = Path(r"D:\02.project\18.ecDNA\02.code\v0.1.1\autocorrelation\output_ws_plot\MYC_PVT1\MYC_CEN8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_MIN_TOTAL_SPOTS = 2
POSITIVE_RED_THRESHOLD = 8
SAMPLE_TYPE_ORDER = ["BRCA", "CRC", "ESCA", "GBM", "PDAC"]

# Step style colors (sample_type colors)
GROUP_COLORS = {
    "BRCA": "#1f77b4",
    "CRC":  "#ff7f0e",
    "ESCA": "#2ca02c",
    "GBM":  "#d62728",
    "PDAC": "#9467bd",
}

# Step (4 violin types) colors
TYPE_COLORS = {
    "g0_exp_red":   "#b14744",  # dark red
    "g0_exp_green": "#097d98",  # dark blue-ish (for green-channel exp; keep distinct)
    "g0_rand":      "#666666",  # grey
    "g0_dapi":      "#000000",  # black
}

# plot geometry to match ecdna_autocorrelation_v4.py
FIGSIZE = (7.2, 5.2)
AX_POS_GR = [0.08, 0.18, 0.15, 0.18]  # [left, bottom, width, height] (square-ish)
RIGHT_PANEL_X = 0.25
LEGEND_Y = 0.35

# binning & curve plot limits
R_PLOT_MAX = 40
BIN_PX = 1  # same meaning as plot_bin_px

np.random.seed(0)

# =========================================================
# STEP 3: scan ROI dirs, merge CSVs, add probability columns
# =========================================================
dfs_per_cell, dfs_gr_per_cell, dfs_gr_pvalues, dfs_gr_long = [], [], [], []

for roi_dir in iter_roi_dirs(PATHA):
    doi = roi_dir.name

    f1 = roi_dir / "per_cell_summary.csv"
    f2 = roi_dir / "gr_per_cell.csv"
    f3 = roi_dir / "gr_pvalues.csv"
    f4 = roi_dir / "gr_long.csv"

    df1 = read_if_exists(f1)
    df2 = read_if_exists(f2)
    df3 = read_if_exists(f3)
    df4 = read_if_exists(f4)

    if df1 is not None and not df1.empty:
        df1["doi"] = doi
        dfs_per_cell.append(df1)

    if df2 is not None and not df2.empty:
        df2["doi"] = doi
        dfs_gr_per_cell.append(df2)

    if df3 is not None and not df3.empty:
        df3["doi"] = doi
        dfs_gr_pvalues.append(df3)

    if df4 is not None and not df4.empty:
        df4["doi"] = doi
        dfs_gr_long.append(df4)

per_cell_all = pd.concat(dfs_per_cell, ignore_index=True) if dfs_per_cell else pd.DataFrame()
gr_per_cell_all = pd.concat(dfs_gr_per_cell, ignore_index=True) if dfs_gr_per_cell else pd.DataFrame()
gr_pvalues_all = pd.concat(dfs_gr_pvalues, ignore_index=True) if dfs_gr_pvalues else pd.DataFrame()
gr_long_all = pd.concat(dfs_gr_long, ignore_index=True) if dfs_gr_long else pd.DataFrame()

# ---- unify fields + add ids ----
per_cell_all = add_sample_type(add_cell_uid(per_cell_all))
gr_per_cell_all = add_sample_type(add_cell_uid(gr_per_cell_all))
gr_long_all = add_sample_type(add_cell_uid(gr_long_all))

# ---- filter to MYC_PVT1 group (original group name) ----
BASE_GROUP = "MYC_PVT1"
if (not per_cell_all.empty) and ("group" in per_cell_all.columns):
    per_cell_all = per_cell_all.loc[per_cell_all["group"].astype(str) == BASE_GROUP].copy()
if (not gr_per_cell_all.empty) and ("group" in gr_per_cell_all.columns):
    gr_per_cell_all = gr_per_cell_all.loc[gr_per_cell_all["group"].astype(str) == BASE_GROUP].copy()
if (not gr_long_all.empty) and ("group" in gr_long_all.columns):
    gr_long_all = gr_long_all.loc[gr_long_all["group"].astype(str) == BASE_GROUP].copy()

# ---- TASK1: remap 4-color -> 3-color (MYC + CEN8) ----
per_cell_all = remap_per_cell_4to3(per_cell_all, mode="MYC_CEN8")
gr_per_cell_all = remap_gr_channels_4to3(gr_per_cell_all, mode="MYC_CEN8")
gr_long_all = remap_gr_channels_4to3(gr_long_all, mode="MYC_CEN8")

# ---- add new probability columns in per_cell_all (OPTIMIZATION) ----
if not per_cell_all.empty:
    for c in [
        "coloc_red_green_frac_red_with_green",
        "coloc_red_green_frac_green_with_red",
        "within_red_frac_with_neighbor",
        "within_green_frac_with_neighbor",
    ]:
        if c in per_cell_all.columns:
            per_cell_all[c] = pd.to_numeric(per_cell_all[c], errors="coerce")

    if ("coloc_red_green_frac_red_with_green" in per_cell_all.columns) and ("coloc_red_green_frac_green_with_red" in per_cell_all.columns):
        per_cell_all["pro_coloc_red-green"] = per_cell_all[[
            "coloc_red_green_frac_red_with_green",
            "coloc_red_green_frac_green_with_red"
        ]].mean(axis=1, skipna=True)
    else:
        per_cell_all["pro_coloc_red-green"] = np.nan

    per_cell_all["pro_coloc_red-red"] = (
        per_cell_all["within_red_frac_with_neighbor"]
        if "within_red_frac_with_neighbor" in per_cell_all.columns else np.nan
    )
    per_cell_all["pro_coloc_green-green"] = (
        per_cell_all["within_green_frac_with_neighbor"]
        if "within_green_frac_with_neighbor" in per_cell_all.columns else np.nan
    )

# ---- output merged tables ----
out_per_cell   = OUT_DIR / f"{OUT_PREFIX}_per_cell_summary.csv"
out_gr_percell = OUT_DIR / f"{OUT_PREFIX}_gr_per_cell.csv"
out_gr_pvalues = OUT_DIR / f"{OUT_PREFIX}_gr_pvalues.csv"
out_gr_long    = OUT_DIR / f"{OUT_PREFIX}_gr_long.csv"

per_cell_all.to_csv(out_per_cell, index=False)
gr_per_cell_all.to_csv(out_gr_percell, index=False)
gr_pvalues_all.to_csv(out_gr_pvalues, index=False)
gr_long_all.to_csv(out_gr_long, index=False)

print("[OK] merged CSV written:")
print(" -", out_per_cell)
print(" -", out_gr_percell)
print(" -", out_gr_pvalues)
print(" -", out_gr_long)

# ---- valid / positive ----
for col in ["n_spots_red", "n_spots_green"]:
    if col in per_cell_all.columns:
        per_cell_all[col] = pd.to_numeric(per_cell_all[col], errors="coerce").fillna(0).astype(int)

per_cell_all["is_valid"] = (per_cell_all["n_spots_red"] + per_cell_all["n_spots_green"]) >= VALID_MIN_TOTAL_SPOTS
valid_cells = per_cell_all.loc[per_cell_all["is_valid"]].copy()
valid_cells["is_positive"] = valid_cells["n_spots_red"] >= POSITIVE_RED_THRESHOLD
pos_cells = valid_cells.loc[valid_cells["is_positive"]].copy()

print(f"[INFO] total cells={len(per_cell_all)}, valid={len(valid_cells)}, positive={len(pos_cells)}")

# ordering
order = [x for x in SAMPLE_TYPE_ORDER if x in valid_cells["sample_type"].astype(str).unique().tolist()]
rest  = [x for x in valid_cells["sample_type"].astype(str).unique().tolist() if x not in order]
order = order + sorted(rest)

# =========================================================
# STEP 4: 阳性比例：5组堆叠条形图（同一sample_type汇总所有ROI）
# =========================================================

ratio_df = (
    valid_cells.groupby("sample_type")["is_positive"]
    .agg(n_valid="count", n_pos="sum")
    .reset_index()
)
ratio_df["n_neg"] = ratio_df["n_valid"] - ratio_df["n_pos"]
ratio_df["pos_ratio"] = ratio_df["n_pos"] / ratio_df["n_valid"].replace(0, np.nan)
ratio_df["neg_ratio"] = ratio_df["n_neg"] / ratio_df["n_valid"].replace(0, np.nan)

order = [x for x in SAMPLE_TYPE_ORDER if x in ratio_df["sample_type"].tolist()]
rest = [x for x in ratio_df["sample_type"].tolist() if x not in order]
order = order + sorted(rest)

ratio_df["sample_type"] = pd.Categorical(ratio_df["sample_type"], categories=order, ordered=True)
ratio_df = ratio_df.sort_values("sample_type").reset_index(drop=True)

out_table = OUT_DIR / f"{GROUP}_positive_ratio_stackedbar_data.csv"
ratio_df.to_csv(out_table, index=False)

apply_plot_style()
fig, ax = plt.subplots(figsize=(7.2, 5.2))
set_small_axes_with_right_panel(fig, ax, width=0.17, height=0.16)

x = np.arange(len(ratio_df))
pos = ratio_df["pos_ratio"].to_numpy(dtype=float)
neg = ratio_df["neg_ratio"].to_numpy(dtype=float)

bar_width = 0.65
edge_lw = 0.6
edge_col = "black"

ax.bar(
    x, pos,
    width=bar_width,
    label="Positive (red>6)",
    color="#b14744",
    edgecolor=edge_col,
    linewidth=edge_lw
)
ax.bar(
    x, neg,
    width=bar_width,
    bottom=pos,
    label="Negative (red<=6)",
    color="#097d98",
    edgecolor=edge_col,
    linewidth=edge_lw
)

ax.set_xticks(x)
ax.set_xticklabels(
    ratio_df["sample_type"].astype(str).tolist(),
    rotation=45,
    ha="right",
    rotation_mode="anchor"
)

ax.set_ylabel("Proportion (valid cells)")
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

for xi, n_valid in zip(x, ratio_df["n_valid"].to_numpy(dtype=int)):
    ax.text(
        xi, 1.02, f"n={n_valid}",
        ha="center", va="bottom"
    )

fig.suptitle(f"{GROUP}: Positive cell proportion by cancer type", x=0.15, y=0.4)

fig.legend(
    loc="center left",
    bbox_to_anchor=(0.3, 0.3),
    frameon=False
)

save_fig(
    fig,
    OUT_DIR / f"{GROUP}_positive_ratio_stackedbar.png",
    OUT_DIR / f"{GROUP}_positive_ratio_stackedbar.pdf"
)

print("[OK] plot saved, data table saved:", out_table)


# =========================================================
# STEP 5: 绘图工具：style/violin（轮廓+点（空心）+中位数虚线）
# =========================================================

def _kde_density(y, y_grid):
    """
    return density on y_grid, and a function dens(y_i) for samples.
    Prefer scipy gaussian_kde; fallback to histogram smoothing.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        dens_grid = np.ones_like(y_grid, dtype=float)
        return dens_grid, (lambda yi: np.ones_like(yi, dtype=float))

    if gaussian_kde is not None:
        kde = gaussian_kde(y)
        dens_grid = kde(y_grid)
        dens_grid = np.clip(dens_grid, 1e-12, None)
        return dens_grid, (lambda yi: np.clip(kde(yi), 1e-12, None))

    bins = max(20, int(np.sqrt(y.size)))
    hist, edges = np.histogram(y, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    k = min(7, len(hist))
    if k >= 3:
        kernel = np.ones(k) / k
        hist_s = np.convolve(hist, kernel, mode="same")
    else:
        hist_s = hist

    dens_grid = np.interp(y_grid, centers, hist_s, left=hist_s[0], right=hist_s[-1])
    dens_grid = np.clip(dens_grid, 1e-12, None)

    def dens_func(yi):
        yi = np.asarray(yi, dtype=float)
        return np.clip(np.interp(yi, centers, hist_s, left=hist_s[0], right=hist_s[-1]), 1e-12, None)

    return dens_grid, dens_func

def violin_outline_points_inside_and_median(
    ax,
    data_by_group,
    group_labels,
    title,
    ylabel,
    n_by_group,
    out_table: Path = None,
    colors_by_group=None,
    violin_width=0.75,
    violin_lw=1,
    point_size=3,
    point_lw=0.75,
    median_lw=1,
):
    clean = []
    for arr in data_by_group:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        clean.append(arr)

    n_groups = len(group_labels)
    positions = np.arange(1, n_groups + 1)

    if colors_by_group is None:
        cmap = plt.get_cmap("tab10")
        colors_by_group = [cmap(i % 10) for i in range(n_groups)]

    parts = ax.violinplot(
        clean,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=violin_width,
    )
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor("none")
        b.set_edgecolor(colors_by_group[i])
        b.set_linewidth(violin_lw)
        b.set_alpha(1.0)

    all_vals = np.concatenate([a for a in clean if a.size > 0], axis=0) if any(a.size > 0 for a in clean) else np.array([0.0, 1.0])
    y_min = float(np.nanmin(all_vals)) if all_vals.size else 0.0
    y_max = float(np.nanmax(all_vals)) if all_vals.size else 1.0
    y_rng = max(y_max - y_min, 1e-9)

    pad = 0.06 * y_rng
    n_y = y_max + pad

    rng = np.random.default_rng(0)

    for i, arr in enumerate(clean):
        if arr.size == 0:
            continue

        col = colors_by_group[i]
        pos = positions[i]
        half_max = violin_width / 2.0

        y_grid = np.linspace(y_min, y_max, 256)
        dens_grid, dens_func = _kde_density(arr, y_grid)
        dens_max = float(np.nanmax(dens_grid)) if dens_grid.size else 1.0
        dens_max = max(dens_max, 1e-12)

        dens_y = dens_func(arr)
        half_w = half_max * (dens_y / dens_max)
        half_w = np.clip(half_w, 0.0, half_max)

        u = rng.uniform(-1.0, 1.0, size=arr.size)
        x = pos + u * half_w

        ax.scatter(
            x, arr,
            s=point_size,
            facecolors="none",
            edgecolors=col,
            linewidths=point_lw,
            zorder=3
        )

        med = float(np.nanmedian(arr))
        ax.hlines(
            y=med,
            xmin=pos - 0.33,
            xmax=pos + 0.33,
            colors=col,
            linestyles="--",
            linewidth=median_lw,
            zorder=4
        )

    for i, nval in enumerate(n_by_group):
        ax.text(
            positions[i],
            n_y,
            f"n={int(nval)}",
            ha="center",
            va="bottom",
            color="black"
        )

    ax.set_title(title, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_ylim(y_min, n_y + 0.08 * y_rng)

    if out_table is not None:
        rows = []
        for st, arr in zip(group_labels, clean):
            for v in arr:
                rows.append({"sample_type": st, "value": float(v)})
        pd.DataFrame(rows).to_csv(out_table, index=False)

def group_arrays_and_n(df, value_col, order):
    data, labels, nlist = [], [], []
    for st in order:
        sub = df.loc[df["sample_type"].astype(str) == st]
        data.append(sub[value_col].to_numpy(dtype=float))
        labels.append(st)
        nlist.append(len(sub))
    return data, labels, nlist


# =========================================================
# STEP 6: red拷贝数：所有有效细胞 vs 阳性细胞（两张图）
# =========================================================
colors = [GROUP_COLORS.get(st, "black") for st in order]

data_all, labels, n_all = group_arrays_and_n(valid_cells, "n_spots_red", order)
apply_plot_style()
fig, ax = plt.subplots(figsize=(7.2, 5.2))
set_small_axes_with_right_panel(fig, ax, width=0.17, height=0.16)

out_table_all = OUT_DIR / f"{GROUP}_violin_red_all_valid_data.csv"
violin_outline_points_inside_and_median(
    ax,
    data_by_group=data_all,
    group_labels=labels,
    title=f"{GROUP}: copy number (all valid cells)",
    ylabel=f"{GROUP} spots per nucleus",
    n_by_group=n_all,
    out_table=out_table_all,
    colors_by_group=colors
)
save_fig(fig, OUT_DIR / f"{GROUP}_violin_red_all_valid.png", OUT_DIR / f"{GROUP}_violin_red_all_valid.pdf")

data_pos, labels, n_pos = group_arrays_and_n(pos_cells, "n_spots_red", order)
apply_plot_style()
fig, ax = plt.subplots(figsize=(7.2, 5.2))
set_small_axes_with_right_panel(fig, ax, width=0.15, height=0.15)

out_table_pos = OUT_DIR / f"{GROUP}_violin_red_positive_only_data.csv"
violin_outline_points_inside_and_median(
    ax,
    data_by_group=data_pos,
    group_labels=labels,
    title=f"{GROUP}: copy number (positive cells only)",
    ylabel=f"{GROUP} spots per nucleus",
    n_by_group=n_pos,
    out_table=out_table_pos,
    colors_by_group=colors
)
save_fig(fig, OUT_DIR / f"{GROUP}_violin_red_positive_only.png", OUT_DIR / f"{GROUP}_violin_red_positive_only.pdf")

print("[OK] saved:")
print(" -", OUT_DIR / f"{GROUP}_violin_red_all_valid.png")
print(" -", OUT_DIR / f"{GROUP}_violin_red_positive_only.png")
print("[OK] data tables saved:")
print(" -", out_table_all)
print(" -", out_table_pos)


# =========================================================
# STEP 7: KDE helper for "points inside shape"
# =========================================================
def _kde_density(y, y_grid):
    """
    return density on y_grid, and a function dens(y_i) for samples.
    Prefer scipy gaussian_kde; fallback to histogram smoothing.
    Robust to singular covariance (e.g., constant values like all 0/1).
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if y.size < 2:
        dens_grid = np.ones_like(y_grid, dtype=float)
        return dens_grid, (lambda yi: np.ones_like(yi, dtype=float))

    if float(np.nanstd(y, ddof=1)) <= 1e-12:
        dens_grid = np.ones_like(y_grid, dtype=float)
        return dens_grid, (lambda yi: np.ones_like(yi, dtype=float))

    if gaussian_kde is not None:
        try:
            kde = gaussian_kde(y)
            dens_grid = kde(y_grid)
            dens_grid = np.clip(dens_grid, 1e-12, None)
            return dens_grid, (lambda yi: np.clip(kde(yi), 1e-12, None))
        except Exception:
            pass

    bins = max(20, int(np.sqrt(y.size)))
    hist, edges = np.histogram(y, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    k = min(7, len(hist))
    if k >= 3:
        kernel = np.ones(k) / k
        hist_s = np.convolve(hist, kernel, mode="same")
    else:
        hist_s = hist

    dens_grid = np.interp(y_grid, centers, hist_s, left=hist_s[0], right=hist_s[-1])
    dens_grid = np.clip(dens_grid, 1e-12, None)

    def dens_func(yi):
        yi = np.asarray(yi, dtype=float)
        return np.clip(
            np.interp(yi, centers, hist_s, left=hist_s[0], right=hist_s[-1]),
            1e-12, None
        )

    return dens_grid, dens_func


# =========================================================
# STEP 8 (Refactor / Clean & Maintainable)
# =========================================================
PROB_COLS = ["pro_coloc_red-green", "pro_coloc_red-red", "pro_coloc_green-green"]
PROB_LABELS = ["red-green", "red-red", "green-green"]

LABEL_MAP = {
    "red-green": "MYC and CEN8",
    "red-red": "MYC and MYC",
    "green-green": "CEN8 and CEN8",
}
def disp_label(lab: str) -> str:
    return LABEL_MAP.get(lab, lab)

METRIC_COLORS = {
    "red-green": "#1f77b4",
    "red-red": "#ff7f0e",
    "green-green": "#2ca02c",
}

AX_POS = [0.08, 0.18, 0.4, 0.2]
GROUP_GAP = 1.0

HEADER_Y_PAD = 0.1
TICK_ROT = 45
HEADER_Y0_ABOVE_1 = 0.40
HEADER_H = 0.18
HEADER_TEXT_YSHIFT_FRAC = -0.08

STAR_UNDER_HEADER_GAP = 0.34
BRACKET_H = 0.018
DY_BRACKET = 0.170

Y_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]
Y_TICK_LABELS = ["0", "", "0.5", "", "1"]

SAMPLETYPE_YMAX = 1.5

LEGEND_NCOL = 3
LEGEND_Y_OFFSET = -0.075
LEGEND_FRAMEON = False

def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def p_to_star(p) -> str:
    if p is None or (not np.isfinite(p)):
        return "na"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def set_prob_y_ticks(ax: plt.Axes) -> None:
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
    ax.yaxis.set_major_formatter(FixedFormatter(Y_TICK_LABELS))

def set_frame(ax: plt.Axes, show_right: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(show_right)

def add_header_box(ax: plt.Axes, x0: float, x1: float, y0: float, h: float, text: str, lw: float = 0.8) -> None:
    rect = plt.Rectangle(
        (x0, y0), (x1 - x0), h,
        fill=False, edgecolor="black", linewidth=lw, clip_on=False
    )
    ax.add_patch(rect)
    ax.text(
        (x0 + x1) / 2.0,
        y0 + h * (0.5 + HEADER_TEXT_YSHIFT_FRAC),
        text,
        ha="center", va="center",
        clip_on=False
    )

def draw_sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str, lw: float = 0.8, fs=None) -> None:
    if x2 < x1:
        x1, x2 = x2, x1
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=lw, clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom", clip_on=False, fontsize=fs)

def draw_box_with_points(
    ax: plt.Axes,
    data,
    pos: float,
    color: str,
    width: float = 0.60,
    point_size: float = 3,
    point_lw: float = 0.7,
    box_lw: float = 1.0,
) -> None:
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    ax.boxplot(
        [arr],
        positions=[pos],
        widths=width,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="none", edgecolor=color, linewidth=box_lw),
        whiskerprops=dict(color=color, linewidth=box_lw),
        capprops=dict(color=color, linewidth=box_lw),
        medianprops=dict(color=color, linewidth=box_lw, linestyle="--"),
    )

    rng = np.random.default_rng(0)
    half_max = width / 2.0

    if arr.size > 1 and np.nanmin(arr) != np.nanmax(arr):
        y_grid = np.linspace(np.nanmin(arr), np.nanmax(arr), 256)
        dens_grid, dens_func = _kde_density(arr, y_grid)
        dens_max = max(float(np.nanmax(dens_grid)), 1e-12)
        dens_y = dens_func(arr)
        half_w = half_max * (dens_y / dens_max)
        half_w = np.clip(half_w, 0.0, half_max)
    else:
        half_w = np.full(arr.shape, half_max, dtype=float)

    u = rng.uniform(-1.0, 1.0, size=arr.size)
    x = pos + u * half_w

    ax.scatter(
        x, arr,
        s=point_size,
        facecolors="none",
        edgecolors=color,
        linewidths=point_lw,
        zorder=3
    )

def compute_positions_metric_group(order: list[str], n_metrics: int, group_gap: float):
    positions, meta, group_ranges, sep_xs = [], [], [], []
    pos_cursor = 1.0
    for gidx in range(n_metrics):
        start = pos_cursor
        for st in order:
            positions.append(pos_cursor)
            meta.append((gidx, st))
            pos_cursor += 1.0
        end = pos_cursor - 1.0
        group_ranges.append((start, end))
        if gidx < n_metrics - 1:
            next_start = end + group_gap
            sep_xs.append((end + next_start) / 2.0)
            pos_cursor = next_start
    return positions, meta, group_ranges, sep_xs

def plot_prob_box_grouped_by_metric(df_cells: pd.DataFrame, tag: str, out_base: Path):
    df = ensure_numeric(df_cells.copy(), PROB_COLS)

    rows = []
    for st in order:
        sub = df.loc[df["sample_type"].astype(str) == st]
        if sub.empty:
            continue
        for col, lab in zip(PROB_COLS, PROB_LABELS):
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(vals)
            vals = vals[m]
            uids = sub.loc[m, "cell_uid"].astype(str).tolist() if "cell_uid" in sub.columns else [""] * len(vals)
            for v, uid in zip(vals, uids):
                rows.append({"sample_type": st, "metric": lab, "value": float(v), "cell_uid": uid})
    plot_df = pd.DataFrame(rows)
    out_table = out_base / f"{GROUP}_prob_box_metricGroup_{tag}_data.csv"
    plot_df.to_csv(out_table, index=False)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_position(AX_POS)

    positions, meta, group_ranges, sep_xs = compute_positions_metric_group(
        order, n_metrics=len(PROB_LABELS), group_gap=GROUP_GAP
    )

    for (gidx, st), x in zip(meta, positions):
        arr = pd.to_numeric(
            df.loc[df["sample_type"].astype(str) == st, PROB_COLS[gidx]],
            errors="coerce"
        ).to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        draw_box_with_points(
            ax, arr, x,
            color=GROUP_COLORS.get(st, "black"),
            width=0.60, point_size=3, point_lw=0.75, box_lw=1.0
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([st for (_, st) in meta], rotation=TICK_ROT, ha="right", rotation_mode="anchor")

    ax.set_ylabel("Probability of colocalization")
    ax.set_title(f"{GROUP}: {tag} (metric-grouped)", pad=10)

    set_frame(ax, show_right=True)
    set_prob_y_ticks(ax)

    ymin, ymax = ax.get_ylim()
    y_rng = max(ymax - ymin, 1e-9)
    header_y0 = ymax + HEADER_Y_PAD * y_rng
    header_h = HEADER_H * y_rng

    for (x0, x1), lab in zip(group_ranges, PROB_LABELS):
        add_header_box(ax, x0 - 0.5, x1 + 0.5, header_y0, header_h, disp_label(lab), lw=0.8)

    ax.set_ylim(ymin, header_y0 + 0.75 * header_h)

    ymin2, ymax2 = ax.get_ylim()
    for xv in sep_xs:
        ax.vlines(xv, ymin2, ymax2, colors="black", linewidth=0.8)

    out_png = out_base / f"{GROUP}_prob_box_metricGroup_{tag}.png"
    out_pdf = out_base / f"{GROUP}_prob_box_metricGroup_{tag}.pdf"
    save_fig(fig, out_png, out_pdf)
    return out_table, out_png

def plot_prob_box_grouped_by_sampletype_with_tests(df_cells: pd.DataFrame, tag: str, out_base: Path):
    df = ensure_numeric(df_cells.copy(), PROB_COLS)

    plot_rows, p_rows = [], []

    apply_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_position(AX_POS)

    positions, colors, datas = [], [], []
    st_group_ranges, sep_xs = [], []
    st_metric_pos = {}

    pos_cursor = 1.0
    for i, st in enumerate(order):
        sub = df.loc[df["sample_type"].astype(str) == st].copy()

        arrs = []
        metric_pos_map = {}

        for col, lab in zip(PROB_COLS, PROB_LABELS):
            arr = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            arrs.append(arr)

            for v in arr:
                plot_rows.append({"sample_type": st, "metric": disp_label(lab), "value": float(v)})

        start = pos_cursor
        for lab, arr in zip(PROB_LABELS, arrs):
            datas.append(arr)
            positions.append(pos_cursor)
            metric_pos_map[lab] = pos_cursor
            colors.append(METRIC_COLORS.get(lab, "black"))
            pos_cursor += 1.0
        end = pos_cursor - 1.0

        st_group_ranges.append((start, end))
        st_metric_pos[st] = metric_pos_map

        if i < len(order) - 1:
            next_start = end + GROUP_GAP
            sep_xs.append((end + next_start) / 2.0)
            pos_cursor = next_start

        arr_rg, arr_rr, arr_gg = arrs
        p_rg_rr = float(ttest_ind(arr_rg, arr_rr, equal_var=False).pvalue) if (ttest_ind and arr_rg.size >= 2 and arr_rr.size >= 2) else np.nan
        p_rg_gg = float(ttest_ind(arr_rg, arr_gg, equal_var=False).pvalue) if (ttest_ind and arr_rg.size >= 2 and arr_gg.size >= 2) else np.nan
        p_rr_gg = float(ttest_ind(arr_rr, arr_gg, equal_var=False).pvalue) if (ttest_ind and arr_rr.size >= 2 and arr_gg.size >= 2) else np.nan

        p_rows.append({
            "sample_type": st,
            "p_RG_vs_RR": p_rg_rr,
            "p_RG_vs_GG": p_rg_gg,
            "p_RR_vs_GG": p_rr_gg,
            "star_RG_vs_RR": p_to_star(p_rg_rr),
            "star_RG_vs_GG": p_to_star(p_rg_gg),
            "star_RR_vs_GG": p_to_star(p_rr_gg),
        })

    for arr, x, c in zip(datas, positions, colors):
        draw_box_with_points(ax, arr, x, c, width=0.55, point_size=3, point_lw=0.75, box_lw=1.0)

    ax.set_xticks([])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax.set_ylabel("Probability of colocalization")
    ax.set_title(f"{GROUP}: {tag} (sample-type grouped)", pad=10)

    set_frame(ax, show_right=True)
    set_prob_y_ticks(ax)

    ax.set_ylim(0.0, SAMPLETYPE_YMAX)

    p_df = pd.DataFrame(p_rows)

    header_y0 = 1.0 + HEADER_Y0_ABOVE_1
    header_h = HEADER_H

    ymin, ymax = ax.get_ylim()
    if header_y0 + header_h > ymax:
        ax.set_ylim(ymin, header_y0 + header_h)
        ymin, ymax = ax.get_ylim()

    for (x0, x1), st in zip(st_group_ranges, order):
        add_header_box(ax, x0 - 0.5, x1 + 0.5, header_y0, header_h, st, lw=0.8)

    base_same = header_y0 - STAR_UNDER_HEADER_GAP
    base_high = base_same + DY_BRACKET
    fs = plt.rcParams.get("font.size", 6)

    for st in order:
        pv = p_df.loc[p_df["sample_type"] == st].iloc[0]
        pos_map = st_metric_pos[st]
        x_rg, x_rr, x_gg = pos_map["red-green"], pos_map["red-red"], pos_map["green-green"]

        draw_sig_bracket(ax, x_rg, x_rr, base_same, BRACKET_H, pv["star_RG_vs_RR"], lw=0.8, fs=fs)
        draw_sig_bracket(ax, x_rr, x_gg, base_same, BRACKET_H, pv["star_RR_vs_GG"], lw=0.8, fs=fs)
        draw_sig_bracket(ax, x_rg, x_gg, base_high, BRACKET_H, pv["star_RG_vs_GG"], lw=0.8, fs=fs)

    ymin2, ymax2 = ax.get_ylim()
    for xv in sep_xs:
        ax.vlines(xv, ymin2, ymax2, colors="black", linewidth=0.8)

    handles = [
        Patch(facecolor="none", edgecolor=METRIC_COLORS["red-green"], linewidth=1.0, label=disp_label("red-green")),
        Patch(facecolor="none", edgecolor=METRIC_COLORS["red-red"], linewidth=1.0, label=disp_label("red-red")),
        Patch(facecolor="none", edgecolor=METRIC_COLORS["green-green"], linewidth=1.0, label=disp_label("green-green")),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_Y_OFFSET),
        ncol=LEGEND_NCOL,
        frameon=LEGEND_FRAMEON,
        handlelength=1.5,
        handletextpad=0.6,
        columnspacing=1.2,
        borderaxespad=0.0,
        fontsize=plt.rcParams.get("font.size", None),
    )

    out_table = out_base / f"{GROUP}_prob_box_sampleGroup_{tag}_data.csv"
    out_pval = out_base / f"{GROUP}_prob_box_sampleGroup_{tag}_pvalues.csv"
    pd.DataFrame(plot_rows).to_csv(out_table, index=False)
    p_df.to_csv(out_pval, index=False)

    out_png = out_base / f"{GROUP}_prob_box_sampleGroup_{tag}.png"
    out_pdf = out_base / f"{GROUP}_prob_box_sampleGroup_{tag}.pdf"
    save_fig(fig, out_png, out_pdf)

    return out_table, out_pval, out_png

def run_step8(valid_cells: pd.DataFrame, pos_cells: pd.DataFrame, out_dir: Path):
    step8_tables = []

    tbl_a1, fig_a1 = plot_prob_box_grouped_by_metric(valid_cells, "all_valid_cells", out_dir)
    tbl_a2, fig_a2 = plot_prob_box_grouped_by_metric(pos_cells, "positive_cells_only", out_dir)

    tbl_b1, p_b1, fig_b1 = plot_prob_box_grouped_by_sampletype_with_tests(valid_cells, "all_valid_cells", out_dir)
    tbl_b2, p_b2, fig_b2 = plot_prob_box_grouped_by_sampletype_with_tests(pos_cells, "positive_cells_only", out_dir)

    step8_tables.extend([tbl_a1, tbl_a2, tbl_b1, p_b1, tbl_b2, p_b2])

    print("[OK][Step8] saved figures:")
    print(" -", fig_a1)
    print(" -", fig_a2)
    print(" -", fig_b1)
    print(" -", fig_b2)
    print("[OK][Step8] saved tables:")
    for t in step8_tables:
        print(" -", t)

    return step8_tables, (fig_a1, fig_a2, fig_b1, fig_b2)

step8_tables, step8_figs = run_step8(valid_cells, pos_cells, OUT_DIR)

# =========================================================
# STEP 9 (Refactor / Maintainability Upgrade)
# =========================================================
G0_TYPES_PLOT: List[str] = ["g0_exp_red", "g0_exp_green", "g0_dapi", "g0_rand"]
G0_TYPES_LEGEND: List[str] = ["g0_exp_red", "g0_exp_green", "g0_dapi", "g0_rand"]

G0_LABEL_MAP: Dict[str, str] = {
    "g0_exp_red":   "MYC",
    "g0_exp_green": "CEN8",
    "g0_dapi":      "DAPI",
    "g0_rand":      "Random",
}

G0_TYPE_COLORS: Dict[str, str] = {
    "g0_exp_red":   "#ff7f0e",
    "g0_exp_green": "#2ca02c",
    "g0_dapi":      "#1f77b4",
    "g0_rand":      "#7f7f7f",
}

def g0_disp_label(type_key: str) -> str:
    return G0_LABEL_MAP.get(type_key, type_key)

@dataclass(frozen=True)
class Step9Params:
    within_spacing: float = 1.0
    group_gap_extra: float = 0.3
    x_pad: float = 0.3
    y_min_override: float | None = None
    y_max_override: float | None = None
    y_range_eps: float = 1e-9
    header_y0_offset_frac: float = 0.65
    header_h_frac: float = 0.20
    bracket_offset_low_frac: float = 0.6
    bracket_offset_mid_frac: float = 0.4
    bracket_offset_high_frac: float = 0.2
    bracket_h: float = 0.02
    legend_y_offset: float = -0.075
    legend_frameon: bool = False
    violin_width: float = 0.70
    point_size: float = 3.0
    point_lw: float = 0.75
    violin_lw: float = 1.0
    median_lw: float = 1.0
    sep_line_lw: float = 0.8
    y_ticks_max_n: int = 4

P = Step9Params()

def filter_gr_by_cells(gr_df: pd.DataFrame, cell_uid_set: set[str]) -> pd.DataFrame:
    return gr_df.loc[gr_df["cell_uid"].astype(str).isin(cell_uid_set)].copy()

def _prep_gr_df(gr_df: pd.DataFrame) -> pd.DataFrame:
    df = gr_df.copy()
    if "channel" in df.columns:
        df["channel"] = df["channel"].astype(str).str.lower()
    for c in ["g0_exp", "g0_rand", "g0_dapi"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _collect_g0_arrays(gr_df: pd.DataFrame) -> Tuple[List[np.ndarray], List[Tuple[str, str]]]:
    arrays: List[np.ndarray] = []
    meta: List[Tuple[str, str]] = []

    for st in order:
        sub = gr_df.loc[gr_df["sample_type"].astype(str) == st].copy()

        exp_red = sub.loc[sub["channel"] == "red", "g0_exp"].to_numpy(dtype=float)
        exp_red = exp_red[np.isfinite(exp_red)]

        exp_green = sub.loc[sub["channel"] == "green", "g0_exp"].to_numpy(dtype=float)
        exp_green = exp_green[np.isfinite(exp_green)]

        pooled = sub.drop_duplicates("cell_uid")

        dapi = pooled["g0_dapi"].to_numpy(dtype=float)
        dapi = dapi[np.isfinite(dapi)]

        rand = pooled["g0_rand"].to_numpy(dtype=float)
        rand = rand[np.isfinite(rand)]

        type_to_arr = {
            "g0_exp_red": exp_red,
            "g0_exp_green": exp_green,
            "g0_dapi": dapi,
            "g0_rand": rand,
        }

        for type_key in G0_TYPES_PLOT:
            arrays.append(type_to_arr[type_key])
            meta.append((st, type_key))

    return arrays, meta

def _finite_minmax(arrays: List[np.ndarray], eps: float) -> Tuple[float, float]:
    mn, mx = None, None
    for a in arrays:
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        vmin = float(np.nanmin(a))
        vmax = float(np.nanmax(a))
        mn = vmin if mn is None else min(mn, vmin)
        mx = vmax if mx is None else max(mx, vmax)

    if mn is None or mx is None:
        return 0.0, 1.0

    if (mx - mn) < eps:
        mid = 0.5 * (mn + mx)
        return mid - 0.5, mid + 0.5

    return mn, mx

def _compute_positions_sampletype_4types(
    order: List[str],
    within_spacing: float,
    group_gap_extra: float,
) -> Tuple[List[float], List[float], Dict[str, Dict[str, float]]]:
    positions: List[float] = []
    sep_xs: List[float] = []
    st_type_pos: Dict[str, Dict[str, float]] = {}

    pos_cursor = 1.0
    for i, st in enumerate(order):
        type_pos: Dict[str, float] = {}

        for type_key in G0_TYPES_PLOT:
            positions.append(pos_cursor)
            type_pos[type_key] = pos_cursor
            pos_cursor += within_spacing

        st_type_pos[st] = type_pos

        end = pos_cursor - within_spacing
        next_start = end + within_spacing + group_gap_extra
        if i < len(order) - 1:
            sep_xs.append((end + next_start) / 2.0)

        pos_cursor = next_start

    return positions, sep_xs, st_type_pos

def _is_almost_constant(x: np.ndarray, eps: float = 1e-12) -> bool:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return True
    return (np.nanmax(x) - np.nanmin(x)) <= eps

def _p_ind(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]

    if a.size < 2 or b.size < 2:
        return np.nan
    if ttest_ind is None:
        return np.nan

    if _is_almost_constant(a, eps=eps) and _is_almost_constant(b, eps=eps):
        return np.nan

    if abs(np.nanmedian(a) - np.nanmedian(b)) <= eps and (
        (np.nanmax(a) - np.nanmin(a)) <= eps and (np.nanmax(b) - np.nanmin(b)) <= eps
    ):
        return np.nan

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation due to catastrophic cancellation.*",
            category=RuntimeWarning,
        )
        return float(ttest_ind(a, b, equal_var=False).pvalue)

def _compute_p_table(arrays: List[np.ndarray]) -> pd.DataFrame:
    rows = []
    idx = 0
    for st in order:
        exp_red   = arrays[idx + 0]
        exp_green = arrays[idx + 1]
        dapi      = arrays[idx + 2]
        rand      = arrays[idx + 3]

        p_rg = _p_ind(exp_red, exp_green)
        p_rd = _p_ind(exp_red, dapi)
        p_rr = _p_ind(exp_red, rand)

        rows.append({
            "sample_type": st,
            "p_expR_vs_expG": p_rg,
            "p_expR_vs_dapi": p_rd,
            "p_expR_vs_rand": p_rr,
            "star_expR_vs_expG": p_to_star(p_rg),
            "star_expR_vs_dapi": p_to_star(p_rd),
            "star_expR_vs_rand": p_to_star(p_rr),
        })
        idx += 4

    return pd.DataFrame(rows)

def _violin_points_inside(
    ax: plt.Axes,
    arr: np.ndarray,
    pos: float,
    color: str,
    p: Step9Params,
) -> None:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    parts = ax.violinplot(
        [arr],
        positions=[pos],
        widths=p.violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    body = parts["bodies"][0]
    body.set_facecolor("none")
    body.set_edgecolor(color)
    body.set_linewidth(p.violin_lw)
    body.set_alpha(1.0)

    rng = np.random.default_rng(0)
    half_max = p.violin_width / 2.0

    if arr.size > 1 and np.nanmin(arr) != np.nanmax(arr):
        y_grid = np.linspace(np.nanmin(arr), np.nanmax(arr), 256)
        dens_grid, dens_func = _kde_density(arr, y_grid)
        dens_max = float(np.nanmax(dens_grid)) if dens_grid.size else 1.0
        dens_max = max(dens_max, 1e-12)
        dens_y = dens_func(arr)
        half_w = half_max * (dens_y / dens_max)
        half_w = np.clip(half_w, 0.0, half_max)
    else:
        half_w = np.full(arr.shape, half_max, dtype=float)

    u = rng.uniform(-1.0, 1.0, size=arr.size)
    x = pos + u * half_w
    ax.scatter(
        x, arr,
        s=p.point_size,
        facecolors="none",
        edgecolors=color,
        linewidths=p.point_lw,
        zorder=3,
    )

    med = float(np.nanmedian(arr))
    ax.hlines(
        med,
        pos - 0.30,
        pos + 0.30,
        colors=color,
        linestyles="--",
        linewidth=p.median_lw,
        zorder=4,
    )

def _tighten_xlim(ax: plt.Axes, positions: List[float], p: Step9Params) -> None:
    min_pos = float(np.min(positions))
    max_pos = float(np.max(positions))
    x_pad = p.violin_width / 2.0 + p.x_pad
    ax.margins(x=0.0)
    ax.set_xlim(min_pos - x_pad, max_pos + x_pad)

def _set_y_limits_with_headroom(
    ax: plt.Axes,
    data_min_y: float,
    data_max_y: float,
    header_y0: float,
    header_h: float,
    p: Step9Params,
) -> Tuple[float, float]:
    base_ymin = data_min_y if p.y_min_override is None else min(float(p.y_min_override), data_min_y)
    base_ymax = data_max_y if p.y_max_override is None else max(float(p.y_max_override), data_max_y)

    full_ymax = max(base_ymax + header_h, header_y0 + header_h)
    ax.set_ylim(base_ymin, full_ymax)
    return base_ymin, full_ymax

def _set_y_ticks_data_range_only(ax: plt.Axes, data_min_y: float, data_max_y: float, max_n: int) -> None:
    locator = MaxNLocator(nbins=max_n)
    candidate = locator.tick_values(float(data_min_y), float(data_max_y))
    ticks = [t for t in candidate if (t >= data_min_y - 1e-12) and (t <= data_max_y + 1e-12)]

    if len(ticks) == 0:
        ticks = np.linspace(float(data_min_y), float(data_max_y), max_n).tolist()

    if len(ticks) > max_n:
        idxs = np.linspace(0, len(ticks) - 1, max_n).round().astype(int)
        ticks = [ticks[i] for i in idxs]

    ax.set_yticks(ticks)

def _draw_separators(ax: plt.Axes, sep_xs: List[float], p: Step9Params) -> None:
    ymin, ymax = ax.get_ylim()
    for xv in sep_xs:
        ax.vlines(xv, ymin, ymax, colors="black", linewidth=p.sep_line_lw)

def _draw_headers(ax: plt.Axes, sep_xs: List[float], header_y0: float, header_h: float) -> None:
    x_left, x_right = ax.get_xlim()
    boundaries = [x_left] + sep_xs + [x_right]
    for i, st in enumerate(order):
        add_header_box(ax, boundaries[i], boundaries[i + 1], header_y0, header_h, st, lw=0.8)

def _draw_brackets(
    ax: plt.Axes,
    p_df: pd.DataFrame,
    st_type_pos: Dict[str, Dict[str, float]],
    header_y0: float,
    bracket_offsets: Dict[str, float],
    bracket_h: float,
) -> None:
    base_low  = header_y0 - bracket_offsets["low"]
    base_mid  = header_y0 - bracket_offsets["mid"]
    base_high = header_y0 - bracket_offsets["high"]
    fs = plt.rcParams.get("font.size", 6)

    for st in order:
        pv = p_df.loc[p_df["sample_type"] == st].iloc[0]
        pos_map = st_type_pos[st]
        x_expR = pos_map["g0_exp_red"]
        x_expG = pos_map["g0_exp_green"]
        x_dapi = pos_map["g0_dapi"]
        x_rand = pos_map["g0_rand"]

        draw_sig_bracket(ax, x_expR, x_expG, base_low,  bracket_h, pv["star_expR_vs_expG"], lw=0.8, fs=fs)
        draw_sig_bracket(ax, x_expR, x_dapi, base_mid,  bracket_h, pv["star_expR_vs_dapi"], lw=0.8, fs=fs)
        draw_sig_bracket(ax, x_expR, x_rand, base_high, bracket_h, pv["star_expR_vs_rand"], lw=0.8, fs=fs)

def _draw_legend(ax: plt.Axes, p: Step9Params) -> None:
    handles = [
        Patch(facecolor="none", edgecolor=G0_TYPE_COLORS[k], linewidth=1.0, label=g0_disp_label(k))
        for k in G0_TYPES_LEGEND
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, p.legend_y_offset),
        ncol=4,
        frameon=p.legend_frameon,
        handlelength=1.25,
        handletextpad=0.6,
        columnspacing=1.2,
        borderaxespad=0.0,
        fontsize=plt.rcParams.get("font.size", None),
    )

def plot_gr_g0_one_axis(gr_df: pd.DataFrame, tag: str, out_base: Path, p: Step9Params = P):
    df = _prep_gr_df(gr_df)

    arrays, meta = _collect_g0_arrays(df)

    positions, sep_xs, st_type_pos = _compute_positions_sampletype_4types(
        order=order,
        within_spacing=p.within_spacing,
        group_gap_extra=p.group_gap_extra,
    )

    rows = []
    for (st, type_key), arr in zip(meta, arrays):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        for v in arr:
            rows.append({"sample_type": st, "type": type_key, "value": float(v)})
    out_table = out_base / f"{GROUP}_gr_g0_{tag}_data.csv"
    pd.DataFrame(rows).to_csv(out_table, index=False)

    p_df = _compute_p_table(arrays)
    out_pval = out_base / f"{GROUP}_gr_g0_{tag}_pvalues.csv"
    p_df.to_csv(out_pval, index=False)

    data_min_y, data_max_y = _finite_minmax(arrays, eps=p.y_range_eps)
    X = max(float(data_max_y - data_min_y), p.y_range_eps)

    header_y0 = float(data_max_y) + p.header_y0_offset_frac * X
    header_h = p.header_h_frac * X
    bracket_offsets = {
        "low":  p.bracket_offset_low_frac * X,
        "mid":  p.bracket_offset_mid_frac * X,
        "high": p.bracket_offset_high_frac * X,
    }

    apply_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_position(AX_POS)

    for i, (arr, (_, type_key)) in enumerate(zip(arrays, meta)):
        _violin_points_inside(ax, arr, positions[i], color=G0_TYPE_COLORS.get(type_key, "black"), p=p)

    ax.set_xticks([])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_ylabel("Autocorrelation g(0)")
    ax.set_title(f"{GROUP}: g(0) per sample-type ({tag})", pad=10)
    set_frame(ax, show_right=True)

    _tighten_xlim(ax, positions, p)
    _set_y_limits_with_headroom(ax, data_min_y, data_max_y, header_y0, header_h, p)
    _set_y_ticks_data_range_only(ax, data_min_y, data_max_y, p.y_ticks_max_n)

    _draw_separators(ax, sep_xs, p)
    _draw_headers(ax, sep_xs, header_y0, header_h)
    _draw_brackets(ax, p_df, st_type_pos, header_y0, bracket_offsets, p.bracket_h)
    _draw_legend(ax, p)

    out_png = out_base / f"{GROUP}_gr_g0_{tag}.png"
    out_pdf = out_base / f"{GROUP}_gr_g0_{tag}.pdf"
    save_fig(fig, out_png, out_pdf)

    return out_png, out_table, out_pval

valid_uid = set(valid_cells["cell_uid"].astype(str).tolist())
pos_uid   = set(pos_cells["cell_uid"].astype(str).tolist())

gr_valid = filter_gr_by_cells(gr_per_cell_all, valid_uid)
gr_pos   = filter_gr_by_cells(gr_per_cell_all, pos_uid)

g0_valid_png, g0_valid_tbl, g0_valid_pv = plot_gr_g0_one_axis(gr_valid, "all_valid_cells", OUT_DIR, p=P)
g0_pos_png,   g0_pos_tbl,   g0_pos_pv   = plot_gr_g0_one_axis(gr_pos,   "positive_cells_only", OUT_DIR, p=P)

print("[OK][Step9] saved:")
print(" -", g0_valid_png)
print(" -", g0_pos_png)

# =========================================================
# STEP 10 (Refactor / Maintainability Upgrade) - mean±SEM g(r) curves
# (keep unchanged; already supports rmax20/40 in your latest file)
# =========================================================
@dataclass(frozen=True)
class Step10Params:
    r_plot_max: int = 30
    bin_px: int = 1
    lw_exp: float = 1.0
    lw_base: float = 0.8
    alpha_exp: float = 0.18
    alpha_rand: float = 0.12
    alpha_dapi: float = 0.10
    point_ms_exp: float = 2.0
    point_ms_base: float = 2.0
    marker_exp: str = "o"
    marker_rand: str = "o"
    marker_dapi: str = "o"
    point_mew: float = 0.0
    legend_y_shift: float = 0.001
    legend_frameon: bool = True
    legend_handlelength: float = 1.2
    x_ticks_n: int = 5
    y_locator_nbins: int = 5

P10_20 = Step10Params(r_plot_max=20)
P10_40 = Step10Params(r_plot_max=40)

def rbin_rule(r_px: np.ndarray, bin_px: int) -> np.ndarray:
    r = np.asarray(r_px, dtype=int)
    rb = np.zeros_like(r)
    pos = r >= 1
    b = int(max(1, bin_px))
    rb[pos] = 1 + ((r[pos] - 1) // b) * b
    return rb

def _normalize_and_filter_long_df(gr_long_df: pd.DataFrame, cell_uid_set: set, p: Step10Params) -> pd.DataFrame:
    df = gr_long_df.copy()
    df["channel"] = df["channel"].astype(str).str.lower()
    df = df.loc[df["cell_uid"].astype(str).isin(cell_uid_set)].copy()
    df = df.loc[df["r_px"].astype(int) <= int(p.r_plot_max)].copy()
    return df

def _select_red_and_prepare_numeric(df: pd.DataFrame, p: Step10Params) -> pd.DataFrame:
    df_red = df.loc[df["channel"] == "red"].copy()
    for c in ["g_exp", "g_rand", "g_dapi"]:
        if c in df_red.columns:
            df_red[c] = pd.to_numeric(df_red[c], errors="coerce")
    df_red["r_bin"] = rbin_rule(df_red["r_px"].to_numpy(dtype=int), p.bin_px)
    return df_red

def _agg_mean_sem_n(sub: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        sub.groupby("r_bin")[col]
        .agg(mean="mean", sem=lambda x: _sem(x.to_numpy()), n="count")
        .reset_index()
        .sort_values("r_bin")
    )

def _build_curves_by_sample_type(df_red: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    curves: Dict[str, pd.DataFrame] = {}
    for st in order:
        sub = df_red.loc[df_red["sample_type"].astype(str) == st].copy()
        if sub.empty:
            continue
        curves[st] = _agg_mean_sem_n(sub, "g_exp")
    return curves

def _build_pooled_baselines(df_red: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agg_rand = _agg_mean_sem_n(df_red, "g_rand")[["r_bin", "mean", "sem", "n"]]
    agg_dapi = _agg_mean_sem_n(df_red, "g_dapi")[["r_bin", "mean", "sem", "n"]]
    return agg_rand, agg_dapi

def _pvalues_at_r0(df_red: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st in order:
        sub0 = df_red.loc[(df_red["sample_type"].astype(str) == st) & (df_red["r_bin"] == 0)].copy()
        if sub0.empty:
            rows.append({"sample_type": st, "p_g0_vs_rand": np.nan, "p_g0_vs_dapi": np.nan, "n_cells": 0})
            continue

        by_cell = sub0.groupby("cell_uid")[["g_exp", "g_rand", "g_dapi"]].mean(numeric_only=True)
        n_cells = int(by_cell.shape[0])

        p_rand = np.nan
        if n_cells >= 2:
            if ttest_1samp is not None:
                p_rand = float(
                    ttest_1samp(by_cell["g_exp"].to_numpy(dtype=float) - 1.0, 0.0, nan_policy="omit").pvalue
                )
            elif ttest_rel is not None:
                m = np.isfinite(by_cell["g_exp"].to_numpy()) & np.isfinite(by_cell["g_rand"].to_numpy())
                if int(np.sum(m)) >= 2:
                    p_rand = float(ttest_rel(by_cell["g_exp"].to_numpy()[m], by_cell["g_rand"].to_numpy()[m]).pvalue)

        p_dapi = np.nan
        if n_cells >= 2 and ttest_rel is not None:
            m = np.isfinite(by_cell["g_exp"].to_numpy()) & np.isfinite(by_cell["g_dapi"].to_numpy())
            if int(np.sum(m)) >= 2:
                p_dapi = float(ttest_rel(by_cell["g_exp"].to_numpy()[m], by_cell["g_dapi"].to_numpy()[m]).pvalue)

        rows.append({"sample_type": st, "p_g0_vs_rand": p_rand, "p_g0_vs_dapi": p_dapi, "n_cells": n_cells})

    return pd.DataFrame(rows)

def _export_used_rows(df_red: pd.DataFrame, out_base: Path, tag: str) -> Path:
    out_path = out_base / f"{GROUP}_gr_long_{tag}_used_rows.csv"
    df_red.to_csv(out_path, index=False)
    return out_path

def _export_pvalues(p_df: pd.DataFrame, out_base: Path, tag: str) -> Path:
    out_path = out_base / f"{GROUP}_gr_long_{tag}_pvalues.csv"
    p_df.to_csv(out_path, index=False)
    return out_path

def _export_agg_curves(
    curves: Dict[str, pd.DataFrame],
    agg_rand: pd.DataFrame,
    agg_dapi: pd.DataFrame,
    out_base: Path,
    tag: str,
) -> Path:
    rows = []
    for st, agg in curves.items():
        for _, r in agg.iterrows():
            rows.append({"sample_type": st, "r_bin": int(r["r_bin"]), "mean": float(r["mean"]), "sem": float(r["sem"])})
    for _, r in agg_rand.iterrows():
        rows.append({"sample_type": "random_all", "r_bin": int(r["r_bin"]), "mean": float(r["mean"]), "sem": float(r["sem"])})
    for _, r in agg_dapi.iterrows():
        rows.append({"sample_type": "dapi_all", "r_bin": int(r["r_bin"]), "mean": float(r["mean"]), "sem": float(r["sem"])})
    out_path = out_base / f"{GROUP}_gr_long_{tag}_binned_mean_sem.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def _set_axes_style(ax: plt.Axes, p: Step10Params) -> None:
    ax.set_xlim(0, int(p.r_plot_max))
    xticks = np.linspace(0, int(p.r_plot_max), int(p.x_ticks_n))
    ax.xaxis.set_major_locator(FixedLocator(list(xticks)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=int(p.y_locator_nbins)))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

def _plot_mean_sem_band_with_points(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    e: np.ndarray,
    *,
    color,
    lw: float,
    alpha: float,
    marker: str,
    ms: float,
    mew: float,
    zorder_line: int,
    zorder_pts: int,
):
    ax.plot(x, y, color=color, linewidth=lw, zorder=zorder_line)
    ax.fill_between(x, y - e, y + e, alpha=alpha, color=color, zorder=zorder_line - 1)
    ax.plot(
        x, y,
        linestyle="None",
        marker=marker,
        markersize=ms,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=mew,
        zorder=zorder_pts,
    )

def _legend_label_for_group(st: str, p_df: pd.DataFrame) -> str:
    pv = p_df.loc[p_df["sample_type"] == st]
    if pv.empty:
        return st
    n = int(pv["n_cells"].iloc[0])
    pr = pv["p_g0_vs_rand"].iloc[0]
    return f"{st} (n={n}, P={_pval_fmt(pr)})"

def _baseline_n_at_r0(df_red: pd.DataFrame) -> int:
    if "r_bin" not in df_red.columns:
        return 0
    return int(df_red.loc[df_red["r_bin"] == 0, "cell_uid"].astype(str).nunique())

def _legend_proxy_dot_line(color, marker: str, ms: float, lw: float, mew: float) -> Line2D:
    return Line2D(
        [],
        [],
        color=color,
        linewidth=lw,
        marker=marker,
        markersize=ms,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=mew,
    )

def plot_gr_curves(gr_long_df: pd.DataFrame, cell_uid_set: set, tag: str, out_base: Path, p: Step10Params):
    df = _normalize_and_filter_long_df(gr_long_df, cell_uid_set, p)
    if df.empty:
        print("[WARN][Step10] empty df for", tag)
        return None

    df_red = _select_red_and_prepare_numeric(df, p)

    out_used_rows = _export_used_rows(df_red, out_base, tag)

    curves = _build_curves_by_sample_type(df_red)
    agg_rand, agg_dapi = _build_pooled_baselines(df_red)

    p_df = _pvalues_at_r0(df_red)
    out_pvals = _export_pvalues(p_df, out_base, tag)

    out_curve_table = _export_agg_curves(curves, agg_rand, agg_dapi, out_base, tag)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_position(AX_POS_GR)
    ax.set_box_aspect(1)

    ax.set_title(f"{GROUP}: mean±SEM g(r) ({tag})")
    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("Autocorrelation g(r)")

    legend_handles: List[Line2D] = []
    legend_labels: List[str] = []
    for st in order:
        if st not in curves:
            continue

        agg = curves[st]
        x = agg["r_bin"].to_numpy(dtype=int)
        y = agg["mean"].to_numpy(dtype=float)
        e = agg["sem"].to_numpy(dtype=float)

        c = GROUP_COLORS.get(st, None)

        _plot_mean_sem_band_with_points(
            ax, x, y, e,
            color=c,
            lw=p.lw_exp,
            alpha=p.alpha_exp,
            marker=p.marker_exp,
            ms=p.point_ms_exp,
            mew=p.point_mew,
            zorder_line=3,
            zorder_pts=4,
        )

        legend_handles.append(_legend_proxy_dot_line(c, p.marker_exp, p.point_ms_exp, p.lw_exp, p.point_mew))
        legend_labels.append(_legend_label_for_group(st, p_df))

    xr = agg_rand["r_bin"].to_numpy(dtype=int)
    yr = agg_rand["mean"].to_numpy(dtype=float)
    er = agg_rand["sem"].to_numpy(dtype=float)
    c_rand = "#666666"
    _plot_mean_sem_band_with_points(
        ax, xr, yr, er,
        color=c_rand,
        lw=p.lw_base,
        alpha=p.alpha_rand,
        marker=p.marker_rand,
        ms=p.point_ms_base,
        mew=p.point_mew,
        zorder_line=2,
        zorder_pts=3,
    )

    xd = agg_dapi["r_bin"].to_numpy(dtype=int)
    yd = agg_dapi["mean"].to_numpy(dtype=float)
    ed = agg_dapi["sem"].to_numpy(dtype=float)
    c_dapi = "black"
    _plot_mean_sem_band_with_points(
        ax, xd, yd, ed,
        color=c_dapi,
        lw=p.lw_exp,
        alpha=p.alpha_dapi,
        marker=p.marker_dapi,
        ms=p.point_ms_base,
        mew=p.point_mew,
        zorder_line=2,
        zorder_pts=3,
    )

    _set_axes_style(ax, p)

    n_base = _baseline_n_at_r0(df_red)
    rand_label = f"Random (n={n_base})"
    dapi_label = f"DAPI (n={n_base})"

    legend_handles2 = legend_handles + [
        _legend_proxy_dot_line(c_rand, p.marker_rand, p.point_ms_base, p.lw_base, p.point_mew),
        _legend_proxy_dot_line(c_dapi, p.marker_dapi, p.point_ms_base, p.lw_exp, p.point_mew),
    ]
    legend_labels2 = legend_labels + [rand_label, dapi_label]

    fig.legend(
        legend_handles2,
        legend_labels2,
        loc="upper left",
        bbox_to_anchor=(RIGHT_PANEL_X, LEGEND_Y + p.legend_y_shift),
        frameon=p.legend_frameon,
        borderaxespad=0.0,
        handlelength=p.legend_handlelength,
    )

    out_png = out_base / f"{GROUP}_gr_mean_sem_{tag}.png"
    out_pdf = out_base / f"{GROUP}_gr_mean_sem_{tag}.pdf"
    save_fig(fig, out_png, out_pdf)

    return out_png, out_used_rows, out_curve_table, out_pvals

step10_valid_r20 = plot_gr_curves(gr_long_all, valid_uid, "all_valid_cells_rmax20", OUT_DIR, p=P10_20)
step10_pos_r20   = plot_gr_curves(gr_long_all, pos_uid,   "positive_cells_only_rmax20", OUT_DIR, p=P10_20)

step10_valid_r40 = plot_gr_curves(gr_long_all, valid_uid, "all_valid_cells_rmax40", OUT_DIR, p=P10_40)
step10_pos_r40   = plot_gr_curves(gr_long_all, pos_uid,   "positive_cells_only_rmax40", OUT_DIR, p=P10_40)

print("[OK][Step10] outputs (valid, rmax20):")
print(" -", step10_valid_r20[0]); print(" -", step10_valid_r20[1]); print(" -", step10_valid_r20[2]); print(" -", step10_valid_r20[3])
print("[OK][Step10] outputs (positive, rmax20):")
print(" -", step10_pos_r20[0]);   print(" -", step10_pos_r20[1]);   print(" -", step10_pos_r20[2]);   print(" -", step10_pos_r20[3])

print("[OK][Step10] outputs (valid, rmax40):")
print(" -", step10_valid_r40[0]); print(" -", step10_valid_r40[1]); print(" -", step10_valid_r40[2]); print(" -", step10_valid_r40[3])
print("[OK][Step10] outputs (positive, rmax40):")
print(" -", step10_pos_r40[0]);   print(" -", step10_pos_r40[1]);   print(" -", step10_pos_r40[2]);   print(" -", step10_pos_r40[3])


# =========================================================
# =========================================================
# RUN B) PVT1 + CEN8 + DAPI  (treat as red+green+DAPI)
# =========================================================
# =========================================================
#
# NOTE:
# - This second run is the SAME workflow, with ONLY name/label/channel remapping differences.
# - Outputs are written to a separate OUT_DIR to avoid overwriting RUN A.
#
# =========================================================

# =========================================================
# STEP 1: USER CONFIG
# =========================================================
PATHA = Path(r"D:\02.project\18.ecDNA\02.code\v0.1.1\autocorrelation\output_ws\MYC_PVT1")
GROUP = "PVT1_CEN8"
OUT_PREFIX = "merged"

OUT_DIR = Path(r"D:\02.project\18.ecDNA\02.code\v0.1.1\autocorrelation\output_ws_plot\MYC_PVT1\PVT1_CEN8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_MIN_TOTAL_SPOTS = 2
POSITIVE_RED_THRESHOLD = 8
SAMPLE_TYPE_ORDER = ["BRCA", "CRC", "ESCA", "GBM", "PDAC"]

GROUP_COLORS = {
    "BRCA": "#1f77b4",
    "CRC":  "#ff7f0e",
    "ESCA": "#2ca02c",
    "GBM":  "#d62728",
    "PDAC": "#9467bd",
}

TYPE_COLORS = {
    "g0_exp_red":   "#b14744",
    "g0_exp_green": "#097d98",
    "g0_rand":      "#666666",
    "g0_dapi":      "#000000",
}

FIGSIZE = (7.2, 5.2)
AX_POS_GR = [0.08, 0.18, 0.15, 0.18]
RIGHT_PANEL_X = 0.25
LEGEND_Y = 0.35

R_PLOT_MAX = 40
BIN_PX = 1

np.random.seed(0)

# =========================================================
# STEP 3: scan ROI dirs, merge CSVs, add probability columns
# =========================================================
dfs_per_cell, dfs_gr_per_cell, dfs_gr_pvalues, dfs_gr_long = [], [], [], []

for roi_dir in iter_roi_dirs(PATHA):
    doi = roi_dir.name

    f1 = roi_dir / "per_cell_summary.csv"
    f2 = roi_dir / "gr_per_cell.csv"
    f3 = roi_dir / "gr_pvalues.csv"
    f4 = roi_dir / "gr_long.csv"

    df1 = read_if_exists(f1)
    df2 = read_if_exists(f2)
    df3 = read_if_exists(f3)
    df4 = read_if_exists(f4)

    if df1 is not None and not df1.empty:
        df1["doi"] = doi
        dfs_per_cell.append(df1)

    if df2 is not None and not df2.empty:
        df2["doi"] = doi
        dfs_gr_per_cell.append(df2)

    if df3 is not None and not df3.empty:
        df3["doi"] = doi
        dfs_gr_pvalues.append(df3)

    if df4 is not None and not df4.empty:
        df4["doi"] = doi
        dfs_gr_long.append(df4)

per_cell_all = pd.concat(dfs_per_cell, ignore_index=True) if dfs_per_cell else pd.DataFrame()
gr_per_cell_all = pd.concat(dfs_gr_per_cell, ignore_index=True) if dfs_gr_per_cell else pd.DataFrame()
gr_pvalues_all = pd.concat(dfs_gr_pvalues, ignore_index=True) if dfs_gr_pvalues else pd.DataFrame()
gr_long_all = pd.concat(dfs_gr_long, ignore_index=True) if dfs_gr_long else pd.DataFrame()

per_cell_all = add_sample_type(add_cell_uid(per_cell_all))
gr_per_cell_all = add_sample_type(add_cell_uid(gr_per_cell_all))
gr_long_all = add_sample_type(add_cell_uid(gr_long_all))

BASE_GROUP = "MYC_PVT1"
if (not per_cell_all.empty) and ("group" in per_cell_all.columns):
    per_cell_all = per_cell_all.loc[per_cell_all["group"].astype(str) == BASE_GROUP].copy()
if (not gr_per_cell_all.empty) and ("group" in gr_per_cell_all.columns):
    gr_per_cell_all = gr_per_cell_all.loc[gr_per_cell_all["group"].astype(str) == BASE_GROUP].copy()
if (not gr_long_all.empty) and ("group" in gr_long_all.columns):
    gr_long_all = gr_long_all.loc[gr_long_all["group"].astype(str) == BASE_GROUP].copy()

# ---- TASK1: remap 4-color -> 3-color (PVT1 + CEN8) ----
per_cell_all = remap_per_cell_4to3(per_cell_all, mode="PVT1_CEN8")
gr_per_cell_all = remap_gr_channels_4to3(gr_per_cell_all, mode="PVT1_CEN8")
gr_long_all = remap_gr_channels_4to3(gr_long_all, mode="PVT1_CEN8")

if not per_cell_all.empty:
    for c in [
        "coloc_red_green_frac_red_with_green",
        "coloc_red_green_frac_green_with_red",
        "within_red_frac_with_neighbor",
        "within_green_frac_with_neighbor",
    ]:
        if c in per_cell_all.columns:
            per_cell_all[c] = pd.to_numeric(per_cell_all[c], errors="coerce")

    if ("coloc_red_green_frac_red_with_green" in per_cell_all.columns) and ("coloc_red_green_frac_green_with_red" in per_cell_all.columns):
        per_cell_all["pro_coloc_red-green"] = per_cell_all[[
            "coloc_red_green_frac_red_with_green",
            "coloc_red_green_frac_green_with_red"
        ]].mean(axis=1, skipna=True)
    else:
        per_cell_all["pro_coloc_red-green"] = np.nan

    per_cell_all["pro_coloc_red-red"] = (
        per_cell_all["within_red_frac_with_neighbor"]
        if "within_red_frac_with_neighbor" in per_cell_all.columns else np.nan
    )
    per_cell_all["pro_coloc_green-green"] = (
        per_cell_all["within_green_frac_with_neighbor"]
        if "within_green_frac_with_neighbor" in per_cell_all.columns else np.nan
    )

out_per_cell   = OUT_DIR / f"{OUT_PREFIX}_per_cell_summary.csv"
out_gr_percell = OUT_DIR / f"{OUT_PREFIX}_gr_per_cell.csv"
out_gr_pvalues = OUT_DIR / f"{OUT_PREFIX}_gr_pvalues.csv"
out_gr_long    = OUT_DIR / f"{OUT_PREFIX}_gr_long.csv"

per_cell_all.to_csv(out_per_cell, index=False)
gr_per_cell_all.to_csv(out_gr_percell, index=False)
gr_pvalues_all.to_csv(out_gr_pvalues, index=False)
gr_long_all.to_csv(out_gr_long, index=False)

print("[OK] merged CSV written:")
print(" -", out_per_cell)
print(" -", out_gr_percell)
print(" -", out_gr_pvalues)
print(" -", out_gr_long)

for col in ["n_spots_red", "n_spots_green"]:
    if col in per_cell_all.columns:
        per_cell_all[col] = pd.to_numeric(per_cell_all[col], errors="coerce").fillna(0).astype(int)

per_cell_all["is_valid"] = (per_cell_all["n_spots_red"] + per_cell_all["n_spots_green"]) >= VALID_MIN_TOTAL_SPOTS
valid_cells = per_cell_all.loc[per_cell_all["is_valid"]].copy()
valid_cells["is_positive"] = valid_cells["n_spots_red"] >= POSITIVE_RED_THRESHOLD
pos_cells = valid_cells.loc[valid_cells["is_positive"]].copy()

print(f"[INFO] total cells={len(per_cell_all)}, valid={len(valid_cells)}, positive={len(pos_cells)}")

order = [x for x in SAMPLE_TYPE_ORDER if x in valid_cells["sample_type"].astype(str).unique().tolist()]
rest  = [x for x in valid_cells["sample_type"].astype(str).unique().tolist() if x not in order]
order = order + sorted(rest)

# STEP 4
ratio_df = (
    valid_cells.groupby("sample_type")["is_positive"]
    .agg(n_valid="count", n_pos="sum")
    .reset_index()
)
ratio_df["n_neg"] = ratio_df["n_valid"] - ratio_df["n_pos"]
ratio_df["pos_ratio"] = ratio_df["n_pos"] / ratio_df["n_valid"].replace(0, np.nan)
ratio_df["neg_ratio"] = ratio_df["n_neg"] / ratio_df["n_valid"].replace(0, np.nan)

order = [x for x in SAMPLE_TYPE_ORDER if x in ratio_df["sample_type"].tolist()]
rest = [x for x in ratio_df["sample_type"].tolist() if x not in order]
order = order + sorted(rest)

ratio_df["sample_type"] = pd.Categorical(ratio_df["sample_type"], categories=order, ordered=True)
ratio_df = ratio_df.sort_values("sample_type").reset_index(drop=True)

out_table = OUT_DIR / f"{GROUP}_positive_ratio_stackedbar_data.csv"
ratio_df.to_csv(out_table, index=False)

apply_plot_style()
fig, ax = plt.subplots(figsize=(7.2, 5.2))
set_small_axes_with_right_panel(fig, ax, width=0.17, height=0.16)

x = np.arange(len(ratio_df))
pos = ratio_df["pos_ratio"].to_numpy(dtype=float)
neg = ratio_df["neg_ratio"].to_numpy(dtype=float)

bar_width = 0.65
edge_lw = 0.6
edge_col = "black"

ax.bar(
    x, pos,
    width=bar_width,
    label="Positive (red>6)",
    color="#b14744",
    edgecolor=edge_col,
    linewidth=edge_lw
)
ax.bar(
    x, neg,
    width=bar_width,
    bottom=pos,
    label="Negative (red<=6)",
    color="#097d98",
    edgecolor=edge_col,
    linewidth=edge_lw
)

ax.set_xticks(x)
ax.set_xticklabels(
    ratio_df["sample_type"].astype(str).tolist(),
    rotation=45,
    ha="right",
    rotation_mode="anchor"
)

ax.set_ylabel("Proportion (valid cells)")
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

for xi, n_valid in zip(x, ratio_df["n_valid"].to_numpy(dtype=int)):
    ax.text(
        xi, 1.02, f"n={n_valid}",
        ha="center", va="bottom"
    )

fig.suptitle(f"{GROUP}: Positive cell proportion by cancer type", x=0.15, y=0.4)

fig.legend(
    loc="center left",
    bbox_to_anchor=(0.3, 0.3),
    frameon=False
)

save_fig(
    fig,
    OUT_DIR / f"{GROUP}_positive_ratio_stackedbar.png",
    OUT_DIR / f"{GROUP}_positive_ratio_stackedbar.pdf"
)

print("[OK] plot saved, data table saved:", out_table)

# STEP 6 (reuse existing helper functions already defined above)
colors = [GROUP_COLORS.get(st, "black") for st in order]

data_all, labels, n_all = group_arrays_and_n(valid_cells, "n_spots_red", order)
apply_plot_style()
fig, ax = plt.subplots(figsize=(7.2, 5.2))
set_small_axes_with_right_panel(fig, ax, width=0.17, height=0.16)

out_table_all = OUT_DIR / f"{GROUP}_violin_red_all_valid_data.csv"
violin_outline_points_inside_and_median(
    ax,
    data_by_group=data_all,
    group_labels=labels,
    title=f"{GROUP}: copy number (all valid cells)",
    ylabel=f"{GROUP} spots per nucleus",
    n_by_group=n_all,
    out_table=out_table_all,
    colors_by_group=colors
)
save_fig(fig, OUT_DIR / f"{GROUP}_violin_red_all_valid.png", OUT_DIR / f"{GROUP}_violin_red_all_valid.pdf")

data_pos, labels, n_pos = group_arrays_and_n(pos_cells, "n_spots_red", order)
apply_plot_style()
fig, ax = plt.subplots(figsize=(7.2, 5.2))
set_small_axes_with_right_panel(fig, ax, width=0.15, height=0.15)

out_table_pos = OUT_DIR / f"{GROUP}_violin_red_positive_only_data.csv"
violin_outline_points_inside_and_median(
    ax,
    data_by_group=data_pos,
    group_labels=labels,
    title=f"{GROUP}: copy number (positive cells only)",
    ylabel=f"{GROUP} spots per nucleus",
    n_by_group=n_pos,
    out_table=out_table_pos,
    colors_by_group=colors
)
save_fig(fig, OUT_DIR / f"{GROUP}_violin_red_positive_only.png", OUT_DIR / f"{GROUP}_violin_red_positive_only.pdf")

print("[OK] saved:")
print(" -", OUT_DIR / f"{GROUP}_violin_red_all_valid.png")
print(" -", OUT_DIR / f"{GROUP}_violin_red_positive_only.png")

# STEP 8: update display labels for PVT1_CEN8 and run (reusing the same functions)
LABEL_MAP = {
    "red-green": "PVT1 and CEN8",
    "red-red": "PVT1 and PVT1",
    "green-green": "CEN8 and CEN8",
}
step8_tables, step8_figs = run_step8(valid_cells, pos_cells, OUT_DIR)

# STEP 9: update legend labels for PVT1_CEN8 and run
G0_LABEL_MAP = {
    "g0_exp_red":   "PVT1",
    "g0_exp_green": "CEN8",
    "g0_dapi":      "DAPI",
    "g0_rand":      "Random",
}

valid_uid = set(valid_cells["cell_uid"].astype(str).tolist())
pos_uid   = set(pos_cells["cell_uid"].astype(str).tolist())

gr_valid = filter_gr_by_cells(gr_per_cell_all, valid_uid)
gr_pos   = filter_gr_by_cells(gr_per_cell_all, pos_uid)

g0_valid_png, g0_valid_tbl, g0_valid_pv = plot_gr_g0_one_axis(gr_valid, "all_valid_cells", OUT_DIR, p=P)
g0_pos_png,   g0_pos_tbl,   g0_pos_pv   = plot_gr_g0_one_axis(gr_pos,   "positive_cells_only", OUT_DIR, p=P)

print("[OK][Step9] saved:")
print(" -", g0_valid_png)
print(" -", g0_pos_png)

# STEP 10: run twice (rmax20/rmax40) as above
step10_valid_r20 = plot_gr_curves(gr_long_all, valid_uid, "all_valid_cells_rmax20", OUT_DIR, p=P10_20)
step10_pos_r20   = plot_gr_curves(gr_long_all, pos_uid,   "positive_cells_only_rmax20", OUT_DIR, p=P10_20)

step10_valid_r40 = plot_gr_curves(gr_long_all, valid_uid, "all_valid_cells_rmax40", OUT_DIR, p=P10_40)
step10_pos_r40   = plot_gr_curves(gr_long_all, pos_uid,   "positive_cells_only_rmax40", OUT_DIR, p=P10_40)

print("[OK][Step10] outputs (valid, rmax20):")
print(" -", step10_valid_r20[0]); print(" -", step10_valid_r20[1]); print(" -", step10_valid_r20[2]); print(" -", step10_valid_r20[3])
print("[OK][Step10] outputs (positive, rmax20):")
print(" -", step10_pos_r20[0]);   print(" -", step10_pos_r20[1]);   print(" -", step10_pos_r20[2]);   print(" -", step10_pos_r20[3])

print("[OK][Step10] outputs (valid, rmax40):")
print(" -", step10_valid_r40[0]); print(" -", step10_valid_r40[1]); print(" -", step10_valid_r40[2]); print(" -", step10_valid_r40[3])
print("[OK][Step10] outputs (positive, rmax40):")
print(" -", step10_pos_r40[0]);   print(" -", step10_pos_r40[1]);   print(" -", step10_pos_r40[2]);   print(" -", step10_pos_r40[3])

