#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator

# -------------------------
# Optional deps (KDE / stats)
# -------------------------
try:
    from scipy.stats import gaussian_kde, ttest_ind, ttest_rel, ttest_1samp
except Exception:
    gaussian_kde = None
    ttest_ind = None
    ttest_rel = None
    ttest_1samp = None


# =========================================================
# CONFIG
# =========================================================
@dataclass(frozen=True)
class Config:
    INPUT_ROOT: Path          # 只读：扫描 ROI 子文件夹
    OUT_DIR: Path             # 只写：所有输出(表+图)
    GROUP: str

    OUT_PREFIX: str = "merged"

    VALID_MIN_TOTAL_SPOTS: int = 2
    POSITIVE_RED_THRESHOLD: int = 6

    SAMPLE_TYPE_ORDER: Tuple[str, ...] = ("BRCA", "CRC", "ESCA", "GBM", "PDAC")

    # Step8 gr_long
    R_PLOT_MAX: int = 40
    BIN_PX: int = 1

    FIGSIZE: Tuple[float, float] = (7.2, 5.2)

    GROUP_COLORS: Dict[str, str] = None  # set by factory
    TYPE_COLORS: Dict[str, str] = None   # set by factory


def make_config(input_root: str, out_dir: str, group: str) -> Config:
    group_colors = {
        "BRCA": "#1f77b4",
        "CRC":  "#ff7f0e",
        "ESCA": "#2ca02c",
        "GBM":  "#d62728",
        "PDAC": "#9467bd",
    }
    type_colors = {
        "g0_exp_red":   "#b14744",
        "g0_exp_green": "#097d98",
        "g0_rand":      "#666666",
        "g0_dapi":      "#000000",
    }
    cfg = Config(
        INPUT_ROOT=Path(input_root),
        OUT_DIR=Path(out_dir),
        GROUP=group,
        GROUP_COLORS=group_colors,
        TYPE_COLORS=type_colors,
    )
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    return cfg


# =========================================================
# STYLE / IO UTILS
# =========================================================
def apply_plot_style() -> None:
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


def save_fig(fig: plt.Figure, out_png: Path, out_pdf: Optional[Path] = None, dpi: int = 300) -> None:
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def set_small_axes_with_right_panel(
    ax: plt.Axes,
    left: float = 0.08,
    bottom: float = 0.18,
    width: float = 0.55,
    height: float = 0.65,
) -> None:
    ax.set_position([left, bottom, width, height])


# =========================================================
# DATA UTILS
# =========================================================
def read_if_exists(fp: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(fp) if fp.exists() else None


def iter_roi_dirs(root: Path) -> Iterable[Path]:
    for d in sorted(root.iterdir()):
        if d.is_dir():
            yield d


def add_sample_type(df: pd.DataFrame, sample_col: str = "sample_id") -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["sample_type"] = df[sample_col].astype(str).str.split("_").str[0]
    return df


def add_cell_uid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "sample_id" in df.columns and "cell_id" in df.columns:
        df["cell_uid"] = df["sample_id"].astype(str) + "::" + df["cell_id"].astype(str)
    return df


def ensure_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    return float(np.nanstd(x, ddof=1) / np.sqrt(x.size))


def pval_fmt(p: float) -> str:
    if p is None or (not np.isfinite(p)):
        return "NA"
    return f"{p:.3g}"


# =========================================================
# STEP 1: Merge ROI CSVs + build probability columns
# =========================================================
def step1_merge(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dfs_per_cell, dfs_gr_per_cell, dfs_gr_pvalues, dfs_gr_long = [], [], [], []

    for roi_dir in iter_roi_dirs(cfg.INPUT_ROOT):
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
            df1 = df1.copy()
            df1["doi"] = doi
            dfs_per_cell.append(df1)

        if df2 is not None and not df2.empty:
            df2 = df2.copy()
            df2["doi"] = doi
            dfs_gr_per_cell.append(df2)

        if df3 is not None and not df3.empty:
            df3 = df3.copy()
            df3["doi"] = doi
            dfs_gr_pvalues.append(df3)

        if df4 is not None and not df4.empty:
            df4 = df4.copy()
            df4["doi"] = doi
            dfs_gr_long.append(df4)

    per_cell_all = pd.concat(dfs_per_cell, ignore_index=True) if dfs_per_cell else pd.DataFrame()
    gr_per_cell_all = pd.concat(dfs_gr_per_cell, ignore_index=True) if dfs_gr_per_cell else pd.DataFrame()
    gr_pvalues_all = pd.concat(dfs_gr_pvalues, ignore_index=True) if dfs_gr_pvalues else pd.DataFrame()
    gr_long_all = pd.concat(dfs_gr_long, ignore_index=True) if dfs_gr_long else pd.DataFrame()

    # ---- build new probability columns on per_cell_all ----
    if not per_cell_all.empty:
        per_cell_all = ensure_numeric(per_cell_all, [
            "coloc_red_green_frac_red_with_green",
            "coloc_red_green_frac_green_with_red",
            "within_red_frac_with_neighbor",
            "within_green_frac_with_neighbor",
        ])

        if {"coloc_red_green_frac_red_with_green", "coloc_red_green_frac_green_with_red"}.issubset(per_cell_all.columns):
            per_cell_all["pro_coloc_red-green"] = per_cell_all[
                ["coloc_red_green_frac_red_with_green", "coloc_red_green_frac_green_with_red"]
            ].mean(axis=1, skipna=True)
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

    # ---- write merged outputs (ONLY to OUT_DIR) ----
    per_cell_all.to_csv(cfg.OUT_DIR / f"{cfg.OUT_PREFIX}_per_cell_summary.csv", index=False)
    gr_per_cell_all.to_csv(cfg.OUT_DIR / f"{cfg.OUT_PREFIX}_gr_per_cell.csv", index=False)
    gr_pvalues_all.to_csv(cfg.OUT_DIR / f"{cfg.OUT_PREFIX}_gr_pvalues.csv", index=False)
    gr_long_all.to_csv(cfg.OUT_DIR / f"{cfg.OUT_PREFIX}_gr_long.csv", index=False)

    print("[OK] Step1 merged CSVs written to:", cfg.OUT_DIR)
    return per_cell_all, gr_per_cell_all, gr_pvalues_all, gr_long_all


# =========================================================
# STEP 2: unify fields + valid/positive + order
# =========================================================
def step2_prepare_cells(cfg: Config, per_cell_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    per_cell_all = add_sample_type(add_cell_uid(per_cell_all))

    if not per_cell_all.empty and "group" in per_cell_all.columns:
        per_cell_all = per_cell_all.loc[per_cell_all["group"].astype(str) == cfg.GROUP].copy()

    for col in ["n_spots_red", "n_spots_green"]:
        if col in per_cell_all.columns:
            per_cell_all[col] = pd.to_numeric(per_cell_all[col], errors="coerce").fillna(0).astype(int)

    per_cell_all["is_valid"] = (
        (per_cell_all["n_spots_red"] + per_cell_all["n_spots_green"]) >= cfg.VALID_MIN_TOTAL_SPOTS
        if not per_cell_all.empty else False
    )
    valid_cells = per_cell_all.loc[per_cell_all["is_valid"]].copy()

    valid_cells["is_positive"] = (
        valid_cells["n_spots_red"] >= cfg.POSITIVE_RED_THRESHOLD
        if not valid_cells.empty else False
    )
    pos_cells = valid_cells.loc[valid_cells["is_positive"]].copy()

    present = valid_cells["sample_type"].astype(str).unique().tolist() if not valid_cells.empty else []
    order = [x for x in cfg.SAMPLE_TYPE_ORDER if x in present]
    order += sorted([x for x in present if x not in order])

    print(f"[INFO] total={len(per_cell_all)}, valid={len(valid_cells)}, positive={len(pos_cells)}")
    return per_cell_all, valid_cells, pos_cells, order


# =========================================================
# STEP 3: positive ratio stacked bar
# =========================================================
def step3_plot_ratio(cfg: Config, valid_cells: pd.DataFrame, order: Sequence[str]) -> None:
    ratio_df = (
        valid_cells.groupby("sample_type")["is_positive"]
        .agg(n_valid="count", n_pos="sum")
        .reset_index()
    )
    ratio_df["n_neg"] = ratio_df["n_valid"] - ratio_df["n_pos"]
    ratio_df["pos_ratio"] = ratio_df["n_pos"] / ratio_df["n_valid"].replace(0, np.nan)
    ratio_df["neg_ratio"] = ratio_df["n_neg"] / ratio_df["n_valid"].replace(0, np.nan)

    ratio_df["sample_type"] = pd.Categorical(ratio_df["sample_type"], categories=list(order), ordered=True)
    ratio_df = ratio_df.sort_values("sample_type").reset_index(drop=True)

    ratio_df.to_csv(cfg.OUT_DIR / f"{cfg.GROUP}_positive_ratio_stackedbar_data.csv", index=False)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=cfg.FIGSIZE)
    set_small_axes_with_right_panel(ax, width=0.17, height=0.16)

    x = np.arange(len(ratio_df))
    pos = ratio_df["pos_ratio"].to_numpy(dtype=float)
    neg = ratio_df["neg_ratio"].to_numpy(dtype=float)

    bar_width = 0.65
    edge_lw = 0.6
    edge_col = "black"

    ax.bar(x, pos, width=bar_width, label="Positive (red>6)",
           color="#b14744", edgecolor=edge_col, linewidth=edge_lw)
    ax.bar(x, neg, width=bar_width, bottom=pos, label="Negative (red<=6)",
           color="#097d98", edgecolor=edge_col, linewidth=edge_lw)

    ax.set_xticks(x)
    ax.set_xticklabels(ratio_df["sample_type"].astype(str).tolist(),
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Proportion (valid cells)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for xi, n_valid in zip(x, ratio_df["n_valid"].to_numpy(dtype=int)):
        ax.text(xi, 1.02, f"n={n_valid}", ha="center", va="bottom")

    fig.suptitle(f"{cfg.GROUP}: Positive cell proportion by cancer type", x=0.15, y=0.4)
    fig.legend(loc="center left", bbox_to_anchor=(0.3, 0.3), frameon=False)

    save_fig(fig,
             cfg.OUT_DIR / f"{cfg.GROUP}_positive_ratio_stackedbar.png",
             cfg.OUT_DIR / f"{cfg.GROUP}_positive_ratio_stackedbar.pdf")

    print("[OK] Step3 ratio plot saved.")


# =========================================================
# KDE + points-inside helper (single source of truth)
# =========================================================
def kde_density(y: np.ndarray, y_grid: np.ndarray):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if y.size < 2:
        dens_grid = np.ones_like(y_grid, dtype=float)
        return dens_grid, (lambda yi: np.ones_like(np.asarray(yi, dtype=float), dtype=float))

    if gaussian_kde is not None:
        kde = gaussian_kde(y)
        dens_grid = np.clip(kde(y_grid), 1e-12, None)
        return dens_grid, (lambda yi: np.clip(kde(yi), 1e-12, None))

    # fallback: histogram density
    bins = max(20, int(np.sqrt(y.size)))
    hist, edges = np.histogram(y, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.clip(hist, 1e-12, None)

    def dens_func(yi):
        yi = np.asarray(yi, dtype=float)
        return np.clip(np.interp(yi, centers, hist, left=hist[0], right=hist[-1]), 1e-12, None)

    return dens_func(y_grid), dens_func


def violin_points_inside(ax: plt.Axes, arr: np.ndarray, pos: float, color,
                         width: float = 0.75,
                         point_size: float = 3,
                         point_lw: float = 0.75,
                         violin_lw: float = 1.0,
                         median_lw: float = 1.0,
                         rng: Optional[np.random.Generator] = None) -> None:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    parts = ax.violinplot([arr], positions=[pos], widths=width,
                          showmeans=False, showmedians=False, showextrema=False)
    body = parts["bodies"][0]
    body.set_facecolor("none")
    body.set_edgecolor(color)
    body.set_linewidth(violin_lw)
    body.set_alpha(1.0)

    if rng is None:
        rng = np.random.default_rng(0)

    half_max = width / 2.0
    y_grid = np.linspace(np.nanmin(arr), np.nanmax(arr), 256) if arr.size > 1 else np.array([0.0, 1.0])
    dens_grid, dens_func = kde_density(arr, y_grid)
    dens_max = max(float(np.nanmax(dens_grid)) if dens_grid.size else 1.0, 1e-12)

    dens_y = dens_func(arr)
    half_w = np.clip(half_max * (dens_y / dens_max), 0.0, half_max)

    u = rng.uniform(-1.0, 1.0, size=arr.size)
    x = pos + u * half_w
    ax.scatter(x, arr, s=point_size, facecolors="none", edgecolors=color, linewidths=point_lw, zorder=3)

    med = float(np.nanmedian(arr))
    ax.hlines(med, pos - 0.30, pos + 0.30, colors=color, linestyles="--", linewidth=median_lw, zorder=4)


# =========================================================
# STEP 4: red copy number violins (reuse your established style)
# =========================================================
def group_arrays_and_n(df: pd.DataFrame, value_col: str, order: Sequence[str]) -> Tuple[List[np.ndarray], List[str], List[int]]:
    data, labels, nlist = [], [], []
    for st in order:
        sub = df.loc[df["sample_type"].astype(str) == st]
        data.append(sub[value_col].to_numpy(dtype=float))
        labels.append(st)
        nlist.append(int(len(sub)))
    return data, labels, nlist


def step4_red_violin(cfg: Config, valid_cells: pd.DataFrame, pos_cells: pd.DataFrame, order: Sequence[str]) -> None:
    if "n_spots_red" not in valid_cells.columns:
        print("[WARN] Step4 skipped: n_spots_red missing.")
        return

    def plot_one(df: pd.DataFrame, tag: str) -> None:
        data, labels, n_by = group_arrays_and_n(df, "n_spots_red", order)
        colors = [cfg.GROUP_COLORS.get(st, "black") for st in labels]

        clean = [np.asarray(a, float)[np.isfinite(a)] for a in data]
        all_vals = np.concatenate([a for a in clean if a.size > 0], axis=0) if any(a.size > 0 for a in clean) else np.array([0.0, 1.0])
        y_min, y_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        y_rng = max(y_max - y_min, 1e-9)
        pad = 0.06 * y_rng
        n_y = y_max + pad

        out_table = cfg.OUT_DIR / f"{cfg.GROUP}_violin_red_{tag}_data.csv"
        rows = [{"sample_type": st, "value": float(v)} for st, arr in zip(labels, clean) for v in arr]
        pd.DataFrame(rows).to_csv(out_table, index=False)

        apply_plot_style()
        fig, ax = plt.subplots(figsize=cfg.FIGSIZE)
        set_small_axes_with_right_panel(ax, width=0.17, height=0.16)
        ax.set_box_aspect(1)

        positions = np.arange(1, len(labels) + 1)
        rng = np.random.default_rng(0)

        for i, arr in enumerate(clean):
            violin_points_inside(ax, arr, positions[i], color=colors[i], rng=rng)

        for i, nval in enumerate(n_by):
            ax.text(positions[i], n_y, f"n={int(nval)}", ha="center", va="bottom", color="black")

        ax.set_title(f"{cfg.GROUP}: copy number ({tag})", pad=10)
        ax.set_ylabel(f"{cfg.GROUP} spots per nucleus")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_ylim(y_min, n_y + 0.08 * y_rng)

        save_fig(fig,
                 cfg.OUT_DIR / f"{cfg.GROUP}_violin_red_{tag}.png",
                 cfg.OUT_DIR / f"{cfg.GROUP}_violin_red_{tag}.pdf")

    plot_one(valid_cells, "all_valid_cells")
    plot_one(pos_cells, "positive_cells_only")
    print("[OK] Step4 red violins saved.")


# =========================================================
# MAIN
# =========================================================
def main():
    # 你只需要改这里：输入目录、输出目录、GROUP
    cfg = make_config(
        input_root=r"D:\02.project\18.ecDNA\02.code\v0.1.1\autocorrelation\output_dc\MUC4",
        out_dir=r"D:\02.project\18.ecDNA\02.code\v0.1.1\autocorrelation\plot_out\MUC4",
        group="MUC4",
    )

    per_cell_all, gr_per_cell_all, gr_pvalues_all, gr_long_all = step1_merge(cfg)
    per_cell_all, valid_cells, pos_cells, order = step2_prepare_cells(cfg, per_cell_all)

    if not valid_cells.empty:
        step3_plot_ratio(cfg, valid_cells, order)
        step4_red_violin(cfg, valid_cells, pos_cells, order)

    print("[DONE] outputs in:", cfg.OUT_DIR)


if __name__ == "__main__":
    main()
