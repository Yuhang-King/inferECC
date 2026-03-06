#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inferECC_pipeline_v1.0.0.py
============================================================
Multi-process optimized inferECC main pipeline (v1.0.0)

Key upgrades vs the original tutorial script:
1) Add --n_cpu (default: 20) to parallelize *per-cell* computations.
   - Each barcode (cell) is processed in parallel for heavy per-cell steps.
2) Standardized IO, args, bilingual comments (中英双语), maintainable structure.
3) Step-level timing + a single consolidated log file in the output directory.
4) Output filenames are kept consistent with the legacy pipeline where possible.

Notes
-----
- This script expects the `inferECC` Python package (and its functions) to be importable.
- Plotting can be expensive on large datasets. Use --plots to enable heavy plots.
- Multiprocessing uses ProcessPoolExecutor; for HPC/cluster usage this is usually robust.

Author: 王宇航 | YUHANG WANG
Contact: wyh.scut@foxmail.com
License: All rights reserved.
"""

from __future__ import annotations

import os
import re
import sys
import time
import math
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

# Optional heavy deps (only used if --plots)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# --- inferECC core (explicit imports; avoid __all__ limitation) ---
import inferECC as ife

# Core IO / preprocessing
from inferECC.io.bgi import read_bgi_as_dataframe
from inferECC.preprocessing.transform import Transform
from inferECC.preprocessing.segmentation import fragments_segmentation
from inferECC.preprocessing.normalize import Normalize

# Core main functions
from inferECC.main.caculate import caculate_fragments_number
from inferECC.main.cutoff import cutoff_fragments_number
from inferECC.main.sample import sample_cell
from inferECC.main.uniform import caculate_uniform
from inferECC.main.tss_site import tss_site
from inferECC.main.gene_structure import tss_region, genebody_region, intergenic_region
from inferECC.main.enrichment_score import tss_score, genebody_score
from inferECC.main.merge import sum_by, Neighbor, neighbor_correlation

# Visualization (plots=True 时会用到；显式导入避免 __all__ 限制)
from inferECC.visualization.tss_enrichment_plot import enrichment_plot
from inferECC.visualization.plotting import bp_from_tss
from inferECC.visualization.density import fragments_length, coverage_density
from inferECC.visualization.heatmap import heatmap_chr
from inferECC.visualization.heatmap_row import (
    ochh_mtx,
    ochh_mtx_ks_test,
    heatmap_raw_plot,
    heatmap_fi_plot,
)

from concurrent.futures import ProcessPoolExecutor, as_completed


# -----------------------------
# Logging / timing utilities
# -----------------------------
class StepTimer:
    """Context manager to time a step and log duration.
    计时器：用于记录每个步骤耗时并写入日志。
    """
    def __init__(self, logger: logging.Logger, step_name: str):
        self.logger = logger
        self.step_name = step_name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info(f"[START] {self.step_name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        if exc_type is None:
            self.logger.info(f"[DONE ] {self.step_name}  ({dt:.2f}s)")
        else:
            self.logger.exception(f"[FAIL ] {self.step_name}  ({dt:.2f}s)")
        return False


def setup_logger(outdir: str) -> logging.Logger:
    """Create logger that writes both to console and to file."""
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "inferECC_pipeline_v1.0.0.log")

    logger = logging.getLogger("inferECC_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_path}")
    return logger


# -----------------------------
# Parallel helpers
# -----------------------------
def _safe_reset_index(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure index becomes normal columns (avoid MultiIndex surprises)."""
    if isinstance(df.index, pd.MultiIndex):
        return df.reset_index(drop=False)
    return df.reset_index(drop=False)


def _split_groups(df: pd.DataFrame, key: str) -> List[Tuple[str, pd.DataFrame]]:
    """Split df by key into (group_value, sub_df)."""
    groups = []
    for g, sub in df.groupby(key, sort=False):
        groups.append((g, sub.copy()))
    return groups


def _apply_func_to_group(args):
    """
    Top-level worker for multiprocessing (pickle-safe).
    args: (group_value, sub_df, func)
    """
    g, sub_df, func = args
    return g, func(sub_df)


def parallel_group_apply(
    df: pd.DataFrame,
    group_key: str,
    func,  # callable
    n_cpu: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Apply a function per barcode in parallel (pickle-safe).

    Key point:
    - Do NOT define local worker inside this function (not picklable).
    - Use a top-level worker: _apply_func_to_group
    """
    if n_cpu <= 1:
        out = [func(sub) for _, sub in df.groupby(group_key, sort=False)]
        return pd.concat(out, axis=0) if out else df.iloc[0:0].copy()

    groups = _split_groups(df, group_key)

    # Prepare payloads: (group, df, func)
    payloads = [(g, sub_df, func) for g, sub_df in groups]

    out_frames = []
    with ProcessPoolExecutor(max_workers=n_cpu) as ex:
        # executor.map preserves order; if you want fastest-first, use submit/as_completed.
        for _, res in ex.map(_apply_func_to_group, payloads):
            out_frames.append(res)

    return pd.concat(out_frames, axis=0) if out_frames else df.iloc[0:0].copy()


# -----------------------------
# Plot helpers (PDF aggregation)
# -----------------------------
def add_png_to_pdf(pdf: PdfPages, png_path: str, title: Optional[str] = None):
    """Insert a PNG image as a page into a PdfPages."""
    import matplotlib.image as mpimg
    img = mpimg.imread(png_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Step I: preprocessing + per-100kb features
# -----------------------------
@dataclass
class Args:
    species: str
    tumor_cell: int
    cell_number: str
    sample_name: str
    fragments_path: str
    outdir: str
    tumor_cell_tb_path: str
    n_cpu: int = 20
    plots: bool = False

    # Tunables (kept as defaults to match original behavior)
    frag_num_cutoff: int = 5000
    coverage_cutoff: int = 6

    # Step I_3 thresholds
    tss_score_pvalue: float = 0.05
    genebody_score_pvalue: float = 0.05
    uniform_distribution_pvalue: float = 0.01


def parse_args(argv: Optional[List[str]] = None) -> Args:
    """Argparse CLI."""
    import argparse

    p = argparse.ArgumentParser(
        prog="inferECC_pipeline_v1.0.0",
        description="inferECC pipeline with per-cell multiprocessing (v1.0.0)."
    )
    p.add_argument("species", help="Genome build: hg19/hg38/mm10")
    p.add_argument("tumor_cell", type=int, choices=[0, 1], help="1=tumor cells, 0=non-tumor cells")
    p.add_argument("cell_number", help='"max" or an integer (sample cells after fragment cutoff)')
    p.add_argument("sample_name", help="frag.file name used in tumor-cell annotation table")
    p.add_argument("fragments_path", help="Path to fragments.tsv(.gz)")
    p.add_argument("outdir", help="Output directory")
    p.add_argument("tumor_cell_tb_path", help="Tumor-cell barcode table (tsv)")

    p.add_argument("--n_cpu", type=int, default=20, help="CPU cores (processes) for per-cell parallelism. Default: 20")
    p.add_argument("--plots", action="store_true", help="Enable heavy plots (density/heatmaps). Default: off")

    p.add_argument("--frag_num_cutoff", type=int, default=5000, help="Fragment-number cutoff per cell. Default: 5000")
    p.add_argument("--coverage_cutoff", type=int, default=6, help="Coverage cutoff per 100kb bin. Default: 6")

    p.add_argument("--tss_score_pvalue", type=float, default=0.05, help="Quantile p for tss_score filtering. Default: 0.05")
    p.add_argument("--uniform_pvalue", type=float, default=0.01, help="KS uniform p-value cutoff. Default: 0.01")

    ns = p.parse_args(argv)

    return Args(
        species=ns.species,
        tumor_cell=int(ns.tumor_cell),
        cell_number=str(ns.cell_number),
        sample_name=str(ns.sample_name),
        fragments_path=str(ns.fragments_path),
        outdir=str(ns.outdir),
        tumor_cell_tb_path=str(ns.tumor_cell_tb_path),
        n_cpu=max(1, int(ns.n_cpu)),
        plots=bool(ns.plots),
        frag_num_cutoff=int(ns.frag_num_cutoff),
        coverage_cutoff=int(ns.coverage_cutoff),
        tss_score_pvalue=float(ns.tss_score_pvalue),
        uniform_distribution_pvalue=float(ns.uniform_pvalue),
    )


# -----------------------------
# Step I_3: per-cell ochh + KS uniform
# -----------------------------
def _worker_ochh_ks_heatmap(
    cb: str,
    df_fi: pd.DataFrame,
    out_png_dir: str,
    make_plots: bool,
) -> Tuple[str, pd.DataFrame, Optional[str], Optional[str]]:
    """
    Worker for a single barcode:
    - build ochh matrix
    - KS uniform test
    - optionally draw raw and filtered heatmaps to PNG

    Returns
    -------
    cb, df_cb_ochh_mtx_uniform, raw_png_path, fi_png_path
    """
    # Local imports inside worker for multiprocessing safety
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_cb = df_fi[df_fi["barcode"] == cb].copy()
    df_cb_ochh_mtx = ochh_mtx(df=df_cb)
    df_cb_ochh_mtx_uniform = ochh_mtx_ks_test(df=df_cb_ochh_mtx)

    raw_png_path = None
    fi_png_path = None

    if make_plots and (not df_cb_ochh_mtx_uniform.empty):
        os.makedirs(out_png_dir, exist_ok=True)

        # Raw heatmap
        try:
            g = heatmap_raw_plot(df=df_cb_ochh_mtx_uniform, cb=cb)
            raw_png_path = os.path.join(out_png_dir, f"{cb}.heatmap_raw.png")
            plt.savefig(raw_png_path, bbox_inches="tight", dpi=180)
            plt.close()
        except Exception:
            plt.close()

        # Filtered heatmap (same df; filtering happens later in main)
        try:
            p = heatmap_fi_plot(df=df_cb_ochh_mtx_uniform, cb=cb)
            fi_png_path = os.path.join(out_png_dir, f"{cb}.heatmap_fi.png")
            plt.savefig(fi_png_path, bbox_inches="tight", dpi=180)
            plt.close()
        except Exception:
            plt.close()

    # Attach identifiers for downstream concat (keep legacy format)
    df_cb_ochh_mtx_uniform = df_cb_ochh_mtx_uniform.copy()
    df_cb_ochh_mtx_uniform["chr_100k"] = df_cb_ochh_mtx_uniform.index
    df_cb_ochh_mtx_uniform["barcode"] = cb

    return cb, df_cb_ochh_mtx_uniform, raw_png_path, fi_png_path


def main(argv: Optional[List[str]] = None) -> int:
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    args = parse_args(argv)
    logger = setup_logger(args.outdir)

    # Avoid noisy warnings
    pd.options.mode.chained_assignment = None
    warnings.simplefilter("ignore", category=FutureWarning)

    os.makedirs(args.outdir, exist_ok=True)
    os.chdir(args.outdir)
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Args: {args}")

    # -----------------------------
    # STEP I_1_2
    # -----------------------------
    with StepTimer(logger, "STEP I_1_2 | Load fragments, filter cells, 100kb segmentation, Normalize, enrichments"):
        # IO: tumor cell barcodes
        tumor_cell_tb_df = pd.read_csv(args.tumor_cell_tb_path, sep="\t")
        tumor_cell_CB_list = list(tumor_cell_tb_df[tumor_cell_tb_df["frag.file"] == args.sample_name]["cellname"])

        # read fragments
        df_fragments = read_bgi_as_dataframe(path=args.fragments_path)

        # filter tumor/non-tumor
        if args.tumor_cell == 1:
            df_fragments = df_fragments[df_fragments.barcode.isin(tumor_cell_CB_list)]
        else:
            df_fragments = df_fragments[(df_fragments.barcode.isin(tumor_cell_CB_list)) ^ True]

        # Transform: drop chrM
        df_fragments = Transform(df_fragments=df_fragments, Delete_chrM_option=True)

        # Optional: fragments length distribution (slow)
        if args.plots:
            fragments_length(df_fragments, lim=False)

        # fragments per cell + cutoff
        df_fragments_number_sort = caculate_fragments_number(df_fragments=df_fragments)
        df_fragments_cutoff = cutoff_fragments_number(
            df_fragments=df_fragments,
            cutoff_value=args.frag_num_cutoff,
            df_fragments_number_sort=df_fragments_number_sort
        )

        # optional sampling cells
        if args.cell_number == "max":
            df_fragments_cutoff_sample = df_fragments_cutoff
        else:
            sample_num = min(int(args.cell_number), len(df_fragments_cutoff["barcode"].unique()))
            df_fragments_cutoff_sample = sample_cell(df_fragments=df_fragments_cutoff, sample_number=sample_num)

        # 100kb segmentation
        df_fragments_cutoff_segmentation = fragments_segmentation(df_fragments=df_fragments_cutoff_sample)

        # Normalize (coverage) in parallel per barcode
        # 原脚本：df.groupby("barcode").apply(Normalize) 单线程，非常慢
        df_fragments_cutoff_normalize = parallel_group_apply(
            df=df_fragments_cutoff_segmentation,
            group_key="barcode",
            func=Normalize,
            n_cpu=args.n_cpu,
            logger=logger,
        )

        # Coverage matrix (legacy)
        df_fragments_cutoff_normalize_dd = df_fragments_cutoff_normalize.drop_duplicates(subset=["barcode", "chr_100k"])
        df_fragments_cutoff_normalize_dd.to_csv("cell_coverage.matrix.tsv", sep="\t", index=True)

        # Coverage density (optional)
        if args.plots:
            coverage_density(df_fragments_cutoff_normalize)

        # Coverage cutoff
        df_02 = df_fragments_cutoff_normalize.copy()
        df_02 = df_02[df_02["Coverage"] >= args.coverage_cutoff]

        # Uniform distribution test (vectorized inside inferECC)
        df_03 = caculate_uniform(df_fragments=df_02)

        # Genomic annotation enrichments (likely non-per-cell heavy)
        df_04 = tss_site(df_fragments=df_03, species=args.species)
        df_05 = tss_region(df_fragments=df_04, species=args.species)
        df_05 = genebody_region(df_fragments=df_05, species=args.species)
        df_05 = intergenic_region(df_fragments=df_05)

        # Optional: TSS distance plots
        if args.plots:
            bp_from_tss(df_05)
            bp_from_tss(df_05, lim=True)

        df_06 = df_05.copy()

        # tss_score / genebody_score are per-cell groupby apply in original => parallelize
        df_07 = parallel_group_apply(
            df=df_06,
            group_key="barcode",
            func=tss_score,
            n_cpu=args.n_cpu,
            logger=logger,
        )
        df_07_dd = df_07.drop_duplicates(subset=["barcode", "chr_100k"])
        if args.plots:
            enrichment_plot(df_07_dd, enrich_arg="tss", show=False)
            
        df_08 = parallel_group_apply(
            df=df_07,
            group_key="barcode",
            func=genebody_score,
            n_cpu=args.n_cpu,
            logger=logger,
        )
        df_08_dd = df_08.drop_duplicates(subset=["barcode", "chr_100k"])
        if args.plots:
            enrichment_plot(df_08_dd, enrich_arg="genebody", show=False)

        df_08.to_csv("cellXecDNA.matrix.tsv", sep="\t", index=True)

        if args.plots:
            heatmap_chr(df_08)

        # Keep in-memory for downstream steps
        df_ch = df_08.copy()
        cell_coverage_dd = df_fragments_cutoff_normalize_dd.copy()

    # -----------------------------
    # STEP I_3
    # -----------------------------
    with StepTimer(logger, "STEP I_3 | KS-uniform per cell + heatmaps + write filtered h5ad"):
        uniform_dir = "ks_uniform_qsub"
        os.makedirs(uniform_dir, exist_ok=True)
        os.chdir(uniform_dir)
        logger.info(f"STEP I_3 workdir: {os.getcwd()}")

        # Step1: filter by tss_score quantile (same as legacy)
        df_ch_dd = df_ch.drop_duplicates(subset=["barcode", "chr_100k"])
        tss_score_vals = df_ch_dd["tss_score"].values.tolist()
        tss_score_vals.sort()
        tss_score_cutoff = tss_score_vals[int(len(tss_score_vals) * (1 - args.tss_score_pvalue))]
        logger.info(f"tss_score_cutoff: {tss_score_cutoff}")
        df_fi = df_ch[df_ch.tss_score <= tss_score_cutoff].copy()

        cb_list = df_fi.barcode.unique().tolist()
        logger.info(f"Cells after tss filter: {len(cb_list)}")

        # Step2: per cell ochh + ks test in parallel
        png_dir = "heatmap_png"
        make_plots = args.plots

        results: List[pd.DataFrame] = []
        raw_pngs: List[str] = []
        fi_pngs: List[str] = []

        if args.n_cpu <= 1:
            for cb in cb_list:
                _, df_cb_ochh, raw_png, fi_png = _worker_ochh_ks_heatmap(cb, df_fi, png_dir, make_plots)
                results.append(df_cb_ochh)
                if raw_png: raw_pngs.append(raw_png)
                if fi_png: fi_pngs.append(fi_png)
        else:
            with ProcessPoolExecutor(max_workers=args.n_cpu) as ex:
                futs = [
                    ex.submit(_worker_ochh_ks_heatmap, cb, df_fi, png_dir, make_plots)
                    for cb in cb_list
                ]
                for fut in as_completed(futs):
                    cb, df_cb_ochh, raw_png, fi_png = fut.result()
                    results.append(df_cb_ochh)
                    if raw_png: raw_pngs.append(raw_png)
                    if fi_png: fi_pngs.append(fi_png)

        combined_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        combined_df.to_csv("ochh_raw.matrix.tsv", sep="\t", index=True)

        # If plots enabled, aggregate PNGs into PDFs (legacy filenames)
        if args.plots:
            # raw heatmaps pdf
            with PdfPages("heatmap_raw_plots.pdf") as pdf:
                for pth in sorted(raw_pngs):
                    add_png_to_pdf(pdf, pth, title=os.path.basename(pth).replace(".heatmap_raw.png", ""))
            # filtered heatmaps pdf
            with PdfPages("heatmap_fi_plots.pdf") as pdf:
                for pth in sorted(fi_pngs):
                    add_png_to_pdf(pdf, pth, title=os.path.basename(pth).replace(".heatmap_fi.png", ""))

        # Step3: filtering by uniform pvalue
        combined_df_fi = combined_df[combined_df["uniform_pvalue"] >= args.uniform_distribution_pvalue].copy()
        combined_df_fi.to_csv("ochh_raw_fi.matrix.tsv", sep="\t", index=True)

        # Build filtered cellXecDNA matrix with uniform_pvalue per (barcode, chr_100k)
        df_fi_dd = df_fi.drop_duplicates(subset=["barcode", "chr_100k"]).copy()
        # Merge is faster than row-by-row loop
        up_map = combined_df[["barcode", "chr_100k", "uniform_pvalue"]].drop_duplicates()
        df_fi_dd = df_fi_dd.merge(up_map, on=["barcode", "chr_100k"], how="left")
        df_fi_dd_uni_fi = df_fi_dd[df_fi_dd["uniform_pvalue"] >= args.uniform_distribution_pvalue].copy()
        df_fi_dd_uni_fi.to_csv("cellXecDNA_fi.matrix.tsv", sep="\t", index=True)

        # AnnData output (legacy)
        mtx_raw_fi_ks = df_fi_dd_uni_fi.copy()
        cxe_mm = pd.pivot_table(
            mtx_raw_fi_ks,
            index=["barcode"],
            columns=["chr_100k"],
            values=["Coverage"],
            fill_value=0
        )
        cxe_mm.columns = cxe_mm.columns.to_frame().chr_100k.to_list()
        cxe_mm.columns.name = "chr_100k"

        adata_sample = sc.AnnData(X=cxe_mm, obs=cxe_mm.index.to_frame(), var=cxe_mm.columns.to_frame())
        adata_sample.obs["sample_raw"] = args.sample_name
        sample = re.sub(r"(-fragments\.tsv\.gz|-atac_fragments\.tsv\.gz)$", "", args.sample_name)
        adata_sample.obs["sample"] = sample
        adata_sample.write_h5ad("cellXecDNA_fi.matrix.h5ad")

        # Binary matrix (>=coverage_cutoff => 1 else 0)
        adata_sample.X = np.where(adata_sample.X >= args.coverage_cutoff, 1, adata_sample.X)
        adata_sample.X = np.where(adata_sample.X < args.coverage_cutoff, 0, adata_sample.X)
        adata_sample.write_h5ad("cellXecDNA_fi.matrix_01.h5ad")

        # Return to outdir root
        os.chdir(args.outdir)

    # -----------------------------
    # STEP III_1 | Merge chr100k into ecdna regions + neighbor correlation
    # -----------------------------
    with StepTimer(logger, "STEP III_1 | chr100k_merge + neighbor_correlation + merged matrices"):
        # We rely on in-memory variables from previous steps:
        # - mtx_raw_fi_ks: df_fi_dd_uni_fi from STEP I_3 (reload to be safe)
        os.chdir(args.outdir)
        mtx_raw_fi_ks = pd.read_csv(os.path.join("ks_uniform_qsub", "cellXecDNA_fi.matrix.tsv"), sep="\t")
        cell_coverage = pd.read_csv("cell_coverage.matrix.tsv", sep="\t")

        # Filter cell_coverage to those chr_100k & barcode present in mtx_raw_fi_ks
        cc_dd = cell_coverage.drop_duplicates(subset=["barcode", "chr_100k"])
        cc_dd_ch = cc_dd[cc_dd["chr_100k"].isin(mtx_raw_fi_ks["chr_100k"].unique())]
        cc_dd_ch_bc = cc_dd_ch[cc_dd_ch["barcode"].isin(mtx_raw_fi_ks["barcode"].unique())]
        cc_dd_ch_bc.to_csv("cellXecDNA_before_merge_long.matrix.tsv", sep="\t", index=True)

        cxe_mtx = cc_dd_ch_bc[["barcode", "chr_100k", "Coverage", "chrom", "start_100k"]]
        cxe_mtx_sv = cxe_mtx.sort_values(by=["chrom", "start_100k"], ascending=(True, True))

        cxe_mm = pd.pivot_table(
            cxe_mtx_sv,
            index=["barcode"],
            columns=["chr_100k"],
            values=["Coverage"],
            fill_value=0
        )
        cxe_mm.columns = cxe_mm.columns.to_frame().chr_100k.to_list()
        cxe_mm.columns.name = "chr_100k"

        adata = sc.AnnData(X=cxe_mm, obs=cxe_mm.index.to_frame(), var=cxe_mm.columns.to_frame())
        sc.pp.filter_genes(adata, min_cells=3)

        ad_var = adata.var.copy()
        ad_var["chrom"] = ad_var.chr_100k.str.split(":", expand=True)[0]
        ad_var["start_100k"] = ad_var.chr_100k.str.split(":", expand=True)[1].str.split("_", expand=True)[0].astype(int)
        ad_var["end_100k"] = ad_var.chr_100k.str.split(":", expand=True)[1].str.split("_", expand=True)[1].astype(int)

        # Neighbor and correlation
        ad_var_nb = ad_var.groupby(ad_var["chrom"]).apply(Neighbor)
        ad_var_nb.index = ad_var_nb.chr_100k
        ad_var_nb_ri = ad_var_nb.reindex(index=list(adata.var.index.values))
        adata.var = ad_var_nb_ri

        # neighbor_correlation uses inferECC default cutoff=0.2 per your v8 notes
        adata = neighbor_correlation(adata)
        adata.var.to_csv("cellXecDNA_before_merge_meta_corr_cf0.2.matrix.tsv", sep="\t", index=True)

        # Density plots (optional; can be heavy but usually OK)
        if args.plots:
            dfv = adata.var.copy()
            dfv["fragLen"] = dfv.neighbor_len
            fragments_length(
                dfv,
                plt_title="Density plot for ecdna length",
                file_name="p01-1_ecdna_length_density.pdf",
                label="ecdna_length(Mbp)",
                xlabel="ecdna_length(Mbp)"
            )
            dfv["fragLen"] = dfv.correlation_neighbor_len
            fragments_length(
                dfv,
                plt_title="Density plot for ecdna correlation_neighbor_len",
                file_name="p01-2_ecdna_correlation_neighbor_len_density.pdf",
                label="ecdna_correlation_neighbor_len(Mbp)",
                xlabel="ecdna_correlation_neighbor_len(Mbp)"
            )
            dfv["fragLen"] = dfv.correlation
            fragments_length(
                dfv,
                plt_title="Density plot for ecdna neighbor_correlation",
                file_name="p01-3_ecdna_neighbor_correlation_density.pdf",
                label="ecdna_neighbor_correlation",
                xlabel="ecdna_neighbor_correlation"
            )
            dfv["fragLen"] = dfv.correlation_pvalue
            fragments_length(
                dfv,
                plt_title="Density plot for ecdna neighbor_correlation_pvalue",
                file_name="p01-4_ecdna_neighbor_correlation_pvalue_density.pdf",
                label="ecdna_neighbor_correlation_pvalue",
                xlabel="ecdna_neighbor_correlation_pvalue"
            )

        adata_t = adata.T.copy()
        adata_sum = sum_by(adata=adata_t, col="correlation_neighbor")
        adata_merge_df = adata_sum.to_df().T.copy()
        adata_merge_df.to_csv("cellXecDNA_merge_df_corr_cf0.2.matrix.tsv", sep="\t", index=True)

    # -----------------------------
    # STEP III_5 | chr_merge to gene mapping
    # -----------------------------
    with StepTimer(logger, "STEP III_5 | chr_merge -> gene mapping"):
        ecc_df = pd.read_csv("cellXecDNA_merge_df_corr_cf0.2.matrix.tsv", sep="\t", index_col=0)
        gene_df = pd.read_csv("cellXecDNA.matrix.tsv", sep="\t")
        merge_df = pd.read_csv("cellXecDNA_before_merge_meta_corr_cf0.2.matrix.tsv", sep="\t")

        chr_merge_df = pd.DataFrame({
            "chr_merge": merge_df["correlation_neighbor"],
            "chr_100k": merge_df["chr_100k"]
        })
        chr_merge_df.to_csv("chr_merge_2_100k_cf0.2.tsv", sep="\t", index=True)

        gene_df_sub = gene_df[gene_df.chr_100k.isin(list(chr_merge_df.chr_100k.unique()))].copy()
        gene_df_sub_new = pd.DataFrame({
            "chr_100k": gene_df_sub["chr_100k"],
            "genebody_region_gene": gene_df_sub["genebody_region_gene"]
        })

        # Clean and explode gene lists
        df_filtered = gene_df_sub_new[gene_df_sub_new["genebody_region_gene"] != "0"].copy()
        df_filtered["genebody_region_gene"] = df_filtered["genebody_region_gene"].apply(
            lambda x: x.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
        )
        df_filtered["genebody_region_gene"] = df_filtered["genebody_region_gene"].apply(lambda x: x.split(","))

        chr_100k_gene_df = df_filtered.groupby("chr_100k")["genebody_region_gene"].agg(lambda x: sum(x, [])).reset_index()
        chr_100k_gene_df.to_csv("chr_100k_2_gene_cf0.2.tsv", sep="\t", index=True)

        chr_merge_df_gene = pd.merge(chr_merge_df, chr_100k_gene_df, on="chr_100k", how="left")
        chr_merge_df_gene_dropna = chr_merge_df_gene.dropna(subset=["genebody_region_gene"])
        chr_merge_df_gene_dropna_merge = chr_merge_df_gene_dropna.groupby("chr_merge")["genebody_region_gene"].agg(
            lambda x: sum(x, [])
        ).reset_index()
        chr_merge_df_gene_dropna_merge["gene_unique"] = chr_merge_df_gene_dropna_merge["genebody_region_gene"].apply(
            lambda x: list(set(x))
        )
        chr_merge_df_gene_dropna_merge.to_csv("chr_merge_2_gene_cf0.2.tsv", sep="\t", index=True)

    # -----------------------------
    # STEP III_6 | mtx tsv -> h5ad + normalized version
    # -----------------------------
    with StepTimer(logger, "STEP III_6 | export h5ad + normalize-by-length"):
        merge_df = pd.read_csv("cellXecDNA_merge_df_corr_cf0.2.matrix.tsv", sep="\t", index_col=0)
        merge_df.columns.name = "ecdna_region"

        adtmg = sc.AnnData(X=merge_df, obs=merge_df.index.to_frame(), var=merge_df.columns.to_frame())
        adtmg.write_h5ad("cellXecDNA_merge_df_cf0.2.matrix.h5ad")

        # Length-normalized version (legacy v6 fix)
        tb0 = merge_df.T.copy()
        tb = tb0.copy()
        tb["ecdna"] = tb.index
        tb["len"] = (
            tb["ecdna"].str.split(":").str[1].str.split("_").str[1].astype(int)
            - tb["ecdna"].str.split(":").str[1].str.split("_").str[0].astype(int)
        ) / 100000

        tb = tb.drop("ecdna", axis=1)
        tb = tb.apply(lambda row: row / row[-1], axis=1)  # divide by length column
        tb = tb.drop("len", axis=1)

        tb_nor = tb.T.copy()
        tb_nor.to_csv("cellXecDNA_merge_cf0.2_df_nor.matrix.tsv", sep="\t", index=True)

        merge_df_nor = tb_nor.copy()
        merge_df_nor.columns.name = "ecdna_region"
        adtmg_nor = sc.AnnData(X=merge_df_nor, obs=merge_df_nor.index.to_frame(), var=merge_df_nor.columns.to_frame())
        adtmg_nor.write_h5ad("cellXecDNA_merge_cf0.2_df_nor.matrix.h5ad")

    logger.info("InferECC finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
