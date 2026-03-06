#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecdna_autocorrelation_v4.py  (final, kernel + rand/dapi export + paired Wilcoxon + multi-channel plot + PNG/PDF)

FFT-based pair auto-correlation g(r) with ROI(mask) normalization,
following Veatch get_autocorr.m (PLOS ONE 2012) and Nature 2021 ecDNA-FISH clustering
(FFT + mask normalization + radial binning).

Standalone post-processing for outputs of ecdna_fish_pipeline_v4.py.

Expected inputs in OUTDIR (unless overridden):
- per_cell_summary.csv
- nuclei_labels.tif
- spots_red.csv / spots_green.csv / spots_cyan.csv (some subset)

Optional input (for DAPI-weighted control):
- DAPI.tif (original image; passed via --dapi)

Outputs written into OUTDIR:
- gr_per_cell.csv     one row per nucleus per channel (includes g0 for exp/rand/dapi)
- gr_long.csv         long-form curves (one row per nucleus per r per channel; includes exp/rand/dapi + raw curves)
- gr_pvalues.csv      paired Wilcoxon p-values at r=0 (exp vs rand, exp vs dapi), per group & channel
- gr_mean_sem.png/.pdf    one figure per group with multiple channels overlaid (mean ± SEM)
  (also saves per-channel-only figures if you pass --plot_per_channel)

Key upgrades in this version
----------------------------
1) Kernelized spot image (most important for "r=0 maximum" shape):
   - Default: gaussian kernel (PSF-like), controlled by --spot_sigma_px
   - Optional: disk kernel, controlled by --spot_disk_radius_px
   - --spot_kernel {gaussian,disk,delta}

2) Random curve export for reproducibility:
   - We export g_raw_rand(r) per nucleus per channel into gr_long.csv.
   - The normalized baseline is still y=1 by construction (after exp/rand normalization),
     but raw rand is exported so you can fully reproduce and audit the normalization.

3) DAPI curve export:
   - We export g_dapi(r) (normalized) and g_raw_dapi(r) (raw) into gr_long.csv.

4) Plotting: mean ± SEM like the paper
   - One figure per group, overlay multiple channels on same axes.
   - r=0 is kept as its own point (not mixed into bins); r>=1 uses binning via --plot_bin_px.

5) P-values (paired Wilcoxon signed-rank at r=0):
   - exp vs rand: test exp_g0 against 1 (since normalized rand baseline is 1)
   - exp vs dapi: paired test exp_g0 vs dapi_g0
   - Results saved to gr_pvalues.csv and annotated on plots.

Color requirements (as requested)
---------------------------------
- channel=green -> green line
- channel=red   -> red line
- channel=cyan  -> cyan line
- DAPI (pooled) -> blue line
- Random (=1)   -> grey line
"""

from __future__ import annotations

import os
import argparse
import warnings
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

try:
    import tifffile as tiff
except Exception:  # pragma: no cover
    tiff = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

# matplotlib is optional unless --plot is used
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# SciPy is strongly recommended (kernel + stats + optional convolution)
try:
    from scipy.ndimage import gaussian_filter
    from scipy.signal import fftconvolve
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    gaussian_filter = None
    fftconvolve = None
    wilcoxon = None


# -----------------------------
# Data container
# -----------------------------
@dataclass
class GRResult:
    """
    Container for a single nucleus+channel g(r) computation.

    Attributes
    ----------
    r : np.ndarray
        Radius array [0..rmax_eff] in pixels.
    g_exp : np.ndarray
        Experimental g(r) normalized by random baseline, so random baseline is ~1.
    g_rand : np.ndarray
        Normalized random baseline (by construction, ones).
    g_dapi : np.ndarray
        DAPI-weighted control g(r) normalized by random baseline.
    dg_exp, dg_dapi : np.ndarray
        Rough propagated uncertainties (mainly for debugging; plotting uses SEM across cells).
    g_raw_exp, g_raw_rand, g_raw_dapi : np.ndarray
        Raw g(r) before exp/rand normalization (important for auditing).
    """
    r: np.ndarray
    g_exp: np.ndarray
    g_rand: np.ndarray
    g_dapi: np.ndarray
    dg_exp: np.ndarray
    dg_dapi: np.ndarray
    g_raw_exp: np.ndarray
    g_raw_rand: np.ndarray
    g_raw_dapi: np.ndarray
    dg_raw_exp: np.ndarray
    dg_raw_rand: np.ndarray
    dg_raw_dapi: np.ndarray


# -----------------------------
# Low-level FFT helpers
# -----------------------------
def _fft_autocorr2d(a: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """
    Compute centered autocorrelation via FFT with zero-padding.

    Parameters
    ----------
    a : np.ndarray
        2D input array.
    out_shape : (int,int)
        FFT shape (should be >= original shape).

    Returns
    -------
    np.ndarray
        fftshift(real(ifft2(abs(fft2(a))^2))) with shape out_shape.
    """
    F = np.fft.fft2(a, s=out_shape)
    ac = np.fft.ifft2(np.abs(F) ** 2)
    return np.fft.fftshift(np.real(ac))


def _crop_center(img: np.ndarray, rmax: int) -> np.ndarray:
    """
    Crop center patch of size (2*rmax+1, 2*rmax+1) from img.
    """
    H, W = img.shape
    cy, cx = H // 2, W // 2
    y0, y1 = cy - rmax, cy + rmax + 1
    x0, x1 = cx - rmax, cx + rmax + 1
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        raise ValueError("rmax too large for padded FFT shape.")
    return img[y0:y1, x0:x1]


def _make_rbin(rmax: int) -> np.ndarray:
    """
    Precompute integer-radius bin index for a (2*rmax+1)x(2*rmax+1) patch.
    Bin rule: floor(radius) (matches MATLAB histc shells behavior).
    """
    size = 2 * rmax + 1
    yy, xx = np.indices((size, size))
    cy = cx = rmax
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return np.floor(rr + 1e-9).astype(np.int32)


def _radial_mean_and_dg(G: np.ndarray, rbin: np.ndarray, rmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radial average g[r] and an SEM-like dg[r] within each integer-radius annulus (0..rmax).

    Parameters
    ----------
    G : np.ndarray
        2D cropped g-map of shape (2*rmax+1, 2*rmax+1).
    rbin : np.ndarray
        Precomputed integer-radius bins (same shape as G).
    rmax : int
        Maximum radius (pixels) to report.

    Returns
    -------
    g : np.ndarray
        Radial mean (length rmax+1).
    dg : np.ndarray
        Std-of-mean within annulus (length rmax+1).
    """
    if G.shape != rbin.shape:
        raise ValueError("G and rbin must have the same shape.")

    valid = np.isfinite(G) & (rbin <= rmax)
    g = np.full(rmax + 1, np.nan, dtype=np.float64)
    dg = np.full(rmax + 1, np.nan, dtype=np.float64)

    if not valid.any():
        return g, dg

    bins = rbin[valid].ravel().astype(np.int32, copy=False)
    vals = G[valid].ravel().astype(np.float64, copy=False)

    cnt = np.bincount(bins, minlength=rmax + 1).astype(np.float64)
    s1 = np.bincount(bins, weights=vals, minlength=rmax + 1).astype(np.float64)
    s2 = np.bincount(bins, weights=vals * vals, minlength=rmax + 1).astype(np.float64)

    ok = cnt > 0
    g[ok] = s1[ok] / cnt[ok]

    var = np.zeros_like(cnt)
    var[ok] = s2[ok] / cnt[ok] - g[ok] ** 2
    var[var < 0] = 0.0
    dg[ok] = np.sqrt(var[ok] / cnt[ok])  # std of mean within annulus

    return g, dg


# -----------------------------
# Core masked autocorr (Veatch style)
# -----------------------------
@dataclass
class _MaskPrep:
    """
    Precomputation for a fixed nucleus mask to speed repeated exp/rand/dapi computations.
    """
    mask: np.ndarray
    A: float
    out_shape: Tuple[int, int]
    NP: np.ndarray
    rbin: np.ndarray


def _prepare_mask(mask: np.ndarray, rmax: int) -> _MaskPrep:
    """
    Precompute NP = autocorr(mask) and r-bin lookup for a given rmax.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D.")
    m = (mask > 0)
    A = float(m.sum())
    if A <= 0:
        raise ValueError("Empty mask.")
    H, W = m.shape
    rmax_eff = int(min(rmax, H - 1, W - 1))
    if rmax_eff < 1:
        #raise ValueError("rmax too large for mask/image shape.")
        raise ValueError("mask/image too small (need at least 2x2) or rmax<1.")
    out_shape = (H + rmax_eff, W + rmax_eff)
    NP = _fft_autocorr2d(m.astype(np.float64), out_shape=out_shape)
    rbin = _make_rbin(rmax_eff)
    return _MaskPrep(mask=m, A=A, out_shape=out_shape, NP=NP, rbin=rbin)


def _autocorr_gr_fft_prepared(
    Im: np.ndarray,
    prep: _MaskPrep,
    rmax: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute masked g-map then radial average, using Veatch-style normalization.

    For an intensity image Im (already zero outside mask):
      NP = autocorr(mask)
      AC = autocorr(Im)
      G_full = (A^2/N^2) * (AC / NP)
    Then crop to center and radially average to get g(r).

    Parameters
    ----------
    Im : np.ndarray
        2D image (float), should be zero outside nucleus mask.
    prep : _MaskPrep
        Precomputed mask autocorr and r-bin.
    rmax : int
        Max radius to report (clipped to image size).

    Returns
    -------
    r : np.ndarray
        Radius array [0..rmax_eff].
    g : np.ndarray
        Radial mean g(r).
    dg : np.ndarray
        Within-annulus std-of-mean (debug-level uncertainty).
    """
    if Im.ndim != 2:
        raise ValueError("Im must be 2D.")
    if Im.shape != prep.mask.shape:
        raise ValueError("Im and mask shapes must match.")

    H, W = prep.mask.shape
    rmax_eff = int(min(rmax, H - 1, W - 1))
    if rmax_eff != prep.rbin.shape[0] // 2:
        prep = _prepare_mask(prep.mask, rmax=rmax_eff)

    Im = Im.astype(np.float64, copy=False)
    N = float(Im.sum())
    if N <= 0:
        r = np.arange(rmax_eff + 1, dtype=int)
        nan = np.full(rmax_eff + 1, np.nan, dtype=np.float64)
        return r, nan, nan

    AC = _fft_autocorr2d(Im, out_shape=prep.out_shape)

    with np.errstate(divide="ignore", invalid="ignore"):
        G_full = (prep.A * prep.A / (N * N)) * (AC / prep.NP)
    G_full[prep.NP == 0] = np.nan

    G = _crop_center(G_full, rmax=rmax_eff)
    g, dg = _radial_mean_and_dg(G, prep.rbin, rmax_eff)
    r = np.arange(rmax_eff + 1, dtype=int)
    return r, g, dg


# -----------------------------
# Spot image rendering (delta + kernel)
# -----------------------------
def points_to_image_counts(
    rc: np.ndarray,
    shape: Tuple[int, int],
    mask: np.ndarray,
) -> np.ndarray:
    """
    Build a delta/count image from (row,col) points, constrained to mask.

    Parameters
    ----------
    rc : np.ndarray
        (K,2) array of (row,col) integer positions.
    shape : (int,int)
        Image shape (H,W).
    mask : np.ndarray
        2D boolean/0-1 nucleus mask of same shape.

    Returns
    -------
    I : np.ndarray
        Delta/count image (float64), zero outside mask.
    """
    H, W = shape
    I = np.zeros((H, W), dtype=np.float64)
    if rc is None or rc.size == 0:
        return I
    rc = np.asarray(rc)
    if rc.ndim != 2 or rc.shape[1] != 2:
        raise ValueError("rc must be (K,2) array of (row,col).")
    r = rc[:, 0].astype(np.int64, copy=False)
    c = rc[:, 1].astype(np.int64, copy=False)
    keep = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    r, c = r[keep], c[keep]
    if r.size == 0:
        return I
    keep2 = mask[r, c] > 0
    r, c = r[keep2], c[keep2]
    if r.size == 0:
        return I
    np.add.at(I, (r, c), 1.0)
    I *= mask.astype(np.float64)
    return I


def _disk_kernel(radius_px: int) -> np.ndarray:
    """
    Create a normalized disk kernel.

    Parameters
    ----------
    radius_px : int
        Disk radius in pixels (>=1).

    Returns
    -------
    K : np.ndarray
        2D kernel with sum(K)=1.
    """
    r = int(radius_px)
    if r < 1:
        raise ValueError("disk radius must be >= 1")
    yy, xx = np.indices((2 * r + 1, 2 * r + 1))
    cy = cx = r
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    K = (d2 <= r * r).astype(np.float64)
    s = K.sum()
    if s > 0:
        K /= s
    return K


def apply_spot_kernel(
    I_delta: np.ndarray,
    kernel: str = "gaussian",
    sigma_px: float = 1.5,
    disk_radius_px: int = 2,
) -> np.ndarray:
    """
    Convert delta/count image into a PSF-like spot image by applying a kernel.

    Parameters
    ----------
    I_delta : np.ndarray
        Delta/count image, float64, typically zero outside mask.
    kernel : str
        One of {"gaussian","disk","delta"}.
    sigma_px : float
        Gaussian sigma in pixels (used when kernel="gaussian").
    disk_radius_px : int
        Disk radius in pixels (used when kernel="disk").

    Returns
    -------
    I : np.ndarray
        Kernelized spot image (float64).
    """
    k = str(kernel).lower()
    if k == "delta":
        return I_delta

    if k == "gaussian":
        if gaussian_filter is None:
            raise RuntimeError("scipy is required for gaussian kernel (scipy.ndimage.gaussian_filter).")
        sig = float(sigma_px)
        if sig <= 0:
            raise ValueError("--spot_sigma_px must be > 0 for gaussian kernel.")
        # mode constant -> no wrap-around; preserves physical meaning
        return gaussian_filter(I_delta, sigma=sig, mode="constant", cval=0.0)

    if k == "disk":
        if fftconvolve is None:
            raise RuntimeError("scipy is required for disk kernel convolution (scipy.signal.fftconvolve).")
        K = _disk_kernel(int(disk_radius_px))
        return fftconvolve(I_delta, K, mode="same")

    raise ValueError(f"Unknown kernel={kernel}. Use gaussian|disk|delta.")


# -----------------------------
# Sampling utilities (random & DAPI)
# -----------------------------
def sample_uniform_points_in_mask(
    n: int,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample n points uniformly from mask pixels (with replacement if needed).
    """
    idx = np.flatnonzero(mask > 0)
    if idx.size == 0 or n <= 0:
        return np.zeros((0, 2), dtype=int)
    replace = n > idx.size
    pick = rng.choice(idx, size=n, replace=replace)
    rr, cc = np.unravel_index(pick, mask.shape)
    return np.stack([rr, cc], axis=1).astype(int)


def sample_weighted_points_in_mask(
    n: int,
    mask: np.ndarray,
    weight_img: np.ndarray,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    DAPI-control: sample positions with probability proportional to DAPI intensity within nucleus mask.

    If the DAPI signal is degenerate (sum<=0), falls back to uniform sampling.

    Parameters
    ----------
    n : int
        Number of points to sample (usually equals n_spots of experimental).
    mask : np.ndarray
        2D nucleus mask.
    weight_img : np.ndarray
        2D DAPI image aligned to mask.
    rng : np.random.Generator
        RNG for reproducibility.
    eps : float
        Stabilizer for probabilities.

    Returns
    -------
    rc : np.ndarray
        (n,2) integer coordinates (row,col).
    """
    if n <= 0:
        return np.zeros((0, 2), dtype=int)
    m = (mask > 0)
    w = np.asarray(weight_img, dtype=np.float64)
    if w.shape != mask.shape:
        raise ValueError("weight_img must match mask shape.")
    w = np.where(m, w, 0.0)
    w = np.clip(w, 0, None)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        return sample_uniform_points_in_mask(n, mask, rng)

    idx = np.flatnonzero(m.ravel())
    p = w.ravel()[idx] + eps
    p = p / p.sum()

    replace = n > idx.size
    pick = rng.choice(idx, size=n, replace=replace, p=p)
    rr, cc = np.unravel_index(pick, mask.shape)
    return np.stack([rr, cc], axis=1).astype(int)


# -----------------------------
# Public API: 3-curve computation (exp / rand / dapi)
# -----------------------------
def compute_gr_three_curves(
    rc_spots: np.ndarray,
    nucleus_mask: np.ndarray,
    dapi_img: Optional[np.ndarray] = None,
    rmax: int = 100,
    rng: Optional[np.random.Generator] = None,
    n_random: int = 5,
    n_dapi: int = 3,
    spot_kernel: str = "gaussian",
    spot_sigma_px: float = 1.5,
    spot_disk_radius_px: int = 2,
) -> GRResult:
    """
    Compute g(r) for experimental spots, random baseline, and DAPI-weighted baseline,
    then normalize exp and dapi by random baseline so that random baseline is ~1.

    Important notes
    --------------
    - Random baseline is estimated per nucleus by repeatedly sampling n_spots points in the mask.
    - DAPI baseline is estimated per nucleus by sampling points with probability ∝ DAPI intensity.
    - Kernelization is applied consistently to exp, rand, and dapi images.

    Parameters
    ----------
    rc_spots : np.ndarray
        (K,2) integer coordinates of experimental spots (row,col).
    nucleus_mask : np.ndarray
        2D nucleus mask (same shape as DAPI and label image), non-zero inside nucleus.
    dapi_img : np.ndarray or None
        2D DAPI image aligned to nucleus mask. If None, g_dapi is NaN.
    rmax : int
        Max radius (pixels) to compute (clipped to image size).
    rng : np.random.Generator or None
        RNG for reproducible random/dapi sampling. If None, uses default seed=0.
    n_random : int
        Number of random replicates to average for baseline stability.
    n_dapi : int
        Number of DAPI replicates to average.
    spot_kernel : str
        {"gaussian","disk","delta"}.
    spot_sigma_px : float
        Gaussian sigma in pixels (kernel="gaussian").
    spot_disk_radius_px : int
        Disk radius in pixels (kernel="disk").

    Returns
    -------
    GRResult
        Holds r, g_exp (normalized), g_rand (ones), g_dapi (normalized), and raw curves.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    mask = (nucleus_mask > 0)
    shape = mask.shape

    n_spots = int(rc_spots.shape[0]) if rc_spots is not None else 0
    rmax_eff = int(min(rmax, shape[0] - 1, shape[1] - 1))
    r = np.arange(rmax_eff + 1, dtype=int)

    if n_spots <= 0:
        nan = np.full(rmax_eff + 1, np.nan, dtype=np.float64)
        ones = np.ones_like(r, dtype=np.float64)
        return GRResult(
            r=r,
            g_exp=nan.copy(),
            g_rand=ones,
            g_dapi=nan.copy(),
            dg_exp=nan.copy(),
            dg_dapi=nan.copy(),
            g_raw_exp=nan.copy(),
            g_raw_rand=nan.copy(),
            g_raw_dapi=nan.copy(),
            dg_raw_exp=nan.copy(),
            dg_raw_rand=nan.copy(),
            dg_raw_dapi=nan.copy(),
        )

    prep = _prepare_mask(mask, rmax=rmax_eff)

    # ---- experimental
    I_exp0 = points_to_image_counts(rc_spots, shape=shape, mask=mask)
    I_exp = apply_spot_kernel(I_exp0, kernel=spot_kernel, sigma_px=spot_sigma_px, disk_radius_px=spot_disk_radius_px)
    I_exp *= mask.astype(np.float64)  # enforce ROI
    _, g_raw_exp, dg_raw_exp = _autocorr_gr_fft_prepared(I_exp, prep, rmax=rmax_eff)

    # ---- random baseline: average over repeats
    n_random = int(max(1, n_random))
    g_rand_reps = []
    for _ in range(n_random):
        rc_rand = sample_uniform_points_in_mask(n_spots, mask=mask, rng=rng)
        I_rand0 = points_to_image_counts(rc_rand, shape=shape, mask=mask)
        I_rand = apply_spot_kernel(I_rand0, kernel=spot_kernel, sigma_px=spot_sigma_px, disk_radius_px=spot_disk_radius_px)
        I_rand *= mask.astype(np.float64)
        _, g_raw_i, _ = _autocorr_gr_fft_prepared(I_rand, prep, rmax=rmax_eff)
        g_rand_reps.append(g_raw_i)

    g_raw_rand = np.nanmean(np.stack(g_rand_reps, axis=0), axis=0)
    dg_raw_rand = (np.nanstd(np.stack(g_rand_reps, axis=0), axis=0, ddof=1) / np.sqrt(n_random)) if n_random > 1 else np.full_like(g_raw_rand, np.nan)

    # ---- DAPI control: average over repeats
    if dapi_img is None:
        g_raw_dapi = np.full_like(g_raw_exp, np.nan)
        dg_raw_dapi = np.full_like(dg_raw_exp, np.nan)
    else:
        n_dapi = int(max(1, n_dapi))
        g_dapi_reps = []
        for _ in range(n_dapi):
            rc_dapi = sample_weighted_points_in_mask(n_spots, mask=mask, weight_img=dapi_img, rng=rng)
            I_dapi0 = points_to_image_counts(rc_dapi, shape=shape, mask=mask)
            I_dapi = apply_spot_kernel(I_dapi0, kernel=spot_kernel, sigma_px=spot_sigma_px, disk_radius_px=spot_disk_radius_px)
            I_dapi *= mask.astype(np.float64)
            _, g_raw_i, _ = _autocorr_gr_fft_prepared(I_dapi, prep, rmax=rmax_eff)
            g_dapi_reps.append(g_raw_i)

        g_raw_dapi = np.nanmean(np.stack(g_dapi_reps, axis=0), axis=0)
        dg_raw_dapi = (np.nanstd(np.stack(g_dapi_reps, axis=0), axis=0, ddof=1) / np.sqrt(n_dapi)) if n_dapi > 1 else np.full_like(g_raw_dapi, np.nan)

    # ---- normalize by random baseline (element-wise)
    eps = 1e-12
    denom = np.where(np.isfinite(g_raw_rand) & (np.abs(g_raw_rand) > eps), g_raw_rand, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        g_exp = g_raw_exp / denom
        g_dapi = g_raw_dapi / denom

        # rough propagation (debug-level); plotting uses SEM across cells instead
        dg_exp = np.sqrt((dg_raw_exp / denom) ** 2 + ((g_raw_exp * dg_raw_rand) / (denom ** 2)) ** 2)
        dg_dapi = np.sqrt((dg_raw_dapi / denom) ** 2 + ((g_raw_dapi * dg_raw_rand) / (denom ** 2)) ** 2)
        
    g_exp[~np.isfinite(g_exp)] = np.nan
    g_dapi[~np.isfinite(g_dapi)] = np.nan
    dg_exp[~np.isfinite(dg_exp)] = np.nan
    dg_dapi[~np.isfinite(dg_dapi)] = np.nan

    g_rand = np.ones_like(g_exp, dtype=np.float64)

    return GRResult(
        r=r,
        g_exp=g_exp,
        g_rand=g_rand,
        g_dapi=g_dapi,
        dg_exp=dg_exp,
        dg_dapi=dg_dapi,
        g_raw_exp=g_raw_exp,
        g_raw_rand=g_raw_rand,
        g_raw_dapi=g_raw_dapi,
        dg_raw_exp=dg_raw_exp,
        dg_raw_rand=dg_raw_rand,
        dg_raw_dapi=dg_raw_dapi,
    )


# -----------------------------
# v4-output adapters (CSV-based)
# -----------------------------
def _require_pandas():  # pragma: no cover
    if pd is None:
        raise RuntimeError("pandas is required. Please `pip install pandas`.")


def load_spots_csv(path: str):
    """
    Load spots_*.csv and ensure required columns exist.

    Expected columns
    ----------------
    row : int
    col : int
    nucleus_id : int
    channel : optional str (if missing, inferred from filename)

    Returns
    -------
    pd.DataFrame
    """
    _require_pandas()
    df = pd.read_csv(path)
    for col in ("row", "col", "nucleus_id"):
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}")
    df["row"] = df["row"].astype(int)
    df["col"] = df["col"].astype(int)
    df["nucleus_id"] = df["nucleus_id"].astype(int)

    if "channel" not in df.columns:
        name = os.path.basename(path).lower()
        if "red" in name:
            df["channel"] = "red"
        elif "green" in name:
            df["channel"] = "green"
        elif "cyan" in name or "aqua" in name:
            df["channel"] = "cyan"
        else:
            df["channel"] = "unknown"
    return df


def compute_gr_from_v4_tables(
    nuclei_labels: np.ndarray,
    dapi_img: Optional[np.ndarray],
    per_cell_df,
    spots_df,
    rmax: int = 100,
    base_seed: int = 0,
    n_random: int = 5,
    n_dapi: int = 3,
    include_dense_extra: bool = True,
    spot_kernel: str = "gaussian",
    spot_sigma_px: float = 1.5,
    spot_disk_radius_px: int = 2,
):
    """
    Compute g(r) per (cell_id, channel) from v4 output tables.

    Parameters
    ----------
    nuclei_labels : np.ndarray
        2D label image, where pixel value == cell_id indicates nucleus membership.
    dapi_img : np.ndarray or None
        2D DAPI image aligned to nuclei_labels (same shape). If None, DAPI control is skipped.
    per_cell_df : pd.DataFrame
        Per-nucleus summary table (must include cell_id; ideally includes group/sample_id).
    spots_df : pd.DataFrame
        Spots table (must include row/col/nucleus_id/channel).
    rmax : int
        Max radius (pixels) to compute.
    base_seed : int
        Base RNG seed; per (cell,channel) seed is derived deterministically for reproducibility.
    n_random : int
        Random baseline repeats per nucleus.
    n_dapi : int
        DAPI baseline repeats per nucleus.
    include_dense_extra : bool
        If False and spots_df has is_dense_extra, exclude those spots.
    spot_kernel, spot_sigma_px, spot_disk_radius_px
        Spot rendering kernel options.

    Returns
    -------
    per_cell_gr_df : pd.DataFrame
        One row per (cell_id, channel) with g0 values and metadata.
    gr_long_df : pd.DataFrame
        Long table: one row per (cell_id, channel, r) with normalized and raw curves.
    """
    _require_pandas()

    if "cell_id" not in per_cell_df.columns:
        raise ValueError("per_cell_df must contain 'cell_id'.")
    if "channel" not in spots_df.columns:
        raise ValueError("spots_df must contain 'channel'.")
    if not include_dense_extra and "is_dense_extra" in spots_df.columns:
        spots_df = spots_df.loc[spots_df["is_dense_extra"].astype(int) == 0].copy()

    if dapi_img is not None and dapi_img.shape != nuclei_labels.shape:
        if dapi_img.ndim == 2 and dapi_img.T.shape == nuclei_labels.shape:
            dapi_img = dapi_img.T
        else:
            raise ValueError(
                f"dapi_img shape {getattr(dapi_img, 'shape', None)} must match nuclei_labels shape {nuclei_labels.shape}."
            )

    records = []
    long_records = []

    channels = sorted([str(c).lower() for c in spots_df["channel"].unique()])
    channels = [c for c in channels if c != "unknown"]
    spots_by_ch = {ch: spots_df.loc[spots_df["channel"].astype(str).str.lower() == ch].copy() for ch in channels}

    for _, row in per_cell_df.iterrows():
        cell_id = int(row["cell_id"])
        if cell_id <= 0:
            continue
        nuc_mask = (nuclei_labels == cell_id)
        if not nuc_mask.any():
            continue

        for ch in channels:
            s_cell = spots_by_ch[ch].loc[spots_by_ch[ch]["nucleus_id"].astype(int) == cell_id]
            if s_cell.shape[0] == 0:
                continue

            rc = s_cell[["row", "col"]].to_numpy(dtype=int, copy=False)

            # deterministic seed per (cell, channel)
            #seed = (int(base_seed) + 1000003 * cell_id + 1009 * (abs(hash(ch)) % 100000))
            CH_ID = {"red": 1, "green": 2, "cyan": 3}
            seed = int(base_seed) + 1000003 * cell_id + 1009 * CH_ID.get(ch, 9)
            rng = np.random.default_rng(seed)

            gr = compute_gr_three_curves(
                rc_spots=rc,
                nucleus_mask=nuc_mask,
                dapi_img=dapi_img,
                rmax=rmax,
                rng=rng,
                n_random=n_random,
                n_dapi=n_dapi,
                spot_kernel=spot_kernel,
                spot_sigma_px=spot_sigma_px,
                spot_disk_radius_px=spot_disk_radius_px,
            )

            area = float(row["nucleus_area_px"]) if "nucleus_area_px" in per_cell_df.columns else float(nuc_mask.sum())
            rho = float(rc.shape[0]) / area if area > 0 else np.nan

            rec = {
                "group": row.get("group", ""),
                "sample_id": row.get("sample_id", ""),
                "cell_id": cell_id,
                "channel": ch,
                "n_spots": int(rc.shape[0]),
                "rho_spots_per_px": rho,

                # r=0 values (normalized)
                "g0_exp": float(gr.g_exp[0]) if gr.g_exp.size else np.nan,
                "g0_rand": 1.0,  # normalized baseline
                "g0_dapi": float(gr.g_dapi[0]) if gr.g_dapi.size else np.nan,

                # r=0 values (raw; for auditing)
                "g0_raw_exp": float(gr.g_raw_exp[0]) if gr.g_raw_exp.size else np.nan,
                "g0_raw_rand": float(gr.g_raw_rand[0]) if gr.g_raw_rand.size else np.nan,
                "g0_raw_dapi": float(gr.g_raw_dapi[0]) if gr.g_raw_dapi.size else np.nan,

                "rmax": int(gr.r.max()) if gr.r.size else int(rmax),

                # reproducibility metadata
                "seed": int(seed),
                "n_random": int(n_random),
                "n_dapi": int(n_dapi),
                "spot_kernel": str(spot_kernel),
                "spot_sigma_px": float(spot_sigma_px),
                "spot_disk_radius_px": int(spot_disk_radius_px),
            }
            for k in ("centroid_row", "centroid_col", "nucleus_area_px"):
                if k in per_cell_df.columns:
                    rec[k] = float(row[k])
            records.append(rec)

            # long-form curves
            for i in range(gr.r.size):
                long_records.append({
                    "group": rec["group"],
                    "sample_id": rec["sample_id"],
                    "cell_id": cell_id,
                    "channel": ch,
                    "r_px": int(gr.r[i]),

                    # normalized curves
                    "g_exp": float(gr.g_exp[i]) if np.isfinite(gr.g_exp[i]) else np.nan,
                    "g_rand": 1.0,
                    "g_dapi": float(gr.g_dapi[i]) if np.isfinite(gr.g_dapi[i]) else np.nan,
                    "dg_exp": float(gr.dg_exp[i]) if np.isfinite(gr.dg_exp[i]) else np.nan,
                    "dg_dapi": float(gr.dg_dapi[i]) if np.isfinite(gr.dg_dapi[i]) else np.nan,

                    # raw curves (export rand for reproducibility)
                    "g_raw_exp": float(gr.g_raw_exp[i]) if np.isfinite(gr.g_raw_exp[i]) else np.nan,
                    "g_raw_rand": float(gr.g_raw_rand[i]) if np.isfinite(gr.g_raw_rand[i]) else np.nan,
                    "g_raw_dapi": float(gr.g_raw_dapi[i]) if np.isfinite(gr.g_raw_dapi[i]) else np.nan,
                })

    per_cell_gr_df = pd.DataFrame(records)
    gr_long_df = pd.DataFrame(long_records)
    return per_cell_gr_df, gr_long_df


def save_gr_outputs(
    outdir: str,
    per_cell_gr_df,
    gr_long_df,
    prefix: str = "gr",
):
    """
    Save:
      - {prefix}_per_cell.csv
      - {prefix}_long.csv
    """
    _require_pandas()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    per_cell_path = os.path.join(outdir, f"{prefix}_per_cell.csv")
    long_path = os.path.join(outdir, f"{prefix}_long.csv")
    per_cell_gr_df.to_csv(per_cell_path, index=False)
    gr_long_df.to_csv(long_path, index=False)
    return per_cell_path, long_path


# -----------------------------
# P-values at r=0 (paired Wilcoxon)
# -----------------------------
def compute_pvalues_g0(
    per_cell_csv: str,
    out_csv: str,
    min_spots: int = 30,
):
    """
    Compute paired Wilcoxon p-values at r=0 per group & channel.

    Tests
    -----
    1) exp vs rand baseline (=1): one-sample signed-rank on (g0_exp - 1)
    2) exp vs dapi: paired signed-rank on (g0_exp, g0_dapi), dropping NaNs

    Parameters
    ----------
    per_cell_csv : str
        Path to gr_per_cell.csv
    out_csv : str
        Output path for gr_pvalues.csv
    min_spots : int
        Only include nuclei with n_spots >= min_spots.

    Returns
    -------
    pd.DataFrame
        P-value table.
    """
    # --- Wilcoxon：diff 全 0 => p=1；并显式 zero_method/mode，避免 RuntimeWarning
    """
    Compute paired Wilcoxon p-values at r=0 per group & channel.

    Patch note
    ----------
    - If all diffs are ~0, Wilcoxon SE can be 0, causing RuntimeWarning and NaN p-values.
      We explicitly return p=1.0 in that degenerate "no difference" case.
    - Use zero_method="zsplit" to handle zero-differences robustly.
    - Use mode="auto" for stability across sample sizes / ties.
    """
    _require_pandas()
    if wilcoxon is None:
        raise RuntimeError("scipy is required for Wilcoxon test (scipy.stats.wilcoxon).")

    pc = pd.read_csv(per_cell_csv)
    if pc.empty:
        out = pd.DataFrame(columns=["group", "channel", "n_cells", "p_exp_vs_rand", "p_exp_vs_dapi"])
        out.to_csv(out_csv, index=False)
        return out

    pc = pc.loc[pc["n_spots"] >= int(min_spots)].copy()
    
    # 如果过滤后为空，直接写一个空结果表，避免 sort_values KeyError
    if pc.empty:
        out = pd.DataFrame(columns=["group", "channel", "n_cells", "p_exp_vs_rand", "p_exp_vs_dapi"])
        out.to_csv(out_csv, index=False)
        return out
    # 兼容：若上游表没有 group 列，则按单组处理
    if "group" not in pc.columns:
        pc["group"] = ""
        
    rows = []
    for (group, ch), sdf in pc.groupby(["group", "channel"]):
        # ---- exp vs rand baseline (=1): one-sample on d = g0_exp - 1
        x = sdf["g0_exp"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        n_cells = int(len(x))

        if n_cells < 2:
            p1 = np.nan
        else:
            d = x - 1.0
            if np.allclose(d, 0.0, rtol=0.0, atol=1e-12):
                p1 = 1.0
            else:
                try:
                    p1 = float(wilcoxon(
                        d,
                        alternative="two-sided",
                        zero_method="zsplit",
                        mode="auto",
                    ).pvalue)
                except Exception:
                    p1 = np.nan

        # ---- exp vs dapi: paired test (drop NaNs)
        x2 = sdf["g0_exp"].to_numpy(dtype=float)
        y2 = sdf["g0_dapi"].to_numpy(dtype=float)
        ok = np.isfinite(x2) & np.isfinite(y2)
        x2 = x2[ok]
        y2 = y2[ok]

        if x2.size < 2:
            p2 = np.nan
        else:
            d2 = x2 - y2
            if np.allclose(d2, 0.0, rtol=0.0, atol=1e-12):
                p2 = 1.0
            else:
                try:
                    p2 = float(wilcoxon(
                        x2,
                        y2,
                        alternative="two-sided",
                        zero_method="zsplit",
                        mode="auto",
                    ).pvalue)
                except Exception:
                    p2 = np.nan

        rows.append({
            "group": group,
            "channel": ch,
            "n_cells": n_cells,
            "p_exp_vs_rand": p1,
            "p_exp_vs_dapi": p2,
        })
        
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["group", "channel", "n_cells", "p_exp_vs_rand", "p_exp_vs_dapi"])
    else:
        out = out.sort_values(["group", "channel"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    return out


# -----------------------------
# Plotting (mean ± SEM like paper)
# - Multi-channel overlay per group (exp curves)
# - Random baseline y=1
# - DAPI pooled baseline
# - Legend and p-values outside the plotting box (right side panel)
# - Square plotting box (axes box aspect = 1)
# - Smaller plotting area (~2/3 of previous) by reserving a right-side panel
# - Fonts unified: Arial regular, size 6
# - Y ticks reduced to ~5–6 major ticks; X ticks fixed: 0,10,20,30,40,50 (up to r_plot)
# -----------------------------

from matplotlib.ticker import MaxNLocator, FixedLocator

# Global matplotlib style (safe to call multiple times)
if plt is not None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 6,
        "font.weight": "normal",
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
    })


def _sem(x: np.ndarray) -> float:
    """
    Standard error of the mean (SEM) ignoring NaNs.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    return float(np.nanstd(x, ddof=1) / np.sqrt(x.size))


def _channel_color_map() -> Dict[str, str]:
    """
    Required line colors:
      green -> green
      red   -> red
      cyan  -> cyan
      DAPI  -> blue
      Random-> grey
    """
    return {
        "green": "green",
        "red": "red",
        "cyan": "cyan",
        "dapi": "blue",
        "random": "grey",
    }


def plot_group_multi_channel_mean_sem(
    outdir: str,
    long_csv: str,
    per_cell_csv: str,
    pval_csv: Optional[str],
    channels: List[str],
    r_plot: int = 50,
    min_spots: int = 30,
    bin_px: int = 4,
    lw_exp: float = 1.0,
    lw_base: float = 0.8,
    prefix: str = "gr",
    save_png: bool = True,
    save_pdf: bool = True,
    plot_per_channel: bool = False,
):
    """
    Plot mean±SEM curves per group, overlaying multiple channels in one axes.
    Also overlays:
      - Random baseline (y=1) in grey dashed
      - DAPI pooled curve (mean±SEM of g_dapi across all channels/cells) in blue dotted

    Important plotting choice:
    - r=0 is kept as its own point (not binned).
    - r>=1 is binned by --plot_bin_px to improve smoothness.
    
    Curves shown
    ------------
    - Experimental g_exp(r): mean±SEM across nuclei, per channel (colored lines).
    - Random baseline: y=1 (grey dashed), with n = number of nuclei in the group after filtering.
    - DAPI pooled: g_dapi(r) mean±SEM pooled across all channels/cells (blue dotted),
      with n = number of nuclei that have finite g_dapi in the filtered set.

    Layout rules (as requested)
    ---------------------------
    1) Legend moved outside the plotting box: right side, upper half.
    2) P-values moved outside the plotting box: right side, lower half.
    3) Plotting box is square (box aspect = 1) and visually smaller by reserving a right-side panel.
    4) Fonts unified: Arial regular, size 6.
    5) X ticks fixed at 0,10,20,30,40,50 (up to r_plot); Y ticks reduced to ~5–6 major ticks.

    Notes
    -----
    - If a channel is absent (no nuclei pass min_spots for that channel in that group),
      it will not appear in the legend. This is expected.

    Parameters
    ----------
    outdir : str
        Output directory.
    long_csv : str
        Path to gr_long.csv.
    per_cell_csv : str
        Path to gr_per_cell.csv (used for filtering and n values).
    pval_csv : str or None
        Path to gr_pvalues.csv (for figure annotations). If None, skip annotation.
    channels : List[str]
        Channels to include (e.g., ["red","green","cyan"]).
    r_plot : int
        Max radius to display.
    min_spots : int
        Only include nuclei with >= this many spots in plot.
    bin_px : int
        Bin size for r>=1.
    prefix : str
        Output name prefix.
    save_png, save_pdf : bool
        Save .png and/or .pdf.
    plot_per_channel : bool
        Additionally save one plot per channel per group.
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Please `pip install matplotlib`.")
    _require_pandas()

    df = pd.read_csv(long_csv)
    pc = pd.read_csv(per_cell_csv)
    pv = pd.read_csv(pval_csv) if (pval_csv and os.path.exists(pval_csv)) else None

    # -------------------------
    # 1) Filter nuclei by min_spots and by requested channels
    # -------------------------
    pc_f = pc.loc[pc["n_spots"] >= int(min_spots)].copy()
    if pc_f.empty:
        print(f"[WARN] no nuclei pass min_spots={min_spots}; skip plotting.")
        return

    channels = [str(c).lower() for c in channels]
    pc_f["channel"] = pc_f["channel"].astype(str).str.lower()
    pc_f = pc_f.loc[pc_f["channel"].isin(channels)].copy()
    if pc_f.empty:
        print("[WARN] no nuclei left after channel filtering; skip plotting.")
        return

    # Keep only selected (cell_id, group, channel) in long table
    df["channel"] = df["channel"].astype(str).str.lower()
    keep_cells = pc_f[["cell_id", "group", "channel"]].drop_duplicates()
    df2 = df.merge(keep_cells, on=["cell_id", "group", "channel"], how="inner")
    df2 = df2.loc[df2["r_px"] <= int(r_plot)].copy()
    if df2.empty:
        print("[WARN] empty long data after filtering; skip plotting.")
        return

    colors = _channel_color_map()

    # -------------------------
    # 2) Radius binning (keep r=0 separate)
    #    r=0 -> r_bin=0
    #    r>=1 -> r_bin = 1 + floor((r-1)/bin_px)*bin_px
    # -------------------------
    bin_px = int(max(1, bin_px))
    r = df2["r_px"].to_numpy(dtype=int)
    r_bin = np.zeros_like(r)
    pos = r >= 1
    r_bin[pos] = 1 + ((r[pos] - 1) // bin_px) * bin_px
    df2["r_bin"] = r_bin

    groups = sorted(pc_f["group"].astype(str).unique().tolist())

    # -------------------------
    # Helpers for consistent legend ordering and right-panel layout
    # -------------------------
    legend_order = ["red", "green", "cyan", "random", "dapi"]

    # Right panel anchors (figure fraction coordinates)
    # Legend in upper half; p-values in lower half.
    right_panel_x = 0.25
    legend_y = 0.35
    pval_y = 0.22

    for g in groups:
        pc_g = pc_f.loc[pc_f["group"].astype(str) == str(g)].copy()
        df_g = df2.loc[df2["group"].astype(str) == str(g)].copy()
        if pc_g.empty or df_g.empty:
            continue

        # -------------------------
        # 3) Create figure with a reserved right-side panel
        #    - Axes box intentionally smaller by setting its position (for paper inch).
        # -------------------------
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        # [left, bottom, width, height] in figure fraction.
        # This makes the plot area smaller and leaves space on the right for legend/pvalues.
        ax.set_position([0.08, 0.18, 0.15, 0.18])
        ax.set_box_aspect(1)  # square plotting box

        # Map of legend items so we can enforce order and avoid odd auto-ordering
        legend_items: Dict[str, Tuple[object, str]] = {}

        # -------------------------
        # 4) Experimental curves per channel (mean ± SEM across nuclei)
        # -------------------------
        for ch in channels:
            pc_gc = pc_g.loc[pc_g["channel"] == ch]
            df_gc = df_g.loc[df_g["channel"] == ch]
            if pc_gc.empty or df_gc.empty:
                continue

            agg = (df_gc.groupby(["r_bin"])["g_exp"]
                   .agg(mean="mean", sem=lambda x: _sem(x.to_numpy()))
                   .reset_index()
                   .sort_values("r_bin"))

            x = agg["r_bin"].to_numpy(dtype=int)
            y = agg["mean"].to_numpy(dtype=float)
            e = agg["sem"].to_numpy(dtype=float)

            n_cells = int(pc_gc["cell_id"].nunique())
            ccol = colors.get(ch, None)

            (h_line,) = ax.plot(x, y, color=ccol, linewidth=lw_exp)
            ax.fill_between(x, y - e, y + e, alpha=0.18, color=ccol)

            legend_items[ch] = (h_line, f"{ch} (n={n_cells})")

        # X support for baselines
        xbase = np.unique(df_g["r_bin"].to_numpy(dtype=int))
        xbase = np.sort(xbase)

        # -------------------------
        # 5) Random baseline (=1), with n = unique nuclei in the filtered group
        # -------------------------
        n_cells_group = int(pc_g["cell_id"].nunique())
        (h_rand,) = ax.plot(
            xbase,
            np.ones_like(xbase, dtype=float),
            linestyle="--",
            color=colors["random"],
            linewidth=lw_base,
        )
        legend_items["random"] = (h_rand, f"Random (n={n_cells_group})")

        # -------------------------
        # 6) DAPI pooled baseline, with n = unique nuclei that have finite g_dapi
        # -------------------------
        # Use per-cell presence of finite DAPI values to define n robustly.
        dapi_valid_cells = (df_g.loc[np.isfinite(df_g["g_dapi"].to_numpy(dtype=float)), "cell_id"]
                            .drop_duplicates())
        n_dapi_cells = int(dapi_valid_cells.shape[0])

        agg_dapi = (df_g.groupby(["r_bin"])["g_dapi"]
                    .agg(mean="mean", sem=lambda x: _sem(x.to_numpy()))
                    .reset_index()
                    .sort_values("r_bin"))

        if (not agg_dapi.empty) and np.isfinite(agg_dapi["mean"].to_numpy()).any():
            xd = agg_dapi["r_bin"].to_numpy(dtype=int)
            yd = agg_dapi["mean"].to_numpy(dtype=float)
            ed = agg_dapi["sem"].to_numpy(dtype=float)

            (h_dapi,) = ax.plot(
                xd, yd,
                linestyle=":",
                color=colors["dapi"],
                linewidth=lw_exp,
            )
            ax.fill_between(xd, yd - ed, yd + ed, alpha=0.12, color=colors["dapi"])
            legend_items["dapi"] = (h_dapi, f"DAPI (n={n_dapi_cells})")

        # -------------------------
        # 7) Axes formatting
        # -------------------------
        ax.set_title(str(g))
        ax.set_xlabel("Radius (pixels)")
        ax.set_ylabel("Autocorrelation g(r)")
        ax.set_xlim(0, int(r_plot))

        # X ticks fixed: 0,10,20,... up to r_plot
        xticks = list(range(0, int(r_plot) + 1, 10))
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        
        ### y轴坐标刻度值选择：
        ### 1)包含y=1且严格均分坐标轴: ax.yaxis.set_major_locator(FixedLocator(ticks))
        ### 2)不严格包含y=1，只需为5个刻度值即可: ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        # Y ticks reduced to ~5–6
        #ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        # -------------------------
        # Y ticks: exactly 5 ticks, MUST include y=1.0
        # (Requires: from matplotlib.ticker import FixedLocator  at module level)
        # - Replace current line:
        #     ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        # - Paste this block AFTER all curves are plotted (after Random/DAPI lines),
        #   and AFTER ax.set_xlim(...). It will overwrite y-limits/ticks consistently.
        # -------------------------
        # 1) Get current y-range from plotted data (includes fills)
        ymin, ymax = ax.get_ylim()
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
            ymin, ymax = 0.0, 2.0
        # 2) Ensure y=1 is within the visible range
        pad = 0.05 * (ymax - ymin + 1e-12)
        ymin2, ymax2 = ymin, ymax
        if 1.0 < ymin2:
            ymin2 = 1.0 - pad
        if 1.0 > ymax2:
            ymax2 = 1.0 + pad
        if ymin2 >= ymax2:
            ymin2, ymax2 = 0.0, 2.0
        # 3) Build 5 ticks with y=1 guaranteed
        # Strategy:
        # - Start with 5 evenly spaced ticks in [ymin2, ymax2]
        # - Replace the closest tick with 1.0
        # - De-duplicate and, if needed, fill back to 5 ticks
        ticks = np.linspace(ymin2, ymax2, 5)
        # Force one tick to be exactly 1.0 (replace nearest)
        idx = int(np.argmin(np.abs(ticks - 1.0)))
        ticks[idx] = 1.0
        # Remove near-duplicates (can happen if range is tiny)
        # Keep order, enforce uniqueness with a small tolerance
        tol = 1e-9
        uniq = []
        for t in ticks:
            if not any(abs(t - u) < tol for u in uniq):
                uniq.append(float(t))
        ticks = np.array(uniq, dtype=float)
        # If uniqueness reduced count (<5), fill with evenly spaced values
        # while keeping 1.0 and endpoints.
        if ticks.size < 5:
            base = np.linspace(ymin2, ymax2, 5)
            base[int(np.argmin(np.abs(base - 1.0)))] = 1.0
            cand = []
            for t in base:
                if not any(abs(t - u) < tol for u in cand):
                    cand.append(float(t))
            ticks = np.array(cand, dtype=float)
        # Still short? add midpoint ticks deterministically (rare edge cases)
        while ticks.size < 5:
            mids = (ticks[:-1] + ticks[1:]) / 2.0
            for m in mids:
                if ticks.size >= 5:
                    break
                if not any(abs(m - u) < tol for u in ticks):
                    ticks = np.sort(np.append(ticks, float(m)))
        # If >5 (also rare), keep endpoints + 1.0 + two evenly spaced others
        if ticks.size > 5:
            ticks = np.sort(ticks)
            # Always keep ymin/ymax and 1.0
            keep = {ticks[0], ticks[-1], 1.0}
            # Add nearest-to-quarter points
            targets = [ymin2 + 0.25 * (ymax2 - ymin2), ymin2 + 0.75 * (ymax2 - ymin2)]
            for tar in targets:
                nearest = float(ticks[int(np.argmin(np.abs(ticks - tar)))])
                keep.add(nearest)
            ticks = np.array(sorted(keep), dtype=float)
            # If still not 5 due to duplicates, fall back to 5 linspace with forced 1.0
            if ticks.size != 5:
                ticks = np.linspace(ymin2, ymax2, 5)
                ticks[int(np.argmin(np.abs(ticks - 1.0)))] = 1.0
        # 4) Apply y-limits and fixed ticks
        ax.set_ylim(float(ymin2), float(ymax2))
        ### 选项：方案1)/2)
        #ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="both", labelsize=6)

        # -------------------------
        # 8) P-values text (right panel, lower half)
        # -------------------------
        pval_text = None
        if pv is not None and (not pv.empty):
            pv_g = pv.loc[pv["group"].astype(str) == str(g)].copy()
            if not pv_g.empty:
                lines = []
                for ch in channels:
                    pv_gc = pv_g.loc[pv_g["channel"].astype(str).str.lower() == ch]
                    if pv_gc.empty:
                        continue
                    p1 = pv_gc["p_exp_vs_rand"].iloc[0]
                    p2 = pv_gc["p_exp_vs_dapi"].iloc[0]

                    if np.isfinite(p1) and np.isfinite(p2):
                        lines.append(f"{ch}: p(rand)={p1:.3g}, p(dapi)={p2:.3g}")
                    elif np.isfinite(p1):
                        lines.append(f"{ch}: p(rand)={p1:.3g}, p(dapi)=NA")
                    elif np.isfinite(p2):
                        lines.append(f"{ch}: p(rand)=NA, p(dapi)={p2:.3g}")
                    else:
                        lines.append(f"{ch}: p(rand)=NA, p(dapi)=NA")

                if lines:
                    pval_text = "\n".join(lines)

        if pval_text:
            fig.text(
                right_panel_x, pval_y,
                pval_text,
                ha="left", va="top",
                fontsize=6, family="Arial",
            )

        # -------------------------
        # 9) Legend (right panel, upper half), enforce stable order
        # -------------------------
        handles, labels = [], []
        for key in legend_order:
            if key in legend_items:
                h, lab = legend_items[key]
                handles.append(h)
                labels.append(lab)

        # Only show legend if there is at least one experimental curve or baseline
        if handles:
            fig.legend(
                handles, labels,
                loc="upper left",
                bbox_to_anchor=(right_panel_x, legend_y),
                frameon=True,
                borderaxespad=0.0,
            )

        # -------------------------
        # 10) Save with tight bounding box so the right panel isn't cut off
        # -------------------------
        base = os.path.join(outdir, f"{prefix}_mean_sem_{g}")
        if save_png:
            plt.savefig(base + ".png", dpi=300, bbox_inches="tight")
        if save_pdf:
            plt.savefig(base + ".pdf", bbox_inches="tight")
        plt.close(fig)
        print("[OK] saved plot:", base + (".png/.pdf" if save_png and save_pdf else ".png" if save_png else ".pdf"))

        # -------------------------
        # 11) Optional: per-channel-only plots
        #     Keep the same right panel style, square plotting box, fonts, and ticks.
        # -------------------------
        if plot_per_channel:
            for ch in channels:
                pc_gc = pc_g.loc[pc_g["channel"] == ch]
                df_gc = df_g.loc[df_g["channel"] == ch]
                if pc_gc.empty or df_gc.empty:
                    continue

                agg = (df_gc.groupby(["r_bin"])["g_exp"]
                       .agg(mean="mean", sem=lambda x: _sem(x.to_numpy()))
                       .reset_index()
                       .sort_values("r_bin"))

                x = agg["r_bin"].to_numpy(dtype=int)
                y = agg["mean"].to_numpy(dtype=float)
                e = agg["sem"].to_numpy(dtype=float)

                n_cells = int(pc_gc["cell_id"].nunique())

                fig2, ax2 = plt.subplots(figsize=(7.2, 5.2))
                ax2.set_position([0.08, 0.18, 0.44, 0.56])
                ax2.set_box_aspect(1)

                (h_line,) = ax2.plot(x, y, color=colors.get(ch, None), linewidth=2)
                ax2.fill_between(x, y - e, y + e, alpha=0.18, color=colors.get(ch, None))

                xbase2 = np.unique(df_gc["r_bin"].to_numpy(dtype=int))
                xbase2 = np.sort(xbase2)

                (h_rand2,) = ax2.plot(
                    xbase2, np.ones_like(xbase2, dtype=float),
                    linestyle="--", color=colors["random"], linewidth=2
                )

                ax2.set_title(f"{g} | {ch}")
                ax2.set_xlabel("Radius (pixels)")
                ax2.set_ylabel("Autocorrelation g(r)")
                ax2.set_xlim(0, int(r_plot))

                xticks2 = list(range(0, int(r_plot) + 1, 10))
                ax2.xaxis.set_major_locator(FixedLocator(xticks2))
                ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
                ax2.tick_params(axis="both", labelsize=6)

                # Right-panel legend for single-channel figure
                fig2.legend(
                    [h_line, h_rand2],
                    [f"{ch} (n={n_cells})", f"Random (n={n_cells})"],
                    loc="upper left",
                    bbox_to_anchor=(right_panel_x, legend_y),
                    frameon=True,
                    borderaxespad=0.0,
                )

                base2 = os.path.join(outdir, f"{prefix}_mean_sem_{g}_{ch}")
                if save_png:
                    plt.savefig(base2 + ".png", dpi=300, bbox_inches="tight")
                if save_pdf:
                    plt.savefig(base2 + ".pdf", bbox_inches="tight")
                plt.close(fig2)

# -----------------------------
# IO helpers (match ecdna_fish_pipeline_v4.py)
# -----------------------------
try:
    import imageio.v3 as iio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


# ==============================
# [PATCH] Unify TIFF->2D conversion and TIFF reading for DAPI semantics
# - Keep ONE ensure_2d() (with squeeze support)
# - Default rgb="b" for DAPI (blue channel)
# - "auto" chooses a single channel (P99 winner), NOT luminance mix
# - Add optional tif_first_page switch to avoid OME discontiguous warning when needed
# ==============================
def ensure_2d(img: np.ndarray, z_project: str = "max", rgb: str = "b") -> np.ndarray:
    """
    Convert microscopy image array to 2D.

    Rules (DAPI-oriented)
    ---------------------
    - If single-channel 2D -> return directly.
    - If RGB(A) -> prefer selecting one channel (default "b" for DAPI).
      "auto" selects a single channel by highest P99 (NOT luminance mixing).
    - If Z-stack (Z,H,W) -> project by z_project ("max" or "mean").
    - Squeezes singleton dims first (OME-like exports).

    Parameters
    ----------
    z_project : {"max","mean"}
    rgb : {"auto","r","g","b","mean","max"}
        For DAPI, use "b" (default). Avoid "auto" unless you truly do not know encoding.
    """
    if img is None:
        raise ValueError("ensure_2d: img is None")
    a = np.asarray(img)

    # absorb OME-like singleton dims
    if a.ndim > 2 and 1 in a.shape:
        a = np.squeeze(a)

    if a.ndim == 2:
        return a

    if a.ndim != 3:
        raise ValueError(f"ensure_2d: unsupported ndim={a.ndim}, shape={a.shape}. Expect 2D or 3D.")

    # 1) RGB(A) channel-last: (H,W,3/4)
    if a.shape[-1] in (3, 4):
        c = a[..., :3].astype(np.float32, copy=False)
        rgb = str(rgb).lower()
        if rgb in ("r", "g", "b"):
            return c[..., {"r": 0, "g": 1, "b": 2}[rgb]]
        if rgb == "mean":
            return c.mean(axis=-1)
        if rgb == "max":
            return c.max(axis=-1)
        if rgb == "auto":
            p99 = np.array([np.percentile(c[..., i], 99) for i in range(3)], dtype=float)
            return c[..., int(np.argmax(p99))]
        raise ValueError("ensure_2d: rgb must be one of auto|r|g|b|mean|max")

    # 2) RGB(A) channel-first: (3/4,H,W)
    if a.shape[0] in (3, 4) and a.shape[1] > 8 and a.shape[2] > 8:
        c = a[:3, ...].astype(np.float32, copy=False)
        rgb = str(rgb).lower()
        if rgb in ("r", "g", "b"):
            return c[{"r": 0, "g": 1, "b": 2}[rgb], ...]
        if rgb == "mean":
            return c.mean(axis=0)
        if rgb == "max":
            return c.max(axis=0)
        if rgb == "auto":
            p99 = np.array([np.percentile(c[i, ...], 99) for i in range(3)], dtype=float)
            return c[int(np.argmax(p99)), ...]
        raise ValueError("ensure_2d: rgb must be one of auto|r|g|b|mean|max")

    # 3) Otherwise treat as Z-stack: (Z,H,W)
    z_project = str(z_project).lower()
    if z_project == "max":
        return a.max(axis=0)
    if z_project == "mean":
        return a.mean(axis=0)
    raise ValueError("ensure_2d: z_project must be 'max' or 'mean'")


def read_tiff_any(
    path: str,
    *,
    tif_first_page: bool = False,
    z_project: str = "max",
    rgb: str = "b",
) -> np.ndarray:
    """
    Read image (prefer TIFF) and return 2D array via ensure_2d().

    Default behavior:
    - Use imageio.imread (if available) or tifffile.imread to read full content (supports stacks).
    Optional:
    - tif_first_page=True reads first TIFF page via TiffFile(...).pages[0].asarray()
      to avoid OME "discontiguous storage" warnings. Use only if you know first page is correct.

    Returns
    -------
    2D np.ndarray
    """
    if path is None:
        raise ValueError("read_tiff_any: path is None")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    ext = p.suffix.lower()
    if ext not in (".tif", ".tiff"):
        # allow other formats through imageio when available
        if _HAS_IMAGEIO:
            arr = iio.imread(str(p))
            return ensure_2d(arr, z_project=z_project, rgb=rgb)
        raise RuntimeError(f"Unsupported image extension: {ext}. Install imageio for non-TIFF formats.")

    if tiff is None:
        raise RuntimeError("tifffile not available. Install: pip install tifffile")

    if tif_first_page:
        with tiff.TiffFile(str(p)) as tif:
            arr = tif.pages[0].asarray()
    else:
        # full read (stack-friendly)
        if _HAS_IMAGEIO:
            try:
                arr = iio.imread(str(p))
            except Exception:
                arr = tiff.imread(str(p))
        else:
            arr = tiff.imread(str(p))

    return ensure_2d(arr, z_project=z_project, rgb=rgb)


# -----------------------------
# ecdna_fish_pipeline_v4.py folder adapter: OUTDIR/sample_id/*
# -----------------------------
def find_sample_dirs(outdir: str) -> list[Path]:
    root = Path(outdir)
    if not root.exists():
        raise FileNotFoundError(outdir)
    sample_dirs = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if (p / "nuclei_labels.tif").exists() and (p / "per_cell_summary.csv").exists():
            sample_dirs.append(p)
    sample_dirs.sort(key=lambda x: x.name)
    return sample_dirs


def load_spots_all_channels(sample_dir: Path) -> "pd.DataFrame":
    _require_pandas()
    dfs = []
    for ch in ("red", "green", "cyan"):
        f = sample_dir / f"spots_{ch}.csv"
        if f.exists():
            dfs.append(load_spots_csv(str(f)))
    if len(dfs) == 0:
        return pd.DataFrame(columns=["row", "col", "nucleus_id", "channel"])
    return pd.concat(dfs, ignore_index=True)


def load_sample_v4(sample_dir: Path, nuclei_name: str | None = None, per_cell_name: str | None = None,
                   spots_basenames: list[str] | None = None):
    _require_pandas()
    if tiff is None:
        raise RuntimeError("tifffile is required. Install: pip install tifffile")

    nuclei_fn = nuclei_name or "nuclei_labels.tif"
    per_cell_fn = per_cell_name or "per_cell_summary.csv"

    per_cell = pd.read_csv(sample_dir / per_cell_fn)
    nuclei_lbl = tiff.imread(str(sample_dir / nuclei_fn))

    # spots
    if spots_basenames is None:
        basenames = _detect_spots_files(str(sample_dir))  # uses sample_dir
    else:
        basenames = spots_basenames

    dfs = []
    for bn in basenames:
        f = sample_dir / bn
        if f.exists():
            dfs.append(load_spots_csv(str(f)))
    spots_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["row","col","nucleus_id","channel"])

    if "cell_id" not in per_cell.columns and "nucleus_id" in per_cell.columns:
        per_cell = per_cell.rename(columns={"nucleus_id": "cell_id"})
    return nuclei_lbl, per_cell, spots_df

# -----------------------------
# 增加：可选读取 manifest，给每个 sample 提供 DAPI 路径（用于 dapi-weighted baseline）
# -----------------------------
def load_manifest_v4(manifest_csv: str) -> "pd.DataFrame":
    _require_pandas()
    df = pd.read_csv(manifest_csv)
    if "sample_id" not in df.columns or "dapi" not in df.columns:
        raise ValueError("manifest must contain columns: sample_id, dapi")

    base = Path(manifest_csv).resolve().parent

    def _resolve(p):
        if pd.isna(p) or str(p).strip() == "":
            return None
        p = Path(str(p))
        if not p.is_absolute():
            p = base / p
        return str(p)

    df["sample_id"] = df["sample_id"].astype(str)
    df["dapi"] = df["dapi"].apply(_resolve)
    return df


def build_dapi_map(manifest_csv: str | None) -> dict[str, str]:
    if manifest_csv is None:
        return {}
    dfm = load_manifest_v4(manifest_csv)
    m = {}
    for _, r in dfm.iterrows():
        sid = str(r["sample_id"])
        dp = r["dapi"]
        if dp:
            m[sid] = str(dp)
    return m

# -----------------------------
# TIFF utilities (_ensure_2d、_read_tif_2d：已弃用，保留以作为维护参考)
# -----------------------------
def _ensure_2d(
    img: np.ndarray,
    z_project: str = "max",
    rgb: str = "auto",
) -> np.ndarray:
    """
    Convert TIFF array to 2D array.

    Supports common microscopy layouts:
      - 2D: (H,W)
      - Z-stack: (Z,H,W) -> project to (H,W)
      - RGB(A): (H,W,3/4) -> choose channel or mean/max or luminance-like
      - Channel-first RGB(A): (3/4,H,W)
      - OME-like with singleton dims: squeezes dims==1 then re-evaluates

    Parameters
    ----------
    z_project : {"max","mean"}
        Projection method for Z-stack.
    rgb : {"auto","r","g","b","mean","max"}
        Selection rule for RGB(A) inputs.

    Returns
    -------
    np.ndarray
        2D image array.
    """
    if img is None:
        raise ValueError("img is None")
    a = np.asarray(img)
    if a.ndim > 2 and 1 in a.shape:
        a = np.squeeze(a)

    if a.ndim == 2:
        return a

    if a.ndim == 3:
        # RGB(A) last
        if a.shape[-1] in (3, 4):
            c = a[..., :3].astype(np.float32, copy=False)
            if rgb == "r":
                return c[..., 0]
            if rgb == "g":
                return c[..., 1]
            if rgb == "b":
                return c[..., 2]
            if rgb == "mean":
                return c.mean(axis=-1)
            if rgb == "max":
                return c.max(axis=-1)
            return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]

        # Channel-first RGB(A)
        if a.shape[0] in (3, 4) and a.shape[1] > 8 and a.shape[2] > 8:
            c = a[:3, ...].astype(np.float32, copy=False)
            if rgb == "r":
                return c[0]
            if rgb == "g":
                return c[1]
            if rgb == "b":
                return c[2]
            if rgb == "mean":
                return c.mean(axis=0)
            if rgb == "max":
                return c.max(axis=0)
            return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

        # Otherwise: Z-stack (Z,H,W)
        if z_project == "max":
            return a.max(axis=0)
        if z_project == "mean":
            return a.mean(axis=0)
        raise ValueError("z_project must be 'max' or 'mean'")

    raise ValueError(f"Unsupported tif array ndim={a.ndim}.")

# --- TIFF 读取：优先用 TiffFile(...).pages[0].asarray()，避免 OME series discontiguous 警告
def _read_tif_2d(path: str, z_project: str = "max", rgb: str = "auto") -> np.ndarray:
    """
    Read a TIFF and convert to 2D via _ensure_2d.

    Patch note
    ----------
    For some OME/RGB TIFFs, tifffile may warn about "OME series cannot handle discontiguous storage".
    Using TiffFile(...).pages[0].asarray() reads the first IFD directly and avoids OME-series parsing.
    """
    if tiff is None:  # pragma: no cover
        raise RuntimeError("tifffile is required to read tif/tiff.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"TIFF not found: {path}")

    with tiff.TiffFile(path) as tif:
        # Most microscopy exports store the actual pixel plane in the first page.
        # This avoids OME-series construction that can trigger discontiguous-storage warnings.
        arr = tif.pages[0].asarray()

    return _ensure_2d(arr, z_project=z_project, rgb=rgb)

# -----------------------------
# Spots file detection helpers
# -----------------------------
def _detect_spots_files(outdir: str) -> List[str]:
    """
    Auto-detect spots_*.csv basenames in outdir.
    """
    return sorted([p.name for p in Path(outdir).glob("spots_*.csv")])


def _basename_to_channel(name: str) -> str:
    """
    Infer channel from filename basename.
    """
    n = name.lower()
    if "red" in n:
        return "red"
    if "green" in n:
        return "green"
    if "cyan" in n or "aqua" in n:
        return "cyan"
    return "unknown"


# -----------------------------
# Standalone CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    
    ap = argparse.ArgumentParser(
        description="Compute FFT-based g(r) autocorrelation from ecdna_fish_pipeline_v4 outputs; write per-sample + global summaries."
    )
    
    # --- CLI additions/changes (match fish_pipeline_v4) ---
    ap.add_argument("--outdir", required=True,
                    help="Same OUTDIR as ecdna_fish_pipeline_v4.py (contains per-sample subfolders).")
    ap.add_argument("--sample_id", default=None,
                    help="If set, only process OUTDIR/sample_id. Otherwise process all sample subfolders.")
    ap.add_argument("--manifest", default=None,
                    help="Optional: same manifest CSV used by fish pipeline; used to load DAPI for dapi-weighted control.")
    ap.add_argument("--z_project", default="max", choices=["max", "mean"],
                    help="If DAPI is 3D, z-projection method (only used when loading DAPI via manifest).")
    ap.add_argument("--dapi_rgb", default="b", choices=["auto","r","g","b","mean","max"],
                    help="If DAPI image is RGB(A), which channel to use (only for dapi-weighted control).")
    
    # tif 
    ap.add_argument("--tif_first_page", action="store_true",
                    help="If set, read TIFF first page only (avoids some OME discontiguous warnings). "
                         "Use only if you know the first page is the correct plane.")
    
    ap.add_argument("--dapi", default=None,
                    help="DAPI tif/tiff (for DAPI-weighted control). If omitted, g_dapi will be NaN.")
    ap.add_argument("--dapi_z_project", default="max", choices=["max", "mean"],
                    help="If DAPI is a Z-stack, projection method.")

    ap.add_argument("--nuclei_labels", default=None,
                    help="Override nuclei_labels.tif basename in OUTDIR (default: nuclei_labels.tif).")
    ap.add_argument("--per_cell", default=None,
                    help="Override per_cell_summary.csv basename in OUTDIR (default: per_cell_summary.csv).")

    # IMPORTANT: basenames only
    ap.add_argument("--spots", nargs="*", default=None,
                    help="Basenames ONLY (no paths). Example: --spots spots_red.csv spots_green.csv. "
                         "If omitted, auto-detect spots_*.csv in OUTDIR.")
    ap.add_argument("--channels", nargs="*", default=None,
                    help="Compute only these channels (e.g. --channels red green cyan). If omitted, use all detected channels.")

    ap.add_argument("--rmax", type=int, default=100, help="Max radius (pixels). Will be clipped to image size.")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    ap.add_argument("--n_random", type=int, default=5, help="Random baseline repeats per nucleus.")
    ap.add_argument("--n_dapi", type=int, default=3, help="DAPI-control repeats per nucleus.")
    ap.add_argument("--exclude_dense_extra", action="store_true",
                    help="If set, exclude spots with is_dense_extra==1 (when column exists).")

    ap.add_argument("--prefix", default="gr", help="Output prefix for CSVs.")

    # ---- kernel options (most important)
    ap.add_argument("--spot_kernel", default="gaussian", choices=["gaussian", "disk", "delta"],
                    help="Spot rendering kernel. gaussian(default) best matches PSF; disk is fixed-radius; delta is original point.")
    ap.add_argument("--spot_sigma_px", type=float, default=1.5,
                    help="Gaussian sigma (pixels) for --spot_kernel gaussian.")
    ap.add_argument("--spot_disk_radius_px", type=int, default=2,
                    help="Disk radius (pixels) for --spot_kernel disk.")

    # ---- plotting options
    ap.add_argument("--plot", action="store_true",
                    help="Generate group-level mean±SEM plots (PNG/PDF) from outputs in OUTDIR.")
    ap.add_argument("--plot_rmax", type=int, default=50,
                    help="Max radius to show in plot (pixels).")
    ap.add_argument("--plot_min_spots", type=int, default=10,
                    help="Only include nuclei with >= this many spots in plot & p-value computation.")
    ap.add_argument("--plot_bin_px", type=int, default=4,
                    help="Radius bin size (pixels) for r>=1 (r=0 kept separate).")
    ap.add_argument("--plot_per_channel", action="store_true",
                    help="Also output per-channel-only plots (in addition to multi-channel overlay).")
    
    return ap
    
# -----------------------------
# main batch driver (OUTDIR/sample_id/*)
# -----------------------------
def main():
    ap = build_argparser()
    args = ap.parse_args()
    _require_pandas()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dapi_map = build_dapi_map(args.manifest)

    # decide which sample dirs
    if args.sample_id:
        sample_dirs = [outdir / args.sample_id]
        if not sample_dirs[0].exists():
            raise FileNotFoundError(f"sample folder not found: {sample_dirs[0]}")
        if not (sample_dirs[0] / "nuclei_labels.tif").exists():
            raise FileNotFoundError(f"missing nuclei_labels.tif in {sample_dirs[0]}")
            
    else:
        sample_dirs = find_sample_dirs(str(outdir))

        # [PATCH] allow outdir itself to be a sample folder
        if len(sample_dirs) == 0:
            if (outdir / "nuclei_labels.tif").exists() and (outdir / "per_cell_summary.csv").exists():
                sample_dirs = [outdir]
            else:
                raise RuntimeError(
                    f"No sample subfolders found in {outdir}.\n"
                    f"Expected either:\n"
                    f"  (1) {outdir}\\<sample_id>\\nuclei_labels.tif + per_cell_summary.csv\n"
                    f"or (2) {outdir}\\nuclei_labels.tif + per_cell_summary.csv (single-sample mode)."
                )
                
    all_pc = []
    all_long = []

    for sd in sample_dirs:
        
        nuclei_lbl, per_cell_df, spots_df = load_sample_v4(
            sd,
            nuclei_name=args.nuclei_labels,
            per_cell_name=args.per_cell,
            spots_basenames=args.spots
        )
        
        # channels filter for computation
        if args.channels:
            want = {c.lower() for c in args.channels}
            spots_df["channel"] = spots_df["channel"].astype(str).str.lower()
            spots_df = spots_df.loc[spots_df["channel"].isin(want)].copy()
        
        # DAPI: args.dapi > manifest map
        dapi_img = None
        dapi_path = None
        if args.dapi:
            dapi_path = args.dapi
        elif sd.name in dapi_map:
            dapi_path = dapi_map[sd.name]
        
        if dapi_path:
            try:
                dapi_img = read_tiff_any(
                    dapi_path,
                    tif_first_page=args.tif_first_page,
                    z_project=args.dapi_z_project if args.dapi else args.z_project,  # 或统一只保留一个 z_project 参数
                    rgb=args.dapi_rgb,
                ).astype(np.float32, copy=False)
            except Exception as e:
                warnings.warn(f"[WARN] failed to load DAPI for {sd.name}: {e}. Will skip dapi-weighted control.",
                              RuntimeWarning)
                dapi_img = None
                
        pc_gr, long_gr = compute_gr_from_v4_tables(
            nuclei_labels=nuclei_lbl,
            dapi_img=dapi_img,
            per_cell_df=per_cell_df,
            spots_df=spots_df,
            rmax=args.rmax,
            base_seed=args.seed,
            n_random=args.n_random,
            n_dapi=args.n_dapi,
            include_dense_extra=(not args.exclude_dense_extra),
            spot_kernel=args.spot_kernel,
            spot_sigma_px=args.spot_sigma_px,
            spot_disk_radius_px=args.spot_disk_radius_px,
        )

        # 统一列顺序：group, sample_id 放前（与 fish pipeline 全局表一致）
        if not pc_gr.empty and "group" in pc_gr.columns and "sample_id" in pc_gr.columns:
            cols = ["group", "sample_id"] + [c for c in pc_gr.columns if c not in ("group", "sample_id")]
            pc_gr = pc_gr.loc[:, cols]
        if not long_gr.empty and "group" in long_gr.columns and "sample_id" in long_gr.columns:
            cols = ["group", "sample_id"] + [c for c in long_gr.columns if c not in ("group", "sample_id")]
            long_gr = long_gr.loc[:, cols]

        # per-sample outputs in OUTDIR/sample_id/
        save_gr_outputs(str(sd), pc_gr, long_gr, prefix=args.prefix)
        
        # per-sample p-values + plots（如果你原来就是全局 outdir 画图，这里改为 sd）
        #pval_path = sd / "gr_pvalues.csv"
        pval_path = sd / f"{args.prefix}_pvalues.csv"
        compute_pvalues_g0(
            per_cell_csv=str(sd / f"{args.prefix}_per_cell.csv"),
            out_csv=str(pval_path),
            min_spots=args.plot_min_spots,
        )
        if args.plot:
            plot_group_multi_channel_mean_sem(
                outdir=str(sd),
                long_csv=str(sd / f"{args.prefix}_long.csv"),
                per_cell_csv=str(sd / f"{args.prefix}_per_cell.csv"),
                pval_csv=str(pval_path),
                channels=[c.lower() for c in (args.channels or ["red","green","cyan"])],
                r_plot=args.plot_rmax,
                min_spots=args.plot_min_spots,
                bin_px=args.plot_bin_px,
                prefix=args.prefix,
                save_png=True,
                save_pdf=True,
                plot_per_channel=args.plot_per_channel,
            )
            
        all_pc.append(pc_gr)
        all_long.append(long_gr)

    # global summaries in OUTDIR/
    if len(all_pc) > 0:
        df_all_pc = pd.concat(all_pc, ignore_index=True)
        #df_all_pc.to_csv(outdir / f"ALL_{args.prefix}_per_cell.csv", index=False)

    if len(all_long) > 0:
        df_all_long = pd.concat(all_long, ignore_index=True)
        #df_all_long.to_csv(outdir / f"ALL_{args.prefix}_long.csv", index=False)
    
    # global p-values from ALL_gr_per_cell.csv（可选但推荐，便于一键比较）
    #all_pc_path = outdir / f"ALL_{args.prefix}_per_cell.csv"
    #if all_pc_path.exists():
    #    compute_pvalues_g0(
    #        per_cell_csv=str(all_pc_path),
    #        out_csv=str(outdir / f"ALL_{args.prefix}_pvalues.csv"),
    #        min_spots=args.plot_min_spots,
    #    )
        
    print("[OK] Done. Outputs written to:", str(outdir))


if __name__ == "__main__":  # pragma: no cover
    main()
