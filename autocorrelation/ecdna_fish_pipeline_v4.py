#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# ============================
ecdna_fish_pipeline_v4.py
# ============================
# PATCH: StarDist-only improvements
# (1) add CLI args
# (2) add StarDist preprocessing + tunable thresholds
# (3) add post-merge to fix over-segmentation
# (4) optional dense-region "rescue" (watershed within under-segmented regions)
# ============================
ecdna_fish_pipeline_v3.py
# ============================
DNA-FISH quantification pipeline (2-color / 3-color) in Python:
- Inputs: aligned single-channel TIFFs (DAPI + each FISH channel) + optional merged composite TIFF
- Nuclei instance segmentation: Cellpose (nuclei) OR StarDist (2D_versatile_fluo) OR classical watershed fallback
- Spot detection: big-fish (detect_spots + automated_threshold) with robust z-score standardization
- Dense region decomposition: big-fish if available; otherwise a conservative heuristic decomposition fallback
- Colocalization:
    * 3-color: red-green, red-cyan, green-cyan
    * 2-color: red-green, red-red, green-green
  computed per nucleus and per image

Outputs per sample:
- nuclei_labels.tif
- nuclei_labels_pseudocolor.png (with in-image IDs + legend)
- fish_{ch}_z_clipped.png (debug)
- spots_{ch}.csv (spot-level table)
- per_cell_summary.csv (cell-level counts + colocalization)
- per_image_summary.csv (image-level totals + QC stats)
- QC overlays (optional export), napari interactive QC (optional)

NOTE:
- The autocorrelation g(r) / FFT part from v2 is intentionally NOT included here. Add later after you provide the corrected logic.

Dependencies:
- numpy, pandas, tifffile, matplotlib, scikit-image, scipy
- optional: cellpose, stardist, bigfish, napari

"""

import os
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label as sklabel, regionprops
from skimage.segmentation import watershed, find_boundaries
from skimage.util import img_as_float
from skimage.morphology import (
    remove_small_objects, remove_small_holes, binary_opening,
    binary_closing, dilation, disk, h_maxima
)
from skimage.feature import peak_local_max

# --- add imports (near top, with other skimage imports) ---v4
from skimage.filters import gaussian
from skimage.measure import label as sklabel, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes, disk, dilation
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt

# ---- optional deps ----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings(
    "ignore",
    message=r".*OME series cannot handle discontiguous storage.*",
    category=UserWarning,
    module=r"tifffile.*"
)

_HAS_CELLPOSE = False
_HAS_STARDIST = False
_HAS_BIGFISH = False
_HAS_NAPARI = False

try:
    from cellpose import models as cellpose_models
    _HAS_CELLPOSE = True
except Exception:
    _HAS_CELLPOSE = False

try:
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize as csbdeep_normalize
    _HAS_STARDIST = True
except Exception:
    _HAS_STARDIST = False

try:
    import bigfish
    from bigfish.detection import detect_spots
    try:
        # some versions may expose these
        from bigfish.detection import decompose_dense, get_dense_region
        _HAS_BIGFISH_DENSE = True
    except Exception:
        _HAS_BIGFISH_DENSE = False
    _HAS_BIGFISH = True
except Exception:
    _HAS_BIGFISH = False
    _HAS_BIGFISH_DENSE = False

try:
    import napari
    _HAS_NAPARI = True
except Exception:
    _HAS_NAPARI = False


# -----------------------------
# Silence known tifffile warning (non-fatal)
# -----------------------------
warnings.filterwarnings(
    "ignore",
    message=r".*OME series cannot handle discontiguous storage.*",
    category=UserWarning,
)

# -----------------------------
# 解决：Cellpose 密集核漏检（提高召回：cellprob/flow/tile 等参数可控）
# -----------------------------
import inspect

def _filter_kwargs(func, kwargs: dict) -> dict:
    """Keep only kwargs accepted by func signature (for cross-version compatibility)."""
    try:
        sig = inspect.signature(func)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        return kwargs

def cap_spots_per_nucleus(df: pd.DataFrame, max_spots: int, rank_col: str = "peak_intensity_raw") -> pd.DataFrame:
    """
    Keep top-K spots per nucleus by rank_col (descending).
    Does NOT change output columns; only drops rows.
    """
    if df is None or df.empty or max_spots is None or max_spots <= 0:
        return df
    if "nucleus_id" not in df.columns:
        return df
    if rank_col not in df.columns:
        # fallback: keep arbitrary first K
        return df.groupby("nucleus_id", group_keys=False).head(max_spots)

    df = df.sort_values(["nucleus_id", rank_col], ascending=[True, False])
    df = df.groupby("nucleus_id", group_keys=False).head(max_spots)
    return df


# -----------------------------
# IO helpers
# -----------------------------
from pathlib import Path

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

def read_tiff_any(path: str, *, tif_first_page: bool = False) -> np.ndarray:
    """
    Read image from common formats:
      - TIFF/TIF (2D/3D, multi-page)
      - JPG/JPEG/PNG/BMP (2D or RGB)

    Returns
    -------
    np.ndarray
        Raw image array as returned by the backend.
        May be:
          (H,W), (Z,H,W), (H,W,3/4), (3/4,H,W), etc.
        Downstream `ensure_2d()` will standardize it.
    """
    if path is None:
        raise ValueError("read_tiff_any: path is None")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    ext = p.suffix.lower()

    # [PATCH] If requested, read the first TIFF page directly to avoid OME-series
    # "discontiguous storage" warnings for some RGB/OME-TIFF exports.
    if tif_first_page and ext in (".tif", ".tiff"):
        if tiff is None:
            raise RuntimeError("tifffile is required to read tif/tiff.")
        with tiff.TiffFile(str(p)) as tif:
            return np.asarray(tif.pages[0].asarray())


    # Prefer imageio for broad format support (incl. jpg/bmp/png/tif)
    if _HAS_IMAGEIO and not (tif_first_page and ext in (".tif", ".tiff")):
        try:
            arr = iio.imread(str(p))
            return np.asarray(arr)
        except Exception:
            pass  # fall through to format-specific readers

    # TIFF fallback
    if ext in (".tif", ".tiff"):
        return tiff.imread(str(p))

    # Raster fallback via PIL (jpg/png/bmp etc.)
    if _HAS_PIL:
        with Image.open(str(p)) as im:
            # keep original mode; convert only if palette, etc.
            # For grayscale jpg/bmp, this will be (H,W); for RGB -> (H,W,3)
            im = im.convert("RGB") if im.mode in ("P", "PA") else im
            return np.asarray(im)

    raise RuntimeError(
        "Failed to read image. Install imageio and pillow: `pip install imageio pillow`."
    )


def ensure_2d(img: np.ndarray, z_project: str = "max", rgb: str = "auto") -> np.ndarray:
    """
    Accept 2D/3D or RGB images.

    Handles common TIFF shapes:
      - (H, W)
      - (Z, H, W)
      - (1, H, W) or (H, W, 1)  -> squeeze to (H, W)
      - (H, W, 3/4) RGB(A) channel-last
      - (3/4, H, W) RGB(A) channel-first
    """
    if img.ndim == 2:
        return img

    if img.ndim != 3:
        raise ValueError(f"Unsupported image ndim={img.ndim}. Expect 2D or 3D.")

    # squeeze singleton
    if img.shape[0] == 1:
        return img[0, ...]
    if img.shape[-1] == 1:
        return img[..., 0]

    # ---------- RGB(A) channel-last: (H, W, 3/4) ----------
    if img.shape[-1] in (3, 4):
        c = img[..., :3].astype(np.float32, copy=False)  # ignore alpha if present
        
        # If RGB channels are essentially identical (grayscale stored as RGB), just take mean
        if np.allclose(c[..., 0], c[..., 1], atol=1e-6) and np.allclose(c[..., 1], c[..., 2], atol=1e-6):
            return c.mean(axis=-1)
            
        # choose best channel by p99
        p99 = np.array([np.percentile(c[..., i], 99) for i in range(3)], dtype=float)
        i_best = int(np.argmax(p99))

        if rgb in ("r", "g", "b"):
            i_pick = {"r": 0, "g": 1, "b": 2}[rgb]
            # if chosen channel is too weak compared to best, fallback to best
            if np.isfinite(p99[i_pick]) and np.isfinite(p99[i_best]) and p99[i_pick] < 0.2 * p99[i_best]:
                return c[..., i_best]
            return c[..., i_pick]

        if rgb == "mean":
            return c.mean(axis=-1)
        if rgb == "max":
            return c.max(axis=-1)

        # rgb == "auto" or anything else -> best channel
        return c[..., i_best]

    # ---------- RGB(A) channel-first: (3/4, H, W) ----------
    if img.shape[0] in (3, 4) and img.shape[1] > 16 and img.shape[2] > 16:
        c = img[:3, ...].astype(np.float32, copy=False)
        if rgb in ("r", "g", "b"):
            return c[{"r": 0, "g": 1, "b": 2}[rgb], ...]
        if rgb == "mean":
            return c.mean(axis=0)
        if rgb == "max":
            return c.max(axis=0)
        p99 = [np.percentile(c[i, ...], 99) for i in range(3)]
        return c[int(np.argmax(p99)), ...]

    # ---------- treat as Z-stack: (Z, H, W) ----------
    if z_project == "max":
        return img.max(axis=0)
    if z_project == "mean":
        return img.mean(axis=0)

    raise ValueError("z_project must be 'max' or 'mean'")

def relabel_sequential(lbl: np.ndarray) -> np.ndarray:
    """Compact labels to 1..N."""
    uniq = np.unique(lbl)
    uniq = uniq[uniq != 0]
    out = np.zeros_like(lbl, dtype=np.int32)
    for new_id, old_id in enumerate(uniq, start=1):
        out[lbl == old_id] = new_id
    return out

def remove_small_labels(lbl: np.ndarray, min_size: int) -> np.ndarray:
    """
    Remove labeled objects smaller than min_size from an integer label image.
    Avoids skimage warning when only one label exists.
    """
    lbl = lbl.astype(np.int32, copy=False)
    if lbl.max() <= 0:
        return lbl
    # compute area per label id (exclude background 0)
    counts = np.bincount(lbl.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    keep[0] = True
    keep[1:] = counts[1:] >= int(min_size)
    out = lbl.copy()
    out[~keep[lbl]] = 0
    return relabel_sequential(out).astype(np.int32)

def save_nuclei_label_figure(
    out_png: str,
    nuclei_lbl: np.ndarray,
    title: str | None = None,
    show_legend: bool = True,
    legend_max_items: int = 30,
    save_pdf: bool = True,
) -> None:
    """
    Save a label image visualization:
      - discrete colormap
      - no colorbar
      - cell id text on nuclei (centroid)
      - optional legend mapping color -> cell id
      - Also saves a PDF version next to PNG by default.
    """
    lbl = nuclei_lbl.astype(np.int32)
    n = int(lbl.max())
    cmap = plt.get_cmap("nipy_spectral", n + 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(lbl, cmap=cmap, interpolation="nearest")
    ax.set_axis_off()

    props = regionprops(lbl)
    for rp in props:
        y, x = rp.centroid
        ax.text(
            x, y, str(rp.label),
            color="white", fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6),
        )

    if title is not None:
        ax.set_title(title)

    if show_legend and n > 0:
        n_show = min(n, legend_max_items)
        handles, labels_ = [], []
        for i in range(1, n_show + 1):
            handles.append(plt.Line2D(
                [0], [0], marker="s", linestyle="",
                markerfacecolor=cmap(i), markeredgecolor="none",
                markersize=8
            ))
            labels_.append(f"cell {i}")
        if n > n_show:
            handles.append(plt.Line2D([0], [0], linestyle=""))
            labels_.append(f"... +{n - n_show} more")

        ax.legend(handles, labels_, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, title="Nuclei")

    plt.tight_layout()

    # --- save PNG ---
    fig.savefig(out_png, dpi=200, bbox_inches="tight")

    # --- save PDF (same basename) ---
    if save_pdf:
        out_pdf = str(Path(out_png).with_suffix(".pdf"))
        fig.savefig(out_pdf, bbox_inches="tight")  # PDF is vector; dpi not needed
        
    plt.close(fig)

# -----------------------------
# Nucleus segmentation options
# -----------------------------
def _prep_for_cellpose_2d(img2d: np.ndarray) -> np.ndarray:
    """Force Cellpose-safe 2D float32, finite, normalized to ~0-1, C-contiguous."""
    x = img2d.astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # robust normalization: 1-99% to [0,1]
    p1, p99 = np.percentile(x, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or (p99 - p1) < 1e-6:
        # fallback: min-max
        mn, mx = float(np.min(x)), float(np.max(x))
        if (mx - mn) < 1e-6:
            return np.ascontiguousarray(np.zeros_like(x, dtype=np.float32))
        x = (x - mn) / (mx - mn)
    else:
        x = (x - p1) / (p99 - p1)

    x = np.clip(x, 0.0, 1.0)
    return np.ascontiguousarray(x)

def segment_nuclei_cellpose(
    dapi_2d: np.ndarray,
    diameter: float | None = None,
    *,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    tile: bool = True,
    tile_overlap: float = 0.1,
    normalize: bool = True,
    augment: bool = False,
) -> np.ndarray:
    """
    Cellpose nuclei segmentation compatible with v4+ and older versions.
    Key knobs to improve recall in dense regions:
      - cellprob_threshold: lower -> more masks (e.g., 0, -1, -2)
      - flow_threshold: lower -> more masks (e.g., 0.4, 0.3, 0.2)
      - tile/tile_overlap: helps large images / dense areas
      - diameter: fixed diameter often improves dense regions

    This function filters kwargs by model.eval() signature for cross-version robustness.
    """
    if not _HAS_CELLPOSE:
        raise RuntimeError("cellpose not available. Install: pip install cellpose")

    # harden input
    if dapi_2d.ndim != 2:
        dapi_2d = np.squeeze(dapi_2d)
        if dapi_2d.ndim != 2:
            raise ValueError(f"Cellpose expects 2D image, got shape {dapi_2d.shape}")
    img = _prep_for_cellpose_2d(dapi_2d)

    # build model (v4+)
    if hasattr(cellpose_models, "CellposeModel"):
        model = cellpose_models.CellposeModel(gpu=False)
    else:
        model = cellpose_models.Cellpose(gpu=False, model_type="nuclei")

    # IMPORTANT: in v4, passing `channels` triggers deprecation warnings; omit it.
    kwargs = dict(
        diameter=diameter,
        channel_axis=None,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        tile=tile,
        tile_overlap=tile_overlap,
        normalize=normalize,
        augment=augment,
    )
    kwargs = _filter_kwargs(model.eval, kwargs)

    try:
        out = model.eval(img, **kwargs)
        masks = out[0] if isinstance(out, (tuple, list)) else out
        if masks is None:
            return np.zeros(img.shape, dtype=np.int32)
        return masks.astype(np.int32)
    except Exception as e:
        # Let caller decide fallback route
        raise RuntimeError(f"Cellpose eval failed: {type(e).__name__}: {e}") from e

# -----------------------------
# NEW: StarDist preproc -- v4
# -----------------------------
def _prep_for_stardist_2d(img2d: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """
    Robust normalize to [0,1] + optional Gaussian smoothing.
    This stabilizes StarDist on DAPI with strong texture/inhomogeneity.
    """
    x = img2d.astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    p1, p99 = np.percentile(x, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or (p99 - p1) < 1e-6:
        mn, mx = float(np.min(x)), float(np.max(x))
        x = (x - mn) / (mx - mn + 1e-12)
    else:
        x = (x - p1) / (p99 - p1 + 1e-12)

    x = np.clip(x, 0.0, 1.0)
    if sigma is not None and sigma > 0:
        x = gaussian(x, sigma=float(sigma), preserve_range=True)
    return x
    

# ----------------------------- v4
# EXISTING: merge_oversegmented_by_dapi(...)
# Keep your implementation as-is.
# -----------------------------
def merge_oversegmented_by_dapi(
    lbl: np.ndarray,
    dapi_s: np.ndarray,
    bw: np.ndarray,
    contact_radius: int = 1,
    quantile: float = 0.65,
) -> np.ndarray:
    """
    Merge adjacent labels if the DAPI intensity along their shared boundary is high.
    Intended to fix over-segmentation while preserving true touching nuclei splits.
    """
    lbl = lbl.copy()
    ids = np.unique(lbl)
    ids = ids[ids != 0]
    if len(ids) < 2:
        return lbl

    vals = dapi_s[bw > 0]
    if vals.size == 0:
        return lbl
    thr = float(np.quantile(vals, quantile))

    se = disk(contact_radius)
    masks = {int(i): (lbl == i) for i in ids}
    parent = {int(i): int(i) for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in ids:
        i = int(i)
        mi = masks[i]
        di = dilation(mi, se)
        for j in ids:
            j = int(j)
            if j <= i:
                continue
            mj = masks[j]
            contact = di & mj
            if not contact.any():
                continue
            boundary_mean = float(dapi_s[contact].mean())
            if boundary_mean >= thr:
                union(i, j)

    mapping = {}
    new_id = 1
    out = np.zeros_like(lbl, dtype=np.int32)
    for i in ids:
        i = int(i)
        r = find(i)
        if r not in mapping:
            mapping[r] = new_id
            new_id += 1
        out[lbl == i] = mapping[r]
    return out

# -----------------------------
# NEW (optional but recommended): rescue missing nuclei in dense/large regions
# -----------------------------
def _stardist_rescue_dense_regions(
    labels: np.ndarray,
    dapi_norm: np.ndarray,
    *,
    max_region_area_px: int = 120000,
    min_region_area_px: int = 20000,
    smooth_sigma: float = 1.0,
    h_maxima_h: float = 6.0,
    min_area_px: int = 5000,
) -> np.ndarray:
    """
    When StarDist under-segments dense regions, a common pattern is:
      - a very large label that likely contains multiple nuclei.
    We detect overly large instances and split them with a distance-transform watershed.
    Conservative by default: only apply to large regions.

    Parameters:
      max_region_area_px: any instance area > this will be considered for splitting
      min_region_area_px: also consider splitting if area is big enough and shape indicates crowding
    """
    out = labels.copy().astype(np.int32)
    if out.max() < 1:
        return out

    props = regionprops(out)
    # decide which objects are suspiciously large
    big_ids = [rp.label for rp in props if rp.area >= max_region_area_px]
    if not big_ids:
        # fallback: if there are medium-large objects, still allow rescue
        big_ids = [rp.label for rp in props if rp.area >= min_region_area_px]

    if not big_ids:
        return out

    current_max = int(out.max())

    for lid in big_ids:
        mask = out == lid
        if mask.sum() < (min_area_px * 2):
            continue

        # use local DAPI to build a binary mask and DT
        # smooth helps make DT peaks more stable in dense DAPI
        dloc = dapi_norm * mask.astype(np.float32)
        if smooth_sigma and smooth_sigma > 0:
            dloc = gaussian(dloc, sigma=float(smooth_sigma), preserve_range=True)

        # build a conservative foreground within this label
        # keep pixels that are not extremely dim inside the label
        vals = dloc[mask]
        if vals.size == 0:
            continue
        thr = float(np.quantile(vals, 0.35))  # conservative
        bw = (dloc > thr) & mask

        bw = remove_small_holes(bw, area_threshold=max(1, min_area_px // 3))
        bw = remove_small_objects(bw, min_size=min_area_px)

        if bw.sum() < (min_area_px * 2):
            continue

        dist = distance_transform_edt(bw)
        # h-maxima markers
        peaks = (h_maxima(dist, h=float(h_maxima_h)) > 0) & bw
        coords = np.argwhere(peaks)
        if coords.shape[0] < 2:
            # cannot split
            continue

        markers = np.zeros_like(dist, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i

        split = watershed(-dist, markers=markers, mask=bw).astype(np.int32)

        # write back: remove old label, assign new ids
        out[mask] = 0
        for sid in np.unique(split):
            if sid == 0:
                continue
            current_max += 1
            out[split == sid] = current_max

    out = relabel_sequential(out)
    return out.astype(np.int32)

def _stardist_rescue_missing_foreground(
    labels: np.ndarray,
    dapi_norm: np.ndarray,
    *,
    fg_quantile: float = 0.80,
    smooth_sigma: float = 1.0,
    h_maxima_h: float = 4.0,
    min_area_px: int = 2500,
) -> np.ndarray:
    """
    Rescue nuclei that StarDist missed completely:
      - build a conservative DAPI foreground mask
      - find foreground pixels not covered by any label
      - split them by DT-watershed
      - append as new labels
    """
    out = labels.copy().astype(np.int32)

    x = dapi_norm.astype(np.float32, copy=False)
    if smooth_sigma and smooth_sigma > 0:
        x = gaussian(x, sigma=float(smooth_sigma), preserve_range=True)

    # conservative foreground within whole image
    thr = float(np.quantile(x[x > 0], fg_quantile)) if np.any(x > 0) else float(np.quantile(x, fg_quantile))
    fg = x > thr
    fg = remove_small_holes(fg, area_threshold=max(1, min_area_px // 2))
    fg = remove_small_objects(fg, min_size=min_area_px)

    missing = fg & (out == 0)
    if missing.sum() < min_area_px:
        return out

    comp = sklabel(missing)
    props = regionprops(comp)
    cur = int(out.max())

    for rp in props:
        if rp.area < min_area_px:
            continue
        m = (comp == rp.label)
        dist = distance_transform_edt(m)
        peaks = (h_maxima(dist, h=float(h_maxima_h)) > 0) & m
        coords = np.argwhere(peaks)
        if coords.shape[0] < 1:
            continue

        markers = np.zeros_like(dist, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i

        ws = watershed(-dist, markers=markers, mask=m).astype(np.int32)
        # append
        for sid in np.unique(ws):
            if sid == 0:
                continue
            if np.sum(ws == sid) < min_area_px:
                continue
            cur += 1
            out[ws == sid] = cur

    out = remove_small_objects(out, min_size=int(min_area_px)).astype(np.int32)
    out = relabel_sequential(out)
    #out = remove_small_labels(out, min_size=int(min_area_px)).astype(np.int32)
    return out

# -----------------------------
# REPLACE your segment_nuclei_stardist with this version
# -----------------------------
def segment_nuclei_stardist(
    dapi_2d: np.ndarray,
    *,
    stardist_prob_thresh: float | None = None,
    stardist_nms_thresh: float | None = 0.7,
    stardist_sigma: float = 1.5,
    min_area_px: int = 5000,
    stardist_merge_quantile: float = 0.75,
    # optional rescue
    stardist_enable_rescue: bool = True,
    rescue_max_region_area_px: int = 120000,
    rescue_h_maxima_h: float = 6.0,
) -> np.ndarray:
    """
    StarDist nuclei segmentation with:
      - robust preprocessing (normalize + smooth)
      - tunable prob_thresh / nms_thresh
      - post-merge to fix over-segmentation (merge_oversegmented_by_dapi)
      - optional dense-region rescue (split abnormally large labels)
      - NEW: rescue missed nuclei foreground (fill blank dense areas)
    """
    if not _HAS_STARDIST:
        raise RuntimeError("stardist not available. Install: pip install stardist csbdeep")

    # 1) preproc: normalize + optional smoothing for StarDist
    x = _prep_for_stardist_2d(dapi_2d, sigma=stardist_sigma)

    # 2) StarDist inference
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    kwargs = {}
    if stardist_prob_thresh is not None:
        kwargs["prob_thresh"] = float(stardist_prob_thresh)
    if stardist_nms_thresh is not None:
        kwargs["nms_thresh"] = float(stardist_nms_thresh)
        
    print(f"[StarDist] kwargs passed to predict_instances: {kwargs}")
    labels, _ = model.predict_instances(x, **kwargs)
    labels = labels.astype(np.int32)

    # 3) remove tiny fragments
    #labels = remove_small_objects(labels, min_size=int(min_area_px)).astype(np.int32)
    #labels = relabel_sequential(labels)
    labels = remove_small_labels(labels, min_size=int(min_area_px))

    # 4) merge obvious over-splits using DAPI boundary intensity
    bw = labels > 0
    labels = merge_oversegmented_by_dapi(
        lbl=labels,
        dapi_s=x,  # normalized/smoothed DAPI
        bw=bw,
        contact_radius=1,
        quantile=float(stardist_merge_quantile),
    )
    #labels = remove_small_objects(labels, min_size=int(min_area_px)).astype(np.int32)
    #labels = relabel_sequential(labels)
    labels = remove_small_labels(labels, min_size=int(min_area_px))

    # 5) optional rescue A: split huge under-segmented dense regions
    if stardist_enable_rescue:
        labels = _stardist_rescue_dense_regions(
            labels=labels,
            dapi_norm=x,
            max_region_area_px=int(rescue_max_region_area_px),
            smooth_sigma=1.0,
            h_maxima_h=float(rescue_h_maxima_h),
            min_area_px=int(min_area_px),
        )
        #labels = remove_small_objects(labels, min_size=int(min_area_px)).astype(np.int32)
        #labels = relabel_sequential(labels)
        labels = remove_small_labels(labels, min_size=int(min_area_px))

    # 6) NEW rescue B: fill "blank" dense foreground missed by StarDist
    #    (requires you added _stardist_rescue_missing_foreground(...) function)
    labels = _stardist_rescue_missing_foreground(
        labels=labels,
        dapi_norm=x,
        fg_quantile=0.80,
        smooth_sigma=1.0,
        h_maxima_h=4.0,
        min_area_px=max(1500, int(min_area_px * 0.5)),
    )
    #labels = remove_small_objects(labels, min_size=int(min_area_px)).astype(np.int32)
    #labels = relabel_sequential(labels)
    labels = remove_small_labels(labels, min_size=int(min_area_px))

    return labels.astype(np.int32)

#### v2
def segment_nuclei_watershed(
    dapi_2d: np.ndarray,
    min_area_px: int = 1500,
    smooth_sigma: float = 1.0,
    opening_radius: int = 1,
    closing_radius: int = 2,
    watershed_separation: bool = True,
    h_maxima_h: float = 5.0,
    merge_quantile: float = 0.65,
) -> np.ndarray:
    """
    Classical nucleus segmentation: Otsu -> morph -> distance watershed -> merge obvious over-splits.
    """
    d = img_as_float(dapi_2d)

    p1, p99 = np.percentile(d, [1, 99])
    d = np.clip((d - p1) / (p99 - p1 + 1e-12), 0, 1)

    d_s = gaussian(d, sigma=smooth_sigma, preserve_range=True)

    thr = threshold_otsu(d_s)
    bw = d_s > thr

    bw = binary_opening(bw, disk(opening_radius))
    bw = binary_closing(bw, disk(closing_radius))
    bw = remove_small_objects(bw, min_size=min_area_px)
    bw = remove_small_holes(bw, area_threshold=max(1, min_area_px // 2))

    if not watershed_separation:
        return sklabel(bw).astype(np.int32)

    dist = distance_transform_edt(bw)
    peaks = (h_maxima(dist, h=h_maxima_h) > 0) & bw
    coords = np.argwhere(peaks)

    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if markers.max() < 2:
        return sklabel(bw).astype(np.int32)

    lbl = watershed(-dist, markers=markers, mask=bw)
    lbl = merge_oversegmented_by_dapi(lbl, d_s, bw, contact_radius=1, quantile=merge_quantile)
    lbl = remove_small_objects(lbl, min_size=min_area_px)
    lbl = relabel_sequential(lbl)
    return lbl.astype(np.int32)


# -----------------------------
# PATCH segment_nuclei_dispatch signature + stardist branch
# -----------------------------
def segment_nuclei_dispatch(
    dapi_2d: np.ndarray,
    method: str,
    cellpose_diameter: float | None,
    min_area_px: int,
    smooth_sigma: float,
    h_maxima_h: float,
    merge_quantile: float,
    watershed_sep: bool,
    # cellpose knobs (kept, but you will not use if method=stardist)
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    tile: bool = True,
    tile_overlap: float = 0.1,
    # NEW: stardist knobs
    stardist_prob_thresh: float | None = None,
    stardist_nms_thresh: float = 0.7,
    stardist_sigma: float = 1.5,
    stardist_merge_quantile: float = 0.75,
    stardist_enable_rescue: bool = True,
    rescue_max_region_area_px: int = 120000,
    rescue_h_maxima_h: float = 6.0,
) -> np.ndarray:
    method = method.lower()

    if method == "cellpose":
        try:
            masks = segment_nuclei_cellpose(
                dapi_2d,
                diameter=cellpose_diameter,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                tile=tile,
                tile_overlap=tile_overlap,
            )
        except Exception as e:
            warnings.warn(f"Cellpose crashed ({e}); fallback to watershed.", RuntimeWarning)
            masks = np.zeros_like(np.squeeze(dapi_2d), dtype=np.int32)
        if masks is None or int(np.max(masks)) == 0:
            warnings.warn("Cellpose returned 0 nuclei; fallback to watershed.", RuntimeWarning)
            masks = segment_nuclei_watershed(
                dapi_2d,
                min_area_px=min_area_px,
                smooth_sigma=smooth_sigma,
                watershed_separation=watershed_sep,
                h_maxima_h=h_maxima_h,
                merge_quantile=merge_quantile,
            )
        return masks
        
    if method == "stardist":
        try:
            masks = segment_nuclei_stardist(
                dapi_2d,
                stardist_prob_thresh=stardist_prob_thresh,
                stardist_nms_thresh=stardist_nms_thresh,
                stardist_sigma=stardist_sigma,
                min_area_px=min_area_px,
                stardist_merge_quantile=stardist_merge_quantile,
                stardist_enable_rescue=stardist_enable_rescue,
                rescue_max_region_area_px=rescue_max_region_area_px,
                rescue_h_maxima_h=rescue_h_maxima_h,
            )
        except Exception as e:
            warnings.warn(f"StarDist crashed ({e}); fallback to watershed.", RuntimeWarning)
            masks = np.zeros_like(np.squeeze(dapi_2d), dtype=np.int32)
        if masks is None or int(np.max(masks)) == 0:
            warnings.warn("StarDist returned 0 nuclei; fallback to watershed.", RuntimeWarning)
            masks = segment_nuclei_watershed(
                dapi_2d,
                min_area_px=min_area_px,
                smooth_sigma=smooth_sigma,
                watershed_separation=watershed_sep,
                h_maxima_h=h_maxima_h,
                merge_quantile=merge_quantile,
            )
        return masks
        
    if method == "watershed":
        return segment_nuclei_watershed(
            dapi_2d,
            min_area_px=min_area_px,
            smooth_sigma=smooth_sigma,
            watershed_separation=watershed_sep,
            h_maxima_h=h_maxima_h,
            merge_quantile=merge_quantile,
        )
        
    raise ValueError("nuclei_method must be one of: cellpose, stardist, watershed")

# -----------------------------
# Robust z-score (v2 style)
# -----------------------------
def robust_zscore(image: np.ndarray, mask: np.ndarray, mad_floor: float = 0.5) -> np.ndarray:
    """
    Robust z-score using median and MAD on pixels within mask.
    z = (I - median) / (1.4826*MAD + eps)
    """
    img = image.astype(np.float32, copy=False)
    vals = img[mask > 0]
    if vals.size == 0:
        raise ValueError("robust_zscore: empty mask.")
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    mad = max(mad, mad_floor)  # 关键：避免MAD塌陷
    scale = 1.4826 * mad + 1e-6
    z = (img - med) / scale
    return z


def clip_for_vis(img: np.ndarray, mask: np.ndarray, q: float = 99.0) -> np.ndarray:
    if mask.sum() == 0:
        hi = np.percentile(img, q)
    else:
        hi = np.percentile(img[mask > 0], q)
    return np.clip(img, 0, hi)


# -----------------------------
# Spot detection (big-fish) + dense decomposition
# -----------------------------
def spot_intensity_peak(img: np.ndarray, rc: np.ndarray, halfwin: int = 1) -> np.ndarray:
    """Peak intensity in (2*halfwin+1)^2 around each spot."""
    H, W = img.shape
    out = np.zeros((rc.shape[0],), dtype=np.float32)
    for i, (r, c) in enumerate(rc):
        r0 = max(0, r - halfwin)
        r1 = min(H, r + halfwin + 1)
        c0 = max(0, c - halfwin)
        c1 = min(W, c + halfwin + 1)
        out[i] = float(np.max(img[r0:r1, c0:c1]))
    return out


def detect_spots_bigfish(
    img: np.ndarray,
    nucleus_mask: np.ndarray,
    *,
    ch: str = "unknown",          # NEW
    fish_smooth_sigma: float,
    fish_bg_sigma: float,
    z_threshold: float | None,
    use_auto_threshold: bool,
    auto_thr_factor: float,
    spot_radius_px: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Returns:
      rc: (N,2) int (row,col)
      thr_used: threshold used
      fish_z: robust z-scored image (computed on nucleus pixels)

    Improvements:
      - background subtraction (gaussian bg) to reduce false positives
      - auto_thr_factor to make auto-threshold stricter if needed
    """
    img_fine = gaussian(img.astype(np.float32, copy=False), sigma=fish_smooth_sigma, preserve_range=True)

    # --- background subtraction (key for suppressing false positives) ---
    if fish_bg_sigma is not None and fish_bg_sigma > 0:
        img_bg = gaussian(img.astype(np.float32, copy=False), sigma=fish_bg_sigma, preserve_range=True)
        img_f = img_fine - img_bg
        img_f = np.clip(img_f, 0, None)
    else:
        img_f = img_fine

    fish_z = robust_zscore(img_f, nucleus_mask.astype(np.uint8))
    z_pos = np.clip(fish_z, 0, None) * nucleus_mask.astype(np.float32)

    if not _HAS_BIGFISH:
        raise RuntimeError("big-fish not available. Install: pip install big-fish")

    voxel_size = (1.0, 1.0)
    spot_radius = (float(spot_radius_px), float(spot_radius_px))

    if use_auto_threshold:
        # 先让 big-fish 给一个阈值（thr_used）
        # ---- robust auto-threshold: big-fish may crash on degenerate images ----
        try:
            spots, thr_used = detect_spots(
                images=z_pos,
                threshold=None,
                return_threshold=True,
                voxel_size=voxel_size,
                spot_radius=spot_radius,
            )
        except Exception as e:
            # big-fish auto threshold sometimes fails (e.g., ValueError: False is not in list)
            thr_used = None
            spots = None
            print(f"[{ch}] big-fish auto-threshold crashed -> fallback. Reason: {type(e).__name__}: {e}")

        # big-fish may return thr_used=None on degenerate images; fallback to a robust threshold
        if thr_used is None or (isinstance(thr_used, float) and not np.isfinite(thr_used)):
            # fallback 1: use fixed z_threshold if provided (recommended)
            thr = float(z_threshold) if z_threshold is not None else 6.0
            print(
                f"[{ch}] auto-threshold failed (thr_used=None). "
                f"Fallback to fixed z_threshold={thr:.3f}."
            )
            spots = detect_spots(
                images=z_pos,
                threshold=float(thr),
                return_threshold=False,
                voxel_size=voxel_size,
                spot_radius=spot_radius,
            )
        else:
            thr_raw = float(thr_used)
            thr = thr_raw * float(auto_thr_factor)
            print(
                f"[{ch}] z_pos max={float(z_pos.max()):.3f}, "
                f"thr={float(thr):.3f}, n_mask_px={int(nucleus_mask.sum())}, "
                f"use_auto=1, auto_thr_factor={float(auto_thr_factor):.3f}"
            )
            # factor != 1 -> rerun with stricter/looser threshold
            if np.isfinite(thr) and abs(float(auto_thr_factor) - 1.0) > 1e-6:
                spots = detect_spots(
                    images=z_pos,
                    threshold=float(thr),
                    return_threshold=False,
                    voxel_size=voxel_size,
                    spot_radius=spot_radius,
                )
                
        # NEW: 立即诊断打印（auto 阈值已确定）
        print(
            f"[{ch}] z_pos max={float(z_pos.max()):.3f}, "
            f"thr={float(thr):.3f}, n_mask_px={int(nucleus_mask.sum())}, "
            f"use_auto=1, auto_thr_factor={float(auto_thr_factor):.3f}"
        )
        
        # 如果 factor != 1，用更严格阈值重跑一次
        # apply stricter factor by re-running if factor != 1
        if np.isfinite(thr) and abs(float(auto_thr_factor) - 1.0) > 1e-6:
            spots = detect_spots(
                images=z_pos,
                threshold=float(thr),
                return_threshold=False,
                voxel_size=voxel_size,
                spot_radius=spot_radius,
            )
            
    else:
        if z_threshold is None:
            raise ValueError("z_threshold is required when use_auto_threshold is False")
        thr = float(z_threshold)
    
        # NEW: 立即诊断打印（fixed 阈值已确定）
        print(
            f"[{ch}] z_pos max={float(z_pos.max()):.3f}, "
            f"thr={float(thr):.3f}, n_mask_px={int(nucleus_mask.sum())}, "
            f"use_auto=0"
        )
    
        spots = detect_spots(
            images=z_pos,
            threshold=thr,
            return_threshold=False,
            voxel_size=voxel_size,
            spot_radius=spot_radius,
        )

    if spots is None or len(spots) == 0:
        return np.zeros((0, 2), dtype=int), float(thr), fish_z

    rc = np.asarray(spots, dtype=float)
    if rc.ndim != 2 or rc.shape[1] < 2:
        return np.zeros((0, 2), dtype=int), float(thr), fish_z

    rc = np.rint(rc[:, :2]).astype(int)

    H, W = img.shape
    keep = (rc[:, 0] >= 0) & (rc[:, 0] < H) & (rc[:, 1] >= 0) & (rc[:, 1] < W)
    rc = rc[keep]
    if rc.size == 0:
        return np.zeros((0, 2), dtype=int), float(thr), fish_z

    keep2 = nucleus_mask[rc[:, 0], rc[:, 1]] > 0
    rc = rc[keep2]
    return rc, float(thr), fish_z

def dense_decompose_bigfish_or_fallback(
    img: np.ndarray,
    nucleus_mask: np.ndarray,
    rc: np.ndarray,
    fish_z: np.ndarray,
    *,
    spot_radius_px: float,
    dense_z: float,
    dense_min_area_px: int,
    max_extra_spots_per_component: int,
    extra_spot_min_dist: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
      rc_all: (N,2) (original + decomposed extra spots)
      is_dense: (N,) bool for extra spots from dense decomposition
    """
    if rc.size == 0:
        return rc, np.zeros((0,), dtype=bool)
        
    # NEW: spots太少直接不走big-fish decompose_dense
    if rc.shape[0] < 20:   # 经验阈值：可设 15~30，自行微调
        return rc, np.zeros((rc.shape[0],), dtype=bool)
        
    # ---- attempt big-fish dense decomposition ----
    if _HAS_BIGFISH and _HAS_BIGFISH_DENSE:
        try:
            z_pos = np.clip(fish_z, 0, None) * nucleus_mask.astype(np.float32)
            
            # 强制：spots 必须是 int64 且二维 (N,2)
            rc_int = np.asarray(rc[:, :2], dtype=np.int64, order="C")
            if rc_int.ndim != 2 or rc_int.shape[1] != 2:
                raise ValueError(f"spots array invalid shape: {rc_int.shape}")
                
            spots_new, dense_regions, ref_spot = decompose_dense(
                image=z_pos,
                #spots=rc,
                spots=rc_int,
                voxel_size=(1.0, 1.0),
                spot_radius=(float(spot_radius_px), float(spot_radius_px)),
            )
            rc_new = np.rint(np.asarray(spots_new)[:, :2]).astype(int)
            orig_set = set(map(tuple, rc.tolist()))
            is_dense = np.array([tuple(p) not in orig_set for p in rc_new.tolist()], dtype=bool)
            return rc_new, is_dense
        except BaseException as e:
            # NEW: 强制降级，不让big-fish异常中断流程
            print(f"[WARN] big-fish decompose_dense failed -> fallback. Reason: {type(e).__name__}: {e}")
            # fall through to heuristic fallback
            pass

    # ---- fallback heuristic decomposition ----
    H, W = img.shape
    dense_mask = (fish_z > dense_z) & (nucleus_mask > 0)
    comp = sklabel(dense_mask)
    props = regionprops(comp)

    orig_set = set(map(tuple, rc.tolist()))
    extra_pts = []

    sup = np.zeros((H, W), dtype=bool)
    for r, c in rc:
        r0 = max(0, r - extra_spot_min_dist)
        r1 = min(H, r + extra_spot_min_dist + 1)
        c0 = max(0, c - extra_spot_min_dist)
        c1 = min(W, c + extra_spot_min_dist + 1)
        sup[r0:r1, c0:c1] = True

    score = np.clip(fish_z, 0, None)

    for rp in props:
        if rp.area < dense_min_area_px:
            continue
        coords = rp.coords
        m = np.zeros((H, W), dtype=bool)
        m[coords[:, 0], coords[:, 1]] = True

        m2 = m & (~sup)
        if m2.sum() == 0:
            continue

        peaks = peak_local_max(
            score,
            labels=m2.astype(np.uint8),
            min_distance=max(1, extra_spot_min_dist),
            threshold_abs=float(dense_z),
            exclude_border=False
        )
        if peaks is None or len(peaks) == 0:
            continue

        zvals = score[peaks[:, 0], peaks[:, 1]]
        order = np.argsort(zvals)[::-1]
        peaks = peaks[order][:max_extra_spots_per_component]

        for p in peaks:
            t = (int(p[0]), int(p[1]))
            if t not in orig_set:
                extra_pts.append(t)
                orig_set.add(t)

    if len(extra_pts) == 0:
        return rc, np.zeros((rc.shape[0],), dtype=bool)

    rc_extra = np.array(extra_pts, dtype=int)
    rc_all = np.vstack([rc, rc_extra])
    is_dense = np.zeros((rc_all.shape[0],), dtype=bool)
    is_dense[rc.shape[0]:] = True
    return rc_all, is_dense


def assign_spots_to_nuclei(labels: np.ndarray, rc: np.ndarray) -> np.ndarray:
    if rc.size == 0:
        return np.zeros((0,), dtype=np.int32)
    rr = np.clip(rc[:, 0].astype(int), 0, labels.shape[0] - 1)
    cc = np.clip(rc[:, 1].astype(int), 0, labels.shape[1] - 1)
    return labels[rr, cc].astype(np.int32)


# -----------------------------
# Colocalization metrics
# -----------------------------
def coloc_pairs_cross_channel(
    rc_a: np.ndarray,
    rc_b: np.ndarray,
    dist_thresh: float,
    one_to_one: bool = True
) -> dict:
    """
    Cross-channel colocalization.
    Outputs:
      n_a, n_b
      n_a_with_b, n_b_with_a
      n_pairs (unique pairs if one_to_one else all A->nearest within threshold)
      frac_a_with_b, frac_b_with_a
      mean_nn_dist_a_to_b (nan if no matches)
    """
    n_a = int(rc_a.shape[0])
    n_b = int(rc_b.shape[0])
    if n_a == 0 or n_b == 0:
        return dict(
            n_a=n_a, n_b=n_b,
            n_a_with_b=0, n_b_with_a=0, n_pairs=0,
            frac_a_with_b=np.nan if n_a == 0 else 0.0,
            frac_b_with_a=np.nan if n_b == 0 else 0.0,
            mean_nn_dist_a_to_b=np.nan
        )

    tree_b = cKDTree(rc_b.astype(np.float32))
    dists, idx = tree_b.query(rc_a.astype(np.float32), k=1)
    within = dists <= dist_thresh

    if not one_to_one:
        n_a_with_b = int(within.sum())
        # approximate b_with_a by whether any a maps to each b within
        b_hit = np.zeros((n_b,), dtype=bool)
        b_hit[idx[within]] = True
        n_b_with_a = int(b_hit.sum())
        return dict(
            n_a=n_a, n_b=n_b,
            n_a_with_b=n_a_with_b, n_b_with_a=n_b_with_a,
            n_pairs=n_a_with_b,
            frac_a_with_b=n_a_with_b / n_a if n_a else np.nan,
            frac_b_with_a=n_b_with_a / n_b if n_b else np.nan,
            mean_nn_dist_a_to_b=float(np.mean(dists[within])) if n_a_with_b else np.nan
        )

    # one-to-one greedy matching by distance
    # build candidate list (a_idx, b_idx, dist)
    cand = [(ai, int(bi), float(di)) for ai, (bi, di, ok) in enumerate(zip(idx, dists, within)) if ok]
    cand.sort(key=lambda x: x[2])  # smallest distance first
    used_a = set()
    used_b = set()
    pairs = []
    for ai, bi, di in cand:
        if ai in used_a or bi in used_b:
            continue
        used_a.add(ai)
        used_b.add(bi)
        pairs.append((ai, bi, di))

    n_pairs = len(pairs)
    n_a_with_b = n_pairs
    n_b_with_a = n_pairs
    mean_nn = float(np.mean([p[2] for p in pairs])) if n_pairs else np.nan
    return dict(
        n_a=n_a, n_b=n_b,
        n_a_with_b=n_a_with_b, n_b_with_a=n_b_with_a,
        n_pairs=n_pairs,
        frac_a_with_b=n_a_with_b / n_a if n_a else np.nan,
        frac_b_with_a=n_b_with_a / n_b if n_b else np.nan,
        mean_nn_dist_a_to_b=mean_nn
    )


def coloc_within_channel(
    rc: np.ndarray,
    dist_thresh: float
) -> dict:
    """
    Within-channel proximity metric (used to satisfy "red-red" and "green-green" requests).
    Definitions:
      - n: number of spots
      - n_with_neighbor: count of spots that have at least one other spot within dist_thresh
      - frac_with_neighbor
      - n_pairs: number of unique unordered pairs within dist_thresh
      - mean_nn_dist: mean nearest-neighbor distance (excluding self); nan if n<2
    """
    n = int(rc.shape[0])
    if n < 2:
        return dict(
            n=n,
            n_with_neighbor=0,
            frac_with_neighbor=np.nan if n == 0 else 0.0,
            n_pairs=0,
            mean_nn_dist=np.nan
        )

    tree = cKDTree(rc.astype(np.float32))
    # nearest neighbor distances excluding self: query k=2 (self + nearest)
    dists2, idx2 = tree.query(rc.astype(np.float32), k=2)
    nn = dists2[:, 1]
    n_with_neighbor = int((nn <= dist_thresh).sum())
    frac = n_with_neighbor / n

    # count unique pairs within dist_thresh
    pairs = tree.query_pairs(r=dist_thresh)
    n_pairs = int(len(pairs))

    return dict(
        n=n,
        n_with_neighbor=n_with_neighbor,
        frac_with_neighbor=frac,
        n_pairs=n_pairs,
        mean_nn_dist=float(np.mean(nn)) if nn.size else np.nan
    )


# -----------------------------
# QC overlay helpers
# -----------------------------
def save_qc_overlay(
    out_path: str,
    base_img: np.ndarray,
    nucleus_mask: np.ndarray,
    rc_by_channel: dict,
    title: str,
) -> None:
    bd = find_boundaries(nucleus_mask.astype(np.int32), mode="outer")

    plt.figure(figsize=(6, 6))
    plt.imshow(base_img, cmap="gray")
    plt.imshow(np.ma.masked_where(~bd, bd), cmap="autumn", alpha=0.9)

    # fixed marker styles per channel
    styles = {
        "red": dict(edgecolors="red"),
        "green": dict(edgecolors="lime"),
        "cyan": dict(edgecolors="cyan"),
    }
    for ch, rc in rc_by_channel.items():
        if rc is None or rc.size == 0:
            continue
        st = styles.get(ch, dict(edgecolors="yellow"))
        plt.scatter(rc[:, 1], rc[:, 0], s=14, marker="o", facecolors="none", linewidths=0.9, **st)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -----------------------------
# Napari QC (interactive)
# -----------------------------
def launch_napari_qc(
    dapi: np.ndarray,
    labels: np.ndarray,
    channels: dict,
    spots: dict,
    coloc_pairs: dict,
) -> None:
    if not _HAS_NAPARI:
        raise RuntimeError("napari not available. Install: pip install napari[all]")

    v = napari.Viewer()
    v.add_image(dapi, name="DAPI", blending="additive")
    v.add_labels(labels, name="nuclei_labels")

    for ch, img in channels.items():
        if img is None:
            continue
        v.add_image(img, name=ch, blending="additive")

    for ch, rc in spots.items():
        if rc is None or rc.size == 0:
            continue
        v.add_points(rc[:, [1, 0]], name=f"spots_{ch}", size=4)  # napari expects (x,y)

    # Optional: add lines for coloc pairs (cross-channel)
    # coloc_pairs expected: { "red_green": [(r0,c0,r1,c1), ...], ... }
    for key, pairs in coloc_pairs.items():
        if pairs is None or len(pairs) == 0:
            continue
        # napari shapes line: ((y1,x1),(y2,x2)) but array format expects (N,2,2) in (y,x)
        arr = np.array([[[p[0], p[1]], [p[2], p[3]]] for p in pairs], dtype=float)
        v.add_shapes(arr, shape_type="line", name=f"coloc_{key}")

    napari.run()


# -----------------------------
# Core per-sample analysis
# -----------------------------
def analyze_sample(
    *,
    sample_id: str,
    dapi_path: str,
    red_path: str,
    green_path: str,
    cyan_path: str | None,
    merged_path: str | None,
    outdir: str,
    group_label: str | None,
    z_project: str,
    tif_first_page: bool,
    
    # nuclei params
    nuclei_method: str,
    cellpose_diameter: float | None,
    min_area_px: int,
    smooth_sigma: float,
    h_maxima_h: float,
    merge_quantile: float,
    watershed_sep: bool,
    
    # stardist params
    stardist_prob_thresh: float | None,
    stardist_nms_thresh: float,
    stardist_sigma: float,
    stardist_merge_quantile: float,
    stardist_enable_rescue: bool,
    rescue_max_region_area_px: int,
    rescue_h_maxima_h: float,
    
    # spot params
    fish_smooth_sigma: float,
    fish_bg_sigma: float,          # NEW
    use_auto_threshold: bool,
    auto_thr_factor: float,        # NEW
    z_threshold: float,
    spot_radius_px: float,
    # centromere constraints
    centromere_max_spots: int,     # NEW (default 4 from CLI)
    centromere_disable_dense: bool,# NEW (default True)

    cellprob_threshold: float,
    flow_threshold: float,
    no_tile: bool,
    tile_overlap: float,

    # dense params
    enable_dense: bool,
    dense_z: float,
    dense_min_area_px: int,
    max_extra_spots_per_component: int,
    extra_spot_min_dist: int,
    
    # coloc params
    coloc_dist_px: float,
    coloc_one_to_one: bool,
    # QC
    qc_nuclei: int,
    qc_seed: int,
    napari_qc: bool,
    # merged visualization options
    merged_rgb: str,

    # RGB
    dapi_rgb: str,
    red_rgb: str,
    green_rgb: str,
    cyan_rgb: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      per_cell_df, per_image_df
    """
    out = Path(outdir) / sample_id
    out.mkdir(parents=True, exist_ok=True)

    # ---- read + ensure 2D ----
    raw_dapi  = read_tiff_any(dapi_path,  tif_first_page=tif_first_page)
    raw_red   = read_tiff_any(red_path,   tif_first_page=tif_first_page)
    raw_green = read_tiff_any(green_path, tif_first_page=tif_first_page)
    raw_cyan  = read_tiff_any(cyan_path,  tif_first_page=tif_first_page) if cyan_path else None

    dapi  = ensure_2d(raw_dapi,  z_project=z_project, rgb=dapi_rgb)
    red   = ensure_2d(raw_red,   z_project=z_project, rgb=red_rgb)
    green = ensure_2d(raw_green, z_project=z_project, rgb=green_rgb)
    cyan  = ensure_2d(raw_cyan,  z_project=z_project, rgb=cyan_rgb) if raw_cyan is not None else None

    print("RAW DAPI shape:", getattr(raw_dapi, "shape", None))
    print("2D  DAPI shape:", dapi.shape, "dtype:", dapi.dtype, "min/max:", float(dapi.min()), float(dapi.max()))
    if dapi.shape != red.shape or dapi.shape != green.shape or (cyan is not None and dapi.shape != cyan.shape):
        raise ValueError("All single-channel TIFFs must be aligned and same shape after z-projection.")

    merged = None
    if merged_path:
        merged = ensure_2d(read_tiff_any(merged_path, tif_first_page=tif_first_page), z_project=z_project, rgb=merged_rgb)

    # ---- nuclei segmentation ----
    nuclei_lbl = segment_nuclei_dispatch(
        dapi,
        method=nuclei_method,
        cellpose_diameter=cellpose_diameter,
        min_area_px=min_area_px,
        smooth_sigma=smooth_sigma,
        h_maxima_h=h_maxima_h,
        merge_quantile=merge_quantile,
        watershed_sep=watershed_sep,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        tile=(not no_tile),
        tile_overlap=tile_overlap,
        # NEW: StarDist --v4
        stardist_prob_thresh=stardist_prob_thresh,
        stardist_nms_thresh=stardist_nms_thresh,
        stardist_sigma=stardist_sigma,
        stardist_merge_quantile=stardist_merge_quantile,
        stardist_enable_rescue=stardist_enable_rescue,
        rescue_max_region_area_px=rescue_max_region_area_px,
        rescue_h_maxima_h=rescue_h_maxima_h,
    )

    n_nuclei = int(nuclei_lbl.max())
    if n_nuclei == 0:
        raise RuntimeError("No nuclei detected. Check DAPI quality / segmentation parameters.")

    tiff.imwrite(str(out / "nuclei_labels.tif"), nuclei_lbl.astype(np.uint16))
    save_nuclei_label_figure(
        out_png=str(out / "nuclei_labels_pseudocolor.png"),
        nuclei_lbl=nuclei_lbl,
        title=f"{sample_id} nuclei labels (n={n_nuclei})",
        show_legend=True,
        legend_max_items=30,
    )

    # ---- mask of all nuclei pixels for z-score ----
    all_nuc_mask = nuclei_lbl > 0

    # ---- per-channel spot detection (+ dense decomposition optional) ----
    channels = {"red": red, "green": green, "cyan": cyan}
    present_channels = [k for k, v in channels.items() if v is not None]
    
    # centromere channel rule:
    # 3-color => cyan is centromere; 2-color => green is centromere
    centromere_channel = "cyan" if (cyan is not None) else "green"
    
    spot_tables = {}
    rc_final = {}
    rc_dense_flag = {}
    thresholds = {}
    fish_z_by_ch = {}

    for ch in present_channels:
        img = channels[ch]
        rc, thr, fish_z = detect_spots_bigfish(
            img,
            all_nuc_mask,
            ch=ch,  # NEW
            fish_smooth_sigma=fish_smooth_sigma,
            fish_bg_sigma=fish_bg_sigma,
            z_threshold=z_threshold,
            use_auto_threshold=use_auto_threshold,
            auto_thr_factor=auto_thr_factor,
            spot_radius_px=spot_radius_px,
        )
        thresholds[ch] = thr
        fish_z_by_ch[ch] = fish_z
        
        do_dense = bool(enable_dense)
        if centromere_disable_dense and (ch == centromere_channel):
            do_dense = False
            
        if do_dense:
            rc2, is_dense = dense_decompose_bigfish_or_fallback(
                img=img,
                nucleus_mask=all_nuc_mask,
                rc=rc,
                fish_z=fish_z,
                spot_radius_px=spot_radius_px,
                dense_z=dense_z,
                dense_min_area_px=dense_min_area_px,
                max_extra_spots_per_component=max_extra_spots_per_component,
                extra_spot_min_dist=extra_spot_min_dist,
            )
        else:
            rc2, is_dense = rc, np.zeros((rc.shape[0],), dtype=bool)
            
        rc_final[ch] = rc2
        rc_dense_flag[ch] = is_dense

        # debug z image
        z_vis = clip_for_vis(np.clip(fish_z, 0, None), all_nuc_mask, q=99.0)
        plt.imsave(str(out / f"fish_{ch}_z_clipped.png"), z_vis, cmap="gray")

        # spot-level table
        nuc_id = assign_spots_to_nuclei(nuclei_lbl, rc2)
        peak = spot_intensity_peak(img.astype(np.float32, copy=False), rc2, halfwin=1) if rc2.size else np.zeros((0,), dtype=np.float32)

        df = pd.DataFrame({
            "group": group_label if group_label is not None else "",
            "sample_id": sample_id,
            "channel": ch,
            "row": rc2[:, 0] if rc2.size else np.array([], dtype=int),
            "col": rc2[:, 1] if rc2.size else np.array([], dtype=int),
            "nucleus_id": nuc_id,
            "peak_intensity_raw": peak,
            "is_dense_extra": is_dense.astype(int) if rc2.size else np.array([], dtype=int),
            "thr_used": thr,
        })
        # keep only spots inside nuclei (nucleus_id > 0)
        df = df[df["nucleus_id"] > 0].copy()

        # --- centromere strict cap (0–4 by default) ---
        if ch == centromere_channel and centromere_max_spots is not None and centromere_max_spots > 0:
            df = cap_spots_per_nucleus(df, max_spots=int(centromere_max_spots), rank_col="peak_intensity_raw")

        spot_tables[ch] = df
        df.to_csv(str(out / f"spots_{ch}.csv"), index=False)

    
    # ---- per-cell summary ----
    props = regionprops(nuclei_lbl)
    nuc_ids = [int(rp.label) for rp in props]

    # counts per nucleus per channel
    counts = {}
    for ch in present_channels:
        df = spot_tables[ch]
        if df.shape[0] == 0:
            counts[ch] = pd.Series(0, index=nuc_ids, dtype=int)
        else:
            counts[ch] = df.groupby("nucleus_id").size().reindex(nuc_ids, fill_value=0).astype(int)

    # nucleus morphology
    nuc_area = pd.Series({int(rp.label): float(rp.area) for rp in props})
    nuc_cy = pd.Series({int(rp.label): float(rp.centroid[0]) for rp in props})
    nuc_cx = pd.Series({int(rp.label): float(rp.centroid[1]) for rp in props})

    per_cell = pd.DataFrame({
        "group": group_label if group_label is not None else "",
        "sample_id": sample_id,
        "cell_id": nuc_ids,
        "nucleus_area_px": [nuc_area[i] for i in nuc_ids],
        "centroid_row": [nuc_cy[i] for i in nuc_ids],
        "centroid_col": [nuc_cx[i] for i in nuc_ids],
    })

    for ch in present_channels:
        per_cell[f"n_spots_{ch}"] = [int(counts[ch].loc[i]) for i in nuc_ids]

    # ---- colocalization per nucleus ----
    def rc_in_nucleus(ch: str, nid: int) -> np.ndarray:
        df = spot_tables[ch]
        if df.shape[0] == 0:
            return np.zeros((0, 2), dtype=int)
        sub = df[df["nucleus_id"] == nid]
        if sub.shape[0] == 0:
            return np.zeros((0, 2), dtype=int)
        return sub[["row", "col"]].to_numpy(dtype=int)

    coloc_pairs_for_napari = {}

    # Pair lists:
    # 3-color: RG, RC, GC
    # 2-color: RG, RR, GG
    if cyan is not None:
        pair_specs = [("red", "green"), ("red", "cyan"), ("green", "cyan")]
    else:
        pair_specs = [("red", "green"), ("red", "red"), ("green", "green")]

    # Prepare columns
    for a, b in pair_specs:
        key = f"{a}_{b}"
        if a != b:
            per_cell[f"coloc_{key}_n_pairs"] = 0
            per_cell[f"coloc_{key}_frac_{a}_with_{b}"] = np.nan
            per_cell[f"coloc_{key}_frac_{b}_with_{a}"] = np.nan
            per_cell[f"coloc_{key}_mean_nn_dist_px"] = np.nan
        else:
            per_cell[f"within_{a}_n_with_neighbor"] = 0
            per_cell[f"within_{a}_frac_with_neighbor"] = np.nan
            per_cell[f"within_{a}_n_pairs"] = 0
            per_cell[f"within_{a}_mean_nn_dist_px"] = np.nan

    for nid in nuc_ids:
        # cross-channel
        for a, b in pair_specs:
            if a != b:
                ra = rc_in_nucleus(a, nid)
                rb = rc_in_nucleus(b, nid)
                m = coloc_pairs_cross_channel(ra, rb, dist_thresh=coloc_dist_px, one_to_one=coloc_one_to_one)
                key = f"{a}_{b}"
                per_cell.loc[per_cell["cell_id"] == nid, f"coloc_{key}_n_pairs"] = int(m["n_pairs"])
                per_cell.loc[per_cell["cell_id"] == nid, f"coloc_{key}_frac_{a}_with_{b}"] = float(m["frac_a_with_b"]) if np.isfinite(m["frac_a_with_b"]) else np.nan
                per_cell.loc[per_cell["cell_id"] == nid, f"coloc_{key}_frac_{b}_with_{a}"] = float(m["frac_b_with_a"]) if np.isfinite(m["frac_b_with_a"]) else np.nan
                per_cell.loc[per_cell["cell_id"] == nid, f"coloc_{key}_mean_nn_dist_px"] = float(m["mean_nn_dist_a_to_b"]) if np.isfinite(m["mean_nn_dist_a_to_b"]) else np.nan
            else:
                r = rc_in_nucleus(a, nid)
                m = coloc_within_channel(r, dist_thresh=coloc_dist_px)
                per_cell.loc[per_cell["cell_id"] == nid, f"within_{a}_n_with_neighbor"] = int(m["n_with_neighbor"])
                per_cell.loc[per_cell["cell_id"] == nid, f"within_{a}_frac_with_neighbor"] = float(m["frac_with_neighbor"]) if np.isfinite(m["frac_with_neighbor"]) else np.nan
                per_cell.loc[per_cell["cell_id"] == nid, f"within_{a}_n_pairs"] = int(m["n_pairs"])
                per_cell.loc[per_cell["cell_id"] == nid, f"within_{a}_mean_nn_dist_px"] = float(m["mean_nn_dist"]) if np.isfinite(m["mean_nn_dist"]) else np.nan

    per_cell.to_csv(str(out / "per_cell_summary.csv"), index=False)

    # ---- per-image summary ----
    per_image = {
        "group": group_label if group_label is not None else "",
        "sample_id": sample_id,
        "n_nuclei": n_nuclei,
        "nuclei_method": nuclei_method,
        "fish_smooth_sigma": fish_smooth_sigma,
        "use_auto_threshold": int(use_auto_threshold),
        "z_threshold": z_threshold,
        "spot_radius_px": spot_radius_px,
        "enable_dense": int(enable_dense),
        "dense_z": dense_z,
        "coloc_dist_px": coloc_dist_px,
        "coloc_one_to_one": int(coloc_one_to_one),
    }
    for ch in present_channels:
        per_image[f"thr_used_{ch}"] = float(thresholds[ch])
        per_image[f"n_spots_{ch}_total_in_nuclei"] = int(spot_tables[ch].shape[0])
        per_image[f"n_spots_{ch}_dense_extra"] = int(spot_tables[ch]["is_dense_extra"].sum()) if spot_tables[ch].shape[0] else 0

    per_image_df = pd.DataFrame([per_image])
    per_image_df.to_csv(str(out / "per_image_summary.csv"), index=False)

    ## ---- QC overlays: export random subset of nuclei ----
    #rng = np.random.default_rng(qc_seed)
    #qc_ids = set(rng.choice(nuc_ids, size=min(qc_nuclei, len(nuc_ids)), replace=False)) if qc_nuclei > 0 else set()
    
    # ---- QC overlays: export TOP-K nuclei by total spot counts (NOT random) ----
    # Count spots per nucleus across all present channels (after filtering nucleus_id>0 and after cap)
    spot_count_total = pd.Series(0, index=nuc_ids, dtype=int)
    for ch in present_channels:
        df = spot_tables[ch]
        if df is None or df.empty:
            continue
        spot_count_total = spot_count_total.add(
            df.groupby("nucleus_id").size().reindex(nuc_ids, fill_value=0).astype(int),
            fill_value=0
        ).astype(int)

    # Tie-breaker: larger nucleus area first, then smaller nucleus id (stable)
    area_series = pd.Series({int(rp.label): int(rp.area) for rp in props}, dtype=int).reindex(nuc_ids).fillna(0).astype(int)

    if qc_nuclei > 0:
        rank_df = pd.DataFrame({
            "cell_id": nuc_ids,
            "spot_total": [int(spot_count_total.loc[i]) for i in nuc_ids],
            "area_px": [int(area_series.loc[i]) for i in nuc_ids],
        })
        rank_df = rank_df.sort_values(
            by=["spot_total", "area_px", "cell_id"],
            ascending=[False, False, True],
            kind="mergesort"
        )
        qc_ids = set(rank_df["cell_id"].head(min(qc_nuclei, len(nuc_ids))).astype(int).tolist())
    else:
        qc_ids = set()
    
    # choose base image: merged if provided else DAPI
    base = merged if merged is not None else dapi

    for nid in qc_ids:
        W = nuclei_lbl == nid
        rc_by = {}
        for ch in present_channels:
            df = spot_tables[ch]
            sub = df[df["nucleus_id"] == nid]
            rc_by[ch] = sub[["row", "col"]].to_numpy(dtype=int) if sub.shape[0] else np.zeros((0, 2), dtype=int)
        save_qc_overlay(
            out_path=str(out / f"QC_cell_{nid:04d}.png"),
            base_img=base,
            nucleus_mask=W,
            rc_by_channel=rc_by,
            title=f"{sample_id} cell {nid}",
        )

    # ---- Napari QC (optional) ----
    if napari_qc:
        # build simplistic colocalization line pairs for visualization for the whole image
        # (not per nucleus). This is for QC only, not metrics.
        # only draw cross-channel pairs (not within-channel).
        coloc_pairs_for_napari = {}
        cross_pairs = [p for p in pair_specs if p[0] != p[1]]
        for a, b in cross_pairs:
            da = spot_tables[a][["row", "col"]].to_numpy(dtype=float) if spot_tables[a].shape[0] else np.zeros((0, 2), dtype=float)
            db = spot_tables[b][["row", "col"]].to_numpy(dtype=float) if spot_tables[b].shape[0] else np.zeros((0, 2), dtype=float)
            if da.shape[0] == 0 or db.shape[0] == 0:
                coloc_pairs_for_napari[f"{a}_{b}"] = []
                continue
            tree = cKDTree(db)
            dists, idx = tree.query(da, k=1)
            ok = dists <= coloc_dist_px
            pairs = []
            for i in np.where(ok)[0]:
                r0, c0 = da[i]
                r1, c1 = db[idx[i]]
                pairs.append((float(r0), float(c0), float(r1), float(c1)))
            coloc_pairs_for_napari[f"{a}_{b}"] = pairs

        launch_napari_qc(
            dapi=dapi,
            labels=nuclei_lbl,
            channels={"red": red, "green": green, "cyan": cyan},
            spots={ch: spot_tables[ch][["row", "col"]].to_numpy(dtype=int) if spot_tables[ch].shape[0] else np.zeros((0, 2), dtype=int)
                  for ch in present_channels},
            coloc_pairs=coloc_pairs_for_napari,
        )

    return per_cell, per_image_df


# -----------------------------
# Batch mode: manifest CSV
# -----------------------------
def load_manifest(manifest_csv: str) -> pd.DataFrame:
    """
    Expected columns:
      sample_id, dapi, red, green,
      cyan (optional), merged (optional), group (optional)

    Paths can be absolute or relative to the manifest directory.
    """
    df = pd.read_csv(manifest_csv)
    required = {"sample_id", "dapi", "red", "green"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")

    base = Path(manifest_csv).resolve().parent

    def _resolve(p):
        if pd.isna(p) or str(p).strip() == "":
            return None
        p = Path(str(p))
        if not p.is_absolute():
            p = base / p
        return str(p)

    for col in ["dapi", "red", "green", "cyan", "merged"]:
        if col in df.columns:
            df[col] = df[col].apply(_resolve)
        else:
            df[col] = None

    if "group" not in df.columns:
        df["group"] = ""

    return df


# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    ap = argparse.ArgumentParser(
        description="DNA-FISH quantification pipeline v4 (nuclei segmentation + big-fish spots + dense decomposition + colocalization)."
    )

    # Single-sample mode
    ap.add_argument("--sample_id", default="sample", help="Sample ID for single-image mode.")
    ap.add_argument("--dapi", default=None, help="DAPI single-channel tif/tiff (2D or 3D)/jpg/jpeg/bmp/png.")
    ap.add_argument("--red", default=None, help="Red channel single-channel tif/tiff (aligned with DAPI)/jpg/jpeg/bmp/png.")
    ap.add_argument("--green", default=None, help="Green channel single-channel tif/tiff (aligned with DAPI)/jpg/jpeg/bmp/png.")
    ap.add_argument("--cyan", default=None, help="Cyan channel single-channel tif/tiff (optional; aligned with DAPI)/jpg/jpeg/bmp/png.")
    ap.add_argument("--merged", default=None, help="Merged composite tif/tiff (optional; for visualization/QC)/jpg/jpeg/bmp/png.")
    ap.add_argument("--group", default=None, help="Optional group label (e.g., Control/R/NR).")

    # Batch mode
    ap.add_argument("--manifest", default=None, help="CSV manifest for batch processing (recommended).")

    # Output
    ap.add_argument("--outdir", required=True, help="Output directory (per-sample subfolders will be created).")

    # Projection / merged RGB
    ap.add_argument("--z_project", default="max", choices=["max", "mean"], help="If 3D, z-projection method.")
    ap.add_argument("--tif_first_page", action="store_true",
                    help="If set, read first TIFF page only (avoids some OME discontiguous warnings). "
                         "Use only if you know the first page is the correct plane.")

    ap.add_argument("--merged_rgb", default="auto", choices=["auto", "r", "g", "b", "mean"],
                    help="If merged is RGB, which channel to visualize as base in QC.")

    # Nuclei segmentation
    ap.add_argument("--nuclei_method", default="cellpose", choices=["cellpose", "stardist", "watershed"],
                    help="Nuclei segmentation method.")
    ap.add_argument("--cellpose_diameter", type=float, default=None, help="Cellpose diameter (px). None=auto.")
    ap.add_argument("--min_area_px", type=int, default=5000, help="Minimum nucleus area (px).")
    ap.add_argument("--smooth_sigma", type=float, default=2.0, help="Gaussian sigma for DAPI smoothing (watershed mode).")
    ap.add_argument("--h_maxima_h", type=float, default=5.0, help="h-maxima suppression on distance map (watershed mode).")
    ap.add_argument("--merge_quantile", type=float, default=0.65, help="Merge quantile for overseg fragments (watershed mode).")
    ap.add_argument("--no_watershed_sep", action="store_true", help="Disable watershed splitting in watershed mode.")
    ap.add_argument("--cellprob_threshold", type=float, default=0.0,
                    help="Cellpose cellprob threshold. Lower -> more nuclei (e.g., 0, -1, -2).")
    ap.add_argument("--flow_threshold", type=float, default=0.4,
                    help="Cellpose flow threshold. Lower -> more nuclei (e.g., 0.4, 0.3, 0.2).")
    ap.add_argument("--no_tile", action="store_true",
                    help="Disable tiling in Cellpose. Default uses tiling for stability on large/dense images.")
    ap.add_argument("--tile_overlap", type=float, default=0.1,
                    help="Cellpose tile overlap (0~0.5).")

    # -----------------------------
    # PATCH CLI: add stardist args in build_argparser() --v4
    # -----------------------------
    ap.add_argument("--stardist_prob_thresh", type=float, default=0.40,
                    help="StarDist prob_thresh. Lower -> higher recall. Typical 0.35-0.50.")
    ap.add_argument("--stardist_nms_thresh", type=float, default=0.70,
                    help="StarDist nms_thresh. Higher -> fewer duplicate/over-split instances. Typical 0.60-0.80.")
    ap.add_argument("--stardist_sigma", type=float, default=1.5,
                    help="Pre-smoothing sigma before StarDist. Typical 1.0-2.0.")
    ap.add_argument("--stardist_merge_quantile", type=float, default=0.75,
                    help="Quantile for boundary-intensity merge after StarDist. Higher -> more aggressive merging. Typical 0.70-0.85.")
    
    ap.add_argument("--stardist_disable_rescue", action="store_true",
                    help="Disable dense-region rescue splitting of huge labels (not recommended if dense misses exist).")
    ap.add_argument("--rescue_max_region_area_px", type=int, default=120000,
                    help="If a nucleus label area exceeds this, attempt watershed split inside it (dense rescue).")
    ap.add_argument("--rescue_h_maxima_h", type=float, default=6.0,
                    help="h value for distance-transform h-maxima in rescue splitting. Typical 5-8.")

    # Spot detection
    ap.add_argument("--fish_smooth_sigma", type=float, default=1.0, help="Gaussian sigma for FISH smoothing.")
    ap.add_argument("--use_auto_threshold", action="store_true", help="Use big-fish automated_threshold (recommended to QC first).")
    ap.add_argument("--z_threshold", type=float, default=6.0, help="Fixed z-threshold for spot detection when not using auto threshold.")
    ap.add_argument("--spot_radius_px", type=float, default=2.5, help="Spot radius (px). ~ spot diameter/2.")
    ap.add_argument("--fish_bg_sigma", type=float, default=8.0,
                    help="Background gaussian sigma for subtraction. 0 to disable. Typical 6~15.")
    ap.add_argument("--auto_thr_factor", type=float, default=1.0,
                    help="Multiply big-fish auto threshold by this factor (>1 makes stricter).")

    # Centromere max spots
    ap.add_argument("--centromere_max_spots", type=int, default=4,
                    help="Max centromere spots per nucleus. 3-color=>cyan, 2-color=>green. Default 4.")
    ap.add_argument("--centromere_disable_dense", action="store_true",
                    help="Disable dense decomposition on centromere channel (recommended).")

    # Dense decomposition
    ap.add_argument("--enable_dense", action="store_true", help="Enable dense region decomposition (big-fish if available, else fallback).")
    ap.add_argument("--dense_z", type=float, default=8.0, help="Dense region z threshold (fallback; also used for big-fish dense detection).")
    ap.add_argument("--dense_min_area_px", type=int, default=25, help="Min component area to consider as dense region (fallback).")
    ap.add_argument("--max_extra_spots_per_component", type=int, default=10, help="Cap extra spots per dense component (fallback).")
    ap.add_argument("--extra_spot_min_dist", type=int, default=2, help="Min distance between extra spots (fallback).")

    # Colocalization
    ap.add_argument("--coloc_dist_px", type=float, default=2.0, help="Distance threshold (px) for colocalization.")
    ap.add_argument("--coloc_one_to_one", action="store_true", help="Use one-to-one greedy matching for cross-channel colocalization.")

    # QC
    ap.add_argument("--qc_nuclei", type=int, default=10, help="Number of nuclei to export QC overlays per sample.")
    ap.add_argument("--qc_seed", type=int, default=0, help="Random seed for QC nucleus sampling.")
    ap.add_argument("--napari_qc", action="store_true", help="Launch napari viewer for interactive QC (blocks until closed).")

    # RGB
    # Per-channel RGB handling (only used when input tif is RGB(A))
    ap.add_argument("--dapi_rgb",  default="b",   choices=["auto","r","g","b","mean","max"],
                    help="If DAPI tif is RGB(A), which channel to use. Default b.")
    ap.add_argument("--red_rgb",   default="r",   choices=["auto","r","g","b","mean","max"],
                    help="If RED tif is RGB(A), which channel to use. Default r.")
    ap.add_argument("--green_rgb", default="g",   choices=["auto","r","g","b","mean","max"],
                    help="If GREEN tif is RGB(A), which channel to use. Default g.")
    ap.add_argument("--cyan_rgb",  default="max", choices=["auto","r","g","b","mean","max"],
                    help="If CYAN tif is RGB(A), which channel to use. Default max.")

    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    rows = []
    if args.manifest:
        dfm = load_manifest(args.manifest)
        for _, r in dfm.iterrows():
            rows.append(dict(
                sample_id=str(r["sample_id"]),
                dapi=r["dapi"],
                red=r["red"],
                green=r["green"],
                cyan=r.get("cyan", None),
                merged=r.get("merged", None),
                group=r.get("group", ""),
            ))
    else:
        if not (args.dapi and args.red and args.green):
            raise SystemExit("Single-sample mode requires --dapi --red --green (and optionally --cyan/--merged).")
        rows.append(dict(
            sample_id=args.sample_id,
            dapi=args.dapi,
            red=args.red,
            green=args.green,
            cyan=args.cyan,
            merged=args.merged,
            group=args.group if args.group is not None else "",
        ))

    all_cells = []
    all_images = []

    for row in rows:
        sample_id = row["sample_id"]
        per_cell, per_image = analyze_sample(
            sample_id=sample_id,
            dapi_path=row["dapi"],
            red_path=row["red"],
            green_path=row["green"],
            cyan_path=row["cyan"],
            merged_path=row["merged"],
            outdir=args.outdir,
            group_label=row["group"],
            
            z_project=args.z_project,
            tif_first_page=args.tif_first_page,
            nuclei_method=args.nuclei_method,
            cellpose_diameter=args.cellpose_diameter,
            min_area_px=args.min_area_px,
            smooth_sigma=args.smooth_sigma,
            h_maxima_h=args.h_maxima_h,
            merge_quantile=args.merge_quantile,
            watershed_sep=(not args.no_watershed_sep),

            # NEW: pass cellpose knobs via dispatch
            # (we pass them into analyze_sample then to segment_nuclei_dispatch)
            fish_smooth_sigma=args.fish_smooth_sigma,
            fish_bg_sigma=args.fish_bg_sigma,
            use_auto_threshold=args.use_auto_threshold,
            auto_thr_factor=args.auto_thr_factor,
            z_threshold=args.z_threshold,
            spot_radius_px=args.spot_radius_px,
            
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
            no_tile=args.no_tile,
            tile_overlap=args.tile_overlap,
            
            # NEW: StarDist params
            stardist_prob_thresh=args.stardist_prob_thresh,
            stardist_nms_thresh=args.stardist_nms_thresh,
            stardist_sigma=args.stardist_sigma,
            stardist_merge_quantile=args.stardist_merge_quantile,
            stardist_enable_rescue=(not args.stardist_disable_rescue),
            rescue_max_region_area_px=args.rescue_max_region_area_px,
            rescue_h_maxima_h=args.rescue_h_maxima_h,
            
            enable_dense=args.enable_dense,
            dense_z=args.dense_z,
            dense_min_area_px=args.dense_min_area_px,
            max_extra_spots_per_component=args.max_extra_spots_per_component,
            extra_spot_min_dist=args.extra_spot_min_dist,
            
            coloc_dist_px=args.coloc_dist_px,
            coloc_one_to_one=args.coloc_one_to_one,
            
            qc_nuclei=args.qc_nuclei,
            qc_seed=args.qc_seed,
            napari_qc=args.napari_qc,
            merged_rgb=args.merged_rgb,
            
            # NEW: centromere constraints
            centromere_max_spots=args.centromere_max_spots,
            centromere_disable_dense=args.centromere_disable_dense,

            dapi_rgb=args.dapi_rgb,
            red_rgb=args.red_rgb,
            green_rgb=args.green_rgb,
            cyan_rgb=args.cyan_rgb,

        )
        all_cells.append(per_cell)
        all_images.append(per_image)

    # write global summaries (batch)
    if len(all_cells) > 0:
        df_all_cells = pd.concat(all_cells, ignore_index=True)
        if "group" in df_all_cells.columns and "sample_id" in df_all_cells.columns:
            cols = ["group", "sample_id"] + [c for c in df_all_cells.columns if c not in ("group", "sample_id")]
            df_all_cells = df_all_cells.loc[:, cols]
        #df_all_cells.to_csv(os.path.join(args.outdir, "ALL_per_cell_summary.csv"), index=False)

    if len(all_images) > 0:
        df_all_images = pd.concat(all_images, ignore_index=True)
        if "group" in df_all_images.columns and "sample_id" in df_all_images.columns:
            cols = ["group", "sample_id"] + [c for c in df_all_images.columns if c not in ("group", "sample_id")]
            df_all_images = df_all_images.loc[:, cols]
        #df_all_images.to_csv(os.path.join(args.outdir, "ALL_per_image_summary.csv"), index=False)

    print("[OK] Done. Outputs written to:", args.outdir)


if __name__ == "__main__":
    main()
