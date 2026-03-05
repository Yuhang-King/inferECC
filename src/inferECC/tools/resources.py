# -*- coding: utf-8 -*-
"""
inferECC/tools/resources.py
=================================
Robust package-resource path resolver.

Purpose / 目的
--------------
When inferECC is installed (editable or normal), code should NOT rely on the current
working directory to locate reference files (bed/tsv/csv). This module provides a
stable way to resolve data file paths shipped inside the inferECC package.

安装（尤其是 pip install -e .）后，不应依赖工作目录来寻找 reference 文件。
本模块提供稳定的包内资源路径解析方式。

Compatibility / 兼容性
----------------------
- Python >= 3.9: uses importlib.resources.files()
- Python 3.7/3.8: falls back to importlib_resources (backport)

If you run on Python 3.7/3.8, install:
    pip install importlib_resources
"""

from __future__ import annotations

import os
from typing import Optional

# --- importlib.resources compat ---
try:
    # Python 3.9+
    from importlib import resources as importlib_resources  # type: ignore
    _HAS_FILES_API = hasattr(importlib_resources, "files")
except Exception:
    importlib_resources = None  # type: ignore
    _HAS_FILES_API = False

if not _HAS_FILES_API:
    # Python 3.7/3.8 backport
    try:
        import importlib_resources as importlib_resources  # type: ignore
    except Exception as e:
        raise ImportError(
            "Python<3.9 detected and importlib_resources is not installed. "
            "Install it via: pip install importlib_resources"
        ) from e


def get_resource_path(rel_path: str, package: str = "inferECC") -> str:
    """
    Return an absolute filesystem path for a resource inside the installed inferECC package.

    Parameters
    ----------
    rel_path : str
        Relative path under the package root.
        Example: "tools/reference/hg38_tss.bed"
    package : str
        Package name. Default: "inferECC"

    Returns
    -------
    str
        Absolute path on the filesystem.

    Notes
    -----
    - For editable install, this points to your source tree.
    - For wheel install, pip may extract resources to a cache location; this function still works.
    """
    rel_path = rel_path.lstrip("/")

    # Preferred API (Python>=3.9 and new importlib_resources backport)
    if hasattr(importlib_resources, "files"):
        return str(importlib_resources.files(package).joinpath(rel_path))

    # Fallback API (older importlib_resources)
    # Using path() context manager ensures resource is available as a file path.
    with importlib_resources.path(package, rel_path) as p:  # type: ignore
        return str(p)


def get_reference_path(filename: str, genome: Optional[str] = None) -> str:
    """
    Convenience wrapper for common reference files.

    Examples
    --------
    get_reference_path("hg38_tss.bed")
    get_reference_path("hg38_gene_pos.tsv")
    get_reference_path("freq_df_2.8wx1w_cutoff.csv")

    If genome is provided, you can pass "tss.bed" or "gene_pos.tsv" and it will format:
      genome + "_" + suffix

    Parameters
    ----------
    filename : str
        Base filename (e.g., "hg38_tss.bed") OR suffix if genome is set (e.g., "tss.bed")
    genome : Optional[str]
        e.g., "hg38", "hg19", "mm10"

    Returns
    -------
    str : absolute path
    """
    if genome is not None and not filename.startswith(genome):
        filename = f"{genome}_{filename}"

    # Decide which subfolder
    if filename.endswith(".csv"):
        rel = f"tools/Uniform_Distribution/{filename}"
    else:
        rel = f"tools/reference/{filename}"
    return get_resource_path(rel)

