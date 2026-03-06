# inferECC
Inferring extrachromosomal circular DNA from single-cell chromatin accessibility sequencing data.

[![Python](https://img.shields.io/badge/python-%3E%3D3.7-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](./pyproject.toml)
[![Build](https://img.shields.io/badge/build-setuptools-informational.svg)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-Proprietary-lightgrey.svg)](./pyproject.toml)

A Python toolkit for **ecDNA (extrachromosomal circular DNA) inference** from **single-cell ATAC-seq fragments**.  
`inferECC` provides a practical workflow from raw fragment tables to region-level scoring and visualization, centered on `pandas.DataFrame` utilities plus optional `AnnData` helpers.

---

## Table of Contents

- [Why inferECC](#why-inferecc)
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Input Schema](#input-schema)
- [Recommended Workflow](#recommended-workflow)
- [API Overview](#api-overview)
- [References and Genomes](#references-and-genomes)
- [FAQ](#faq)
- [Citation](#citation)
- [Changelog](#changelog)

---

## Why inferECC

`inferECC` is designed for fragment-level scATAC workflows where you need to:

- clean and standardize raw fragments,
- bin fragments into 1 kb / 100 kb windows,
- compute coverage and uniformity-based filters,
- annotate TSS / gene body / intergenic context,
- score enrichment signals,
- merge neighboring regions and evaluate correlation,
- generate publication-ready distribution and heatmap plots.

---

## Features

### 1) I/O
- Read BGI-style fragments with standardized output columns.

### 2) Preprocessing
- Chromosome name normalization (e.g. `1 -> chr1`),
- non-reference chromosome filtering,
- mitochondrial chromosome removal,
- fragment-length calculation.

### 3) Coverage + Uniformity
- Compute `fragnum_1k`, `fragnum_100k`, and `Coverage`,
- classify regions with built-in uniform-distribution cutoffs (`is_UD`).

### 4) Functional Annotation
- TSS proximity and region annotation,
- gene body annotation,
- intergenic inference.

### 5) Enrichment Scoring
- TSS enrichment (`tss_score`),
- gene-body enrichment (`genebody_score`).

### 6) Region Merging / Correlation
- Neighbor merge in genomic order,
- correlation-constrained merging in `AnnData` space.

### 7) Visualization
- Fragment length density,
- coverage density,
- TSS-relative-position distribution,
- region heatmaps.

### 8) Advanced (`crc_deg`)
- Scanpy-oriented helpers for clustering, DEG formatting, volcano plots, and overlap significance.

---

## Installation

### Requirements

- Python `>=3.7`
- Dependencies (core): `numpy`, `pandas`, `matplotlib`, `scanpy`, `anndata`, `scipy`

### Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[plot]"
```

---

## Quickstart

```python
import inferECC as ie

# 1) Load raw fragments
f = ie.read_bgi_as_dataframe("fragments.tsv")

# 2) Preprocess + binning + coverage
f = ie.Transform(f)
f = ie.fragments_segmentation(f)
f = ie.Normalize(f)

# 3) Uniformity inference (adds is_UD)
f = ie.caculate_uniform(f)

# 4) Annotation + enrichment scores
f = ie.tss_site(f, species="hg38")
f = ie.tss_region(f, species="hg38")
f = ie.genebody_region(f, species="hg38")
f = ie.intergenic_region(f)
f = ie.tss_score(f)
f = ie.genebody_score(f)

# 5) Plot examples
ie.fragments_length(f, show=False)
ie.coverage_density(f, show=False)
ie.bp_from_tss(f, show=False)
```

---

## Input Schema

`read_bgi_as_dataframe()` returns standardized fragment columns:

- `chrom`
- `chromStart`
- `chromEnd`
- `barcode`
- `readSupport`

Recommended data quality assumptions:

- no missing chromosome/start/end/barcode fields,
- genomic coordinates in the same reference build as your selected `species`.

---

## Recommended Workflow

```python
from inferECC import (
    read_bgi_as_dataframe,
    Transform,
    fragments_segmentation,
    Normalize,
    caculate_fragments_number,
    cutoff_fragments_number,
    sample_cell,
    caculate_uniform,
    tss_site,
    tss_region,
    genebody_region,
    intergenic_region,
    tss_score,
    genebody_score,
)

# Load
x = read_bgi_as_dataframe("fragments.tsv")

# Preprocess
x = Transform(
    x,
    Normalize_Chromosome_name=True,
    Delete_other_chromosome_option=True,
    Delete_chrM_option=True,
)

# Bin and normalize
x = fragments_segmentation(x)
x = Normalize(x)

# Optional cell filtering / sampling
n = caculate_fragments_number(x)
x = cutoff_fragments_number(x, cutoff_value=5000, df_fragments_number_sort=n)
# x = sample_cell(x, sample_number=50, top_sample=False)

# Uniformity + annotation + scoring
x = caculate_uniform(x)
x = tss_site(x, species="hg38")
x = tss_region(x, species="hg38")
x = genebody_region(x, species="hg38")
x = intergenic_region(x)
x = tss_score(x, expand=True, adj=True, intergenic=False)
x = genebody_score(x, expand=True, adj=True)
```

---

## API Overview

### I/O
- `read_bgi_as_dataframe(path, label_column=None)`

### Preprocessing
- `Transform(...)`
- `fragments_segmentation(df)`
- `Normalize(df)`

### Counts and filtering
- `caculate_fragments_number(df)`
- `cutoff_fragments_number(df, cutoff_value=5000, ...)`
- `sample_cell(df, sample_number=10, top_sample=False)`

### Uniformity
- `caculate_uniform(df, Uniform_Distribution=...)`

### Annotation and scoring
- `tss_site(df, species="hg38")`
- `tss_region(df, species="hg38")`
- `genebody_region(df, species="hg38")`
- `intergenic_region(df)`
- `tss_score(df, ...)`
- `genebody_score(df, ...)`

### Merging and correlation
- `Neighbor(df)`
- `sum_by(adata, col)`
- `sum_by_sparse(adata, col)`
- `neighbor_correlation(adata, ...)`

### Visualization
- `fragments_length(df, ...)`
- `coverage_density(df, ...)`
- `bp_from_tss(df, ...)`
- `enrichment_plot(df, enrich_arg="tss"|"genebody", ...)`
- `heatmap_chr(df, ...)`
- `heatmap_chr_fi(df, ...)`
- `ochh_mtx(...)`, `ochh_mtx_ks_test(...)`, `heatmap_raw_plot(...)`, `heatmap_fi_plot(...)`

### Advanced
- `inferECC.crc_deg.ecdna_mtx` (Scanpy downstream helper functions)

---

## References and Genomes

Built-in packaged references include:

- TSS BED for `hg38`, `hg19`, `mm10`,
- gene-body TSV for `hg38`, `hg19`, `mm10`,
- uniformity cutoff CSV (`freq_df_2.8wx1w_cutoff.csv`).

Use:

```python
from inferECC.tools.resources import get_reference_path
```

to resolve stable resource paths independent of your working directory.

---

## FAQ

### Q1: What order should I run functions in?
At minimum: `Transform -> fragments_segmentation -> Normalize -> caculate_uniform -> annotation -> scoring`.

### Q2: Why does `caculate_uniform` raise out-of-range errors?
Your observed `fragnum_100k` exceeds the cutoff table range. Use stricter filtering/sampling or a broader reference table.

### Q3: Which genome should I set in `species`?
Use the same build as your fragments (one of `hg38`, `hg19`, `mm10`).

### Q4: Why are some plotting outputs saved in unexpected folders?
Some plotting functions create directories and call `os.chdir()` internally; set/restore working directory explicitly in your scripts.

### Q5: Is this package compatible with AnnData workflows?
Yes. `sum_by`, `sum_by_sparse`, and `neighbor_correlation` support AnnData-centric matrix aggregation and neighborhood-correlation annotation.

---

## Citation

If you use `inferECC` in academic work, please:

1. cite the software repository,
2. include the exact version used (e.g. `1.0.0`),
3. describe major parameters (species, cell/filter thresholds, scoring settings).

> A formal manuscript citation entry can be added here when available.

---

## Changelog

- See commit history and release notes in your repository.
- Suggested entry point: `git log --oneline`.

---

## Version

Current package version: **1.0.0**