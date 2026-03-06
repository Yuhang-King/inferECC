"""
Miscellaneous non-normalizing data transformations on AnnData objects
"""
import os
import gzip
import math
import random
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
#from anndata import AnnData
#from scipy.sparse import csr_matrix
from typing_extensions import Literal

from ..tools import (
    chr_raw_relation,
    chr_relation,
    TSS_table,
    GeneBody_table,
)

# ------------------------------------------ Log 基因结构原始计数： ------------------------------------------ #

def tss_region(
    df_fragments: pd.DataFrame,
    species: Literal["hg38", "hg19", "mm10"] = "hg38",
    label_column: Optional[str] = None
):
    """
    Funuction：caculate_uniform Uniform_Distribution

    Args:
        df_fragments: df_fragments.
        label_column:

    Returns:
        Pandas Dataframe with the following standardized column names.
            * `chrom`: Chromosome.
            * `chromStart`, `chromEnd`: Fragments position in chromosome.
            * `barcode`: Cell barcode.
            * `readSupport`: De-duplicated counts for each Fragment in one cell.
            
    """
    TSS = TSS_table(species = species)
    df_fragments = df_fragments.copy()
    a = df_fragments["chr_raw"]
    
    chr_raw_relation_return=a.apply(chr_raw_relation,args=(TSS,chr_relation,"alternate_parameters"))
    
    region=[]
    region_gene=[]
    for value in chr_raw_relation_return:
        region.append(value[0])
        region_gene.append(value[1])
        pass
    df_fragments["tss_region"]=region
    df_fragments["tss_region_gene"] = region_gene
    
    return df_fragments


def genebody_region(
    df_fragments: pd.DataFrame,
    species: Literal["hg38", "hg19", "mm10"] = "hg38",
    label_column: Optional[str] = None
):
    """
    Funuction：caculate_uniform Uniform_Distribution

    Args:
        df_fragments: df_fragments.
        label_column:

    Returns:
        Pandas Dataframe with the following standardized column names.
            * `chrom`: Chromosome.
            * `chromStart`, `chromEnd`: Fragments position in chromosome.
            * `barcode`: Cell barcode.
            * `readSupport`: De-duplicated counts for each Fragment in one cell.
            
    """
    GeneBody = GeneBody_table(species = species)
    df_fragments = df_fragments.copy()
    a = df_fragments["chr_raw"]
    
    chr_raw_relation_return=a.apply(chr_raw_relation,args=(GeneBody,chr_relation,"alternate_parameters"))
    
    region=[]
    region_gene=[]
    for value in chr_raw_relation_return:
        region.append(value[0])
        region_gene.append(value[1])
        pass
    df_fragments["genebody_region"]=region
    df_fragments["genebody_region_gene"] = region_gene
    
    return df_fragments

# ------------------------------------------ Log intergenic_region： ------------------------------------------ #

def bool_exchange(TF_bool):
    if(TF_bool):
        return False
    else:
        return True
    pass



def intergenic_region(
    df_fragments: pd.DataFrame,
    label_column: Optional[str] = None
):
    """
    Funuction：intergenic_region

    Args:
        df_fragments: df_fragments.
        label_column:

    Returns:
        Pandas Dataframe with the following standardized column names.
            * `chrom`: Chromosome.
            * `chromStart`, `chromEnd`: Fragments position in chromosome.
            * `barcode`: Cell barcode.
            * `readSupport`: De-duplicated counts for each Fragment in one cell.
            
    """
    df_fragments = df_fragments.copy()

    df_fragments["intergenic_region"]=((df_fragments["tss_region"])|(df_fragments["genebody_region"])).astype(bool).apply(bool_exchange)
    
    return df_fragments


