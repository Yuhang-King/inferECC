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

from ..tools import Enrichment_around_TSS,TSS_table


# ------------------------------------------ Log Uniform_Distribution ------------------------------------------ #

def tss_site(
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
    input_a = df_fragments.loc[:,"chr_raw"]
    Enrichment_around_TSS_return=input_a.apply(Enrichment_around_TSS,args=(TSS,"alternate_parameters"))
    
    region=[]
    region_gene=[]
    position_from_TSS=[]
    for value in Enrichment_around_TSS_return:
        region.append(value[0])
        region_gene.append(value[1])
        position_from_TSS.append(value[2])
        pass
    df_fragments.loc[:,"tss_site"] = region
    df_fragments.loc[:,"tss_site_gene"] = region_gene
    df_fragments.loc[:,"position_from_tss"] = position_from_TSS
    
    return df_fragments

