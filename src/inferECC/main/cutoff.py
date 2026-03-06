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

from .caculate import caculate_fragments_number

# ------------------------------------------ Log Transformation ------------------------------------------ #

def cutoff_fragments_number(
    df_fragments: pd.DataFrame, 
    cutoff_value: Optional[str] = 5000,
    df_fragments_number_sort: Optional[pd.DataFrame] = None,
    label_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Funuction：cutoff_fragments_number

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
    df_sort = df_fragments_number_sort.copy()
    
    if not df_sort.empty:
        df_sort_cutoff = df_sort[df_sort["fragments_number"] >= cutoff_value]
        df_fragments_cutoff = df_fragments[df_fragments.barcode.isin(df_sort_cutoff.barcode)]
        pass
    else:
        df_sort=caculate_fragments_number(df_fragments=df_fragments)
        df_sort_cutoff = df_sort[df_sort["fragments_number"] >= cutoff_value]
        df_fragments_cutoff = df_fragments[df_fragments.barcode.isin(df_sort_cutoff.barcode)]
        pass
        
    return df_fragments_cutoff

