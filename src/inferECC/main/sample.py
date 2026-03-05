"""
Miscellaneous non-normalizing data transformations on AnnData objects
"""
import os
import gzip
import math
import random
from random import sample
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

def sample_cell(
    df_fragments: pd.DataFrame, 
    sample_number: Optional[int] = 10,
    top_sample: Optional[bool] = False,
    label_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Funuction：sample_cell

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
    
    if top_sample:
        df_sort = caculate_fragments_number(df_fragments=df_fragments)
        df_sort_sample = df_sort.iloc[len(df_sort)-sample_number:len(df_sort)]
        df_fragments_sample = df_fragments[df_fragments.barcode.isin(df_sort_sample.barcode)]
        pass
    else:
        barcode_list = list(df_fragments.barcode.unique())
        sample_number = min(len(barcode_list),sample_number)
        barcode_list_sample = sample(barcode_list, sample_number)
        df_fragments_sample = df_fragments[df_fragments.barcode.isin(barcode_list_sample)]
        pass
        
    return df_fragments_sample

