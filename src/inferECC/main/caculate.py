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


# ------------------------------------------ Log Transformation ------------------------------------------ #

def caculate_fragments_number(
    df_fragments: pd.DataFrame,
    label_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Funuction：caculate_fragments_number

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
    df_fragments_cutoff = df_fragments.copy()
    
    df_fragments_number = pd.DataFrame({"barcode":df_fragments.barcode.value_counts().index,
                            "fragments_number":df_fragments.barcode.value_counts().values})
    df_fragments_number.fragments_number = pd.to_numeric(df_fragments_number.fragments_number)
    df_fragments_number_sort = df_fragments_number.sort_values(by = ["fragments_number"])
    
    return df_fragments_number_sort


