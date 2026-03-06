"""
fragment_segmentation functions for BGI scATAC-seq fragments data.

"""
import gzip
import math
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

#from anndata import AnnData
#from scipy.sparse import csr_matrix
from typing_extensions import Literal

def fragments_segmentation(
    df_fragments: pd.DataFrame,
    label_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Funuction：fragment_segmentation

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
    df_fragments_cutoff["start_1k"] = (((df_fragments_cutoff["chromStart"])/1000).astype(int))*1000
    df_fragments_cutoff["end_1k"] = (((df_fragments_cutoff["chromStart"])/1000).astype(int)+1)*1000
    df_fragments_cutoff["start_100k"] = (((df_fragments_cutoff["chromStart"])/100000).astype(int))*100000
    df_fragments_cutoff["end_100k"] = (((df_fragments_cutoff["chromStart"])/100000).astype(int)+1)*100000
    
    df_fragments_cutoff["chr_raw"]=(df_fragments_cutoff["chrom"].astype(str)+":"+
                                    df_fragments_cutoff["chromStart"].astype(str)+"_"+
                                    df_fragments_cutoff["chromEnd"].astype(str))
    df_fragments_cutoff["chr_1k"]=(df_fragments_cutoff["chrom"].astype(str)+":"+
                                   df_fragments_cutoff["start_1k"].astype(str)+"_"+
                                   df_fragments_cutoff["end_1k"].astype(str))
    df_fragments_cutoff["chr_100k"]=(df_fragments_cutoff["chrom"].astype(str)+":"+
                                     df_fragments_cutoff["start_100k"].astype(str)+"_"+
                                     df_fragments_cutoff["end_100k"].astype(str))
    
    return df_fragments_cutoff

