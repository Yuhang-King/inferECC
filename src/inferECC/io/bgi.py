"""
IO functions for BGI scATAC-seq fragments data.

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

def read_bgi_as_dataframe(path: str, label_column: Optional[str] = None) -> pd.DataFrame:
    """
    Read a BGI scATAC-seq fragments data file as a pandas DataFrame.

    Args:
        path: Path to read file.
        label_column: Column name containing positive cell labels.

    Returns:
        Pandas Dataframe with the following standardized column names.
            * `chrom`: Chromosome.
            * `chromStart`, `chromEnd`: Fragments position in chromosome.
            * `barcode`: Cell barcode.
            * `readSupport`: De-duplicated counts for each Fragment in one cell.
            
    """
    names = ["chrom","chromStart","chromEnd","barcode","readSupport"]
    dtype = {
        "chrom":"string",
        "chromStart":int,
        "chromEnd":int,
        "barcode":"string",
        "readSupport":"string"
    }
    # Use first 10 rows for validation.
    df_fragments = pd.read_csv(path,
                      sep="\t",
                      header = None, 
                      comment = "#",
                      names = names,
                      dtype = dtype,
                      nrows=10)
    
    if label_column:
        dtype.update({label_column: np.uint32})
        
        if label_column not in df_fragments.columns:
            raise IOError(f"Column `{label_column}` is not present.")
            pass
        pass
    
    df_fragments = pd.read_csv(path,
                      sep = "\t",
                      header = None, 
                      comment = "#",
                      names = names,
                      dtype = dtype)

    return df_fragments

