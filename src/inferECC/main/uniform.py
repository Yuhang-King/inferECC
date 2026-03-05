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

import sys
sys.path.append("..")
from ..tools import Uniform_Distribution_table
ud_cutoff_table=Uniform_Distribution_table(exe=True)

# ------------------------------------------ Log Uniform_Distribution ------------------------------------------ #

def caculate_uniform(
    df_fragments: pd.DataFrame,
    Uniform_Distribution: pd.DataFrame = ud_cutoff_table,
    label_column: Optional[str] = None
) -> pd.DataFrame:
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
    df_04 = df_fragments.copy()
    Uniform_Distribution = Uniform_Distribution.copy()
    UD_max = Uniform_Distribution["frag_num_100k"].max()
    
    ### 判定，判定每个CB中的chr_100k是否为is_ecDNA_frag_100k
    ### 增加：freq_df_cutoff 的模拟平均分布Uniform_Distribution参考中，是否含有测试的"fragnum_100k"数；
    ### 若有，继续执行；若无，报错: The value of "fragnum_100k" is outside the value range of the reference table.
    df_05 = pd.DataFrame()
    for CB_temp in df_04["barcode"].unique():
        df_temp = df_04.loc[df_04["barcode"]==CB_temp].copy()
        df_temp_new = pd.DataFrame()
        for chr_temp in df_temp["chr_100k"].unique():
            df_temp_chr = df_temp.loc[df_temp["chr_100k"]==chr_temp].copy()
            chr_temp_flag = list(df_temp_chr["fragnum_100k"])[0]
            
            if(chr_temp_flag>UD_max):
                print("The value of fragnum_100k:",chr_temp_flag)
                print("Uniform_Distribution_max:",UD_max)
                raise Exception("The value of fragnum_100k is outside the value range of the Uniform_Distribution reference table.")
                pass
            else:
                UD_ref = Uniform_Distribution[Uniform_Distribution["frag_num_100k"]==chr_temp_flag]
                UD_ref_max = list(UD_ref["frag_num_1k_max"])[0]
                df_temp_chr["is_UD"] = max(df_temp_chr["fragnum_1k"]) <= UD_ref_max
                df_temp_new = pd.concat([df_temp_new, df_temp_chr])
                pass
            pass
        df_05 = pd.concat([df_05, df_temp_new])
        pass
    
    return df_05

