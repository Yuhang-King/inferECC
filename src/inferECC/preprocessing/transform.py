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

from ..tools import TSS_table

# ------------------------------------------ Log Transformation ------------------------------------------ #

#染色体名称标准化：
TSS = TSS_table()
chromosome_list = list(TSS.chromosome.unique())

def Normalize_Chromosome(
    df_fragments: pd.DataFrame,
    chromosome_list=chromosome_list
) -> pd.DataFrame:
    '''
    @Normalize_Chromosome: 染色体名称标准化
    @param df_fragments {pandas.core.frame.DataFrame} 一个表格对象：df_fragments
    @param chromosome_list {list} 一个list:(TSS.chromosome.unique())
    @return: df_fragments {pandas.core.frame.DataFrame} 一个染色体名称标准化后的表格对象：df_fragments
    '''
    df_fragments = df_fragments.copy()
    a=chromosome_list
    b=list(df_fragments.chrom.unique())
    d=[c in b for c in a]
    if True in d:
        print("df_fragments.chrom character content contains chr.")
        pass
    else:
        print("df_fragments.chrom character content does not contain chr.")
        df_fragments["chrom"] = "chr" + df_fragments["chrom"]
        pass
    
    return df_fragments

def Delete_other_chromosome(
    df_fragments: pd.DataFrame,
    chromosome_list=chromosome_list
) -> pd.DataFrame:
    '''
    @Delete_other_chromosome: 删除非(常染色体、性染色体、线粒体DNA)
    @param df_fragments {pandas.core.frame.DataFrame} 一个表格对象：df_fragments
    @param chromosome_list {list} 一个list:(TSS.chromosome.unique())
    @return: df_fragments {pandas.core.frame.DataFrame} 一个删除非(常染色体、性染色体、线粒体DNA)后的表格对象：df_fragments
    '''
    df_fragments = df_fragments.copy()
    df_fragments=df_fragments[df_fragments.chrom.isin(chromosome_list)]
    
    return df_fragments

def Delete_chrM(
    df_fragments: pd.DataFrame,
    chromosome_list=chromosome_list
) -> pd.DataFrame:
    '''
    @Delete_chrM: 删除线粒体DNA
    @param df_fragments {pandas.core.frame.DataFrame} 一个表格对象：df_fragments
    @param chromosome_list {list} 一个list:(TSS.chromosome.unique())
    @return: df_fragments {pandas.core.frame.DataFrame} 一个删除线粒体DNA后的表格对象：df_fragments
    '''
    df_fragments = df_fragments.copy()
    a=chromosome_list
    chrM_list=["chrM","chrm","M","m"]
    for chrM_list_temp in chrM_list:
        if(chrM_list_temp in a): a.remove(chrM_list_temp)
    df_fragments=df_fragments[df_fragments.chrom.isin(a)]
    
    return df_fragments

def Transform(
    df_fragments: pd.DataFrame, 
    Normalize_Chromosome_name: Optional[bool] = True,
    Delete_other_chromosome_option: Optional[bool] = True,
    Delete_chrM_option: Optional[bool] = True,
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
    df_fragments = df_fragments.copy()
    if Normalize_Chromosome_name:
        df_fragments = Normalize_Chromosome(df_fragments=df_fragments,chromosome_list=chromosome_list)
        pass
    if Delete_other_chromosome_option:
        df_fragments = Delete_other_chromosome(df_fragments=df_fragments,chromosome_list=chromosome_list)
        pass
    if Delete_chrM_option:
        df_fragments = Delete_chrM(df_fragments=df_fragments,chromosome_list=chromosome_list)
        pass
    
    df_fragments["fragLen"] = df_fragments["chromEnd"] - df_fragments["chromStart"]
    
    return df_fragments


    
    
