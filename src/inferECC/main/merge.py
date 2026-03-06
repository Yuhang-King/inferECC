"""
merge ecdna

"""
import os
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from typing_extensions import Literal

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# ------------------------------------------ merge ecdna ------------------------------------------ #

import pandas as pd
import numpy as np
import anndata as ad
from scipy import sparse
from pandas import CategoricalDtype

### dense matrix X
def sum_by(adata: ad.AnnData, col: str) -> ad.AnnData:
    adata.strings_to_categoricals()
    #assert isinstance(adata.obs[col], CategoricalDtype)
    assert pd.api.types.is_categorical_dtype(adata.obs[col])
    
    indicator = pd.get_dummies(adata.obs[col])
    
    return ad.AnnData(
        indicator.values.T @ adata.X,
        var=adata.var,
        obs=pd.DataFrame(index=indicator.columns)
    )

### sparse matrix X
def sum_by_sparse(adata: ad.AnnData, col: str) -> ad.AnnData:
    adata.strings_to_categoricals()
    #assert isinstance(adata.obs[col], CategoricalDtype)
    assert pd.api.types.is_categorical_dtype(adata.obs[col])

    cat = adata.obs[col].values
    indicator = sparse.coo_matrix(
        (
            np.broadcast_to(True, adata.n_obs),
            (cat.codes, np.arange(adata.n_obs))
        ),
        shape=(len(cat.categories), adata.n_obs),
    )

    return ad.AnnData(
        indicator @ adata.X,
        var=adata.var,
        obs=pd.DataFrame(index=cat.categories)
    )


from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from typing_extensions import Literal

def Neighbor(
    df: pd.DataFrame,
    label: Optional[str] = None
) -> pd.DataFrame:
    """
    Funuction：Neighbor

    Args:
        df: pd.DataFrame,
        label: Optional[str] = None

    Returns:
        Pandas Dataframe with the following standardized column names.
            * `chrom`: Chromosome.
            * `chromStart`, `chromEnd`: Fragments position in chromosome.
            * `barcode`: Cell barcode.
            * `readSupport`: De-duplicated counts for each Fragment in one cell.
            
    """
    temp = df.copy()
    temp = temp.sort_values(by=["chrom","start_100k"],ascending=(True,True))
    
    a=list(temp.start_100k.values)
    del(a[0])
    a.append(float("inf"))
    temp["next_100k"] = a
    temp["gap_100k"] = temp["next_100k"]-temp["start_100k"]
    temp["neighbor_bool"] = temp["gap_100k"] <= 500000
    
    neighbor = []
    chrom_temp = list(temp.chrom.values)[0]
    start_temp = list(temp.start_100k.values)[0]
    end_temp = list(temp.end_100k.values)[0]
    region_len = 1
    Len = len(temp.neighbor_bool.values)
    for i in range(Len):
        if list(temp.neighbor_bool.values)[i]:
            region_len += 1
            start_temp = start_temp
            end_temp = list(temp.end_100k.values)[i+1]
            pass
        else:
            pre_region = chrom_temp+":"+str(start_temp)+"_"+str(end_temp)
            for j in range(region_len): 
                neighbor.append(pre_region)
                pass
            region_len = 1
            start_temp = list(temp.start_100k.values)[(i+1)%Len]
            end_temp = list(temp.end_100k.values)[(i+1)%Len]
            pass
        pass
    
    temp["neighbor"] = neighbor
    neighbor_len_temp = temp.neighbor.str.split(':',expand=True)[1]
    neighbor_len_temp_e = neighbor_len_temp.str.split('_',expand=True)[1].astype(int)
    neighbor_len_temp_s = neighbor_len_temp.str.split('_',expand=True)[0].astype(int)
    temp["neighbor_len"] = (neighbor_len_temp_e-neighbor_len_temp_s)/1000000

    return temp

#from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from scipy.stats import pearsonr
def neighbor_correlation(
    adata: ad.AnnData,
    label: Optional[str] = None,
    correlation_bool_cutoff: Optional[int] = 0.2,
    correlation_pvalue_bool_cutoff: Optional[int] = 0.05
) -> pd.DataFrame:
    """
    Funuction：neighbor_correlation

    Args:
        df: anndata._core.anndata.AnnData,
        label: Optional[str] = None

    Returns:
        Pandas Dataframe with the following standardized column names.
            * `chrom`: Chromosome.
            * `chromStart`, `chromEnd`: Fragments position in chromosome.
            * `barcode`: Cell barcode.
            * `readSupport`: De-duplicated counts for each Fragment in one cell.
            
    """
    
    ad = adata.copy()
    ad_var = ad.var.copy()
    ad_var_sort = ad_var.sort_values(by=["chrom","start_100k"],ascending=(True,True))

    correlation_list = []
    correlation_pvalue_list = []
    # 计算顺序排序的相邻两个chr_100k的相关性 correlation, pvalue
    for i in range(len(ad_var_sort.chr_100k)-1):
        
        # 假设 adata 是你的 AnnData 对象，gene1 和 gene2 是你要比较的基因
        gene1 = ad_var_sort.chr_100k.to_list()[i]
        gene2 = ad_var_sort.chr_100k.to_list()[i+1]
        
        # 获取基因的表达值
        ad.raw = ad.copy()
        gene1_expression = ad.raw[:, gene1].X.flatten()
        gene2_expression = ad.raw[:, gene2].X.flatten()
        
        # 计算相关性
        #correlation = pd.Series(gene1_expression).corr(pd.Series(gene2_expression))
        #print(f' {gene1} 和 {gene2} 的表达值的相关性为: {correlation}')
        
        # 计算相关性和 p值
        correlation, pvalue = pearsonr(gene1_expression, gene2_expression)
        #print(f' {gene1} 和 {gene2} 的coverage的相关性为: {correlation}, p值为: {pvalue}')

        correlation_list.append(correlation)
        correlation_pvalue_list.append(pvalue)
        
        pass
    correlation_list.append(0)
    correlation_pvalue_list.append(0)
    ad_var_sort["correlation"] = correlation_list
    ad_var_sort["correlation_pvalue"] = correlation_pvalue_list

    # temp 暂代 ad_var_sort 
    temp = ad_var_sort
    # 将 correlation, pvalue 通过cutoff值 转为 bool类型
    temp["correlation_bool"] = temp["correlation"] >= correlation_bool_cutoff
    temp["correlation_pvalue_bool"] = temp["correlation_pvalue"] <= correlation_pvalue_bool_cutoff
    temp["neighbor_correlation_pvalue_bool"] = temp['neighbor_bool'] & temp['correlation_bool'] & temp['correlation_pvalue_bool']

    ### 通过 neighbor_correlation_pvalue_bool 合并后的 chr neighbor
    neighbor = []
    chrom_temp = list(temp.chrom.values)[0]
    start_temp = list(temp.start_100k.values)[0]
    end_temp = list(temp.end_100k.values)[0]
    region_len = 1
    Len = len(temp.neighbor_correlation_pvalue_bool.values)
    for i in range(Len):
        if list(temp.neighbor_correlation_pvalue_bool.values)[i]:
            region_len += 1
            start_temp = start_temp
            end_temp = list(temp.end_100k.values)[i+1]
            pass
        else:
            #pre_region = chrom_temp+":"+str(start_temp)+"_"+str(end_temp)  ### 这一步骤出现错误，等待v9修正：：chr编号错误
            pre_region = ":"+str(start_temp)+"_"+str(end_temp)  ### v9：已修正以上错误
            for j in range(region_len): 
                neighbor.append(pre_region)
                pass
            region_len = 1
            start_temp = list(temp.start_100k.values)[(i+1)%Len]
            end_temp = list(temp.end_100k.values)[(i+1)%Len]
            pass
        pass
    
    temp["correlation_neighbor"] = neighbor
    temp["correlation_neighbor"] = temp["chrom"] + temp["correlation_neighbor"] ### v9：已修正以上错误
    neighbor_len_temp = temp.correlation_neighbor.str.split(':',expand=True)[1]
    neighbor_len_temp_e = neighbor_len_temp.str.split('_',expand=True)[1].astype(int)
    neighbor_len_temp_s = neighbor_len_temp.str.split('_',expand=True)[0].astype(int)
    temp["correlation_neighbor_len"] = (neighbor_len_temp_e-neighbor_len_temp_s)/1000000

    # 结束 ad_var_sort 取代 temp
    ad_var_sort = temp
    ad_var_sort.index = ad_var_sort.chr_100k
    ad_var_sort_ri = ad_var_sort.reindex(index=list(ad.var.index.values))
    ad.var = ad_var_sort_ri
    
    return ad
    
