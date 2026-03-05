"""
enrichment_score.py

"""
import os
import math
import random
import numpy as np
import pandas as pd

# ------------------------------------------  caculate enrichment_score  ------------------------------------------ #

def tss_enrichment_score(
    df:pd.DataFrame,
    expand:bool=True,
    adj:bool=True,
    intergenic:bool=False
):
    """
    计算tss富集分数
    输入：
    df数据框 tss_region
    tss site/region 比 非tss位点/区域 计数
    
    默认：
    expand=True 拓展 tss site 位点到 tss region
    adj=True 调整非tss区域，分母加tss
    intergenic=False 调整分母，基因body—>基因间区
    """
    df = df.copy()
    
    if expand:
        son = sum(df.tss_region)
        pass
    else:
        son = sum(df.tss_site)
        pass
        
    if not intergenic:
        under = sum(df.genebody_region)
        pass
    else:
        under = sum(df.intergenic_region)
        pass
        
    if adj:
        mom = son + under
        pass
    else:
        mom = under
        pass
        
    if mom == 0:
        score = -1
        pass
    else:
        score = son/mom
        pass
        
    df["tss_score"]=score
    return df

def genebody_enrichment_score(
    df:pd.DataFrame,
    expand:bool=True,
    adj:bool=True
):
    """
    基因体富集分数
    输入：df默认数据框
    拓展参数 True
    调整参数 True
    """
    df = df.copy()
    tss = sum(df.tss_region)
    gb = sum(df.genebody_region)
    ig = sum(df.intergenic_region)
    
    if expand:
        son = gb+tss
        pass
    else:
        son = gb
        pass
        
    if adj:
        mom = son+ig
        pass
    else:
        mom = ig
        pass
        
    if mom == 0:
        score = -1
        pass
    else:
        score = son/mom
        pass
    df["genebody_score"]=score
    return df

def tss_score(
    df:pd.DataFrame,
    expand:bool=True,
    adj:bool=True,
    intergenic:bool=False
):
    """函数文档字符串"""
    # 函数代码块
    df=df.copy()
    df=df.groupby(df["chr_100k"]).apply(
        tss_enrichment_score,
        expand=expand,
        adj=adj,
        intergenic=intergenic)
    return df

def genebody_score(
    df:pd.DataFrame,
    expand:bool=True,
    adj:bool=True
):
    """
    函数文档字符串
    
    """
    # 函数代码块
    df=df.copy()
    df=df.groupby(df["chr_100k"]).apply(
        genebody_enrichment_score,
        expand=expand,
        adj=adj)
    return df
