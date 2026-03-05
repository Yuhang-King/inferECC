"""
heatmap

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

def heatmap_chr(
    df: pd.DataFrame,
    lim: Optional[bool] = False,
    file_name: Optional[str] = "heatmap",
    show: Optional[bool] = False,
):
    """heatmap"""
    
    if(os.path.exists(file_name) != True): 
        os.makedirs(file_name)
        pass
    os.chdir(file_name)
    
    df_08 = df.copy()
    for cb in df_08.barcode.unique():
        print(cb)
        df_cb = df_08[df_08["barcode"]==cb]
                
        df_cb_uni=df_cb[["barcode","start_1k","end_1k",
                         "start_100k","end_100k","chr_1k",
                         "chr_100k","fragnum_100k","fragnum_1k",
                         "Coverage","is_UD"]]
        df_cb_uni.loc[:,"index_num"]=(df_cb_uni.loc[:,"end_1k"]-df_cb_uni.loc[:,"start_100k"])/1000
        df_cb_uni.loc[:,"index_num"]=df_cb_uni.loc[:,"index_num"].astype(int)
        df_cb_uni_dd=df_cb_uni.drop_duplicates(subset=['chr_1k','chr_100k'])
        
        df_cb_uni_dd_tri=df_cb_uni_dd[["chr_100k","index_num","fragnum_1k"]]
        df_cb_uni_dd_tri_ref=pd.DataFrame( 
            { 'chr_100k':["chr:references"]*100,
             'index_num': list(range(1,101)),  
             'fragnum_1k': [0]*100}
        )
        df_cb_uni_dd_tri_ref_con = pd.concat(
            [df_cb_uni_dd_tri,df_cb_uni_dd_tri_ref],
            keys=['chr_100k', 'index_num', 'fragnum_1k']
        )
        df_mtx=pd.pivot_table(
            df_cb_uni_dd_tri_ref_con,
            index=["chr_100k"],
            columns=["index_num"],
            values=["fragnum_1k"],
            fill_value=0
        )
        
        plt.figure(figsize=(30,df_mtx.fragnum_1k.shape[0]/5+2), dpi= 180)
        sns.heatmap(data=df_mtx.fragnum_1k,
                    #annot=True,
                    #fmt="d",
                    linewidths=0.05,
                    square=True,
                    linecolor="grey",
                    cmap="Oranges")
        plt.xlabel('index_number')
        plt.title(cb)
        plt.savefig(cb+".pdf")
        pass
    pass

### after filter
def heatmap_chr_fi(
    df: pd.DataFrame,
    lim: Optional[bool] = False,
    file_name: Optional[str] = "heatmap_fi",
    show: Optional[bool] = False,
):
    """heatmap"""
    
    if(os.path.exists(file_name) != True): 
        os.makedirs(file_name)
        pass
    os.chdir(file_name)
    
    df_08 = df.copy()
    for cb in df_08.barcode.unique():
        print(cb)
        df_cb = df_08[df_08["barcode"]==cb]
                
        df_cb_uni=df_cb[["barcode","start_1k","end_1k",
                         "start_100k","end_100k","chr_1k",
                         "chr_100k","fragnum_100k","fragnum_1k",
                         "Coverage","is_UD"]]
        df_cb_uni.loc[:,"index_num"]=(df_cb_uni.loc[:,"end_1k"]-df_cb_uni.loc[:,"start_100k"])/1000
        df_cb_uni.loc[:,"index_num"]=df_cb_uni.loc[:,"index_num"].astype(int)
        df_cb_uni_dd=df_cb_uni.drop_duplicates(subset=['chr_1k','chr_100k'])
        
        df_cb_uni_dd_tri=df_cb_uni_dd[["chr_100k","index_num","fragnum_1k"]]
        df_cb_uni_dd_tri_ref=pd.DataFrame( 
            { 'chr_100k':["chr:references"]*100,
             'index_num': list(range(1,101)),  
             'fragnum_1k': [0]*100}
        )
        df_cb_uni_dd_tri_ref_con = pd.concat(
            [df_cb_uni_dd_tri,df_cb_uni_dd_tri_ref],
            keys=['chr_100k', 'index_num', 'fragnum_1k']
        )
        df_mtx=pd.pivot_table(
            df_cb_uni_dd_tri_ref_con,
            index=["chr_100k"],
            columns=["index_num"],
            values=["fragnum_1k"],
            fill_value=0
        )
        
        plt.figure(figsize=(30,df_mtx.fragnum_1k.shape[0]/5+2), dpi= 180)
        sns.heatmap(data=df_mtx.fragnum_1k,
                    #annot=True,
                    #fmt="d",
                    linewidths=0.05,
                    square=True,
                    linecolor="grey",
                    cmap="Oranges")
        plt.xlabel('index_number')
        plt.title(cb)
        plt.savefig(cb+".pdf")
        pass
    pass

