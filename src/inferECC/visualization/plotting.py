"""
Relative Position (bp from TSS)

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

def bp_from_tss(
    df: pd.DataFrame,
    lim: Optional[bool] = False,
    xlim: Optional[int] = 50000,
    file_name: Optional[str] = "p03_around_tss.pdf",
    show: Optional[bool] = True,
) -> plt:
    """
    Funuction：Relative Position (bp from TSS)

    Args:
        df_fragments: df_fragments.
        label_column: position_from_tss

    Returns:
        Pandas Dataframe with the following standardized column names.
            * `chrom`: Chromosome.
            * `chromStart`, `chromEnd`: Fragments position in chromosome.
            * `barcode`: Cell barcode.
            * `readSupport`: De-duplicated counts for each Fragment in one cell.
            
    """
    if lim == False:
        df_06 = df.copy()
        xlim=min(abs(min(df_06.position_from_tss)),abs(max(df_06.position_from_tss)))
        a=df_06[(-xlim<df_06.position_from_tss)&(df_06.position_from_tss<+xlim)]
        
        fig = plt.figure(figsize=(10,5), dpi= 200)
        ax = fig.add_subplot()
        ax.plot = a.position_from_tss.plot(kind="hist",bins=100,color="steelblue",edgecolor="black",density=False,label="Frequency")
        ax.set_xlabel("Relative Position (bp from TSS)")
        ax.set_ylabel("Fragments Number")
        ax.set_xlim(-xlim-1, +xlim+1)
        
        ax2 = ax.twinx()
        ax2.plot = a.position_from_tss.plot(kind="kde",color="red",label="Density")
        ax2.set_ylabel("Density")
        ax2.set_title("Enrichment_around_TSS")
        #ax2.set_xlim(-xlim,+xlim)
        
        fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
        plt.savefig(file_name)
        if show:
            return plt.show()
        pass
    
    elif lim == True:
        df_06 = df.copy()
        xlim=xlim
        a=df_06[(-xlim<df_06.position_from_tss)&(df_06.position_from_tss<+xlim)]
        
        fig = plt.figure(figsize=(10,5), dpi= 200)
        ax = fig.add_subplot()
        ax.plot = a.position_from_tss.plot(kind="hist",bins=100,color="steelblue",edgecolor="black",density=False,label="Frequency")
        ax.set_xlabel("Relative Position (bp from TSS)")
        ax.set_ylabel("Fragments Number")
        ax.set_xlim(-xlim-1, +xlim+1)
        
        ax2 = ax.twinx()
        ax2.plot = a.position_from_tss.plot(kind="kde",color="red",label="Density")
        ax2.set_ylabel("Density")
        ax2.set_title("Enrichment_around_TSS")
        #ax2.set_xlim(-xlim,+xlim)
        
        fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
        plt.savefig('p04_around_tss_2.pdf')
        if show:
            return plt.show()
        pass
    else:
        print("Wrong args!")
        pass
