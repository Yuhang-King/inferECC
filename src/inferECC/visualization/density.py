"""
fragments_length

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

def fragments_length(
    df: pd.DataFrame,
    lim: Optional[bool] = False,
    xlim: Optional[int] = 2000,
    plt_title: Optional[str] = "Density plot for fragments length",
    file_name: Optional[str] = "p01_fragments_length_density.pdf",
    label: Optional[str] = "fragments",
    xlabel: Optional[str] = "fragments_length",
    show: Optional[bool] = False,
) -> plt:
    """
    Funuction：fragments_length

    Args:
        df_fragments: df_fragments.
        label_column: fragments_length

    Returns:
        Pandas Dataframe with the following standardized column names.
            * `chrom`: Chromosome.
            * `chromStart`, `chromEnd`: Fragments position in chromosome.
            * `barcode`: Cell barcode.
            * `readSupport`: De-duplicated counts for each Fragment in one cell.
            
    """
    df_fragments = df.copy()
    if lim==False:  
        # Draw Plot
        plt.figure(figsize=(15,5), dpi= 200)
        sns.kdeplot(df_fragments.fragLen, fill=True, color="red", label=label, alpha=0.7)
        plt.title(plt_title, fontsize=18)
        plt.xlabel(xlabel)
        plt.legend()
        plt.savefig(file_name)
        #plt.show()
        if show:
            return plt.show()
        pass
    elif lim==True:
        # Draw Plot
        plt.figure(figsize=(15,5), dpi= 200)
        sns.kdeplot(df_fragments.fragLen, fill=True, color="red", label=label, alpha=0.7)
        plt.title(plt_title, fontsize=18)
        plt.xlabel(xlabel)
        plt.legend()
        plt.xlim(-100, xlim)
        plt.savefig(file_name)
        #plt.show()
        if show:
            return plt.show()
        pass
    else:
        print("Wrong args!")
        pass
    pass

def coverage_density(
    df: pd.DataFrame,
    lim: Optional[bool] = False,
    xlim: Optional[int] = 2000,
    file_name: Optional[str] = "p02_coverage_density.pdf",
    show: Optional[bool] = True,
) -> plt:
    """
    Funuction：coverage_density

    Args:
        df_fragments: df_fragments.
        label_column: coverage_density

    Returns:
        Pandas Dataframe with the following standardized column names.
            * `chrom`: Chromosome.
            * `chromStart`, `chromEnd`: Fragments position in chromosome.
            * `barcode`: Cell barcode.
            * `readSupport`: De-duplicated counts for each Fragment in one cell.
            
    """
    df_02 = df.copy()
    
    # Draw Plot
    plt.figure(figsize=(15,5), dpi= 200)
    sns.kdeplot(df_02.Coverage, fill=True, color="red", label="Coverage", alpha=0.7)
    #plt.vlines(0,0,6, colors = "balck", linestyles = "dashed")
    plt.axvline(6, color = "b", linestyle = "-.")
    plt.xlim(-3, min(max(df_02.Coverage),max(df_02.Coverage))+10)
    #plt.ylim( , )
    plt.title('coverage_density', fontsize=18)
    plt.legend()
    plt.savefig(file_name)
                
    if show:
        return plt.show()
    else:
        print("Wrong args!")
        pass
    pass
