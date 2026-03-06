"""
enrichment_plot

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

    
def enrichment_plot(
    df: pd.DataFrame,
    enrich_arg: Optional[str] = "tss",
    tss_file_name: Optional[str] = "p05_tss_enrichment_score.pdf",
    gb_file_name: Optional[str] = "p06_gb_enrichment_score.pdf",
    show: Optional[bool] = True,
) -> plt:
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
    if enrich_arg == "tss":
        df = df.copy()
        a=df[(-1<df.tss_score)]
        
        fig = plt.figure(figsize=(10,5), dpi= 200)
        ax = fig.add_subplot()
        ax.plot = a.tss_score.plot(kind="hist",bins=100,color="steelblue",edgecolor="black",density=False,label="Frequency")
        ax.set_xlabel("tss_enrichment_score")
        ax.set_ylabel("Fragments_100kbp Number")
        ax.set_xlim(-0.25, +1.25)
        
        ax2 = ax.twinx()
        ax2.plot = a.tss_score.plot(kind="kde",color="red",label="Density")
        ax2.set_ylabel("Density")
        ax2.set_title("tss_enrichment_score")
        
        fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
        plt.savefig(tss_file_name)
        if show:
            return plt.show()
        pass
    elif enrich_arg == "genebody":
        df = df.copy()
        a=df[(-1<df.genebody_score)]
        
        fig = plt.figure(figsize=(10,5), dpi= 200)
        ax = fig.add_subplot()
        ax.plot = a.genebody_score.plot(kind="hist",bins=100,color="steelblue",edgecolor="black",density=False,label="Frequency")
        ax.set_xlabel("genebody_enrichment_score")
        ax.set_ylabel("Fragments_100kbp Number")
        ax.set_xlim(-0.25, +1.25)
        
        ax2 = ax.twinx()
        ax2.plot = a.genebody_score.plot(kind="kde",color="red",label="Density")
        ax2.set_ylabel("Density")
        ax2.set_title("genebody_enrichment_score")
        
        fig.legend(loc=1, bbox_to_anchor=(0.18,1), bbox_transform=ax.transAxes)
        plt.savefig(gb_file_name)
        if show:
            return plt.show()
        pass
    else:
        print("Wrong args!")
        pass
