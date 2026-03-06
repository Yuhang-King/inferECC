# -*- coding: utf-8 -*-
"""
inferECC/tools/ref_data.py
=================================
Gene structure references:
- TSS table (hg38/hg19/mm10)
- Gene body table (hg38/hg19/mm10)

改造要点（中英）:
- Use package resource path resolver (no working-directory dependence).
  使用包内资源路径解析，避免依赖当前工作目录或 __file__ 路径拼接。
- Remove hard-coded "/reference/..." relative paths and file_path_abs().
  删除硬编码相对路径与 file_path_abs()。
"""

from __future__ import annotations

import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from inferECC.tools.resources import get_reference_path


def read_TSS_table(path: str) -> pd.DataFrame:
    """
    Read TSS BED table.

    Parameters
    ----------
    path : str
        Absolute path of the BED file.

    Returns
    -------
    DataFrame
        Parsed TSS table with additional columns:
        - start/end: +-25 bp window (as in legacy logic)
        - chr_raw: "chr:start_end"
    """
    df_temp = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chromosome", "TSSstart", "TSSend", "gene", "point", "pos_neg", "gene_type"],
        dtype={
            "chromosome": "string",
            "TSSstart": "int64",
            "TSSend": "int64",
            "gene": "string",
            "point": "string",
            "pos_neg": "string",
            "gene_type": "string",
        },
    )

    # Keep legacy window definition: [-24, +25]
    df_temp["start"] = df_temp["TSSstart"] - 24
    df_temp["end"] = df_temp["TSSstart"] + 25
    df_temp["chr_raw"] = (
        df_temp["chromosome"].astype(str)
        + ":"
        + df_temp["start"].astype(str)
        + "_"
        + df_temp["end"].astype(str)
    )
    return df_temp


def read_GeneBody_table(path: str) -> pd.DataFrame:
    """
    Read gene body TSV table.

    Parameters
    ----------
    path : str
        Absolute path of the TSV file.

    Returns
    -------
    DataFrame
        Parsed gene body table with additional column:
        - chr_raw: "chr:start_end"
    """
    df_temp = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["gene", "chromosome", "start", "end"],
        dtype={
            "gene": "string",
            "chromosome": "string",
            "start": "int64",
            "end": "int64",
        },
    )
    df_temp["chr_raw"] = (
        df_temp["chromosome"].astype(str)
        + ":"
        + df_temp["start"].astype(str)
        + "_"
        + df_temp["end"].astype(str)
    )
    return df_temp


def TSS_table(species: Literal["hg38", "hg19", "mm10"] = "hg38") -> pd.DataFrame:
    """
    Load TSS reference table by genome.

    Parameters
    ----------
    species : {"hg38","hg19","mm10"}

    Returns
    -------
    DataFrame
    """
    print("species value:", species)
    # package resource path (stable)
    path = get_reference_path("tss.bed", genome=species)
    return read_TSS_table(path=path)


def GeneBody_table(species: Literal["hg38", "hg19", "mm10"] = "hg38") -> pd.DataFrame:
    """
    Load gene body reference table by genome.

    Parameters
    ----------
    species : {"hg38","hg19","mm10"}

    Returns
    -------
    DataFrame
    """
    # package resource path (stable)
    path = get_reference_path("gene_pos.tsv", genome=species)
    return read_GeneBody_table(path=path)


"""
基因结构 TSS_region/gene_body/intergenic region
"""

'''
import os
import numpy as np
import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# ------------------------------------------  make Reference  ------------------------------------------ #

reff=("/reference/hg38_gene_pos.tsv",
      "/reference/hg19_gene_pos.tsv",
      "/reference/mm10_gene_pos.tsv",
      "/reference/hg38_tss.bed",
      "/reference/hg19_tss.bed",
      "/reference/mm10_tss.bed")
hg38_gene_pos_path=reff[0]
hg19_gene_pos_path=reff[1]
mm10_gene_pos_path=reff[2]
hg38_tss_path=reff[3]
hg19_tss_path=reff[4]
mm10_tss_path=reff[5]

# file_path_abs
def file_path_abs(file_path: str):
    module_path = os.path.dirname(__file__)
    file_path_abs = module_path + file_path
    
    return file_path_abs

def read_TSS_table(path: str):
    """
    Function: 
    
    Args:
        path
        
    Returns:
        r
    """
    df_temp = pd.read_csv(path,
                   sep = "\t",
                   header = None,
                   names = ["chromosome","TSSstart","TSSend","gene","point","pos_neg","gene_type"],
                   dtype = {"chromosome":"string",
                         "TSSstart":int,
                         "TSSend":int,
                         "gene":"string",
                         "point":"string",
                         "pos_neg":"string",
                         "gene_type":"string"}) 
    df_temp["start"] = df_temp["TSSstart"]-24
    df_temp["end"] = df_temp["TSSstart"]+25
    df_temp["chr_raw"] = df_temp["chromosome"].astype(str)+":"+df_temp["start"].astype(str)+"_"+df_temp["end"].astype(str)
    
    return df_temp

def read_GeneBody_table(path: str):
    """
    Function: 
    
    Args:
        path
        
    Returns:
        r
    """
    df_temp = pd.read_csv(path,
                   sep = "\t",
                   header = None, 
                   names=["gene","chromosome","start","end"],
                   dtype = {"gene":"string",
                         "chromosome":"string",
                         "start":int,
                         "end":int})
    df_temp["chr_raw"] = df_temp["chromosome"].astype(str)+":"+df_temp["start"].astype(str)+"_"+df_temp["end"].astype(str)
    
    return df_temp

def TSS_table(
species: Literal["hg38", "hg19", "mm10"] = "hg38"
):
    """
    Function: 
    
    Args:
        species
        
    Returns:
        r
    """
    species = species
    print("species value:", species)
    print("species == 'mm10': ", species == "mm10"  )
    if species == "hg38":
        path = file_path_abs(file_path=hg38_tss_path)
        df_tss = read_TSS_table(path=path)
        return df_tss
    elif species == "hg19":
        path = file_path_abs(file_path=hg19_tss_path)
        df_tss = read_TSS_table(path=path)
        return df_tss
    elif species == "mm10":
        path = file_path_abs(file_path=mm10_tss_path)
        df_tss = read_TSS_table(path=path)
        return df_tss
    else:
        raise Exception("The received 'species' is not a parameter within the specified genome reference range.")
        pass
    pass
    

def GeneBody_table(
species: Literal["hg38", "hg19", "mm10"] = "hg38"
):
    """
    Function: 
    
    Args:
        species
        
    Returns:
        r
    """
    species = species
    print("species value:",species)
    if species == "hg38":
        path = file_path_abs(file_path=hg38_gene_pos_path)
        df_genebody = read_GeneBody_table(path=path)
        return df_genebody
    elif species == "hg19":
        path = file_path_abs(file_path=hg19_gene_pos_path)
        df_genebody = read_GeneBody_table(path=path)
        return df_genebody
    elif species == "mm10":
        path = file_path_abs(file_path=mm10_gene_pos_path)
        df_genebody = read_GeneBody_table(path=path)
        return df_genebody
    else:
        raise Exception("The received 'species' is not a parameter within the specified genome reference range.")
        pass
    pass
'''

