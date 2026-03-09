import os
from sys import argv

### argv[1]：fragments_file name:: string 
sample=argv[1]
cancer=argv[2]
path=argv[3]
os.chdir(path)
print(os.getcwd())

################################## mtx_tsv2h5ad ###########################
### inferecc_h5ad.py
#%%time
import os
import gzip
#import math
#import random
#import numpy as np
import pandas as pd
import scanpy as sc
#import seaborn as sns
#import matplotlib.pyplot as plt

# 临时设置，忽略 FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

###
# real input 
#outfile_dir=outfile_dir

# IO
#mtx_dir=outfile_dir
#os.chdir(mtx_dir)
#print(os.getcwd())
#lib_path_temp = "./cellXecDNA_merge_df.matrix.tsv"
merge_df = pd.read_csv("cellXecDNA_merge_cf0.2_df_nor_chrright.matrix.tsv",sep="\t",index_col=0)
#merge_df = adata_merge_df.copy()

merge_df.columns.name = "ecdna_region"
adtmg = sc.AnnData(X = merge_df,
                   obs = merge_df.index.to_frame(),
                   var = merge_df.columns.to_frame())
#sc.pp.filter_cells(adtmg, min_genes=3) 
#sc.pp.filter_genes(adtmg, min_cells=3)
adtmg.write_h5ad("cellXecDNA_merge_cf0.2_df_nor_chrright.matrix.h5ad")

"""
### v6 修正：：
#merge_df = pd.read_csv(lib_path_temp,sep="\t",index_col=0)
tb0 = merge_df.T.copy()
tb = tb0.copy()
tb['ecdna'] = tb.index
tb['len']=(tb['ecdna'].str.split(':').str[1].str.split('_').str[1].astype(int)-tb['ecdna'].str.split(':').str[1].str.split('_').str[0].astype(int))/100000
# 删除'ecdna'列
tb = tb.drop('ecdna', axis=1)
# 将每一行的每个元素都除以该行的最后一个元素:len
tb = tb.apply(lambda row: row / row[-1], axis=1)
# 删除'len'列
tb = tb.drop('len', axis=1)
tb_nor = tb.T.copy()
tb_nor.to_csv("cellXecDNA_merge_cf0.2_df_nor.matrix.tsv",sep="\t",index=True)

merge_df = tb_nor.copy()
merge_df.columns.name = "ecdna_region"
adtmg = sc.AnnData(X = merge_df,
                   obs = merge_df.index.to_frame(),
                   var = merge_df.columns.to_frame())
#sc.pp.filter_cells(adtmg, min_genes=3) 
#sc.pp.filter_genes(adtmg, min_cells=3)
adtmg.write_h5ad("cellXecDNA_merge_cf0.2_df_nor.matrix.h5ad")
"""

print("III_6 Done!!!")
