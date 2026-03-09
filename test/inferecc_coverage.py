#%%time
import os
import gzip
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sys import argv

import inferECC
from inferECC import *

### argv[1]：fragments path
### argv[2]：output dir
### argv[3]：species：hg19/hg38/mm10
fragments=argv[1]
outfile_dir=argv[2]
species=argv[3]
sample_num=int(argv[4])

### run:
if(os.path.exists(outfile_dir) != True): os.makedirs(outfile_dir)
os.chdir(outfile_dir)
print(os.getcwd())
df_fragments = read_bgi_as_dataframe(path=fragments)

df_mtx = pd.read_csv("./cellXecDNA.matrix.tsv",sep="\t")

df_fragments = df_fragments[df_fragments.barcode.isin(df_mtx.barcode.unique())]

### Transform: 删除chrM
df_fragments = Transform(df_fragments=df_fragments,Delete_chrM_option=True)

### 统计细胞片段数：
df_fragments_number_sort = caculate_fragments_number(df_fragments=df_fragments)
df_fragments_number_sort

### cutoff 过滤片段数量低的细胞：
df_fragments_cutoff = cutoff_fragments_number(df_fragments=df_fragments,
                                              cutoff_value=5000,
                                              df_fragments_number_sort=df_fragments_number_sort)

### segmentation 片段分割:
df_fragments_cutoff_segmentation = fragments_segmentation(df_fragments=df_fragments_cutoff)


### new::
#Normalize(计算覆盖度Coverage)：
def Coverage_temp(Dataframe_Cov):
    """函数文档字符串"""
    # 函数代码块
    Dataframe_Cov["fragnum_100k"]=Dataframe_Cov.shape[0]
    #Dataframe_Cov=Dataframe_Cov.groupby(Dataframe_Cov["chr_1k"]).apply(Coverage_1k)
    return Dataframe_Cov
def Normalize_temp(Dataframe_Nor):
    """函数文档字符串"""
    # 函数代码块
    Dataframe_Nor["fragnum_raw"]=Dataframe_Nor.shape[0]
    Dataframe_Nor=Dataframe_Nor.groupby(Dataframe_Nor["chr_100k"]).apply(Coverage_temp)
    Dataframe_Nor["Coverage"]=10000*Dataframe_Nor["fragnum_100k"]/Dataframe_Nor["fragnum_raw"]
    return Dataframe_Nor

### Normalize(计算覆盖度Coverage)：
#单线程
df_fragments_cutoff_normalize = df_fragments_cutoff_segmentation.groupby(df_fragments_cutoff_segmentation["barcode"]).apply(Normalize_temp)

df_fragments_cutoff_normalize_dd = df_fragments_cutoff_normalize.drop_duplicates(subset=['barcode','chr_100k'])

df_fragments_cutoff_normalize_dd.to_csv("cell_coverage.matrix.tsv",sep="\t",index=True)

print("finish!")

