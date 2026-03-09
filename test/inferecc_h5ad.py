### inferecc_h5ad.py

#%%time
import os
import gzip
import math
import random
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

# 临时设置，忽略 FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

lib_dir="C:/Users/wangyuhang/03.project/01.ecDNA/02.results/result_1k/CRC/"
os.chdir(lib_dir)
print(os.getcwd())

def get_all_folders(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return folders

# 指定路径
target_path = lib_dir # '/your/target/directory'

# 获取所有文件夹
folders_list = get_all_folders(target_path)

# 打印结果
# print("所有文件夹列表:")
for folder in folders_list:
    print(folder)
    lib_path_temp = lib_dir+"/"+folder+"/"+"cellXecDNA_merge_df.matrix.tsv"
    merge_df = pd.read_csv(lib_path_temp,sep="\t",index_col=0)
    merge_df.columns.name = "ecdna_region"
    adtmg = sc.AnnData(X = merge_df,
                       obs = merge_df.index.to_frame(),
                       var = merge_df.columns.to_frame())
    #sc.pp.filter_cells(adtmg, min_genes=3) 
    #sc.pp.filter_genes(adtmg, min_cells=3)
    adtmg.write_h5ad(lib_dir+"/"+folder+"/"+"cellXecDNA_merge_df.matrix.h5ad")
    pass
    