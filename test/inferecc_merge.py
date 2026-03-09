#%%time
import os
import gzip
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
from sys import argv

# 禁用 SettingWithCopyWarning 警告  
pd.options.mode.chained_assignment = None  # 或 'raise' 表示引发异常  

# 画图参数
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
from matplotlib.backends.backend_pdf import PdfPages

import inferECC
from inferECC import *

### argv[1]：fragments path
### argv[2]：output dir
### argv[3]：species：hg19/hg38/mm10
fragments=argv[1]
outfile_dir=argv[2]
species=argv[3]
sample_num=int(argv[4])

# IO
#mtx_dir="C:/Users/wangyuhang/03.project/01.ecDNA/02.results/result_1k/CRC/GSM4861351_COLO320DM_rep1_atac_fragments.tsv.gz"
mtx_dir=outfile_dir
os.chdir(mtx_dir)
print(os.getcwd())


# 经过 ks uniform 检测过的 mtx
mtx_raw_fi_ks = pd.read_csv("./ks_uniform_qsub/cellXecDNA_fi.matrix.tsv",sep="\t")
# 包含 coverage<6 部分的mtx
cell_coverage = pd.read_csv("./cell_coverage.matrix.tsv",sep="\t")

# 将 cell_coverage 按照 mtx_raw_fi_ks 的 chr_100k、barcode 过滤 
cc_dd = cell_coverage.drop_duplicates(subset=['barcode','chr_100k'])
cc_dd_ch = cc_dd[cc_dd["chr_100k"].isin(mtx_raw_fi_ks["chr_100k"].unique())]
cc_dd_ch_bc = cc_dd_ch[cc_dd_ch["barcode"].isin(mtx_raw_fi_ks["barcode"].unique())]
cc_dd_ch_bc.to_csv("cellXecDNA_before_merge_long.matrix.tsv",sep="\t",index=True)

cxe_mtx = cc_dd_ch_bc[["barcode","chr_100k","Coverage","chrom","start_100k"]]
cxe_mtx_sv = cxe_mtx.sort_values(by=["chrom","start_100k"],ascending=(True,True))
# cxe_mm 稠密矩阵
cxe_mm = pd.pivot_table(
        cxe_mtx,
        index=["barcode"],
        columns=["chr_100k"],
        values=["Coverage"],
        fill_value=0
    )
cxe_mm.columns = cxe_mm.columns.to_frame().chr_100k.to_list()
cxe_mm.columns.name = "chr_100k"


g = sns.clustermap(data=cxe_mm.corr(),
                   col_cluster=True,
                   row_cluster=True
               )
print(os.getcwd())
g.savefig("p07_cxe_mm_correlation.png")


import anndata
import scanpy as sc
adata = sc.AnnData(X = cxe_mm,
                   obs = cxe_mm.index.to_frame(),
                   var = cxe_mm.columns.to_frame())
#sc.pp.filter_cells(adata, min_genes=3) 
sc.pp.filter_genes(adata, min_cells=3)

ad_var = adata.var.copy()
ad_var["chrom"] = ad_var.chr_100k.str.split(':',expand=True)[0]
ad_var["start_100k"] = ad_var.chr_100k.str.split(':',expand=True)[1].str.split('_',expand=True)[0]
ad_var['start_100k'] = ad_var['start_100k'].astype(int)
ad_var["end_100k"] = ad_var.chr_100k.str.split(':',expand=True)[1].str.split('_',expand=True)[1]
ad_var['end_100k'] = ad_var['end_100k'].astype(int)

ad_var_nb = ad_var.groupby(ad_var["chrom"]).apply(Neighbor)
ad_var_nb.index = ad_var_nb.chr_100k
ad_var_nb_ri = ad_var_nb.reindex(index=list(adata.var.index.values))
adata.var = ad_var_nb_ri
adata.var.to_csv("cellXecDNA_before_merge_meta.matrix.tsv",sep="\t",index=True)

adata_t = adata.T.copy()
adata_sum = sum_by(adata = adata_t, col = "neighbor")
adata_merge_df = adata_sum.to_df().T.copy()
adata_merge_df.to_csv("cellXecDNA_merge_df.matrix.tsv",sep="\t",index=True)

mdt = sns.clustermap(data=adata_merge_df.corr(),
                   col_cluster=True,
                   row_cluster=True
               )
print(os.getcwd())
mdt.savefig("p08_merge_ecdna_correlation.png")

print("Done!!!")

