### v5:: 合并原本的 shell_1_2_3 步骤为T_I，适合集群多线程/多任务并行计算，100cells/task；
### v5:: 添加shell：T_II 将多线程/多任务结果合并，按照 library/sample 合并；
### v5:: 合并原本的 shell_4_5_6 步骤为T_III；最终输出格式为：cellXecDNA_merge_df.matrix.h5ad
### v6:: merge 步骤修正，修正为normalnize后的合并值
### V7:: 添加步骤，计算 chr100 neighbor_correlation
### V8:: 更改阈值，计算 chr100 neighbor_correlation cutoff 0.2

################################## inferecc_PDAC_ecc_rna_corr ###########################
import time
time_s = time.time()

import os
import gzip
import math
import random
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sys import argv

# 画图参数
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
from matplotlib.backends.backend_pdf import PdfPages

import warnings
# 忽略 FutureWarning 类型警告
warnings.simplefilter(action='ignore', category=FutureWarning)
# 忽略特定类型的警告：忽略 scanpy包中含有 ignore 的 UserWarning 类型警告
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")
# 禁用 pandas 包中的 SettingWithCopyWarning 类型警告  
pd.options.mode.chained_assignment = None  # 或 'raise' 表示引发异常

# 设置scanpy输出图片的分辨率
sc.set_figure_params(dpi=100)

# 引入 inferECC
import inferECC
from inferECC import *

### argv[1]：ecd_path name:: string 
### argv[2]：rna_path:: dir
### argv[3]：outfile_dir:: abbr:: dir
ecc_path = argv[1]
rna_path = argv[2]
outfile_dir = argv[3]
rna_number_max = int(argv[4]) #500
n_cpu = int(argv[5]) #40 / -1:all cpu / st.q 100
scrna_lib = argv[6]

###
if(os.path.exists(outfile_dir) != True): os.makedirs(outfile_dir)
os.chdir(outfile_dir)
print(os.getcwd())

### ecdna
#lib_dir = "D:/02.project/18.ecDNA/01.result/result_htan/htan_atac_merge0.2_add/PDAC/ht264p1-s1h2-atac_fragments.tsv.gz/"
#lib_dir = "D:/02.project/18.ecDNA/01.result/result_htan/htan_atac_merge0.2_add/PDAC/ht306p1-s1h1-atac_fragments.tsv.gz/"
lib_dir = ecc_path
adata_ecc = sc.read(lib_dir + "/cellXecDNA_merge_cf0.2_df_nor_chrright.matrix.h5ad")
adata_ecc.obs["sc_barcodes"] = adata_ecc.obs.index
adata_ecc.var["ecdna_region"] = adata_ecc.var.index
adata_ecc.var["ecdna_len"] = adata_ecc.var.ecdna_region.str.split(':', expand=True)[1].str.split('_', expand=True)[1].astype(int)-adata_ecc.var.ecdna_region.str.split(':', expand=True)[1].str.split('_', expand=True)[0].astype(int)
adata_ecc.var["ecdna_len"] = adata_ecc.var["ecdna_len"]/1000000
### scrna 细胞只保留eccdna细胞
# 筛选满足条件的基因
adata_ecc_sub = adata_ecc[:, adata_ecc.var_names[adata_ecc.var["ecdna_len"] >= 0.2]].copy()

### scrna
#lib_dir = "E:\\05.project\\04.ecDNA\\01.data\\04.htan_data\\Multiome\\Multiome_rna"
lib_dir = rna_path
#adata_rna = sc.read_10x_mtx(lib_dir,prefix="ht264p1-s1h2fc2a2n1z1_1bmn1-")
adata_rna = sc.read_10x_mtx(lib_dir,prefix=scrna_lib)
adata_rna.obs["sc_barcodes"]=adata_rna.obs.index
# scrna 细胞只保留eccdna细胞
adata_rna_sub = adata_rna[adata_rna.obs["sc_barcodes"].isin(adata_ecc_sub.obs["sc_barcodes"].tolist())].copy()
# 将稀疏矩阵转换为稠密矩阵
adata_rna_sub.X = adata_rna_sub.X.toarray()
adata_rna_sub.raw = adata_rna_sub.copy()
# 过滤低质量细胞
sc.pp.filter_cells(adata_rna_sub, min_genes=3)
sc.pp.filter_genes(adata_rna_sub, min_cells=rna_number_max)

# 假设adata_rna_sub和 adata_ecc_sub是AnnData对象，将它们的obs索引（cellID）对齐
############function::
def align_cell_ids(adata_rna_sub, adata_ecc_sub):
    # 获取cellID
    cell_ids_rna = adata_rna_sub.obs.index
    cell_ids_ecc = adata_ecc_sub.obs.index
    # 找到共同的cellID
    common_cell_ids = np.intersect1d(cell_ids_rna, cell_ids_ecc)
    # 筛选数据集
    adata_rna_aligned = adata_rna_sub[common_cell_ids]
    adata_ecc_aligned = adata_ecc_sub[common_cell_ids]
    return adata_rna_aligned, adata_ecc_aligned
##################
# 调用函数
aligned_rna, aligned_ecc = align_cell_ids(adata_rna_sub, adata_ecc_sub)

from joblib import Parallel, delayed
from scipy import stats
# 假设adata_rna_sub和adata_ecc_sub是AnnData对象，并且它们的obs索引（cellID）已经对齐
# adata_rna_sub::k cells X n genes
# adata_ecc_sub::k cells X m ecdnas
# 初始化一个n乘以m的零矩阵
#correlations = np.zeros((aligned_rna.shape[1], aligned_ecc.shape[1]))
# 定义一个函数来计算相关性
def compute_corr(i, j):
    expr_a = aligned_rna[:, aligned_rna.var.index[i]].X
    expr_b = aligned_ecc[:, aligned_ecc.var.index[j]].X
    corr, _ = stats.spearmanr(expr_a, expr_b)
    return corr

# 使用并行计算
results = Parallel(n_jobs=n_cpu)(delayed(compute_corr)(i, j) for i in range(aligned_rna.shape[1]) for j in range(aligned_ecc.shape[1]))
# 将结果转换为numpy数组并调整形状::n乘以m的矩阵
correlations = np.array(results).reshape(aligned_rna.shape[1], aligned_ecc.shape[1])
# 假设gene_names_rna和gene_names_ecc分别是两个adata的基因名
gene_names_rna = aligned_rna.var.index
gene_names_ecc = aligned_ecc.var.index
# 创建一个新的DataFrame
df_correlations = pd.DataFrame(correlations, index=gene_names_rna, columns=gene_names_ecc).T
df_correlations.to_csv('corr_ecc-rna.tsv', sep='\t',index=True)

# 使用seaborn的clustermap函数来画热图并聚类
sns.clustermap(df_correlations)
# 显示图像
#plt.show()
plt.savefig("corr_ecc-rna.pdf")

########################################## step 2 ##########################################
from sklearn.metrics import adjusted_rand_score, jaccard_score
from joblib import Parallel, delayed
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### corr_rna-rna_ari
# 初始化一个空的DataFrame来存储ARI值
ari_values = pd.DataFrame(index=df_correlations.columns, columns=df_correlations.columns)
# 定义一个函数来计算ARI值
def compute_ari(col1, col2):
    return adjusted_rand_score(df_correlations[col1] > 0, df_correlations[col2] > 0)
    #return jaccard_score(df_correlations[col1] > 0, df_correlations[col2] > 0)
# 使用并行计算
results = Parallel(n_jobs=n_cpu)(delayed(compute_ari)(col1, col2) for col1 in df_correlations.columns for col2 in df_correlations.columns)
# 将结果转换为numpy数组并调整形状::n乘以m的矩阵
ari_values = np.array(results).reshape(len(df_correlations.columns), len(df_correlations.columns))
# 将结果存为tsv文件
ari_values_df = pd.DataFrame(ari_values, index=df_correlations.columns, columns=df_correlations.columns)
ari_values_df.to_csv('corr_rna-rna_ari.tsv', sep='\t')
# 将数据类型转换为float
ari_values_df = ari_values_df.astype(float)
# 画出ARI值/jaccard系数矩阵的热图
# 使用seaborn的clustermap函数来画热图并聚类
sns.clustermap(ari_values_df)
#plt.show()
plt.savefig("corr_rna-rna_ari.pdf")

### corr_ecc-ecc_ari
df_correlations = df_correlations.T

# 初始化一个空的DataFrame来存储ARI值
ari_values = pd.DataFrame(index=df_correlations.columns, columns=df_correlations.columns)
# 定义一个函数来计算ARI值
def compute_ari(col1, col2):
    return adjusted_rand_score(df_correlations[col1] > 0, df_correlations[col2] > 0)
    #return jaccard_score(df_correlations[col1] > 0, df_correlations[col2] > 0)
# 使用并行计算
results = Parallel(n_jobs=n_cpu)(delayed(compute_ari)(col1, col2) for col1 in df_correlations.columns for col2 in df_correlations.columns)
# 将结果转换为numpy数组并调整形状::n乘以m的矩阵
ari_values = np.array(results).reshape(len(df_correlations.columns), len(df_correlations.columns))
# 将结果存为tsv文件
ari_values_df = pd.DataFrame(ari_values, index=df_correlations.columns, columns=df_correlations.columns)
ari_values_df.to_csv('corr_ecc-ecc_ari.tsv', sep='\t')
# 将数据类型转换为float
ari_values_df = ari_values_df.astype(float)
# 画出ARI值/jaccard系数矩阵的热图
# 使用seaborn的clustermap函数来画热图并聚类
sns.clustermap(ari_values_df)
#plt.show()
plt.savefig("corr_ecc-ecc_ari.pdf")


time_e = time.time()
print("RunningTime:" + str((time_e - time_s)/60) + " minutes")
print("ecc_rna_corr Done!!!")
