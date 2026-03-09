################################## chr_merge_2_gene ###########################
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
#lib_dir="C:/Users/wangyuhang/03.project/01.ecDNA/02.results/result_1k/CRC/"
os.chdir(mtx_dir)
print(os.getcwd())

# mtx:: chr_merge name
ecc_df = pd.read_csv("./cellXecDNA_merge_df.matrix.tsv",sep="\t")
# mtx:: chr_100k::gene
gene_df = pd.read_csv("./cellXecDNA.matrix.tsv",sep="\t")
# mtx:: chr_merge::chr_100k
merge_df = pd.read_csv("./cellXecDNA_before_merge_meta.matrix.tsv",sep="\t")

###### step 1
chr_merge_df = pd.DataFrame()
chr_merge_df["chr_merge"] = merge_df["neighbor"]
chr_merge_df["chr_100k"] = merge_df["chr_100k"]
chr_merge_df.to_csv("chr_merge_2_100k.tsv",sep="\t",index=True)

gene_df_sub = gene_df[gene_df.chr_100k.isin(list(chr_merge_df.chr_100k.unique()))]
gene_df_sub_new = pd.DataFrame()
gene_df_sub_new["chr_100k"] = gene_df_sub["chr_100k"]
gene_df_sub_new["genebody_region_gene"] = gene_df_sub["genebody_region_gene"]
#gene_df_sub_new

###### step 2
#%%time 2 mins
# 根据条件筛选删除包含字符格式的0的行
df_filtered = gene_df_sub_new[gene_df_sub_new['genebody_region_gene'] != '0']
df = df_filtered.copy()

# 使用 apply 函数替换字符串中的方括号和单引号，并去除空格
df['genebody_region_gene'] = df['genebody_region_gene'].apply(lambda x: x.replace("[", "").replace("]", "").replace("'", "").replace(" ", ""))

# 使用 apply 函数将 'A' 列的字符串转换为列表
df['genebody_region_gene'] = df['genebody_region_gene'].apply(lambda x: x.split(','))

# 按照 'A' 列进行分组，将 'B' 列的列表合并为一个列表
chr_100k_gene_df = df.groupby('chr_100k')['genebody_region_gene'].agg(lambda x: sum(x, [])).reset_index()

chr_100k_gene_df.to_csv("chr_100k_2_gene.tsv",sep="\t",index=True)

###### step 3
# 使用 merge 函数按照 df；A\B的共有列：chr_100k 列进行匹配，并将匹配的 genebody_region_gene 列添加到 DataFrame A
chr_merge_df_gene = pd.merge(chr_merge_df, chr_100k_gene_df, on='chr_100k', how='left')

# 删除包含空值（NaN）的行
chr_merge_df_gene_dropna = chr_merge_df_gene.dropna(subset=['genebody_region_gene'])

# 按照 'A' 列进行分组，将 'B' 列的列表合并为一个列表
chr_merge_df_gene_dropna_merge = chr_merge_df_gene_dropna.groupby('chr_merge')['genebody_region_gene'].agg(lambda x: sum(x, [])).reset_index()

# 使用 apply 函数对 genebody_region_gene 中的每个 list 进行 unique，并添加为新的列 gene_unique
chr_merge_df_gene_dropna_merge['gene_unique'] = chr_merge_df_gene_dropna_merge['genebody_region_gene'].apply(lambda x: list(set(x)))
chr_merge_df_gene_dropna_merge.to_csv("chr_merge_2_gene.tsv",sep="\t",index=True)

print("Done!!!")

