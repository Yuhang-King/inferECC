### v2:: 较于前版本v1，加入了: sample_num=int(argv[4])
### v3:: 较于前版本v2，加入（合并）了:：cell_coverage.py 步骤 
### v4:: 删除argv[4] 删除细胞抽样步骤，采用全部细胞
### v5:: 合并原本的 shell_1_2_3 步骤为T_I，适合集群多线程/多任务并行计算，100cells/task；
### v5:: 添加shell：T_II 将多线程/多任务结果合并，按照 library/sample 合并；
### v5:: 合并原本的 shell_4_5_6 步骤为T_III；最终输出格式为：cellXecDNA_merge_df.matrix.h5ad
### v6:: merge 步骤修正，修正为normalnize后的合并值
### V7:: 添加步骤，计算 chr100 neighbor_correlation
### V8:: 更改阈值，计算 chr100 neighbor_correlation cutoff 0.2
### v9:: 合并原本的 inferECC_Tutorials_5_I_12.py、inferECC_Tutorials_5_I_3.py、inferECC_Tutorials_8_III.py

#%%time
import os
import re
import gzip
import math
import random
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sys import argv

import inferECC
from inferECC import *

### argv[1]：species = argv[1] # species:: abbr:: hg19/hg38/mm10
### argv[2]：tumor_cell = argv[2] # int 0/1
### argv[3]：cell_number = argv[3] # "max"/int
### argv[4]：sample_name = argv[4] # character frag.file
### argv[5]：fragments_path = argv[5] # fragments_path:: path 
### argv[6]：outfile_path = argv[6] # outfile_dir-path:: dir-path 
### argv[7]：tumor_cell_tb_path = argv[7] # path:: tumor cell CB table(anno file)

species = argv[1] # species:: abbr:: hg19/hg38/mm10
tumor_cell = argv[2] # int 0/1
cell_number = argv[3] # "max"/int
sample_name = argv[4] # character frag.file
fragments_path = argv[5] # fragments_path:: path 
outfile_path = argv[6] # outfile_dir-path:: dir-path 
tumor_cell_tb_path = argv[7] # path:: tumor cell CB table(anno file)

######################################## step-I_1_2 ############################################
### IO
tumor_cell_tb_df = pd.read_csv(tumor_cell_tb_path,sep="\t")
tumor_cell_CB_list = list(tumor_cell_tb_df[tumor_cell_tb_df["frag.file"]==sample_name]["cellname"])

### run:
if(os.path.exists(outfile_path) != True): os.makedirs(outfile_path)
os.chdir(outfile_path)
print(os.getcwd())
df_fragments = read_bgi_as_dataframe(path=fragments_path)

### CB_list
if(int(tumor_cell)==1):
    df_fragments = df_fragments[df_fragments.barcode.isin(tumor_cell_CB_list)]
else:
    df_fragments = df_fragments[(df_fragments.barcode.isin(tumor_cell_CB_list))^True]
    pass

### Transform: 删除chrM
df_fragments = Transform(df_fragments=df_fragments,Delete_chrM_option=True)

### 片段长度分布：
# fragments_length density
# 耗时久，notebook 内运行建议关闭，
fragments_length(df_fragments,lim=False)

### 统计细胞片段数：
df_fragments_number_sort = caculate_fragments_number(df_fragments=df_fragments)
df_fragments_number_sort

### cutoff 过滤片段数量低的细胞：
df_fragments_cutoff = cutoff_fragments_number(df_fragments=df_fragments,
                                              cutoff_value=5000,
                                              df_fragments_number_sort=df_fragments_number_sort)

### 细胞抽样：
if(cell_number=="max"): #不抽样
    df_fragments_cutoff_sample = df_fragments_cutoff
else:
    sample_num = min(int(cell_number),len(df_fragments_cutoff["barcode"].unique())) #抽样
    df_fragments_cutoff_sample = sample_cell(df_fragments=df_fragments_cutoff,sample_number=sample_num)
    pass

### segmentation 片段分割:
df_fragments_cutoff_segmentation = fragments_segmentation(df_fragments=df_fragments_cutoff_sample)

### Normalize(计算覆盖度Coverage)：
#单线程
df_fragments_cutoff_normalize = df_fragments_cutoff_segmentation.groupby(df_fragments_cutoff_segmentation["barcode"]).apply(Normalize)

### v3：：较于前版本v2，加入（合并）了:：cell_coverage.py 步骤 
df_fragments_cutoff_normalize_dd = df_fragments_cutoff_normalize.drop_duplicates(subset=['barcode','chr_100k'])
df_fragments_cutoff_normalize_dd.to_csv("cell_coverage.matrix.tsv",sep="\t",index=True)

### 覆盖度Coverage分布密度图：
# coverage density plot
coverage_density(df_fragments_cutoff_normalize)

### cutoff：过滤片段覆盖度Coverage较低的片段
Coverage_cutoff = 6
df_02 = df_fragments_cutoff_normalize.copy()
df_02 = df_02[df_02["Coverage"] >= Coverage_cutoff]

### 计算片段是否符合均匀分布：
df_03 = caculate_uniform(df_fragments=df_02)

### 计算基因结构tss等富集分布：
df_04 = tss_site(df_fragments=df_03,species=species)
df_05 = tss_region(df_fragments=df_04,species=species)
df_05 = genebody_region(df_fragments=df_05,species=species)
df_05 = intergenic_region(df_fragments=df_05)

### tss区域富集可视化：
bp_from_tss(df_05)
bp_from_tss(df_05,lim=True)
df_06=df_05.copy()

### 基因结构分布可视化：
df_07 = df_06.groupby(df_06["barcode"]).apply(tss_score)
df_07_dd=df_07.drop_duplicates(subset=['barcode','chr_100k'])
enrichment_plot(df_07_dd,enrich_arg="tss",show=False)

df_08 = df_07.groupby(df_07["barcode"]).apply(genebody_score)
df_08_dd=df_08.drop_duplicates(subset=['barcode','chr_100k'])
enrichment_plot(df_08_dd,enrich_arg="genebody",show=False)
df_08.to_csv("cellXecDNA.matrix.tsv",sep="\t",index=True)

### 100kpb heatmap可视化:
heatmap_chr(df_08)
print("step-I_1_2 Done!!!")

######################################## step-I_3 ############################################
# 禁用 SettingWithCopyWarning 警告 
pd.options.mode.chained_assignment = None  # 或 'raise' 表示引发异常
from matplotlib.backends.backend_pdf import PdfPages

### argv[1]：fragments:: path
### argv[2]：outfile_dir:: dir
### argv[3]：species:: abbr:: hg19/hg38/mm10
### argv[4]：frag_index:: 100 cells frag index in single raw frag file
### argv[5]：tumor_cell_tb:: tumor cell anno file

### tss_score_pvalue = 0.05  ### argv::tss_score_pvalue
tss_score_pvalue = 0.05
### genebody_score_pvalue = 0.05  ### argv::genebody_score_pvalue
genebody_score_pvalue = 0.05
### uniform_distribution_pvalue = 0.05  ### argv::uniform_distribution_pvalue
uniform_distribution_pvalue = 0.01

###
# real input
outfile_path=outfile_path
### run:
#if(os.path.exists(outfile_dir) != True): os.makedirs(outfile_dir)
os.chdir(outfile_path)
print(os.getcwd())
#df = pd.read_csv("./cellXecDNA.matrix.tsv",sep="\t")
df_ch = df_08.copy()

### run:
uniform_dir = "ks_uniform_qsub"
if(os.path.exists(uniform_dir) != True): os.makedirs(uniform_dir)
os.chdir(uniform_dir)
print(os.getcwd())

### step1：过滤 tss enrichment chr_100k
# tss score
tss_score_pvalue = tss_score_pvalue  ### argv::tss_score_pvalue
df_ch_dd = df_ch.drop_duplicates(subset=['barcode','chr_100k'])
tss_score = df_ch_dd["tss_score"].values.tolist()
tss_score.sort()
tss_score_cutoff = tss_score[int(len(tss_score)*(1-tss_score_pvalue))]
print("tss_score_cutoff: " + str(tss_score_cutoff))
df_fi = df_ch[df_ch.tss_score <= tss_score_cutoff]

"""
# genebody score
genebody_score_pvalue = 0.05  ### argv::genebody_score_pvalue
df_ch_dd = df_ch.drop_duplicates(subset=['barcode','chr_100k'])
genebody_score = df_ch_dd["genebody_score"].values.tolist()
genebody_score.sort()
genebody_score_cutoff = genebody_score[int(len(genebody_score)*(1-genebody_score_pvalue))]
print("genebody_score_cutoff: " + str(genebody_score_cutoff)) 
df_fi = df_ch[df_ch.genebody_score <= tss_score_cutoff]
"""

import matplotlib
matplotlib.use("Agg")
from matplotlib.pyplot import plot,savefig

### step2：过滤前原始 heatmap
# 创建空列表 temp_df_list
temp_df_list = []
cb_list = df_fi.barcode.unique()
# 创建一个 PDF 文档
pdf_pages = PdfPages("heatmap_raw_plots.pdf")
for cb in cb_list:
    df_cb = df_fi[df_fi["barcode"]==cb]
    df_cb_ochh_mtx = ochh_mtx(df=df_cb)
    df_cb_ochh_mtx_uniform = ochh_mtx_ks_test(df = df_cb_ochh_mtx)
    
    if not df_cb_ochh_mtx_uniform.empty:
        g=heatmap_raw_plot(df=df_cb_ochh_mtx_uniform,cb=cb)
        pdf_pages.savefig()  # 保存当前热图到 PDF 文档
        pass
    else:
        print(cb+" dataframe is empty!")
        pass
    df_cb_ochh_mtx_uniform['chr_100k'] = df_cb_ochh_mtx_uniform.index
    df_cb_ochh_mtx_uniform['barcode'] = cb
    temp_df_list.append(df_cb_ochh_mtx_uniform)
    pass
# 关闭 PDF 文档
pdf_pages.close()
matplotlib.pyplot.close()
# 合并 temp_df_list
combined_df = pd.concat(temp_df_list, ignore_index=True)
combined_df.to_csv("ochh_raw.matrix.tsv",sep="\t",index=True)

### step3：过滤uniform后 heatmap
# 创建空列表 temp_df_list
#combined_df_fi = combined_df[combined_df["uniform_pvalue"]>=uniform_distribution_pvalue]
combined_df_fi = combined_df.copy()
combined_df_fi = combined_df_fi.set_index('chr_100k')
combined_df_fi["chr_100k"] = combined_df_fi.index
combined_df_fi_cb_list = combined_df_fi.barcode.unique()
# 创建一个 PDF 文档
pdf_fi_pages = PdfPages("heatmap_fi_plots.pdf")
for cb in combined_df_fi_cb_list:
    combined_df_fi_cb = combined_df_fi[combined_df_fi["barcode"]==cb]
    if not combined_df_fi_cb.empty:
        p=heatmap_fi_plot(df=combined_df_fi_cb,cb=cb)
        pdf_fi_pages.savefig()  # 保存当前热图到 PDF 文档
        pass
    else:
        print(cb+" dataframe is empty!")
        pass
    pass
# 关闭 PDF 文档
pdf_fi_pages.close()
matplotlib.pyplot.close()

## 过滤后 ochh mtx
combined_df_fi = combined_df[combined_df["uniform_pvalue"]>=uniform_distribution_pvalue]
combined_df_fi.to_csv("ochh_raw_fi.matrix.tsv",sep="\t",index=True)

## 过滤后 cellXecDNA_fi mtx
df_fi_dd = df_fi.drop_duplicates(subset=['barcode','chr_100k'])
uniform_pvalue_list = []
# 方法: 使用iloc遍历
for index in range(len(df_fi_dd)):
    row = df_fi_dd.iloc[index]
    combined_df_up_temp = combined_df[(combined_df["chr_100k"]==row.chr_100k)&(combined_df["barcode"]==row.barcode)].uniform_pvalue
    if len(combined_df_up_temp)==1 :
        up_temp = combined_df_up_temp.to_list()[0]
        pass
    else:
        print(row.barcode+row.chr_100k+"uniform_pvalue Error!")
        pass
    uniform_pvalue_list.append(up_temp)
    pass
df_fi_dd["uniform_pvalue"] = uniform_pvalue_list
df_fi_dd_uni_fi = df_fi_dd[df_fi_dd["uniform_pvalue"]>=uniform_distribution_pvalue]
df_fi_dd_uni_fi.to_csv("cellXecDNA_fi.matrix.tsv",sep="\t",index=True)

# 经过 ks uniform 检测过的 mtx
#mtx_raw_fi_ks = pd.read_csv(lib_dir+"/ks_uniform_qsub/cellXecDNA_fi.matrix.tsv",sep="\t")
mtx_raw_fi_ks = df_fi_dd_uni_fi.copy()
# 避免 barcode/chr_100k 同时存在于 index level 和列名导致 pivot_table 歧义
mtx_raw_fi_ks = mtx_raw_fi_ks.reset_index(drop=True)
### step 1
# cxe_mm 稠密矩阵
cxe_mm = pd.pivot_table(mtx_raw_fi_ks,
                        index=["barcode"],
                        columns=["chr_100k"],
                        values=["Coverage"],
                        fill_value=0)
cxe_mm.columns = cxe_mm.columns.to_frame().chr_100k.to_list()
cxe_mm.columns.name = "chr_100k"
### step 2
adata_sample = sc.AnnData(X = cxe_mm,
                          obs = cxe_mm.index.to_frame(),
                          var = cxe_mm.columns.to_frame())
adata_sample.obs["sample_raw"]=sample_name
sample = re.sub("(-fragments.tsv.gz|-atac_fragments.tsv.gz)$", "", sample_name)
# adata list
#adata_sample.obs["cancer"]=cancer
adata_sample.obs["sample"]=sample
adata_sample.write_h5ad("cellXecDNA_fi.matrix.h5ad")
### step 3
# 将adata.X中>=6的表达值替换为1
# 将adata.X中 <6的表达值替换为0
adata_sample.X = np.where(adata_sample.X >= 6, 1, adata_sample.X)
adata_sample.X = np.where(adata_sample.X < 6, 0, adata_sample.X)
adata_sample.write_h5ad("cellXecDNA_fi.matrix_01.h5ad")

print("I_3 Done!!!")

######################################## step-III_1 ############################################
### v5:: 合并原本的 shell_1_2_3 步骤为T_I，适合集群多线程/多任务并行计算，100cells/task；
### v5:: 添加shell：T_II 将多线程/多任务结果合并，按照 library/sample 合并；
### v5:: 合并原本的 shell_4_5_6 步骤为T_III；最终输出格式为：cellXecDNA_merge_df.matrix.h5ad
### v6:: merge 步骤修正，修正为normalnize后的合并值
### V7:: 添加步骤，计算 chr100 neighbor_correlation
### V8:: 更改阈值，计算 chr100 neighbor_correlation cutoff 0.2

################################## chr100k_merge ###########################
#%%time
import anndata
import scanpy as sc

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

### argv[1]：fragments_file name:: string 
### argv[2]：cancer_type_path_sub_frag:: dir
### argv[3]：cancer_type_path_merge_frag:: abbr:: dir
###
# real input
os.chdir(outfile_path)
print(os.getcwd())
# 经过 ks uniform 检测过的 mtx
#mtx_raw_fi_ks = pd.read_csv("./ks_uniform_qsub/cellXecDNA_fi.matrix.tsv",sep="\t")
mtx_raw_fi_ks = df_fi_dd_uni_fi.copy()
# 避免 barcode/chr_100k 同时存在于 index level 和列名导致后续筛选与透视歧义
mtx_raw_fi_ks = mtx_raw_fi_ks.reset_index(drop=True)
# 包含 coverage<6 部分的mtx
#cell_coverage = pd.read_csv("./cell_coverage.matrix.tsv",sep="\t")
cell_coverage = df_fragments_cutoff_normalize_dd.copy()

# 将 cell_coverage 按照 mtx_raw_fi_ks 的 chr_100k、barcode 过滤 
cc_dd = cell_coverage.drop_duplicates(subset=['barcode','chr_100k'])
cc_dd_ch = cc_dd[cc_dd["chr_100k"].isin(mtx_raw_fi_ks["chr_100k"].unique())]
cc_dd_ch_bc = cc_dd_ch[cc_dd_ch["barcode"].isin(mtx_raw_fi_ks["barcode"].unique())]
cc_dd_ch_bc.to_csv("cellXecDNA_before_merge_long.matrix.tsv",sep="\t",index=True)

cxe_mtx = cc_dd_ch_bc[["barcode","chr_100k","Coverage","chrom","start_100k"]].copy()
# 避免 barcode/chr_100k 同时存在于 index level 和列名导致 pivot_table 歧义
cxe_mtx = cxe_mtx.reset_index(drop=True)
cxe_mtx_sv = cxe_mtx.sort_values(by=["chrom","start_100k"],ascending=(True,True))
# cxe_mm 稠密矩阵
cxe_mm = pd.pivot_table(cxe_mtx,
                        index=["barcode"],
                        columns=["chr_100k"],
                        values=["Coverage"],
                        fill_value=0)
cxe_mm.columns = cxe_mm.columns.to_frame().chr_100k.to_list()
cxe_mm.columns.name = "chr_100k"

"""
import seaborn as sns
import pandas as pd
# 假设 cxe_mm 是您的数据
data = cxe_mm.corr()
# 使用列的平均值填充缺失值
data_filled = data.fillna(data.mean())
# 进行层次聚类
g = sns.clustermap(data=data_filled, col_cluster=True, row_cluster=True)
print(os.getcwd())
g.savefig("p07_cxe_mm_correlation.png")
"""

### step 2
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

# 计算chr100 Neighbor
ad_var_nb = ad_var.groupby(ad_var["chrom"]).apply(Neighbor)
ad_var_nb.index = ad_var_nb.chr_100k
ad_var_nb_ri = ad_var_nb.reindex(index=list(adata.var.index.values))
adata.var = ad_var_nb_ri
#adata.var.to_csv("cellXecDNA_before_merge_meta.matrix.tsv",sep="\t",index=True)

# 计算chr100 neighbor_correlation
### V8:: 更新 neighbor_correlation 函数参数默认阈值 neighbor_correlation cutoff 0.2
adata = neighbor_correlation(adata)
adata.var.to_csv("cellXecDNA_before_merge_meta_corr_cf0.2.matrix.tsv",sep="\t",index=True)

## 可视化，片段长度、correlation
df=adata.var.copy()
df["fragLen"]=df.neighbor_len
fragments_length(df,
                 plt_title = "Density plot for ecdna length",
                 file_name = "p01-1_ecdna_length_density.pdf",
                 label = "ecdna_length(Mbp)",
                 xlabel = "ecdna_length(Mbp)")
df["fragLen"]=df.correlation_neighbor_len
fragments_length(df,
                 plt_title = "Density plot for ecdna correlation_neighbor_len",
                 file_name = "p01-2_ecdna_correlation_neighbor_len_density.pdf",
                 label = "ecdna_correlation_neighbor_len(Mbp)",
                 xlabel = "ecdna_correlation_neighbor_len(Mbp)")
df["fragLen"]=df.correlation
fragments_length(df,
                 plt_title = "Density plot for ecdna neighbor_correlation",
                 file_name = "p01-3_ecdna_neighbor_correlation_density.pdf",
                 label = "ecdna_neighbor_correlation",
                 xlabel = "ecdna_neighbor_correlation")
df["fragLen"]=df.correlation_pvalue
fragments_length(df,
                 plt_title = "Density plot for ecdna neighbor_correlation_pvalue",
                 file_name = "p01-4_ecdna_neighbor_correlation_pvalue_density.pdf",
                 label = "ecdna_neighbor_correlation_pvalue",
                 xlabel = "ecdna_neighbor_correlation_pvalue")

adata_t = adata.T.copy()
adata_sum = sum_by(adata = adata_t, col = "correlation_neighbor")
adata_merge_df = adata_sum.to_df().T.copy()
adata_merge_df.to_csv("cellXecDNA_merge_df_corr_cf0.2.matrix.tsv",sep="\t",index=True)

"""
import seaborn as sns
import pandas as pd
# 假设 cxe_mm 是您的数据
data = adata_merge_df.corr()
# 使用列的平均值填充缺失值
data_filled = data.fillna(data.mean())
# 进行层次聚类
mdt = sns.clustermap(data=data_filled, col_cluster=True, row_cluster=True)
print(os.getcwd())
g.savefig("p08_merge_ecdna_correlation.png")
"""

print("III_4 Done!!!")

################################## chr_merge_2_gene ###########################
"""
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

###
# real input 
outfile_dir=outfile_dir
# IO
mtx_dir=outfile_dir
os.chdir(mtx_dir)
print(os.getcwd())
"""

# mtx:: chr_merge name
#ecc_df = pd.read_csv("./cellXecDNA_merge_df.matrix.tsv",sep="\t")
ecc_df = adata_merge_df.copy()

# mtx:: chr_100k::gene
gene_df = pd.read_csv("./cellXecDNA.matrix.tsv",sep="\t")
# mtx:: chr_merge::chr_100k
merge_df = pd.read_csv("./cellXecDNA_before_merge_meta_corr_cf0.2.matrix.tsv",sep="\t")

###### step 1
chr_merge_df = pd.DataFrame()
chr_merge_df["chr_merge"] = merge_df["correlation_neighbor"]
chr_merge_df["chr_100k"] = merge_df["chr_100k"]
chr_merge_df.to_csv("chr_merge_2_100k_cf0.2.tsv",sep="\t",index=True)

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

chr_100k_gene_df.to_csv("chr_100k_2_gene_cf0.2.tsv",sep="\t",index=True)

###### step 3
# 使用 merge 函数按照 df；A\B的共有列：chr_100k 列进行匹配，并将匹配的 genebody_region_gene 列添加到 DataFrame A
chr_merge_df_gene = pd.merge(chr_merge_df, chr_100k_gene_df, on='chr_100k', how='left')

# 删除包含空值（NaN）的行
chr_merge_df_gene_dropna = chr_merge_df_gene.dropna(subset=['genebody_region_gene'])

# 按照 'A' 列进行分组，将 'B' 列的列表合并为一个列表
chr_merge_df_gene_dropna_merge = chr_merge_df_gene_dropna.groupby('chr_merge')['genebody_region_gene'].agg(lambda x: sum(x, [])).reset_index()

# 使用 apply 函数对 genebody_region_gene 中的每个 list 进行 unique，并添加为新的列 gene_unique
chr_merge_df_gene_dropna_merge['gene_unique'] = chr_merge_df_gene_dropna_merge['genebody_region_gene'].apply(lambda x: list(set(x)))
chr_merge_df_gene_dropna_merge.to_csv("chr_merge_2_gene_cf0.2.tsv",sep="\t",index=True)

print("III_5 Done!!!")

################################## mtx_tsv2h5ad ###########################
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

###
# real input 
# IO
mtx_dir=outfile_path
os.chdir(mtx_dir)
print(os.getcwd())
#lib_path_temp = "./cellXecDNA_merge_df.matrix.tsv"
#merge_df = pd.read_csv(lib_path_temp,sep="\t",index_col=0)
merge_df = adata_merge_df.copy()

merge_df.columns.name = "ecdna_region"
adtmg = sc.AnnData(X = merge_df,
                   obs = merge_df.index.to_frame(),
                   var = merge_df.columns.to_frame())
#sc.pp.filter_cells(adtmg, min_genes=3) 
#sc.pp.filter_genes(adtmg, min_cells=3)
adtmg.write_h5ad("cellXecDNA_merge_df_cf0.2.matrix.h5ad")

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

print("III_6 Done!!!")
