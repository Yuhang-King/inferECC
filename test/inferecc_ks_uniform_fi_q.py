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

# 禁用 SettingWithCopyWarning 警告 
pd.options.mode.chained_assignment = None  # 或 'raise' 表示引发异常
from matplotlib.backends.backend_pdf import PdfPages 

### argv[1]：mtx path 
### argv[2]：output dir
### argv[3]：uniform_distribution_pvalue

### tss_score_pvalue = 0.05  ### argv::tss_score_pvalue
tss_score_pvalue = 0.05
### genebody_score_pvalue = 0.05  ### argv::genebody_score_pvalue
genebody_score_pvalue = 0.05
### uniform_distribution_pvalue = 0.05  ### argv::uniform_distribution_pvalue
uniform_distribution_pvalue = 0.01

mtx=argv[1]
outfile_dir=argv[2]

### run:
#if(os.path.exists(outfile_dir) != True): os.makedirs(outfile_dir)
os.chdir(outfile_dir)
print(os.getcwd())
df = pd.read_csv("./cellXecDNA.matrix.tsv",sep="\t")
df_ch = df.copy()

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
combined_df_fi = combined_df
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

print("Done!!!")
