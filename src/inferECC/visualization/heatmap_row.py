#%%time
import os
import gzip
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from sys import argv

# 禁用 SettingWithCopyWarning 警告
pd.options.mode.chained_assignment = None  # 或 'raise' 表示引发异常
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from typing_extensions import Literal

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

def ochh_mtx(
    df: pd.DataFrame,
    lim: Optional[bool] = False,
    file_name: Optional[str] = "ochh_mtx",
    show: Optional[bool] = False
):
    """
    输入：df：单个细胞 cell barcode 的长矩阵
    输出：df：单个细胞 cell barcode 100kbp X 100计数分布矩阵：
    输出简称：one_cell_100kbp_100stat_mtx: ochh_mtx
    行：100kbp
    列：1.2.3....100
    参考序列: chr:references: list(range(1,101))
    参考序列：列值全为0的一行（100个0）
    """
    df_cb = df.copy()
    df_cb_uni=df_cb[["barcode","start_1k","end_1k",
                 "start_100k","end_100k","chr_1k",
                 "chr_100k","fragnum_100k","fragnum_1k"]]
    df_cb_uni.loc[:,"index_num"]=(df_cb_uni.loc[:,"end_1k"]-df_cb_uni.loc[:,"start_100k"])/1000
    df_cb_uni.loc[:,"index_num"]=df_cb_uni.loc[:,"index_num"].astype(int)
    df_cb_uni_dd=df_cb_uni.drop_duplicates(subset=['chr_1k','chr_100k'])
    
    df_cb_uni_dd_tri=df_cb_uni_dd[["chr_100k","index_num","fragnum_1k"]]
    df_cb_uni_dd_tri_ref=pd.DataFrame(
        { 'chr_100k':["chr:references"]*100,
         'index_num': list(range(1,101)),  
         'fragnum_1k': [0]*100}
    )
    df_cb_uni_dd_tri_ref_con = pd.concat(
        [df_cb_uni_dd_tri, df_cb_uni_dd_tri_ref],
        axis=0
    )
    df_cb_uni_dd_tri_ref_con['index_num']=df_cb_uni_dd_tri_ref_con['index_num'].astype(int)
    #print(df_cb_uni_dd_tri_ref_con)
    df_mtx=pd.pivot_table(
        df_cb_uni_dd_tri_ref_con,
        index=["chr_100k"],
        columns=["index_num"],
        values=["fragnum_1k"],
        fill_value=0
    )
    return df_mtx.fragnum_1k.drop("chr:references")

from scipy import stats
def ochh_mtx_ks_test(
    df: pd.DataFrame,
    lim: Optional[bool] = False,
    file_name: Optional[str] = "ochh_mtx_ks_test",
    show: Optional[bool] = False,
):
    """
    输入：df：单个细胞 cell barcode 100kbp X 100计数分布矩阵
    输出：一维向量：100kbp ks-test 检测符合平均分布uniform的p-value
    输出简称：ochh_mtx_ks_p_value
    """
    df_ochh = df.copy()
    
    P_value = []
    for i in range(0,len(df_ochh)):
        row = df_ochh.iloc[i].to_numpy()
        # 示例数组 A
        A = np.array(list(range(1,101)))/100
        # 对应的元素重复次数
        repeats = np.array(list(row)).astype(int)
        # 使用repeat函数创建新数组 B
        B = np.repeat(A, repeats)
        # 使用K-S检验来检验均匀性
        kstest_result = stats.kstest(B, 'uniform')
        P_value.append(kstest_result.pvalue)
        pass
    df_ochh["uniform_pvalue"]=P_value
    return df_ochh

def heatmap_raw_plot(
    df: pd.DataFrame,
    cb: Optional[str] = "one cell",
    lim: Optional[bool] = False,
    file_name: Optional[str] = "heatmap_raw_plot",
    show: Optional[bool] = False,
):
    """heatmap_raw_plot"""
    df_mtx = df.copy()

    # 创建示例行标签
    row_labels = df_mtx['uniform_pvalue']
    # 定义三个阈值
    thresholds = [0.001, 0.01, 0.05]
    
    # 创建一个函数，根据阈值分类行标签
    def classify_row_colors(label):
        if label <= thresholds[0]:
            return 'Group 1'
        elif thresholds[0] < label <= thresholds[1]:
            return 'Group 2'
        elif thresholds[1] < label <= thresholds[2]:
            return 'Group 3'
        else:
            return 'Group 4'
    
    # 为行标签分类并指定颜色
    row_colors = [classify_row_colors(label) for label in row_labels]
    # 指定每个分组的颜色
    palette = {'Group 1': 'red', 'Group 2': 'tomato', 'Group 3': 'LightCoral', 'Group 4': 'grey'}
    # 创建调色板
    row_color_palette = [palette[x] for x in row_colors]
    
    # 创建一个自定义的连续颜色映射，渐变色：白色、橙色、红色
    cmap = colors.LinearSegmentedColormap.from_list("brw", ["white", "orange","red"], N=256)
    
    ### clustermap size 在函数内，heatmap size 在函数外
    #plt.figure(figsize=(80,df_mtx.shape[0]/5+2), dpi=80)
    
    g = sns.clustermap(data=df_mtx.iloc[:,:100],
                       figsize=(35, df_mtx.shape[0]/3.5+1.5), 
                       #annot=True,
                       #fmt="d",
                       linewidths=0.05,
                       #square=True,
                       linecolor="grey",
                       cmap=cmap,
                       row_colors=[row_color_palette],
                       dendrogram_ratio=(0.1, 0.1),
                       row_cluster=False,
                       col_cluster=False,
               )
    # 创建图例项
    legend_elements = [mpatches.Patch(color='grey', label='p_value:*'),
                       mpatches.Patch(color='LightCoral', label='p_value:**'),
                       mpatches.Patch(color='tomato', label='p_value:***'),
                       mpatches.Patch(color='red', label='p<0.001')]
    
    # 添加图例
    g.ax_heatmap.legend(handles=legend_elements, 
                        title='Uniform Distribution p_value', 
                        bbox_to_anchor=(-0.04, 1.0),  #调整legend位置
                        ncol=1, # 调整legend列数
                        #loc='lower left',
                       )
    
    # 调整行标签的宽和高
    #g.ax_row_dendrogram.set_yticks([])  # 隐藏默认行标签
    #g.ax_row_dendrogram.set_yticklabels([])  # 隐藏默认行标签
    
    # 调整clustermap图例位置和大小
    g.cax.set_position([.03, .1, .01, .45])
    
    # 添加 xlabel 和 title
    g.ax_heatmap.set_xlabel('index_number', fontsize=14)  # 添加 xlabel
    g.ax_heatmap.set_title(cb, fontsize=16)  # 添加 title
    
    # 显示图形
    #plt.show()
    #plt.savefig("heatmap_row_plot.pdf")
    return g

def heatmap_fi_plot(
    df: pd.DataFrame,
    cb: Optional[str] = "one cell",
    lim: Optional[bool] = False,
    file_name: Optional[str] = "heatmap_fi_plot",
    show: Optional[bool] = False,
):
    """heatmap_fi_plot"""
    df_mtx = df.copy()
    df_mtx=df_mtx.sort_values(by='uniform_pvalue')
    
    # 创建示例行标签
    row_labels = df_mtx['uniform_pvalue']
    # 定义三个阈值
    thresholds = [0.001, 0.01, 0.05]
    
    # 创建一个函数，根据阈值分类行标签
    def classify_row_colors(label):
        if label <= thresholds[0]:
            return 'Group 1'
        elif thresholds[0] < label <= thresholds[1]:
            return 'Group 2'
        elif thresholds[1] < label <= thresholds[2]:
            return 'Group 3'
        else:
            return 'Group 4'
    
    # 为行标签分类并指定颜色
    row_colors = [classify_row_colors(label) for label in row_labels]
    # 指定每个分组的颜色
    palette = {'Group 1': 'red', 'Group 2': 'LightCoral', 'Group 3': 'LightGray', 'Group 4': 'DimGrey'}
    #palette = {'Group 1': 'red', 'Group 2': 'tomato', 'Group 3': 'LightCoral', 'Group 4': 'grey'}
    # 创建调色板
    row_color_palette = [palette[x] for x in row_colors]
    
    # 创建一个自定义的连续颜色映射，渐变色：白色、橙色、红色
    cmap = colors.LinearSegmentedColormap.from_list("brw", ["white", "orange","red"], N=256)
    
    ### clustermap size 在函数内，heatmap size 在函数外
    #plt.figure(figsize=(80,df_mtx.shape[0]/5+2), dpi=80)
    
    g = sns.clustermap(data=df_mtx.iloc[:,:100],
                       figsize=(35, df_mtx.shape[0]/3.5+1.5), 
                       #annot=True,
                       #fmt="d",
                       linewidths=0.05,
                       #square=True,
                       linecolor="grey",
                       cmap=cmap,
                       row_colors=[row_color_palette],
                       dendrogram_ratio=(0.1, 0.1),
                       row_cluster=False,
                       col_cluster=False,
               )
    # 创建图例项
    legend_elements = [mpatches.Patch(color='DimGrey', label='p_value:*'),
                       mpatches.Patch(color='LightGray', label='p_value:**'),
                       mpatches.Patch(color='LightCoral', label='p_value:***'),
                       mpatches.Patch(color='red', label='p<0.001')]
    
    # 添加图例
    g.ax_heatmap.legend(handles=legend_elements, 
                        title='Uniform Distribution p_value', 
                        bbox_to_anchor=(-0.04, 1.0),  #调整legend位置
                        ncol=1, # 调整legend列数
                        #loc='lower left',
                       )
    
    # 调整行标签的宽和高
    #g.ax_row_dendrogram.set_yticks([])  # 隐藏默认行标签
    #g.ax_row_dendrogram.set_yticklabels([])  # 隐藏默认行标签
    
    # 调整clustermap图例位置和大小
    g.cax.set_position([.03, .1, .01, .45])
    
    # 添加 xlabel 和 title
    g.ax_heatmap.set_xlabel('index_number', fontsize=14)  # 添加 xlabel
    g.ax_heatmap.set_title(cb, fontsize=16)  # 添加 title
    
    # 显示图形
    #plt.show()
    #plt.savefig("heatmap_row_plot.pdf")
    return g
