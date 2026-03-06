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
from matplotlib.pyplot import rc_context

import warnings
# 忽略 FutureWarning 类型警告
warnings.simplefilter(action='ignore', category=FutureWarning)
# 忽略特定类型的警告：忽略 scanpy包中含有 ignore 的 UserWarning 类型警告
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")
# 禁用 pandas 包中的 SettingWithCopyWarning 类型警告  
pd.options.mode.chained_assignment = None  # 或 'raise' 表示引发异常

# 设置scanpy输出图片的分辨率
sc.set_figure_params(dpi=100)

########################################### step1::##########################################
def find_resolution(adata, n_clusters, tol=0.01, max_iter=100):
    # 初始化resolution的值
    min_res, max_res = 0.0, 1.0

    # 初始化最佳的resolution和聚类数量
    best_res = None
    best_clusters = np.inf

    for _ in range(max_iter):
        # 计算当前的resolution
        cur_res = (min_res + max_res) / 2.0

        # 使用当前的resolution进行聚类
        sc.tl.leiden(adata, resolution=cur_res)

        # 计算当前的聚类数量
        cur_clusters = adata.obs['leiden'].nunique()

        # 如果当前的聚类数量更接近目标聚类数量，就更新最佳的resolution和聚类数量
        if np.abs(cur_clusters - n_clusters) < np.abs(best_clusters - n_clusters):
            best_res = cur_res
            best_clusters = cur_clusters

        # 如果当前的聚类数量接近期望的数量，就返回当前的resolution
        if np.abs(cur_clusters - n_clusters) <= tol:
            return cur_res

        # 否则，根据当前的聚类数量调整resolution的值
        if cur_clusters < n_clusters:
            min_res = cur_res
        else:
            max_res = cur_res

    # 如果经过max_iter次迭代后，仍然没有找到满足条件的resolution，就返回最佳的resolution
    return best_res
"""
# 假设adata是你的AnnData对象
# 假设你期望的聚类数量是 4
resolution = find_resolution(adata, n_clusters=4)
"""

def get_highest_expr_top_mean_genes(adata, n_top=10):
    # 计算每个基因的平均表达量
    mean_expr = pd.DataFrame(adata.X.mean(axis=0), index=adata.var_names, columns=['mean_expr'])

    # 对平均表达量进行排序并选择前n_top个
    top_genes = mean_expr.sort_values(by='mean_expr', ascending=False).head(n_top)

    # 返回前n_top个基因的列表
    return top_genes.index.tolist()
    
"""
# 假设adata是你的AnnData对象
# 获取并打印前10个基因的列表
top_highest_n = get_top_highest_expr_genes(adata, n_top=10)
print(top10_genes)
"""

def get_highly_variable_top_mean_genes(adata, n_top=10):
    # 假设adata是你的AnnData对象
    # 计算高变异基因
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=np.inf, min_disp=0.25)
    
    # 获取高变异基因的布尔型掩码
    mask = adata.var['highly_variable']
    
    # 使用这个掩码来获取高变异基因的列表
    highly_variable_genes = adata.var[mask].index.tolist()
    
    # 计算每个高变异基因的平均表达量
    mean_expr = pd.DataFrame(adata[:, highly_variable_genes].X.mean(axis=0), index=highly_variable_genes, columns=['mean_expr'])
    
    # 对平均表达量进行排序并选择前10个
    top_genes = mean_expr.sort_values(by='mean_expr', ascending=False).head(n_top)
    
    # 返回前n_top个基因的列表
    return top_genes.index.tolist()

def get_highly_variable_top_disp_genes(adata, n_top=10):
    # 计算高变异基因
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=np.inf, min_disp=0.25)

    # 获取高变异基因的布尔型掩码
    mask = adata.var['highly_variable']

    # 使用这个掩码来获取高变异基因的列表
    highly_variable_genes = adata.var[mask]

    # 对dispersions进行排序并选择前n_top个
    top_genes = highly_variable_genes.sort_values(by='dispersions', ascending=False).head(n_top)

    # 返回前n_top个基因的列表
    return top_genes.index.tolist()


########################################### step2::##########################################
# DEG dataframe
def get_deg_df(rank_genes_groups_dic):
    result = rank_genes_groups_dic
    
    # 1.获取字典的键值对迭代器
    items_iterator = result.items()
    # 将迭代器转换为列表，并从第二个元素开始
    items_list = list(items_iterator)[1:]
    deg_df = pd.DataFrame()
    # 遍历从第二个键开始的子集
    for key, value in items_list:
        temp_df = pd.DataFrame(value)
        deg_df[key] = temp_df[temp_df.columns[0]]
        pass
    
    # 2.pvalue 对数转换
    deg_df["logpvals"] = -deg_df.pvals.apply(math.log10)
    deg_df["logpvals_adj"] = -deg_df.pvals_adj.apply(math.log10)
    #deg_df["logscores"] = -deg_df.scores.apply(math.log2)
    
    # 3.获取 cell type标签
    vs_label_temp = pd.DataFrame(result["names"]).columns.to_list()
    vs_label = [vs_label_temp[0]+".vs."+vs_label_temp[1]+":"+vs_label_temp[0]+"_up",
                vs_label_temp[0]+".vs."+vs_label_temp[1]+":"+vs_label_temp[1]+"_up"]
    
    # 4.尝试写循环筛选上下调基因分类赋值给 "up" 和 "down" 和 "nosig" 加入pvalue条件
    deg_df.loc[(deg_df.logfoldchanges>0.5)&(deg_df.pvals<0.05),'type']=vs_label[0]
    deg_df.loc[(deg_df.logfoldchanges<-0.5)&(deg_df.pvals<0.05),'type']=vs_label[1]
    deg_df.loc[(abs(deg_df.logfoldchanges)<=0.5)|(deg_df.pvals>=0.05),'type']='Not_significant'
    # 将'type'列的数据类型更改为category
    deg_df.type = deg_df.type.astype('category')
    
    return(deg_df)

# 绘制火山图
def volcano_plot(deg_df,features = "ecDNAs"):
    deg_df = deg_df.copy()
    
    # 5.设置火山图三种点的颜色
    colors = ["#01c5c4","#ff414d","#686d76"]
    sns.set_palette(sns.color_palette(colors))
    # 画图风格：ticks 无网格线
    sns.set_style("ticks")
    
    # 6.绘图
    ax=sns.scatterplot(x='logfoldchanges',
                       #x="scores",
                       y='logpvals',
                       #y="logpvals_adj",
                       data=deg_df,
                       hue='type', #颜色映射
                       edgecolor = None, #点边界颜色
                       s=10,#点大小
                       )
    # 标签
    ax.set_title("Differentially Upregulated "+features) 
    ax.set_xlabel("log2(foldchange)")
    ax.set_ylabel("-log10(pvalue)")
    ax.set_xlim(-abs(deg_df.logfoldchanges).max()*1.55,+abs(deg_df.logfoldchanges).max()*1.55)
    ax.set_ylim(0,deg_df.logpvals.max()*1.05)
    # 图例位置
    ax.legend(loc='center right',
              #bbox_to_anchor=(0.98,0.85),
              bbox_to_anchor=(1.35,0.55),
              fontsize=8,
              ncol=1)
    plt.show()


########################################### step3::##########################################
# 计算列 'b' 中每个元素列表的重叠p-value
from scipy.stats import fisher_exact

def calculate_overlap_pvalue(list1, list2):
    # 计算两个列表的交集
    overlap = set(list1) & set(list2)
    
    # 构建两个列表的 contingency table
    contingency_table = [
        [len(overlap), len(list1) - len(overlap)],
        [len(list2) - len(overlap), 0]  # 这里第二行不需要overlap的个数，因为已经包含在第一行中了
    ]
    
    # 进行 Fisher's Exact Test
    _, p_value = fisher_exact(contingency_table)
    
    return p_value

def self_Overlap_pvalue_Heatmap(result_df, self_keys="gene_unique_list",heatmap_index="type",font_size=1):
    df = result_df.copy()
    
    # 创建一个矩阵来存储 p-value
    matrix_size = len(df[self_keys])
    p_value_matrix = [[0] * matrix_size for _ in range(matrix_size)]
    
    # 填充矩阵中的 p-value
    for i in range(matrix_size):
        for j in range(i, matrix_size):
            p_value = calculate_overlap_pvalue(df[self_keys][i], df[self_keys][j])
            p_value_matrix[i][j] = p_value
            p_value_matrix[j][i] = p_value
    
    # 将 p-value 矩阵转换为 DataFrame
    p_value_df = pd.DataFrame(p_value_matrix, columns=df[heatmap_index], index=df[heatmap_index])
    
    # 绘制热图
    font_size = font_size
    sns.heatmap(p_value_df, 
                annot=False, 
                #cmap='viridis', 
                #cmap='YlGnBu',
                cmap='coolwarm',
                square=True,
                linewidths=1, linecolor='black',
                xticklabels=font_size, yticklabels=font_size,
                fmt=".3f")
    plt.title('Overlap p-value: '+ self_keys)
    plt.show()

from collections import Counter
# 定义频数统计函数
def count_element_frequency(element):
    return dict(Counter(element))
"""
# 对列 'c' 中的每个元素应用频数统计函数
df['frequency_counts'] = df['c'].apply(count_element_frequency)
"""
# 提取每个元素中频数最高的 2 个键
def top_n_keys(dictionary,n=100):
    sorted_keys = sorted(dictionary, key=dictionary.get, reverse=True)
    return sorted_keys[:n]
"""
# 创建新的列 'top_keys'，该列包含每个元素中频数最高的 2 个键
df['top_keys'] = df['frequency_counts'].apply(top_2_keys)
"""

