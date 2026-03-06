"""
基因结构 关系函数
"""
import os
import gzip
import math
import random
import numpy as np
import pandas as pd

# ------------------------------------------  make Reference  ------------------------------------------ #

#TSS_region/gene_body识别函数：
def relation(
    interval1,
    interval2
):
    '''
    @msg: 判断两个区间的关系
    @param intervals {list} 一个二维数组，每一项代表一个区间
    @param interval1 {list} 第一个区间
    @param interval2 {list} 第二个区间
    @return: {int}  返回两个区间的关系，0:两个区间相等、1:两个区间相离、2:两个区间相交、3:两个区间为包含关系
    '''
    min1,max1=sorted(interval1)
    min2,max2=sorted(interval2)
    if(min1==min2 and max1==max2):return 0
    if(max1<min2 or max2<min1):return 1
    if(min1<min2<=max1<max2 or min2<min1<=max2<max1):return 2
    if(min1<=min2<=max2<=max1 or min2<=min1<=max1<=max2):return 3
    pass

def chr_relation(
    chr1,
    chr2,
    alternate_parameters
):
    '''
    @msg: 判断两个chr的关系
    @param chr {字符串} 一个字符串，例如，"chr8:127677756_129627654"
    @param chr1 {字符串} 第一个染色体片段区间
    @param chr2 {字符串} 第二个染色体片段区间
    @param alternate_parameters 备用参数，防止pandas\core\apply.py args参数数量的错误识别导致报错
    @return: {bool}  返回两个区间的关系，True:两个片段区间不相离、False:两个片段区间相离
    '''
    alternate_parameters="alternate_parameters"
    
    chr_index_1=chr1.split(":")[0]
    chr_index_2=chr2.split(":")[0]
    interval_1=int(chr1.split(":")[1].split("_")[0]),int(chr1.split(":")[1].split("_")[1])
    interval_2=int(chr2.split(":")[1].split("_")[0]),int(chr2.split(":")[1].split("_")[1])
    if(chr_index_1==chr_index_2 and relation(interval_1,interval_2)!=1):
        return True
    else:
        return False
    pass

def chr_raw_relation(
    chr_raw,
    chr_raw_ref,
    chr_relation,
    alternate_parameters):
    '''
    @msg: 判断chr_raw、chr_raw_ref的关系
    @param chr_raw {字符串} 一个字符串，例如，"chr8:127677756_129627654"
    @param chr_raw_ref {pandas.core.frame.DataFrame} TSS/gene_pos
    @param alternate_parameters 备用参数，防止pandas\core\apply.py args参数数量的错误识别导致报错
    @return: {bool,list}  bool：是否位于TSS/gene_pos：list：如果位于，则return对应的gene list
    '''
    alternate_parameters="alternate_parameters"
    
    chr_index=chr_raw.split(":")[0]
    chr_raw_ref_chr_index=chr_raw_ref[chr_raw_ref["chromosome"]==chr_index]
    chr_raw_ref_chr_index["chr_relation_TF"]=chr_raw_ref_chr_index["chr_raw"].apply(chr_relation,args=(chr_raw,alternate_parameters))
    
    if(chr_raw_ref_chr_index["chr_relation_TF"].sum()<1):
        chr_relation = 0
        gene_list = 0
        return chr_relation,gene_list
    else:
        chr_relation = chr_raw_ref_chr_index["chr_relation_TF"].sum()
        chr_raw_ref_chr_index_True = chr_raw_ref_chr_index[chr_raw_ref_chr_index["chr_relation_TF"]]
        gene_list = list(chr_raw_ref_chr_index_True.gene.unique())
        return chr_relation,gene_list
    pass

#fragment TSS富集函数：
#最小绝对值
def findMinAbs(array):
    '''
    @findMinAbs: 寻找array数字队列中绝对值最小的数。
    @param array {array} 一个数字队列
    @return: {int} 返回 array 数字队列中绝对值最小的数。
    '''
    if array == None or len(array) <= 0:
        print("输入参数不合理")
        return 0
    mins = 2**32
    i = 0
    while i < len(array):
        if abs(array[i]) < abs(mins):
            mins = array[i]
        i += 1
    return mins

# TSS富集分数
def Enrichment_around_TSS(
    chr_raw,
    chr_raw_ref,
    alternate_parameters
):
    '''
    @Enrichment_around_TSS: 计算 fragments 距离 TSS_regions 的距离
    @param chr_raw {chr} 一个字符串，例如，"chr8:127677756_129627654"
    @param chr_raw_ref {pandas.core.frame.DataFrame} 一个表格对象：TSS
    @param alternate_parameters 备用参数，防止pandas\core\apply.py args参数数量的错误识别导致报错
    @return: {int} 返回 fragments chr_raw 与 TSS_start point 的距离
    '''
    #备用参数：
    alternate_parameters="alternate_parameters"
    #获取chr信息：
    chr_index=chr_raw.split(":")[0]
    if(bool(chr_index in chr_raw_ref.chromosome.unique())==False):
        print("The TSS reference file does not contain this chromosome.")
        return -1,-1,-1
    else:
        chr_raw_ref_chr_index=chr_raw_ref[chr_raw_ref.loc[:,"chromosome"]==chr_index]
        #print(chr_raw)
        
        #获取TSS位点信息：
        work_df=chr_raw_ref_chr_index[["TSSstart","gene"]]
        #print(work_df)
        
        #计算fragment起止点距离同chr上的TSS距离：
        work_df.loc[:,"chr_raw_start"]=int(chr_raw.split(":")[1].split("_")[0])
        work_df.loc[:,"chr_raw_end"]=int(chr_raw.split(":")[1].split("_")[1])
        work_df.loc[:,"s_tss"]=work_df.loc[:,"chr_raw_start"]-work_df.loc[:,"TSSstart"]
        work_df.loc[:,"e_tss"]=work_df.loc[:,"chr_raw_end"]-work_df.loc[:,"TSSstart"]
        #判断fragment是否与各个TSS相交：
        work_df.loc[:,"s_tss_X_e_tss"]=work_df.loc[:,"s_tss"]*work_df.loc[:,"e_tss"]
        work_df.loc[(work_df.loc[:,"s_tss_X_e_tss"] <= 0), "get_tss_TF"] = True
        work_df.loc[(work_df.loc[:,"s_tss_X_e_tss"] > 0), "get_tss_TF"] = False
        
        #统计、判断该fragment是否与所有TSS相交，并输出该fragment距离TSS最小距离：
        value_counts=work_df.get_tss_TF.value_counts()
        #print(value_counts)
        if(True in value_counts.index):
            around_tss_position = 0
            chr_relation = value_counts[True]
            work_df_True = work_df[work_df.loc[:,"get_tss_TF"]]
            gene_list = list(work_df_True.gene.unique())
            return chr_relation,gene_list,around_tss_position
        else:
            chr_relation = 0
            gene_list = 0
            s_tss_abs_min = findMinAbs(list(work_df.loc[:,"s_tss"]))
            e_tss_abs_min = findMinAbs(list(work_df.loc[:,"e_tss"]))
            around_tss_position = findMinAbs(list((s_tss_abs_min,e_tss_abs_min)))
            return chr_relation,gene_list,around_tss_position
        pass
    pass

