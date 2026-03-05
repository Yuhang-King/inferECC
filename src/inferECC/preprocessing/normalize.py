"""
Functions to either scale single-cell data or normalize such that the row-wise sums are identical.
"""
import os
import gzip
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns

# ------------------------------------------ Log Normalize ------------------------------------------ #

#Normalize(计算覆盖度Coverage)：
def Normalize(Dataframe_Nor):
    """函数文档字符串"""
    # 函数代码块
    Dataframe_Nor["fragnum_raw"]=Dataframe_Nor.shape[0]
    Dataframe_Nor=Dataframe_Nor.groupby(Dataframe_Nor["chr_100k"]).apply(Coverage)
    Dataframe_Nor["Coverage"]=10000*Dataframe_Nor["fragnum_100k"]/Dataframe_Nor["fragnum_raw"]
    return Dataframe_Nor

def Coverage(Dataframe_Cov):
    """函数文档字符串"""
    # 函数代码块
    Dataframe_Cov["fragnum_100k"]=Dataframe_Cov.shape[0]
    Dataframe_Cov=Dataframe_Cov.groupby(Dataframe_Cov["chr_1k"]).apply(Coverage_1k)
    return Dataframe_Cov

def Coverage_1k(Dataframe_Cov_1k):
    """函数文档字符串"""
    # 函数代码块
    Dataframe_Cov_1k["fragnum_1k"]=Dataframe_Cov_1k.shape[0]
    return Dataframe_Cov_1k


# ------------------------------------------ Log Normalize_Multi ------------------------------------------ #
"""
import multiprocessing
from tqdm import tqdm

def Normalize_Multi(DF_Normalize,cpu_number=10):

    cpu_num = cpu_number
    
    ###该部分即把入参DF_Normalize拆分成10分，
    ###每一份放在一个list里面，把所有list结果统统放在data_list中
    barcode_list = DF_Normalize.barcode.unique()
    task_list = pd.Series(range(0, len(barcode_list)), index=barcode_list)
    job_num = min(cpu_num, len(task_list.index))
    task_group = pd.qcut(task_list, job_num, labels=range(0, job_num))
    #data_list
    data_list = []
    for i in range(0, job_num):
        data_list.append(list(task_group[task_group == i].index))
        pass
    #data_list --> Data_list
    Data_list = []
    for i in range(0, job_num):
        Data_list.append(DF_Normalize[DF_Normalize.barcode.isin(list(data_list[i]))])
        pass
    
    ###开启多线程，每一份同时调用Normalize函数
    mp = multiprocessing.Pool(job_num)
    mplist = []
    for i in range(0, job_num):
        mplist.append(
            mp.apply_async(
                func=Normalize,
                kwds={'Dataframe_Nor':Data_list[i]}))
    mp.close()
    mp.join()

    ###结果合并
    res = pd.DataFrame()
    df_res_list = []
    #df_res = pd.DataFrame()
    for result in tqdm(mplist):
        part_res = result.get()
        if len(part_res)>=1:
            res = pd.concat([res,part_res[1]])
            df_res_list.append(part_res[0])
            #df_res = pd.concat([df_res,part_res[0]],axis=1)
            pass
        pass
    
    df_res = pd.concat(df_res_list,axis=1)

    print('Normalize_Multi: FINISH!')
    return df_res,res
"""

"""
import time
start_time = time.time()
df_fragments_cutoff_Normalize,df_fragments_cutoff_Normalize_info = Normalize_Multi(df_fragments_cutoff)
end_time = time.time()
print('耗时{}分钟'.format((end_time-start_time)/60))
"""



