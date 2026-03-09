import os
from sys import argv
import pandas as pd

### argv[1]：fragments_file name:: string 
sample=argv[1]
cancer=argv[2]
path=argv[3]
os.chdir(path)
print(os.getcwd())

# 读取tsv文件
df = pd.read_csv('chr_merge_2_100k_cf0.2.tsv', sep='\t')
# 分割列
df['chr_merge_split'] = df['chr_merge'].str.split(':')
df['chr_100k_split'] = df['chr_100k'].str.split(':')
# 生成新的列
df['chr_merge_right'] = df['chr_100k_split'].str[0] + ':' + df['chr_merge_split'].str[1]
# 提取chr_merge和new_column两列
new_df = df[['chr_merge', 'chr_merge_right']]
# 根据new_column列进行去重
new_df = new_df.drop_duplicates(subset='chr_merge_right')
# 保存新的dataframe到tsv文件
new_df.to_csv('chr_2_right.tsv', sep='\t', index=False)

# 假设df3已经存在
df3 = pd.read_csv('cellXecDNA_merge_cf0.2_df_nor.matrix.tsv', sep='\t', index_col=0)
# 创建一个字典，键为new_df的chr_merge列的值，值为chr_merge_right列的值
rename_dict = pd.Series(new_df.chr_merge_right.values,index=new_df.chr_merge).to_dict()
# 使用字典的get方法和DataFrame的rename方法对df3的列名进行更新
df3.rename(columns=rename_dict, inplace=True)
df3.to_csv('cellXecDNA_merge_cf0.2_df_nor_chrright.matrix.tsv', sep='\t', index=True)

# 假设df2已经存在
df2 = pd.read_csv('chr_merge_2_gene_cf0.2.tsv', sep='\t', index_col=0)
# 使用字典的get方法和DataFrame的rename方法对df2的chr_merge列的值进行更新
df2['chr_merge'] = df2['chr_merge'].map(rename_dict)
# 保存新的dataframe到tsv文件
df2.to_csv('chr_merge_2_gene_cf0.2_chrright.tsv', sep='\t', index=True)

print("done.")