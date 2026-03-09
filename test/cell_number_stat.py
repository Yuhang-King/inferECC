import warnings
import scanpy as sc

# 忽略来自 scanpy 的警告
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    #all_path = "D:/02.project/18.ecDNA/01.result/result_1k/CRC/"
    #all_path = "D:/02.project/18.ecDNA/01.result/result_htan/htan_atac_merge0.2_pdac/"
    #all_path = "D:/02.project/18.ecDNA/01.result/result_htan/htan_atac_merge0.2_s3/"
    all_path = "D:/02.project/18.ecDNA/01.result/result_htan/htan_atac_merge0.2_add/"
    #all_path = "D:/02.project/18.ecDNA/01.result/result_htan/htan_atac_merge/"
    cancer_list = get_all_folders(all_path)
    adata_list = list()
    for cancer in cancer_list:
        sample_list = get_all_folders(all_path+cancer)
        print(cancer)
        for sample in sample_list:
            lib_dir = all_path+cancer+"/"+sample
            print(lib_dir)
            #adata_sample = sc.read(lib_dir+"/ks_uniform_qsub/cellXecDNA_fi.matrix.h5ad")
            adata_sample = sc.read(lib_dir+"/cellXecDNA_merge_cf0.2_df_nor_chrright.matrix.h5ad")
            #adata_sample = sc.read(lib_dir+"/cellXecDNA_merge_df_nor.matrix.h5ad")
            # 筛选在大于3个细胞中存在的ecDNA
            #sc.pp.filter_genes(adata_sample, min_cells=3)
            #sc.pp.filter_cells(adata_sample, min_genes=3)
            adata_sample.obs["sample_raw"]=sample
            sample = re.sub("(-fragments.tsv.gz|-atac_fragments.tsv.gz)$", "", sample)
            # adata list
            adata_sample.obs["cancer"]=cancer
            adata_sample.obs["sample"]=sample
            # adata list
            adata_list.append(adata_sample)
            pass
        pass