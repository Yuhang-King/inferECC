# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# inferecc_pipeline.py

__author__ = "Yuhang Wang (biyuhangwang [at] mail.scut.edu.cn)"

import os
import argparse
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

import inferECC
from inferECC import *

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline for inferring ecDNA from single-cell ATAC-seq data / 单细胞ATAC-seq推断ecDNA分析流程")

    parser.add_argument("--species", required=True, choices=["hg19", "hg38", "mm10"],
                        help="Reference genome (e.g. hg19, hg38, mm10) / 参考基因组")
    parser.add_argument("--tumor_cell", type=int, choices=[0, 1], required=True,
                        help="Whether to analyze tumor cells: 1 for tumor, 0 for normal / 是否为肿瘤细胞")
    parser.add_argument("--cell_number", required=True,
                        help="Number of cells to sample (int) or 'max' for all / 抽样细胞数（整数）或'max'代表全部")
    parser.add_argument("--sample_name", required=True,
                        help="Sample name matching frag.file column in annotation / 样本名，匹配注释文件的 frag.file 列")
    parser.add_argument("--fragments_path", required=True,
                        help="Path to fragments.tsv.gz file / 片段文件路径")
    parser.add_argument("--outfile_path", required=True,
                        help="Directory for output files / 输出文件目录")
    parser.add_argument("--tumor_cell_tb_path", required=True,
                        help="Path to tumor/normal cell annotation file / 肿瘤/正常细胞注释表路径，一个tsv文件，列名为cellname的列为肿瘤细胞的cell barcode")

    return parser.parse_args()


def main():
    args = parse_args()

    # === I/O preparation / 输入输出准备 ===
    os.makedirs(args.outfile_path, exist_ok=True)
    os.chdir(args.outfile_path)
    print(f"[INFO] Output directory set to: {os.getcwd()}")

    # === Load tumor/normal cell barcode list / 读取细胞注释表 ===
    tumor_cell_tb_df = pd.read_csv(args.tumor_cell_tb_path, sep="\t")
    tumor_cell_CB_list = list(
        tumor_cell_tb_df[tumor_cell_tb_df["frag.file"] == args.sample_name]["cellname"]
    )

    # === Read and filter fragments / 读取并过滤片段 ===
    df_fragments = read_bgi_as_dataframe(path=args.fragments_path)

    if args.tumor_cell == 1:
        df_fragments = df_fragments[df_fragments.barcode.isin(tumor_cell_CB_list)]
    else:
        df_fragments = df_fragments[~df_fragments.barcode.isin(tumor_cell_CB_list)]

    # === Remove chrM and transform / 去除线粒体并转换 ===
    df_fragments = Transform(df_fragments=df_fragments, Delete_chrM_option=True)

    # === Plot fragment length distribution / 绘制片段长度密度图 ===
    fragments_length(df_fragments, lim=False)

    # === Count and filter low-fragment cells / 计算和过滤低覆盖细胞 ===
    df_fragments_number_sort = caculate_fragments_number(df_fragments)
    df_fragments_cutoff = cutoff_fragments_number(
                                            df_fragments, cutoff_value=5000, df_fragments_number_sort=df_fragments_number_sort
    )

    # === Cell sampling / 细胞抽样 ===
    if args.cell_number == "max":
        df_fragments_cutoff_sample = df_fragments_cutoff
    else:
        sample_num = min(int(args.cell_number), df_fragments_cutoff["barcode"].nunique())
        df_fragments_cutoff_sample = sample_cell(df_fragments_cutoff, sample_number=sample_num)

    # === Fragment segmentation / 片段分割 ===
    df_fragments_cutoff_segmentation = fragments_segmentation(df_fragments_cutoff_sample)

    # === Normalize coverage per cell / 归一化覆盖度计算 ===
    df_normalized = df_fragments_cutoff_segmentation.groupby("barcode").apply(Normalize)
    df_normalized_dd = df_normalized.drop_duplicates(subset=["barcode", "chr_100k"])
    df_normalized_dd.to_csv("cell_coverage.matrix.tsv", sep="\t", index=True)

    # === Plot coverage density / 覆盖度分布密度图 ===
    coverage_density(df_normalized)

    # === Filter low-coverage fragments / 过滤低覆盖度片段 ===
    df_filtered = df_normalized[df_normalized["Coverage"] >= 6]

    # === Uniformity test / 均匀性评估 ===
    df_uniform = caculate_uniform(df_fragments=df_filtered)

    # === Feature region enrichment / 基因结构区域富集计算 ===
    df_tss = tss_site(df_uniform, species=args.species)
    df_tss = tss_region(df_tss, species=args.species)
    df_tss = genebody_region(df_tss, species=args.species)
    df_tss = intergenic_region(df_tss)

    # === TSS enrichment visualization / TSS富集可视化 ===
    bp_from_tss(df_tss)
    bp_from_tss(df_tss, lim=True)

    # === Plot TSS score and other enrichments / 各类富集评分与可视化 ===
    df_tss_score = df_tss.groupby("barcode").apply(tss_score).drop_duplicates(subset=['barcode', 'chr_100k'])
    enrichment_plot(df_tss_score, enrich_arg="tss", show=False)

    df_genebody_score = df_tss.groupby("barcode").apply(genebody_score).drop_duplicates(subset=['barcode', 'chr_100k'])
    enrichment_plot(df_genebody_score, enrich_arg="genebody", show=False)
    df_genebody_score.to_csv("cellXecDNA.matrix.tsv", sep="\t", index=True)

    # === Final visualization: 100kb window heatmap / 可视化100kb窗口热图 ===
    heatmap_chr(df_genebody_score)

    print("[✔] ecDNA inference pipeline completed successfully.")

if __name__ == "__main__":
    main()
