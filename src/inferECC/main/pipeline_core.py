import os
import gzip
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from sys import argv


#定义对象：inferECC
'''
class inferECC:
    """
    @define class inferECC, data type: pandas.core.frame.DataFrame
    @input:single cell ATAC sequencing fragments quantization matrix, 
    @including fragments with start and end site information, cell barcode information, and fragment counts.
    @for BGI data: Counts after removing duplicates,
    @for 10X data: Counts without de-duplicating.
    @row: one fragment per row,
    @col: meta data for fragments.
    @return: {pandas.core.frame.DataFrame} class inferECC
    @tech_pltaform: Technology platform,including "BGI", "10X"
    """
'''    
