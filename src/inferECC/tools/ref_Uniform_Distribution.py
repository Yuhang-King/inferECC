"""
Uniform_Distribution
"""
import os
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
#from anndata import AnnData
#from scipy.sparse import csr_matrix

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# ------------------------------------------  get Uniform_Distribution  ------------------------------------------ #

#reff=("/Uniform_Distribution/freq_df_2.8wx1w_cutoff.csv",)
#ud_cutoff_path=reff[0]

# file_path_abs
def file_path_abs(file_path: str):
    module_path = os.path.dirname(__file__)
    file_path_abs = module_path + file_path
    return file_path_abs

#path_abs = file_path_abs(file_path=ud_cutoff_path)

from inferECC.tools.resources import get_reference_path
path_abs = get_reference_path("freq_df_2.8wx1w_cutoff.csv")


def Uniform_Distribution_table(
    path: str = path_abs,
    exe: Optional[bool] = True,
) -> pd.DataFrame:
    
    if exe:
        Uniform_Distribution = pd.read_csv(path,sep = ",")
        return Uniform_Distribution
    pass



