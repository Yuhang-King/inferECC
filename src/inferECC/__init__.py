# inferECC/__init__.py
__version__ = "1.0.0"

# --- IO ---
from .io.bgi import read_bgi_as_dataframe

# --- preprocessing ---
from .preprocessing.transform import Transform
from .preprocessing.segmentation import fragments_segmentation
from .preprocessing.normalize import Normalize

# --- main ---
from .main.caculate import caculate_fragments_number
from .main.cutoff import cutoff_fragments_number
from .main.sample import sample_cell
from .main.uniform import caculate_uniform
from .main.tss_site import tss_site
from .main.gene_structure import (
    tss_region,
    genebody_region,
    intergenic_region,
)
from .main.enrichment_score import (
    tss_score,
    genebody_score,
)
from .main.merge import (
    sum_by,
    sum_by_sparse,
    Neighbor,
    neighbor_correlation,
)

# --- visualization (可选：不建议默认导出所有plot函数) ---
from .visualization.tss_enrichment_plot import (
    enrichment_plot,
)
from .visualization.plotting import (
    bp_from_tss,
)
from .visualization.density import (
    fragments_length,
    coverage_density,
)
from .visualization.heatmap import (
    heatmap_chr,
    heatmap_chr_fi,
)
from .visualization.heatmap_row import (
    ochh_mtx,
    ochh_mtx_ks_test,
    heatmap_raw_plot,
    heatmap_fi_plot,
)

__all__ = [
  "read_bgi_as_dataframe",
  "Transform", "fragments_segmentation", "Normalize",
  "caculate_fragments_number", "cutoff_fragments_number", "sample_cell",
  "caculate_uniform",
  "tss_site", "tss_region", "genebody_region", "intergenic_region",
  "tss_score", "genebody_score",
  "Neighbor", "neighbor_correlation", "sum_by","sum_by_sparse",
  "ochh_mtx", "ochh_mtx_ks_test","coverage_density",
]

