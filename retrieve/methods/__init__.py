
from .align import *
from .set_similarity import *
from .vsm import *

try:
    import pyemd
    from .wmd import get_wmd, get_wmd_histograms
except ModuleNotFoundError:
    import warnings
    warnings.warn("Couldn't import `pyemd` module")
    
from .pairwise_chunked import pairwise_kernels_chunked
from .transportation import parallel_linear_sum_assignment
from .transportation import get_linear_sum_assignment_score
