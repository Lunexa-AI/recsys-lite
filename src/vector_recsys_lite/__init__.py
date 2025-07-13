"""
vector_recsys_lite: Fast, zero-dep SVD recommender + ANN benchmark (2 s on 100k MovieLens)

Features:
- Zero dependencies (NumPy only)
- SVD, kNN, and hybrid recommenders
- CLI and Python API
- Handles large, sparse datasets (100k x 10k+)
"""

__version__ = "0.1.4"

from .algo import svd_reconstruct, top_n
from .io import load_ratings

__all__ = [
    "__version__",
    "svd_reconstruct",
    "top_n",
    "load_ratings",
]
