"""
vector_recsys_lite: Fast, zero-dep SVD recommender + ANN benchmark (2 s on 100k MovieLens)

Features:
- Zero dependencies (NumPy only)
- SVD, kNN, and hybrid recommenders
- CLI and Python API
- Handles large, sparse datasets (100k x 10k+)
"""

__version__ = "0.1.4"

from .algo import svd_reconstruct, top_n, compute_rmse, compute_mae, RecommenderSystem
from .io import load_ratings, save_ratings, create_sample_ratings

__all__ = [
    "__version__",
    "svd_reconstruct",
    "top_n",
    "compute_rmse",
    "compute_mae",
    "load_ratings",
    "save_ratings",
    "create_sample_ratings",
    "RecommenderSystem",
]
