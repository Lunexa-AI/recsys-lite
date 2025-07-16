"""
vector_recsys_lite: Fast, zero-dep SVD recommender + ANN benchmark (2 s on 100k MovieLens)

Features:
- Zero dependencies (NumPy only)
- SVD, kNN, and hybrid recommenders
- CLI and Python API
- Handles large, sparse datasets (100k x 10k+)
"""

__version__ = "0.1.4"

from .algo import RecommenderSystem, compute_mae, compute_rmse, svd_reconstruct, top_n
from .explain import visualize_svd
from .io import create_sample_ratings, load_ratings, save_ratings

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
    "visualize_svd",
]
