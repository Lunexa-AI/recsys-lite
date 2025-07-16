"""
vector_recsys_lite: Fast, zero-dep SVD recommender + ANN benchmark (2 s on 100k MovieLens)

Features:
- Zero dependencies (NumPy only)
- SVD, kNN, and hybrid recommenders
- CLI and Python API
- Handles large, sparse datasets (100k x 10k+)
"""

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("vector_recsys_lite")
    except PackageNotFoundError:
        __version__ = "unknown"
except ImportError:
    __version__ = "unknown"

from .algo import RecommenderSystem, compute_mae, compute_rmse, svd_reconstruct, top_n
from .explain import visualize_svd
from .io import create_sample_ratings, load_ratings, save_ratings
from .tools import (
    RecsysPipeline,
    grid_search,
    load_toy_dataset,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    train_test_split_ratings,
)

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
    "load_toy_dataset",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "train_test_split_ratings",
    "RecsysPipeline",
    "grid_search",
]
