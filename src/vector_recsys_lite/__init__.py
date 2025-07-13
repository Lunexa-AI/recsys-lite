"""Vector recommender system plugin - Fast SVD-based collaborative filtering.

A lightweight, zero-dependency recommender system built on NumPy with optional
Numba JIT acceleration for high-performance collaborative filtering.

Features:
    - Truncated SVD matrix factorization
    - Top-N recommendation generation
    - Optional Numba JIT acceleration
    - Rich CLI interface
    - Comprehensive benchmarking tools

Example:
    >>> from vector_recsys_lite import svd_reconstruct, top_n
    >>> import numpy as np
    >>>
    >>> # Load your rating matrix (users × items)
    >>> ratings = np.random.rand(100, 50).astype(np.float32)
    >>>
    >>> # Factorize with rank-10 SVD
    >>> reconstructed = svd_reconstruct(ratings, k=10)
    >>>
    >>> # Get top-5 recommendations per user
    >>> recommendations = top_n(reconstructed, ratings, n=5)
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
