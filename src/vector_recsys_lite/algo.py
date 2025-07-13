"""
Core algorithms for vector-based recommender systems.

This module provides fast SVD-based matrix factorization algorithms
for collaborative filtering recommender systems.
"""

from typing import Any, Optional, Union

import joblib  # type: ignore[import-untyped]
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import svds

# Type aliases for better type checking
FloatMatrix = Union[np.ndarray, csr_matrix]
FloatMatrixOptional = Optional[FloatMatrix]

# Try to import Numba for optional acceleration
try:
    from numba import jit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Create a dummy jit decorator
    def jit(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func

        return decorator


class RecommenderSystem:
    """
    Fast SVD-based recommender system with optional Numba acceleration.

    This class provides a unified interface for collaborative filtering
    using truncated SVD matrix factorization with support for both
    dense and sparse matrices.
    """

    def __init__(
        self,
        algorithm: str = "svd",
        random_state: Optional[int] = None,
        use_sparse: bool = True,
    ) -> None:
        """
        Initialize the recommender system.

        Parameters
        ----------
        algorithm : str, default="svd"
            Algorithm to use: "svd" for truncated SVD.
        random_state : Optional[int], default=None
            Random seed for reproducible results.
        use_sparse : bool, default=True
            Whether to use sparse matrix operations.
        """
        self.algorithm = algorithm
        self.random_state = random_state
        self.use_sparse = use_sparse
        self._model: Optional[dict[str, Any]] = None
        self._fitted = False

        if algorithm not in ["svd"]:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._fitted

    def fit(self, ratings: FloatMatrix, k: Optional[int] = None) -> "RecommenderSystem":
        """
        Fit the recommender system to the rating matrix.

        Parameters
        ----------
        ratings : FloatMatrix
            Rating matrix of shape (n_users, n_items).
        k : Optional[int], default=None
            Number of singular values to use. If None, uses min(n_users, n_items).

        Returns
        -------
        RecommenderSystem
            Self for method chaining.
        """
        if self.algorithm == "svd":
            if issparse(ratings):
                self._model = self._fit_svd_sparse(ratings, k)
            else:
                self._model = self._fit_svd_dense(ratings, k)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        self._fitted = True
        return self

    def predict(self, ratings: FloatMatrix) -> FloatMatrix:
        """
        Generate predictions for the rating matrix.

        Parameters
        ----------
        ratings : FloatMatrix
            Rating matrix to predict for.

        Returns
        -------
        FloatMatrix
            Predicted ratings matrix.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        if self.algorithm == "svd":
            if self._model is None:
                raise RuntimeError("Model not fitted")
            return self._predict_svd(ratings)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def recommend(
        self, ratings: FloatMatrix, n: int = 10, exclude_rated: bool = True
    ) -> np.ndarray:
        """
        Generate top-N recommendations for users.

        Parameters
        ----------
        ratings : FloatMatrix
            Rating matrix.
        n : int, default=10
            Number of recommendations to generate.
        exclude_rated : bool, default=True
            Whether to exclude items the user has already rated.

        Returns
        -------
        np.ndarray
            Array of shape (n_users, n) containing item indices.
        """
        if n <= 0:
            raise ValueError("n must be positive")

        predictions = self.predict(ratings)

        if exclude_rated:
            if issparse(ratings):
                known = ratings.toarray()
            else:
                known = ratings

            # Use top_n function with known items masking
            return top_n(predictions, known, n=n)
        else:
            # Simple top-n without exclusion
            if issparse(predictions):
                predictions = predictions.toarray()

            # Get top-n items for each user
            result = np.argsort(predictions, axis=1)[:, -n:][:, ::-1]
            return result

    def _fit_svd_dense(self, ratings: np.ndarray, k: Optional[int]) -> dict[str, Any]:
        """Fit SVD model for dense matrices."""
        n_users, n_items = ratings.shape

        if k is None:
            k = max(1, min(n_users, n_items) // 4)

        # Compute SVD
        u, s, vt = np.linalg.svd(ratings.astype(np.float32), full_matrices=False)

        return {"u": u[:, :k], "s": s[:k], "vt": vt[:k, :], "k": k}

    def _fit_svd_sparse(self, ratings: csr_matrix, k: Optional[int]) -> dict[str, Any]:
        """Fit SVD model for sparse matrices."""
        n_users, n_items = ratings.shape

        if k is None:
            k = max(1, min(n_users, n_items) // 4)

        # Use scipy's svds for sparse matrices
        u, s, vt = svds(ratings.astype(np.float32), k=k, random_state=self.random_state)

        return {"u": u, "s": s, "vt": vt, "k": k}

    def _predict_svd(self, ratings: FloatMatrix) -> FloatMatrix:
        """Generate SVD predictions."""
        if self._model is None:
            raise RuntimeError("Model not fitted")

        model = self._model
        u = model["u"]
        s = model["s"]
        vt = model["vt"]

        # Reconstruct the matrix
        reconstructed = u @ (s[:, np.newaxis] * vt)

        if self.use_sparse and not issparse(reconstructed):
            return csr_matrix(reconstructed)
        elif not self.use_sparse and issparse(reconstructed):
            return reconstructed.toarray()
        else:
            return reconstructed

    def save(self, path: str) -> None:
        """Save model to file."""
        with open(path, "wb") as f:
            joblib.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "RecommenderSystem":
        """Load model from file."""
        with open(path, "rb") as f:
            result = joblib.load(f)
            if isinstance(result, cls):
                return result
            raise ValueError(f"Loaded object is not a {cls.__name__}")

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "algorithm": getattr(self, "algorithm", None),
            "use_sparse": getattr(self, "use_sparse", False),
        }

    def clone(self) -> "RecommenderSystem":
        """Return a new instance with the same parameters, but not fitted."""
        return RecommenderSystem(
            algorithm=self.algorithm,
            random_state=self.random_state,
            use_sparse=self.use_sparse,
        )

    def reset(self) -> None:
        """Reset the model to unfitted state."""
        self._model = None
        self._fitted = False


def svd_reconstruct(
    mat: FloatMatrix,
    *,
    k: Optional[int] = None,
    random_state: Optional[int] = None,
    use_sparse: bool = True,
) -> FloatMatrix:
    """
    Truncated SVD reconstruction for collaborative filtering.

    Performs singular value decomposition and reconstructs the matrix using
    the top-k singular values and vectors. This is the core algorithm for
    matrix factorization-based recommendation systems.

    Parameters
    ----------
    mat : FloatMatrix
        Rating matrix of shape (n_users, n_items) where 0 indicates unrated items.
        Can be dense numpy array or sparse scipy matrix.
    k : int, optional
        Rank of the SVD approximation. If None, uses min(n_users, n_items) // 4.
        Must be between 1 and min(n_users, n_items).
    random_state : int, optional
        Random seed for reproducible results.
    use_sparse : bool, default=True
        Whether to return sparse matrix for memory efficiency.

    Returns
    -------
    FloatMatrix
        Reconstructed rating matrix of same shape as input.

    Raises
    ------
    ValueError
        If k is invalid or matrix is not 2D.
    RuntimeError
        If SVD computation fails.

    Examples
    --------
    >>> import numpy as np
    >>> from vector_recsys_lite import svd_reconstruct
    >>>
    >>> # Create sample rating matrix
    >>> ratings = np.array([[5, 3, 0, 1], [0, 0, 4, 5]], dtype=np.float32)
    >>>
    >>> # Reconstruct with rank-2 SVD
    >>> reconstructed = svd_reconstruct(ratings, k=2)
    >>> print(reconstructed.shape)  # (2, 4)
    >>>
    >>> # Use sparse matrices for large datasets
    >>> from scipy import sparse
    >>> sparse_ratings = sparse.csr_matrix(ratings)
    >>> sparse_reconstructed = svd_reconstruct(sparse_ratings, k=2)
    """
    if mat.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got {mat.ndim}D")

    n_users, n_items = mat.shape
    if n_users == 0 or n_items == 0:
        raise ValueError("Matrix cannot have zero dimensions")

    # Auto-determine k if not provided
    if k is None:
        k = max(1, min(n_users, n_items) // 4)

    if not 1 <= k <= min(n_users, n_items):
        raise ValueError(
            f"k must be between 1 and min(n_users, n_items)={min(n_users, n_items)}, "
            f"got {k}"
        )

    try:
        if issparse(mat):
            # Use scipy's svds for sparse matrices
            u, s, vt = svds(mat.astype(np.float32), k=k, random_state=random_state)
        else:
            # Use numpy's svd for dense matrices
            u, s, vt = np.linalg.svd(mat.astype(np.float32), full_matrices=False)
            u, s, vt = u[:, :k], s[:k], vt[:k, :]

        # Reconstruct: U * S * V^T
        reconstructed = u @ (s[:, np.newaxis] * vt)

        # Convert to sparse if requested
        if use_sparse and not issparse(reconstructed):
            return csr_matrix(reconstructed)
        elif not use_sparse and issparse(reconstructed):
            return reconstructed.toarray()
        else:
            return reconstructed

    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"SVD computation failed: {e}") from e


@jit(cache=True, fastmath=True)  # type: ignore[misc]
def _top_n_numba(
    est: np.ndarray,
    known: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Numba-accelerated top-N selection with known item masking.

    This function is JIT-compiled for performance. It finds the top-N items
    for each user while excluding items they have already rated.

    Parameters
    ----------
    est : np.ndarray
        Estimated ratings matrix (n_users, n_items).
    known : np.ndarray
        Known ratings matrix where >0 indicates rated items.
    n : int
        Number of top items to return per user.

    Returns
    -------
    np.ndarray
        Array of shape (n_users, n) containing item indices.
    """
    n_users, n_items = est.shape
    output = np.empty((n_users, n), dtype=np.int32)

    for i in range(n_users):
        # Create a copy to avoid modifying original
        user_estimates = est[i].copy()

        # Mask known items with -inf so they won't be selected
        for j in range(n_items):
            if known[i, j] > 0.0:
                user_estimates[j] = -np.inf

        # Use argpartition for O(n) partial sort
        idx = np.argpartition(user_estimates, -n)[-n:]

        # Sort the top-n items in descending order
        sorted_indices = np.argsort(user_estimates[idx])[::-1]
        output[i] = idx[sorted_indices]

    return output


def top_n(est: FloatMatrix, known: FloatMatrix, *, n: int = 10) -> np.ndarray:
    """
    Get top-N items for each user, excluding known items.

    Parameters
    ----------
    est : FloatMatrix
        Estimated ratings matrix of shape (n_users, n_items).
    known : FloatMatrix
        Known ratings matrix where >0 indicates rated items.
    n : int, default=10
        Number of top items to return per user.

    Returns
    -------
    np.ndarray
        Array of shape (n_users, n) containing item indices.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    # Convert to dense for processing
    if issparse(est):
        est = est.toarray()
    if issparse(known):
        known = known.toarray()

    # Use Numba-accelerated function if available
    if HAS_NUMBA:
        result: np.ndarray = _top_n_numba(est, known, n)
        return result
    else:
        # Fallback implementation
        result = []
        for user_preds, user_known in zip(est, known):
            # Mask out known items
            masked_preds = user_preds.copy()
            masked_preds[user_known > 0] = -np.inf

            # Get top-n items
            top_indices = np.argsort(masked_preds)[::-1][:n]
            result.append(top_indices)

        return np.array(result)


def compute_rmse(predictions: FloatMatrix, actual: FloatMatrix) -> float:
    """
    Compute Root Mean Square Error between predictions and actual ratings.

    Parameters
    ----------
    predictions : FloatMatrix
        Predicted ratings matrix.
    actual : FloatMatrix
        Actual ratings matrix.

    Returns
    -------
    float
        RMSE value.
    """
    if predictions.shape != actual.shape:
        raise ValueError("Matrices must have same shape")

    # Convert to dense for computation
    if issparse(predictions):
        predictions = predictions.toarray()
    if issparse(actual):
        actual = actual.toarray()

    # Check for infinite values
    if np.any(np.isinf(predictions)) or np.any(np.isinf(actual)):
        raise ValueError("Matrix contains infinite values")

    # Check for NaN values
    if np.any(np.isnan(predictions)) or np.any(np.isnan(actual)):
        raise ValueError("Matrix contains NaN values")

    # Only consider non-zero (rated) items
    mask = actual > 0
    if not np.any(mask):
        return 0.0

    diff = predictions[mask] - actual[mask]
    return float(np.sqrt(np.mean(diff**2)))


def compute_mae(predictions: FloatMatrix, actual: FloatMatrix) -> float:
    """
    Compute Mean Absolute Error between predictions and actual ratings.

    Parameters
    ----------
    predictions : FloatMatrix
        Predicted ratings matrix.
    actual : FloatMatrix
        Actual ratings matrix.

    Returns
    -------
    float
        MAE value.
    """
    if predictions.shape != actual.shape:
        raise ValueError("Matrices must have same shape")

    # Convert to dense for computation
    if issparse(predictions):
        predictions = predictions.toarray()
    if issparse(actual):
        actual = actual.toarray()

    # Check for infinite values
    if np.any(np.isinf(predictions)) or np.any(np.isinf(actual)):
        raise ValueError("Matrix contains infinite values")

    # Check for NaN values
    if np.any(np.isnan(predictions)) or np.any(np.isnan(actual)):
        raise ValueError("Matrix contains NaN values")

    # Only consider non-zero (rated) items
    mask = actual > 0
    if not np.any(mask):
        return 0.0

    diff = np.abs(predictions[mask] - actual[mask])
    return float(np.mean(diff))


def benchmark_algorithm(
    algorithm: str, ratings: FloatMatrix, k: int = 10, n_runs: int = 3
) -> dict[str, Any]:
    """
    Benchmark algorithm performance and accuracy.

    Parameters
    ----------
    algorithm : str
        Algorithm to benchmark: "svd", "svd_sparse".
    ratings : FloatMatrix
        Rating matrix to test on.
    k : int, default=10
        Rank for matrix factorization.
    n_runs : int, default=3
        Number of runs for averaging.

    Returns
    -------
    dict[str, Any]
        Benchmark results including timing and memory usage.
    """
    import time

    import psutil

    results = {
        "algorithm": algorithm,
        "matrix_shape": ratings.shape,
        "sparsity": (
            1 - (np.count_nonzero(ratings) / ratings.size)
            if hasattr(ratings, "size")
            else 0
        ),
        "runs": [],
    }

    for run in range(n_runs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        if algorithm == "svd":
            reconstructed = svd_reconstruct(ratings, k=k)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Compute accuracy metrics
        if issparse(ratings):
            actual = ratings.toarray()
        else:
            actual = ratings

        if issparse(reconstructed):
            predictions = reconstructed.toarray()
        else:
            predictions = reconstructed

        rmse = compute_rmse(predictions, actual)
        mae = compute_mae(predictions, actual)

        results["runs"].append(
            {
                "run": run + 1,
                "time": end_time - start_time,
                "memory_peak": end_memory - start_memory,
                "rmse": rmse,
                "mae": mae,
            }
        )

    # Compute averages
    avg_time = np.mean([run["time"] for run in results["runs"]])
    avg_memory = np.mean([run["memory_peak"] for run in results["runs"]])
    avg_rmse = np.mean([run["rmse"] for run in results["runs"]])
    avg_mae = np.mean([run["mae"] for run in results["runs"]])

    results.update(
        {
            "avg_time": avg_time,
            "avg_memory_mb": avg_memory,
            "avg_rmse": avg_rmse,
            "avg_mae": avg_mae,
        }
    )

    return results
