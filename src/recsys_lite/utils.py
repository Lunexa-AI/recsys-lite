from typing import Any

from scipy.sparse import issparse


def as_dense(matrix: Any) -> Any:
    """
    Convert a matrix to a dense numpy array if it is sparse, otherwise return as is.
    Args:
        matrix: np.ndarray or scipy.sparse matrix
    Returns:
        np.ndarray if input is sparse or already dense, otherwise returns input as-is
    """
    if issparse is not None and issparse(matrix):
        return matrix.toarray()
    return matrix


__all__ = ["as_dense"]
