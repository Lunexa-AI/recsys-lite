"""
Educational explanations for recommender system concepts.
"""

from typing import Optional

import numpy as np
from scipy.sparse.linalg import svds

from .utils import as_dense


def visualize_svd(mat: np.ndarray, k: int, random_state: Optional[int] = None) -> None:
    """
    Visualize and explain SVD components.
    """
    mat_dense = as_dense(mat)
    u, s, vt = svds(mat_dense, k=k, random_state=random_state)
    print("SVD Explanation:")
    print("- U: User latent factors (shape:", u.shape, ")")
    print("Sample U (first 3 users x factors):")
    print(u[:3])
    print("\n- S: Singular values (importance of factors):")
    print(s)
    print("\n- V^T: Item latent factors (shape:", vt.shape, ")")
    print("Sample V^T (first 3 factors x items):")
    print(vt[:, :3])
    print("\nReconstruction: U * diag(S) * V^T approximates original matrix.")


def ascii_heatmap(matrix: np.ndarray, title: str = "Matrix Heatmap") -> None:
    """
    Print ASCII heatmap of matrix.
    """
    print(f"\n{title}:")
    max_val = np.max(matrix)
    chars = " .:-=+*#%@"
    for row in matrix:
        line = " ".join(chars[int(val / max_val * (len(chars) - 1))] for val in row)
        print(line)
