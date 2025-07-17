"""
Educational explanations for recommender system concepts.
"""

from typing import Optional

import numpy as np
from scipy.sparse.linalg import svds

from .utils import as_dense


def visualize_svd(
    mat: np.ndarray, k: int, random_state: Optional[int] = None, plot: bool = False
) -> None:
    """
    Visualize and explain SVD components. If plot=True and matplotlib is available, show SVD plots.
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
    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(1, 3, figsize=(12, 3))
            axs[0].imshow(u, aspect="auto", cmap="viridis")
            axs[0].set_title("U (users x factors)")
            axs[1].bar(range(len(s)), s)
            axs[1].set_title("Singular values")
            axs[2].imshow(vt, aspect="auto", cmap="viridis")
            axs[2].set_title("V^T (factors x items)")
            plt.tight_layout()
            plt.show()
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Please install it with 'pip install matplotlib'."
            )


def ascii_heatmap(
    matrix: np.ndarray, title: str = "Matrix Heatmap", plot: bool = False
) -> None:
    """
    Print ASCII heatmap of matrix. If plot=True and matplotlib is available, show color heatmap.
    """
    print(f"\n{title}:")
    max_val = np.max(matrix)
    chars = " .:-=+*#%@"
    for row in matrix:
        line = " ".join(chars[int(val / max_val * (len(chars) - 1))] for val in row)
        print(line)
    if plot:
        try:
            import matplotlib.pyplot as plt

            plt.imshow(matrix, aspect="auto", cmap="viridis")
            plt.title(title)
            plt.colorbar()
            plt.show()
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Please install it with 'pip install matplotlib'."
            )
