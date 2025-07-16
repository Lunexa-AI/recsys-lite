"""
ML tooling utilities for vector-recsys-lite.
"""

from typing import Any, List, Optional

import numpy as np

from .algo import RecommenderSystem, compute_mae, compute_rmse, top_n

TOY_DATASETS = {
    "tiny_example": np.array(
        [
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [0, 0, 5, 4],
        ],
        dtype=np.float32,
    ),
    "small_movielens": np.array(
        [  # Truncated example
            [5, 4, 0, 0, 3],
            [0, 0, 5, 4, 0],
            [2, 0, 4, 5, 1],
        ],
        dtype=np.float32,
    ),
}


def load_toy_dataset(name: str = "tiny_example") -> np.ndarray:
    """
    Load a small toy dataset for teaching and testing.

    Args:
        name: Dataset name ('tiny_example', 'small_movielens')

    Returns:
        NumPy array of ratings matrix.
    """
    if name not in TOY_DATASETS:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(TOY_DATASETS.keys())}"
        )
    return TOY_DATASETS[name]


def precision_at_k(recs: list[list[int]], actual: list[set[int]], k: int) -> float:
    """
    Compute average precision@k.

    Args:
        recs: List of recommended item lists per user
        actual: List of sets of actual relevant items per user
        k: Top k to consider

    Returns:
        Mean precision@k
    """
    precisions = []
    for r, a in zip(recs, actual):
        relevant = sum(1 for item in r[:k] if item in a)
        precisions.append(relevant / k if k > 0 else 0)
    return np.mean(precisions)


def recall_at_k(recs: list[list[int]], actual: list[set[int]], k: int) -> float:
    """
    Compute average recall@k.
    """
    recalls = []
    for r, a in zip(recs, actual):
        relevant = sum(1 for item in r[:k] if item in a)
        recalls.append(relevant / len(a) if len(a) > 0 else 0)
    return np.mean(recalls)


def ndcg_at_k(recs: list[list[int]], actual: list[set[int]], k: int) -> float:
    """
    Compute average NDCG@k (Normalized Discounted Cumulative Gain).
    """

    def dcg(items: list[int], rel: set[int]) -> float:
        return sum(
            (1 / np.log2(i + 2) if item in rel else 0) for i, item in enumerate(items)
        )

    ndcgs = []
    for r, a in zip(recs, actual):
        if not a:
            ndcgs.append(0.0)
            continue
        dcg_val = dcg(r[:k], a)
        ideal = sorted([1 if i in a else 0 for i in range(len(r))], reverse=True)[:k]
        idcg = dcg(ideal, set(range(len(ideal))))
        ndcgs.append(dcg_val / idcg if idcg > 0 else 0)
    return np.mean(ndcgs)


def train_test_split_ratings(
    matrix: np.ndarray, test_size: float = 0.2, random_state: Optional[int] = None
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """
    Split ratings matrix into train/test by masking test ratings.

    Args:
        matrix: Input ratings matrix
        test_size: Fraction of non-zero ratings to mask for test
        random_state: Seed for reproducibility

    Returns:
        train_matrix, test_list (list of (user, item, rating))
    """
    if random_state is not None:
        np.random.seed(random_state)

    train = matrix.copy()
    non_zero = np.argwhere(matrix > 0)
    n_test = int(len(non_zero) * test_size)
    test_idx = np.random.choice(len(non_zero), n_test, replace=False)

    test = []
    for idx in test_idx:
        i, j = non_zero[idx]
        test.append((i, j, matrix[i, j]))
        train[i, j] = 0

    return train, test


class RecsysPipeline:
    def __init__(self, steps: List[tuple[str, Any]]):
        self.steps = steps

    def fit(self, X: np.ndarray, **fit_params) -> "RecsysPipeline":
        data = X
        for name, step in self.steps:
            if hasattr(step, "fit"):
                step.fit(data, **fit_params)
            data = step.transform(data) if hasattr(step, "transform") else data
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        data = X
        for name, step in self.steps:
            data = (
                step.predict(data) if hasattr(step, "predict") else step.transform(data)
            )
        return data

    def recommend(self, X: np.ndarray, n: int = 10) -> list[list[int]]:
        preds = self.predict(X)
        return top_n(preds, X, n=n)


def grid_search_k(
    matrix: np.ndarray,
    k_values: list[int],
    metric: str = "rmse",
    cv: int = 3,
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """
    Grid search for best k using cross-validation.

    Args:
        matrix: Ratings matrix
        k_values: List of k to try
        metric: 'rmse' or 'mae'
        cv: Number of folds
        random_state: Seed

    Returns:
        Dict with best_k and scores
    """
    scores = {k: [] for k in k_values}
    for _ in range(cv):
        train, test = train_test_split_ratings(
            matrix, test_size=1 / cv, random_state=random_state
        )
        for k in k_values:
            rec = RecommenderSystem().fit(train, k=k)
            preds = rec.predict(train)
            if metric == "rmse":
                score = np.mean(
                    [
                        compute_rmse(np.array([[preds[u, i]]]), np.array([[r]]))
                        for u, i, r in test
                    ]
                )
            elif metric == "mae":
                score = np.mean(
                    [
                        compute_mae(np.array([[preds[u, i]]]), np.array([[r]]))
                        for u, i, r in test
                    ]
                )
            else:
                raise ValueError(f"Unknown metric: {metric}")
            scores[k].append(score)
    mean_scores = {k: np.mean(v) for k, v in scores.items()}
    # For error metrics like rmse/mae, lower is better - use min
    best_k = min(mean_scores, key=lambda x: mean_scores[x])
    return {"best_k": best_k, "scores": mean_scores}
