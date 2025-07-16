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
    """
    if not recs or not actual or k <= 0:
        return 0.0
    precisions = []
    for r, a in zip(recs, actual):
        if not r or not a:
            precisions.append(0.0)
            continue
        relevant = sum(1 for item in r[:k] if item in a)
        precisions.append(relevant / k if k > 0 else 0)
    return float(np.nan_to_num(np.mean(precisions), nan=0.0))


def recall_at_k(recs: list[list[int]], actual: list[set[int]], k: int) -> float:
    """
    Compute average recall@k.
    """
    if not recs or not actual or k <= 0:
        return 0.0
    recalls = []
    for r, a in zip(recs, actual):
        if not r or not a:
            recalls.append(0.0)
            continue
        relevant = sum(1 for item in r[:k] if item in a)
        recalls.append(relevant / len(a) if len(a) > 0 else 0)
    return float(np.nan_to_num(np.mean(recalls), nan=0.0))


def ndcg_at_k(recs: list[list[int]], actual: list[set[int]], k: int) -> float:
    """
    Compute average NDCG@k (Normalized Discounted Cumulative Gain).
    """
    if not recs or not actual or k <= 0:
        return 0.0

    def dcg(items: list[int], rel: set[int]) -> float:
        return sum(
            (1 / np.log2(i + 2) if item in rel else 0) for i, item in enumerate(items)
        )

    ndcgs = []
    for r, a in zip(recs, actual):
        if not a or not r:
            ndcgs.append(0.0)
            continue
        dcg_val = dcg(r[:k], a)
        ideal = sorted([1 if i in a else 0 for i in range(len(r))], reverse=True)[:k]
        idcg = dcg(ideal, set(range(len(ideal))))
        ndcgs.append(dcg_val / idcg if idcg > 0 else 0)
    return float(np.nan_to_num(np.mean(ndcgs), nan=0.0))


def intra_list_diversity(
    recs: list[list[int]], item_features: Optional[np.ndarray] = None
) -> float:
    """
    Compute average intra-list diversity.
    If item_features provided, use cosine distance; else assume uniform.
    """
    diversities = []
    for r in recs:
        if len(r) < 2:
            diversities.append(0.0)
            continue
        if item_features is not None:
            feats = item_features[r]
            dist = 1 - (feats @ feats.T) / (
                np.linalg.norm(feats, axis=1)[:, np.newaxis]
                * np.linalg.norm(feats, axis=1)
            )
            diversities.append(np.mean(dist[np.triu_indices(len(r), k=1)]))
        else:
            diversities.append(1.0)  # Assume max diversity if no features
    return np.mean(diversities)


def coverage(recs: list[list[int]], total_items: int) -> float:
    """
    Fraction of unique items recommended.
    """
    unique = set()
    for r in recs:
        unique.update(r)
    return len(unique) / total_items if total_items > 0 else 0.0


def train_test_split_ratings(
    matrix: np.ndarray,
    test_size: float = 0.2,
    folds: int = 1,
    stratified: bool = False,
    random_state: Optional[int] = None,
) -> list[tuple[np.ndarray, list[tuple[int, int, float]]]]:
    """
    Split ratings matrix into train/test by masking test ratings.
    Support multiple folds for CV.
    """
    if folds is None:
        folds = 1
    if matrix.size == 0 or np.all(matrix == 0):
        return [(matrix.copy(), [])]
    if folds == 1:
        if random_state is not None:
            np.random.seed(random_state)
        train = matrix.copy()
        non_zero = np.argwhere(matrix > 0)
        n_test = int(len(non_zero) * test_size)
        if n_test == 0:
            return [(train, [])]
        test_idx = np.random.choice(len(non_zero), n_test, replace=False)
        test = []
        for idx in test_idx:
            i, j = non_zero[idx]
            test.append((i, j, matrix[i, j]))
            train[i, j] = 0
        return [(train, test)]
    else:
        non_zero = np.argwhere(matrix > 0)
        if len(non_zero) == 0:
            return [(matrix.copy(), []) for _ in range(folds)]
        np.random.shuffle(non_zero)
        fold_size = len(non_zero) // folds
        splits = []
        for f in range(folds):
            test_idx = non_zero[f * fold_size : (f + 1) * fold_size]
            train = matrix.copy()
            test = []
            for u, i in test_idx:
                r = train[u, i]
                test.append((u, i, r))
                train[u, i] = 0
            splits.append((train, test))
        return splits
    # For stratified: If true, bin ratings and sample proportionally (add logic)


class RecsysPipeline:
    def __init__(self, steps: List[tuple[str, Any]]):
        if not steps:
            raise ValueError("Pipeline must have at least one step")
        self.steps = steps

    def fit(self, X: np.ndarray, **fit_params) -> "RecsysPipeline":
        data = X
        for name, step in self.steps:
            try:
                if hasattr(step, "fit"):
                    step.fit(data, **fit_params)
                if hasattr(step, "transform"):
                    data = step.transform(data)
            except Exception as e:
                raise RuntimeError(f"Error in pipeline step '{name}': {str(e)}") from e
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        data = X
        for name, step in self.steps:
            try:
                if hasattr(step, "predict"):
                    data = step.predict(data)
                elif hasattr(step, "transform"):
                    data = step.transform(data)
                else:
                    raise AttributeError(f"Step '{name}' lacks predict or transform")
            except Exception as e:
                raise RuntimeError(f"Error in pipeline step '{name}': {str(e)}") from e
        return data

    def recommend(self, X: np.ndarray, n: int = 10) -> list[list[int]]:
        preds = self.predict(X)
        return top_n(preds, X, n=n)


def grid_search(
    matrix: np.ndarray,
    param_grid: dict[str, list[Any]],
    metric: str = "rmse",
    cv: int = 3,
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """
    Grid search over parameter grid.
    param_grid example: {'k': [10,20], 'algorithm': ['svd','als']}
    """
    from itertools import product

    METRIC_FUNCS = {"rmse": compute_rmse, "mae": compute_mae}
    if matrix.size == 0 or not param_grid or cv <= 0:
        return {"best_params": None, "best_score": None, "results": []}
    params_list = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())
    scores = []
    for params in params_list:
        param_dict = dict(zip(param_names, params))
        fold_scores = []
        for _ in range(cv):
            splits = train_test_split_ratings(matrix, 1 / cv, random_state)
            train, test = splits[0] if splits else (matrix.copy(), [])
            rec = RecommenderSystem(algorithm=param_dict.get("algorithm", "svd"))
            rec.fit(train, k=param_dict.get("k", 10))
            preds = rec.predict(train)
            if not test:
                score = 0.0
            else:
                score = np.mean(
                    [
                        METRIC_FUNCS[metric](np.array([[preds[u, i]]]), np.array([[r]]))
                        for u, i, r in test
                    ]
                )
            fold_scores.append(score)
        scores.append(np.mean(fold_scores))
    best_idx = int(np.argmin(scores)) if scores else 0
    return {
        "best_params": (
            dict(zip(param_names, params_list[best_idx])) if scores else None
        ),
        "best_score": scores[best_idx] if scores else None,
        "results": scores,
        "all_scores": scores,
    }


PRETRAINED_MODELS = {
    "tiny_svd": {
        "u": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "s": np.array([1.0, 0.5]),
        "vt": np.array([[0.5, 0.6], [0.7, 0.8]]),
        "k": 2,
        "global_bias": 3.0,
        "user_bias": np.array([0.1, -0.1]),
        "item_bias": np.array([0.2, -0.2]),
    },
}


def load_pretrained_model(name: str = "tiny_svd") -> RecommenderSystem:
    """
    Load pre-fitted model for instant use.
    """
    if name not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown model: {name}")
    rec = RecommenderSystem(algorithm="svd")
    rec._model = PRETRAINED_MODELS[name]
    rec._fitted = True
    return rec
