import numpy as np
import pytest

from vector_recsys_lite import RecommenderSystem
from vector_recsys_lite.tools import (
    RecsysPipeline,
    grid_search,
    load_toy_dataset,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    train_test_split_ratings,
)


def test_load_toy_dataset():
    mat = load_toy_dataset("tiny_example")
    assert mat.shape == (4, 4)
    assert mat.dtype == np.float32
    with pytest.raises(ValueError):
        load_toy_dataset("invalid")


def test_precision_at_k():
    recs = [[1, 2, 3], [4, 5, 6]]
    actual = [{1, 3}, {4}]
    assert precision_at_k(recs, actual, k=2) == pytest.approx(0.5)


def test_recall_at_k():
    recs = [[1, 2, 3], [4, 5, 6]]
    actual = [{1, 3, 7}, {4, 8}]
    assert recall_at_k(recs, actual, k=3) == pytest.approx(0.5833333333333333)


def test_ndcg_at_k():
    recs = [[1, 2, 3], [4, 5, 6]]
    actual = [{1, 3}, {4, 6}]
    assert ndcg_at_k(recs, actual, k=3) > 0.7  # Adjusted approximate


def test_train_test_split_ratings():
    mat = np.random.rand(5, 5)
    mat[mat < 0.5] = 0
    train, test = train_test_split_ratings(mat, test_size=0.2)
    assert len(test) > 0
    assert np.all(train[test[0][0], test[0][1]] == 0)


def test_recsys_pipeline():
    class DummyTransformer:
        def transform(self, X):
            return X * 2

    pipe = RecsysPipeline([("trans", DummyTransformer()), ("rec", RecommenderSystem())])
    mat = np.random.rand(3, 3)
    pipe.fit(mat, k=2)
    preds = pipe.predict(mat)
    assert preds.shape == mat.shape
    recs = pipe.recommend(mat, n=2)
    assert len(recs) == 3


def test_grid_search():
    param_grid = {"k": [1, 2], "algorithm": ["svd"]}
    mat = np.random.rand(4, 4)
    mat[mat < 0.5] = 0
    result = grid_search(mat, param_grid, cv=2)
    assert "best_params" in result
    assert len(result["all_scores"]) == 2


@pytest.mark.parametrize("name", ["tiny_example", "small_movielens"])
def test_load_toy_dataset_param(name):
    mat = load_toy_dataset(name)
    assert mat.ndim == 2
    assert np.all(mat >= 0)


def test_precision_at_k_empty():
    assert precision_at_k([], [], 5) == 0.0
    assert precision_at_k([[1]], [set()], 1) == 0.0


def test_recall_at_k_all_relevant():
    recs = [[1, 2]]
    actual = [{1, 2}]
    assert recall_at_k(recs, actual, 2) == 1.0


def test_ndcg_at_k_perfect():
    recs = [[1, 2]]
    actual = [{1, 2}]
    assert ndcg_at_k(recs, actual, 2) == 1.0


def test_train_test_split_ratings_empty():
    mat = np.zeros((3, 3))
    train, test = train_test_split_ratings(mat)
    assert len(test) == 0


def test_recsys_pipeline_error():
    class BadStep:
        def fit(self, X):
            raise ValueError("Test error")

    pipe = RecsysPipeline([("bad", BadStep())])
    with pytest.raises(RuntimeError):
        pipe.fit(np.zeros((2, 2)))
