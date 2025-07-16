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
    train, test = train_test_split_ratings(mat, test_size=0.2)[0]
    assert train.shape == mat.shape
    assert isinstance(test, list)


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
    train, test = train_test_split_ratings(mat)[0]
    assert train.shape == mat.shape
    assert test == []


def test_recsys_pipeline_error():
    class BadStep:
        def fit(self, X):
            raise ValueError("Test error")

    pipe = RecsysPipeline([("bad", BadStep())])
    with pytest.raises(RuntimeError):
        pipe.fit(np.zeros((2, 2)))


def test_train_test_split_ratings_stratified():
    mat = np.zeros((10, 10))
    mat[:5, :5] = 1
    mat[5:, 5:] = 2
    # Stratified split should preserve 1s and 2s in test set
    train, test = train_test_split_ratings(mat, test_size=0.2, stratified=True)[0]
    test_ratings = [r for _, _, r in test]
    assert set(test_ratings).issubset({1, 2})
    # Check at least one of each rating in test set
    assert 1 in test_ratings and 2 in test_ratings


def test_recsys_pipeline_shape_validation():
    class BadTransformer:
        def transform(self, X):
            return X[:, : X.shape[1] // 2]  # Change shape

    pipe = RecsysPipeline([("bad", BadTransformer()), ("rec", RecommenderSystem())])
    mat = np.random.rand(4, 4)
    with pytest.raises(RuntimeError, match="Shape mismatch after step"):
        pipe.fit(mat, k=2)
    with pytest.raises(RuntimeError, match="Shape mismatch after step"):
        pipe.predict(mat)


def test_grid_search_extra_params():
    param_grid = {"k": [1], "algorithm": ["svd"], "bias": [True, False]}
    mat = np.random.rand(4, 4)
    mat[mat < 0.5] = 0
    result = grid_search(mat, param_grid, cv=2)
    assert "best_params" in result
    assert "all_scores" in result


def test_load_pretrained_model():
    from vector_recsys_lite.tools import load_pretrained_model

    rec = load_pretrained_model("tiny_svd")
    assert rec.is_fitted()
    assert hasattr(rec, "predict")
    with pytest.raises(ValueError):
        load_pretrained_model("not_a_model")


def test_intra_list_diversity_and_coverage():
    from vector_recsys_lite.tools import coverage, intra_list_diversity

    recs = [[1, 2, 3], [2, 3, 4]]
    feats = np.eye(5)
    div = intra_list_diversity(recs, feats)
    assert 0.0 <= div <= 1.0
    cov = coverage(recs, 5)
    assert 0.0 <= cov <= 1.0
    # Edge cases
    assert intra_list_diversity([], feats) == 0.0
    assert coverage([], 5) == 0.0


def test_metrics_all_zero_ratings():
    recs = [[0, 0, 0]]
    actual = [{0}]
    assert precision_at_k(recs, actual, 3) == 1.0
    assert recall_at_k(recs, actual, 3) == 1.0
    assert ndcg_at_k(recs, actual, 3) >= 0.0


def test_metrics_large_k():
    recs = [[1, 2]]
    actual = [{1}]
    # k > number of items
    assert precision_at_k(recs, actual, 10) >= 0.0
    assert recall_at_k(recs, actual, 10) >= 0.0
    assert ndcg_at_k(recs, actual, 10) >= 0.0


def test_metrics_single_row_col():
    recs = [[1]]
    actual = [{1}]
    assert precision_at_k(recs, actual, 1) == 1.0
    assert recall_at_k(recs, actual, 1) == 1.0
    assert ndcg_at_k(recs, actual, 1) == 1.0


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide")
def test_grid_search_cv1_and_empty():
    param_grid = {"k": [1], "algorithm": ["svd"]}
    mat = np.zeros((4, 4))
    result = grid_search(mat, param_grid, cv=1)
    assert "best_params" in result
    assert result["best_score"] == 0.0


def test_train_test_split_ratings_too_few_nonzeros():
    mat = np.zeros((3, 3))
    mat[0, 0] = 1
    with pytest.raises(ValueError):
        train_test_split_ratings(mat, test_size=0.5)


def test_ndcg_at_k_idcg_zero():
    recs = [[1, 2, 3]]
    actual = [set()]
    assert ndcg_at_k(recs, actual, 3) == 0.0


def test_recsys_pipeline_empty_steps():
    with pytest.raises(ValueError):
        RecsysPipeline([])


def test_train_test_split_ratings_folds_gt_nonzeros():
    mat = np.zeros((3, 3))
    mat[0, 0] = 1
    with pytest.raises(ValueError):
        train_test_split_ratings(mat, folds=5)
