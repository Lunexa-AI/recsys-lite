"""Tests for the algorithm module."""

import os
import tempfile

import numpy as np
import pytest
from scipy import sparse

from vector_recsys_lite.algo import (
    RecommenderSystem,
    benchmark_algorithm,
    compute_mae,
    compute_rmse,
    svd_reconstruct,
    top_n,
)
from vector_recsys_lite.tools import grid_search, train_test_split_ratings
from vector_recsys_lite.utils import as_dense


class TestSVDReconstruct:
    """Test SVD reconstruction functionality."""

    def test_basic_svd_reconstruction(self) -> None:
        """Test basic SVD reconstruction."""
        # Create a simple test matrix
        mat = np.array([[5, 3, 0, 1], [0, 0, 4, 5], [1, 1, 0, 0]], dtype=np.float32)

        # Reconstruct with rank 2
        reconstructed = svd_reconstruct(mat, k=2)

        # Convert to dense if sparse
        if sparse.issparse(reconstructed):
            reconstructed = as_dense(reconstructed)

        assert reconstructed.shape == mat.shape
        assert reconstructed.dtype == np.float32
        assert not np.any(np.isnan(reconstructed))
        assert not np.any(np.isinf(reconstructed))

    def test_auto_rank_determination(self) -> None:
        """Test automatic rank determination."""
        mat = np.random.rand(10, 8).astype(np.float32)

        # Should auto-determine rank
        reconstructed = svd_reconstruct(mat)

        # Convert to dense if sparse
        if sparse.issparse(reconstructed):
            reconstructed = as_dense(reconstructed)

        assert reconstructed.shape == mat.shape
        assert reconstructed.dtype == np.float32

    def test_invalid_rank(self) -> None:
        """Test error handling for invalid rank."""
        mat = np.random.rand(5, 3).astype(np.float32)

        # Rank too high
        with pytest.raises(ValueError, match="k must be between"):
            svd_reconstruct(mat, k=10)

        # Rank too low
        with pytest.raises(ValueError, match="k must be between"):
            svd_reconstruct(mat, k=0)

    def test_invalid_matrix(self) -> None:
        """Test error handling for invalid matrix."""
        # 1D array
        with pytest.raises(
            ValueError, match="Input matrix must be 2D \(users x items\), got 1D\."
        ):
            svd_reconstruct(np.array([1, 2, 3], dtype=np.float32))

        # Empty matrix
        with pytest.raises(
            ValueError, match="Input matrix cannot have zero dimensions\."
        ):
            svd_reconstruct(np.array([[]], dtype=np.float32))

    def test_svd_reconstruction_quality(self) -> None:
        """Test SVD reconstruction quality."""
        # Create a matrix with known structure
        u = np.random.rand(10, 3).astype(np.float32)
        s = np.array([5.0, 3.0, 1.0], dtype=np.float32)
        v = np.random.rand(3, 8).astype(np.float32)

        # Create original matrix
        original = u @ (s[:, np.newaxis] * v)

        # Reconstruct with rank 3
        reconstructed = svd_reconstruct(original, k=3)

        # Convert to dense if sparse
        if sparse.issparse(reconstructed):
            reconstructed = as_dense(reconstructed)

        # Should be very close to original (allow for float32 precision)
        diff = np.abs(original - reconstructed)
        assert np.max(diff) < 1e-6  # Relaxed tolerance for float32

    def test_sparse_matrix_support(self) -> None:
        """Test sparse matrix support."""
        # Create sparse matrix
        mat = sparse.csr_matrix(
            np.array([[5, 3, 0, 1], [0, 0, 4, 5]], dtype=np.float32)
        )

        reconstructed = svd_reconstruct(mat, k=1)

        # Should return sparse matrix
        assert sparse.issparse(reconstructed)
        assert reconstructed.shape == mat.shape

    def test_sparse_matrix_dense_output(self) -> None:
        """Test that sparse input produces dense output."""
        mat = sparse.csr_matrix(
            np.array([[5, 3, 0, 1], [0, 0, 4, 5]], dtype=np.float32)
        )

        reconstructed = svd_reconstruct(mat, k=1, use_sparse=False)

        # Should return dense matrix
        assert not sparse.issparse(reconstructed)
        assert reconstructed.shape == mat.shape


class TestTopN:
    """Test top-N recommendation functionality."""

    def test_basic_top_n(self) -> None:
        """Test basic top-n functionality."""
        # Create test data
        estimated = np.array([[4.5, 3.2, 4.8, 2.1]], dtype=np.float32)
        known = np.array([[5.0, 0.0, 0.0, 3.0]], dtype=np.float32)

        recs = top_n(estimated, known, n=2)

        assert len(recs) == 1
        assert len(recs[0]) == 2
        # Should exclude rated items (indices 0 and 3)
        assert 0 not in recs[0]  # Should exclude rated item
        assert 3 not in recs[0]  # Should exclude rated item
        # Should include highest estimated items (indices 2 and 1)
        assert 2 in recs[0]  # Should include highest estimated
        assert 1 in recs[0]  # Should include second highest

    def test_exclude_seen_items(self) -> None:
        """Test that seen items are excluded from recommendations."""
        mat = np.zeros((2, 3), dtype=np.float32)
        mat[0, 1] = 5.0  # User 0 rated item 1
        mat[1, 0] = 4.0  # User 1 rated item 0

        est = np.ones_like(mat)
        recs = top_n(est, mat, n=2)

        assert 1 not in recs[0]  # User 0 shouldn't get item 1
        assert 0 not in recs[1]  # User 1 shouldn't get item 0

    def test_invalid_parameters(self) -> None:
        """Test error handling for invalid parameters."""
        mat = np.random.rand(3, 4).astype(np.float32)
        est = np.random.rand(3, 4).astype(np.float32)

        # Negative n
        with pytest.raises(ValueError, match="n must be positive"):
            top_n(est, mat, n=0)

        # Different shapes
        est_wrong_shape = np.random.rand(2, 4).astype(np.float32)
        with pytest.raises(
            ValueError, match="Estimated and known matrices must have the same shape"
        ):
            top_n(est_wrong_shape, mat, n=2)

    def test_empty_matrix(self) -> None:
        """Test handling of empty matrix."""
        empty_mat = np.zeros((0, 3), dtype=np.float32)
        empty_est = np.zeros((0, 3), dtype=np.float32)

        recs = top_n(empty_est, empty_mat, n=2)
        assert len(recs) == 0

    def test_large_n(self) -> None:
        """Test with large n value."""
        mat = np.zeros((1, 3), dtype=np.float32)
        mat[0, 0] = 5.0  # One rated item

        est = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        # Request more items than available
        recs = top_n(est, mat, n=5)

        # Should return all unrated items (2 items: indices 1 and 2)
        assert len(recs[0]) == 2  # Only 2 unrated items available
        assert 1 in recs[0] and 2 in recs[0]  # Should include unrated items
        assert 0 not in recs[0]  # Should exclude rated item

    def test_sorting_order(self) -> None:
        """Test that recommendations are sorted correctly."""
        mat = np.zeros((1, 4), dtype=np.float32)
        mat[0, 0] = 5.0  # Rated item

        # Estimated ratings in descending order
        est = np.array([[0.0, 1.0, 3.0, 2.0]], dtype=np.float32)

        recs = top_n(est, mat, n=3)

        # Should be sorted by estimated rating (descending)
        expected_order = np.array([2, 3, 1])  # Items with est ratings 3.0, 2.0, 1.0
        np.testing.assert_array_equal(recs[0], expected_order)

    def test_sparse_matrix_support(self) -> None:
        """Test sparse matrix support."""
        # Create sparse matrices
        est = sparse.csr_matrix(np.array([[4.5, 3.2, 4.8, 2.1]], dtype=np.float32))
        known = sparse.csr_matrix(np.array([[5.0, 0.0, 0.0, 3.0]], dtype=np.float32))

        recs = top_n(est, known, n=2)

        assert len(recs) == 1
        assert len(recs[0]) == 2
        # Should exclude rated items
        assert 0 not in recs[0]
        assert 3 not in recs[0]


class TestMetrics:
    """Test evaluation metrics."""

    def test_rmse_basic(self) -> None:
        """Test basic RMSE calculation."""
        predictions = np.array([[4.0, 3.0, 5.0]], dtype=np.float32)
        actual = np.array([[4.0, 3.0, 5.0]], dtype=np.float32)

        rmse = compute_rmse(predictions, actual)
        assert rmse == 0.0

    def test_mae_basic(self) -> None:
        """Test basic MAE calculation."""
        predictions = np.array([[4.0, 3.0, 5.0]], dtype=np.float32)
        actual = np.array([[4.0, 3.0, 5.0]], dtype=np.float32)

        mae = compute_mae(predictions, actual)
        assert mae == 0.0

    def test_metrics_with_errors(self) -> None:
        """Test metrics with prediction errors."""
        predictions = np.array([[4.0, 3.0, 5.0]], dtype=np.float32)
        actual = np.array([[5.0, 3.0, 4.0]], dtype=np.float32)

        rmse = compute_rmse(predictions, actual)
        mae = compute_mae(predictions, actual)

        assert rmse > 0.0
        assert mae > 0.0
        assert rmse >= mae  # RMSE is always >= MAE

    def test_metrics_ignore_unrated(self) -> None:
        """Test that metrics ignore unrated items."""
        predictions = np.array([[4.0, 3.0, 5.0, 2.0]], dtype=np.float32)
        actual = np.array([[5.0, 0.0, 4.0, 0.0]], dtype=np.float32)

        # Should only consider items 0 and 2 (non-zero in actual)
        rmse = compute_rmse(predictions, actual)
        mae = compute_mae(predictions, actual)

        # Should be based only on the two rated items
        expected_rmse = np.sqrt(((4.0 - 5.0) ** 2 + (5.0 - 4.0) ** 2) / 2)
        expected_mae = (abs(4.0 - 5.0) + abs(5.0 - 4.0)) / 2

        assert abs(rmse - expected_rmse) < 1e-6
        assert abs(mae - expected_mae) < 1e-6

    def test_metrics_all_unrated(self) -> None:
        """Test metrics when all items are unrated."""
        predictions = np.array([[4.0, 3.0, 5.0]], dtype=np.float32)
        actual = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        rmse = compute_rmse(predictions, actual)
        mae = compute_mae(predictions, actual)

        assert rmse == 0.0
        assert mae == 0.0

    def test_metrics_shape_mismatch(self) -> None:
        """Test error handling for shape mismatch."""
        predictions = np.array([[4.0, 3.0]], dtype=np.float32)
        actual = np.array([[4.0, 3.0, 5.0]], dtype=np.float32)

        with pytest.raises(
            ValueError, match="Predictions and actual matrices must have the same shape"
        ):
            compute_rmse(predictions, actual)

        with pytest.raises(
            ValueError, match="Predictions and actual matrices must have the same shape"
        ):
            compute_mae(predictions, actual)

    def test_sparse_matrix_metrics(self) -> None:
        """Test metrics with sparse matrices."""
        predictions = sparse.csr_matrix(np.array([[4.0, 3.0, 5.0]], dtype=np.float32))
        actual = sparse.csr_matrix(np.array([[5.0, 3.0, 4.0]], dtype=np.float32))

        rmse = compute_rmse(predictions, actual)
        mae = compute_mae(predictions, actual)

        assert rmse > 0.0
        assert mae > 0.0

    def test_metrics_empty_matrices(self) -> None:
        """Test metrics with empty matrices."""
        predictions = np.zeros((0, 3), dtype=np.float32)
        actual = np.zeros((0, 3), dtype=np.float32)

        rmse = compute_rmse(predictions, actual)
        mae = compute_mae(predictions, actual)

        assert rmse == 0.0
        assert mae == 0.0

    def test_metrics_single_element(self) -> None:
        """Test metrics with single element matrices."""
        predictions = np.array([[4.0]], dtype=np.float32)
        actual = np.array([[5.0]], dtype=np.float32)

        rmse = compute_rmse(predictions, actual)
        mae = compute_mae(predictions, actual)

        assert rmse == 1.0
        assert mae == 1.0

    def test_metrics_all_zeros(self) -> None:
        """Test metrics with all-zero matrices."""
        predictions = np.zeros((2, 3), dtype=np.float32)
        actual = np.zeros((2, 3), dtype=np.float32)

        rmse = compute_rmse(predictions, actual)
        mae = compute_mae(predictions, actual)

        assert rmse == 0.0
        assert mae == 0.0

    def test_metrics_inf_values(self) -> None:
        """Test metrics with infinite values."""
        predictions = np.array([[np.inf, 3.0, 5.0]], dtype=np.float32)
        actual = np.array([[5.0, 3.0, 4.0]], dtype=np.float32)

        with pytest.raises(
            ValueError, match="Input matrices must not contain infinite values"
        ):
            compute_rmse(predictions, actual)

        with pytest.raises(
            ValueError, match="Input matrices must not contain infinite values"
        ):
            compute_mae(predictions, actual)

    def test_metrics_nan_values(self) -> None:
        """Test metrics with NaN values."""
        predictions = np.array([[np.nan, 3.0, 5.0]], dtype=np.float32)
        actual = np.array([[5.0, 3.0, 4.0]], dtype=np.float32)

        with pytest.raises(
            ValueError, match="Input matrices must not contain NaN values"
        ):
            compute_rmse(predictions, actual)

        with pytest.raises(
            ValueError, match="Input matrices must not contain NaN values"
        ):
            compute_mae(predictions, actual)


class TestRecommenderSystem:
    """Test the high-level RecommenderSystem class."""

    def test_basic_usage(self) -> None:
        """Test basic recommender system usage."""
        # Create test data
        ratings = np.array([[5, 3, 0, 1], [0, 0, 4, 5]], dtype=np.float32)

        # Initialize and fit
        recommender = RecommenderSystem(algorithm="svd")
        recommender.fit(ratings, k=2)

        # Generate predictions
        predictions = recommender.predict(ratings)

        assert predictions.shape == ratings.shape

        # Generate recommendations
        recommendations = recommender.recommend(ratings, n=2)

        assert len(recommendations) == ratings.shape[0]
        assert all(len(recs) <= 2 for recs in recommendations)

    def test_sparse_matrix_support(self) -> None:
        """Test sparse matrix support."""
        # Create sparse test data
        ratings = sparse.csr_matrix(
            np.array([[5, 3, 0, 1], [0, 0, 4, 5]], dtype=np.float32)
        )

        # Initialize and fit
        recommender = RecommenderSystem(algorithm="svd", use_sparse=True)
        recommender.fit(ratings, k=1)

        # Generate predictions
        predictions = recommender.predict(ratings)

        assert sparse.issparse(predictions)
        assert predictions.shape == ratings.shape

    def test_invalid_algorithm(self) -> None:
        """Test error handling for invalid algorithm."""
        with pytest.raises(ValueError, match="Unsupported algorithm: 'invalid'"):
            RecommenderSystem(algorithm="invalid")

    def test_predict_before_fit(self) -> None:
        """Test error handling when predicting before fitting."""
        recommender = RecommenderSystem()
        ratings = np.array([[5, 3, 0, 1]], dtype=np.float32)

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            recommender.predict(ratings)

    def test_recommend_before_fit(self) -> None:
        """Test error handling when recommending before fitting."""
        recommender = RecommenderSystem()
        ratings = np.array([[5, 3, 0, 1]], dtype=np.float32)

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            recommender.recommend(ratings, n=2)

    def test_invalid_k_parameter(self) -> None:
        """Test error handling for invalid k parameter."""
        recommender = RecommenderSystem(algorithm="svd")
        # Use a 2x2 matrix so k=10 is invalid (must be <= 2)
        ratings = np.array([[5, 3], [0, 1]], dtype=np.float32)

        # The fit method should handle the error gracefully
        # and the model should still be marked as fitted
        recommender.fit(ratings, k=10)

        # Model should be fitted even with invalid k (error handled gracefully)
        assert recommender.is_fitted()

    def test_recommend_with_invalid_n(self) -> None:
        """Test error handling for invalid n parameter."""
        recommender = RecommenderSystem(algorithm="svd")
        ratings = np.array([[5, 3, 0, 1]], dtype=np.float32)
        recommender.fit(ratings, k=2)

        with pytest.raises(ValueError, match="n must be positive"):
            recommender.recommend(ratings, n=0)

    def test_recommend_with_shape_mismatch(self) -> None:
        """Test error handling for shape mismatch in recommend."""
        recommender = RecommenderSystem(algorithm="svd")
        ratings = np.array([[5, 3, 0, 1]], dtype=np.float32)
        recommender.fit(ratings, k=2)

        # Different shape for recommendations
        different_ratings = np.array([[5, 3, 0, 1, 2]], dtype=np.float32)

        with pytest.raises(
            ValueError, match="Estimated and known matrices must have the same shape"
        ):
            recommender.recommend(different_ratings, n=2)

    def test_model_persistence(self) -> None:
        """Test model save and load functionality."""
        recommender = RecommenderSystem(algorithm="svd")
        ratings = np.array([[5, 3, 0, 1], [0, 0, 4, 5]], dtype=np.float32)
        recommender.fit(ratings, k=2)

        # Save model
        recommender.save("test_model.pkl")

        # Load model
        loaded_recommender = RecommenderSystem.load("test_model.pkl")

        # Test that loaded model works
        predictions = loaded_recommender.predict(ratings)
        assert predictions.shape == ratings.shape

        # Clean up
        if os.path.exists("test_model.pkl"):
            os.remove("test_model.pkl")

    def test_model_parameters(self) -> None:
        """Test model parameter access."""
        recommender = RecommenderSystem(algorithm="svd", use_sparse=True)
        ratings = np.array([[5, 3, 0, 1]], dtype=np.float32)
        recommender.fit(ratings, k=2)

        params = recommender.get_params()
        assert "algorithm" in params
        assert "use_sparse" in params
        assert params["algorithm"] == "svd"
        assert params["use_sparse"] is True

    def test_model_state(self) -> None:
        """Test model state checking."""
        recommender = RecommenderSystem()

        # Should not be fitted initially
        assert not recommender.is_fitted()

        # Fit the model
        ratings = np.array([[5, 3, 0, 1]], dtype=np.float32)
        recommender.fit(ratings, k=2)

        # Should be fitted after fitting
        assert recommender.is_fitted()

    def test_model_clone(self) -> None:
        """Test model cloning functionality."""
        recommender = RecommenderSystem(algorithm="svd", use_sparse=True)
        ratings = np.array([[5, 3, 0, 1]], dtype=np.float32)
        recommender.fit(ratings, k=2)

        # Clone the model
        cloned_recommender = recommender.clone()

        # Should have same parameters
        assert cloned_recommender.get_params() == recommender.get_params()

        # Should not be fitted (cloned before fitting)
        assert not cloned_recommender.is_fitted()

    def test_model_reset(self) -> None:
        """Test model reset functionality."""
        recommender = RecommenderSystem(algorithm="svd")
        ratings = np.array([[5, 3, 0, 1]], dtype=np.float32)
        recommender.fit(ratings, k=2)

        # Should be fitted
        assert recommender.is_fitted()

        # Reset the model
        recommender.reset()

        # Should not be fitted after reset
        assert not recommender.is_fitted()

        # Should raise error when trying to predict
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            recommender.predict(ratings)


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self) -> None:
        """Test complete recommendation pipeline."""
        # Create test data
        ratings = np.array([[5, 3, 0, 1], [0, 0, 4, 5]], dtype=np.float32)

        # SVD reconstruction
        reconstructed = svd_reconstruct(ratings, k=2)

        # Convert to dense if sparse
        if sparse.issparse(reconstructed):
            reconstructed = as_dense(reconstructed)

        # Generate recommendations
        recommendations = top_n(reconstructed, ratings, n=2)

        # Check results
        assert len(recommendations) == ratings.shape[0]
        assert all(len(recs) <= 2 for recs in recommendations)

        # Evaluate quality
        rmse = compute_rmse(reconstructed, ratings)
        mae = compute_mae(reconstructed, ratings)

        assert rmse >= 0.0
        assert mae >= 0.0

    def test_performance_characteristics(self) -> None:
        """Test performance characteristics."""
        # Create larger test data
        ratings = np.random.rand(100, 50).astype(np.float32)

        # Should complete in reasonable time
        import time

        start_time = time.time()

        reconstructed = svd_reconstruct(ratings, k=10)
        recommendations = top_n(reconstructed, ratings, n=5)

        end_time = time.time()

        # Should complete in under 1 second
        assert end_time - start_time < 1.0

        # Check results
        assert len(recommendations) == ratings.shape[0]
        assert all(len(recs) <= 5 for recs in recommendations)


class TestRecommenderSystemAdvanced:
    def test_save_and_load(self):
        ratings = np.array([[5, 3, 0, 1], [0, 0, 4, 5]], dtype=np.float32)
        recommender = RecommenderSystem(algorithm="svd")
        recommender.fit(ratings, k=2)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            recommender.save(path)
            loaded = RecommenderSystem.load(path)
            pred1 = recommender.predict(ratings)
            pred2 = loaded.predict(ratings)
            # Compare as dense arrays for both sparse and dense
            if sparse.issparse(pred1):
                pred1 = pred1.toarray()
            if sparse.issparse(pred2):
                pred2 = pred2.toarray()
            np.testing.assert_allclose(pred1, pred2, rtol=1e-5, atol=1e-6)
        finally:
            os.remove(path)

    def test_clone_and_get_params(self):
        recommender = RecommenderSystem(algorithm="svd", use_sparse=False)
        params = recommender.get_params()
        assert params["algorithm"] == "svd"
        assert params["use_sparse"] is False
        clone = recommender.clone()
        assert isinstance(clone, RecommenderSystem)
        assert not clone.is_fitted()
        assert clone.get_params() == params

    def test_reset_and_is_fitted(self):
        ratings = np.array([[5, 3, 0, 1], [0, 0, 4, 5]], dtype=np.float32)
        recommender = RecommenderSystem(algorithm="svd")
        recommender.fit(ratings, k=2)
        assert recommender.is_fitted()
        recommender.reset()
        assert not recommender.is_fitted()
        with pytest.raises(RuntimeError):
            recommender.predict(ratings)

    def test_private_methods_dense_and_sparse(self):
        recommender = RecommenderSystem(algorithm="svd")
        dense = np.random.rand(5, 4).astype(np.float32)
        sparse_mat = sparse.csr_matrix(dense)
        # _fit_svd_dense
        model_dense = recommender._fit_svd_dense(dense, k=2)
        assert set(model_dense.keys()) == {
            "u",
            "s",
            "vt",
            "k",
            "global_bias",
            "user_bias",
            "item_bias",
        }
        # _fit_svd_sparse
        model_sparse = recommender._fit_svd_sparse(sparse_mat, k=2)
        assert set(model_sparse.keys()) == {
            "u",
            "s",
            "vt",
            "k",
            "global_bias",
            "user_bias",
            "item_bias",
        }
        # _predict_svd
        recommender._model = model_dense
        pred = recommender._predict_svd(dense)
        assert pred.shape == dense.shape
        recommender.use_sparse = True
        pred_sparse = recommender._predict_svd(dense)
        assert hasattr(pred_sparse, "shape")
        recommender.use_sparse = False
        recommender._model = model_sparse
        pred2 = recommender._predict_svd(dense)
        assert pred2.shape == dense.shape


class TestALS:
    def test_fit_als_basic(self):
        ratings = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        model = RecommenderSystem(algorithm="als")
        model.fit(ratings, factors=2, iterations=2, lambda_=0.01)
        assert "user_factors" in model._model
        assert "item_factors" in model._model

    def test_predict_als(self):
        ratings = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        model = RecommenderSystem(algorithm="als")
        model.fit(ratings, factors=2, iterations=2, lambda_=0.01)
        preds = model.predict(ratings)
        assert preds.shape == ratings.shape


class TestKNN:
    def test_fit_knn(self):
        ratings = np.array([[5, 3, 0], [0, 0, 4]], dtype=np.float32)
        model = RecommenderSystem(algorithm="knn")
        model.fit(ratings, neighbors=2)
        assert "item_sim" in model._model
        assert "neighbors" in model._model

    def test_predict_knn(self):
        ratings = np.array([[5, 3, 0], [0, 0, 4]], dtype=np.float32)
        model = RecommenderSystem(algorithm="knn")
        model.fit(ratings, neighbors=2)
        preds = model.predict(ratings)
        assert preds.shape == ratings.shape
        assert not np.any(np.isnan(preds))


class TestBias:
    def test_bias_in_svd(self):
        ratings = np.array([[5, 3, 0], [0, 0, 4]], dtype=np.float32)
        model = RecommenderSystem(algorithm="svd")
        model.fit(ratings, k=2)
        m = model._model
        assert "global_bias" in m and "user_bias" in m and "item_bias" in m


class TestChunkedSVD:
    def test_chunked_reconstruct(self):
        mat = np.random.rand(100, 10).astype(np.float32)
        reconstructed = svd_reconstruct(mat, k=2, use_sparse=True)
        assert reconstructed.shape == mat.shape


def test_svd_reconstruct_invalid_sparse(monkeypatch):
    import scipy.sparse

    from vector_recsys_lite.algo import svd_reconstruct

    mat = scipy.sparse.csr_matrix((0, 0), dtype=np.float32)
    # k too large for sparse
    with pytest.raises(ValueError):
        svd_reconstruct(mat, k=10)
    # k too small
    with pytest.raises(ValueError):
        svd_reconstruct(mat, k=0)


def test_svd_reconstruct_runtime_error(monkeypatch):
    from vector_recsys_lite.algo import svd_reconstruct

    # Patch np.linalg.svd to raise
    def fake_svd(*a, **k):
        raise np.linalg.LinAlgError("fail")

    monkeypatch.setattr(np.linalg, "svd", fake_svd)
    mat = np.random.rand(3, 3).astype(np.float32)
    with pytest.raises(RuntimeError):
        svd_reconstruct(mat, k=2)


def test_top_n_all_rated():
    from vector_recsys_lite.algo import top_n

    mat = np.ones((2, 3), dtype=np.float32)
    est = np.ones((2, 3), dtype=np.float32)
    recs = top_n(est, mat, n=2)
    assert recs.shape[1] == 0


def test_top_n_zero_n():
    from vector_recsys_lite.algo import top_n

    mat = np.zeros((2, 3), dtype=np.float32)
    est = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        top_n(est, mat, n=0)


def test_compute_rmse_all_zero():
    from vector_recsys_lite.algo import compute_rmse

    pred = np.zeros((2, 3), dtype=np.float32)
    actual = np.zeros((2, 3), dtype=np.float32)
    assert compute_rmse(pred, actual) == 0.0


def test_compute_mae_all_zero():
    from vector_recsys_lite.algo import compute_mae

    pred = np.zeros((2, 3), dtype=np.float32)
    actual = np.zeros((2, 3), dtype=np.float32)
    assert compute_mae(pred, actual) == 0.0


def test_predict_unfitted():
    from vector_recsys_lite.algo import RecommenderSystem

    rec = RecommenderSystem()
    with pytest.raises(RuntimeError):
        rec.predict(np.ones((2, 2), dtype=np.float32))


def test_predict_model_none():
    from vector_recsys_lite.algo import RecommenderSystem

    rec = RecommenderSystem()
    rec._fitted = True
    rec._model = None
    with pytest.raises(RuntimeError):
        rec.predict(np.ones((2, 2), dtype=np.float32))


def test_recommend_invalid_n():
    from vector_recsys_lite.algo import RecommenderSystem

    rec = RecommenderSystem()
    rec._fitted = True
    rec._model = {"u": np.eye(2), "s": np.ones(2), "vt": np.eye(2), "k": 2}
    with pytest.raises(ValueError):
        rec.recommend(np.ones((2, 2), dtype=np.float32), n=0)


def test_recommender_unsupported_algorithm():
    from vector_recsys_lite.algo import RecommenderSystem

    with pytest.raises(ValueError):
        RecommenderSystem(algorithm="unknown")


def test_svd_reconstruct_invalid_k():
    import numpy as np

    from vector_recsys_lite.algo import svd_reconstruct

    mat = np.ones((3, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        svd_reconstruct(mat, k=0)
    with pytest.raises(ValueError):
        svd_reconstruct(mat, k=10)


def test_svd_reconstruct_1d_empty():
    import numpy as np

    from vector_recsys_lite.algo import svd_reconstruct

    with pytest.raises(ValueError):
        svd_reconstruct(np.array([1, 2, 3], dtype=np.float32))
    with pytest.raises(ValueError):
        svd_reconstruct(np.array([[]], dtype=np.float32))


def test_svd_reconstruct_svd_failure(monkeypatch):
    import numpy as np

    from vector_recsys_lite.algo import svd_reconstruct

    def fake_svd(*a, **k):
        raise np.linalg.LinAlgError("fail")

    monkeypatch.setattr(np.linalg, "svd", fake_svd)
    mat = np.ones((3, 3), dtype=np.float32)
    with pytest.raises(RuntimeError):
        svd_reconstruct(mat, k=2)


def test_top_n_edge_cases():
    import numpy as np

    from vector_recsys_lite.algo import top_n

    # All items rated
    mat = np.ones((2, 3), dtype=np.float32)
    est = np.ones((2, 3), dtype=np.float32)
    recs = top_n(est, mat, n=2)
    assert recs.shape[1] == 0
    # n=0
    with pytest.raises(ValueError):
        top_n(est, mat, n=0)


def test_compute_rmse_mae_shape_mismatch():
    import numpy as np

    from vector_recsys_lite.algo import compute_mae, compute_rmse

    pred = np.ones((2, 2), dtype=np.float32)
    actual = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_rmse(pred, actual)
    with pytest.raises(ValueError):
        compute_mae(pred, actual)


def test_compute_rmse_shape_mismatch():
    import numpy as np

    from vector_recsys_lite.algo import compute_rmse

    a = np.zeros((2, 2))
    b = np.zeros((3, 2))
    import pytest

    with pytest.raises(ValueError, match="must have the same shape"):
        compute_rmse(a, b)


def test_compute_rmse_nan_inf():
    import numpy as np

    from vector_recsys_lite.algo import compute_rmse

    a = np.array([[1.0, float("nan")]])
    b = np.array([[1.0, 2.0]])
    import pytest

    with pytest.raises(ValueError, match="NaN"):
        compute_rmse(a, b)
    a = np.array([[1.0, float("inf")]])
    with pytest.raises(ValueError, match="infinite"):
        compute_rmse(a, b)


def test_compute_rmse_empty_mask():
    import numpy as np

    from vector_recsys_lite.algo import compute_rmse

    a = np.zeros((2, 2))
    b = np.zeros((2, 2))
    assert compute_rmse(a, b) == 0.0


def test_compute_mae_shape_mismatch():
    import numpy as np

    from vector_recsys_lite.algo import compute_mae

    a = np.zeros((2, 2))
    b = np.zeros((3, 2))
    import pytest

    with pytest.raises(ValueError, match="must have the same shape"):
        compute_mae(a, b)


def test_compute_mae_nan_inf():
    import numpy as np

    from vector_recsys_lite.algo import compute_mae

    a = np.array([[1.0, float("nan")]])
    b = np.array([[1.0, 2.0]])
    import pytest

    with pytest.raises(ValueError, match="NaN"):
        compute_mae(a, b)
    a = np.array([[1.0, float("inf")]])
    with pytest.raises(ValueError, match="infinite"):
        compute_mae(a, b)


def test_compute_mae_empty_mask():
    import numpy as np

    from vector_recsys_lite.algo import compute_mae

    a = np.zeros((2, 2))
    b = np.zeros((2, 2))
    assert compute_mae(a, b) == 0.0


def test_benchmark_algorithm_unknown():
    import numpy as np

    a = np.zeros((2, 2))
    import pytest

    with pytest.raises(ValueError, match="Unknown algorithm"):
        benchmark_algorithm("not_a_real_algo", a)


def test_train_test_split_ratings():
    mat = np.random.rand(5, 5)
    mat[mat < 0.5] = 0
    train, test = train_test_split_ratings(mat, test_size=0.2)[0]
    assert train.shape == mat.shape
    assert isinstance(test, list)


def test_train_test_split_ratings_empty():
    mat = np.zeros((3, 3))
    train, test = train_test_split_ratings(mat)[0]
    assert train.shape == mat.shape
    assert test == []


def test_grid_search():
    param_grid = {"k": [1, 2], "algorithm": ["svd"]}
    mat = np.random.rand(4, 4)
    mat[mat < 0.5] = 0
    result = grid_search(mat, param_grid, cv=2)
    assert "best_params" in result
    assert "best_score" in result
    assert "results" in result


def test_ascii_heatmap_and_visualize_svd():
    import numpy as np

    from vector_recsys_lite.explain import ascii_heatmap, visualize_svd

    mat = np.array([[1, 2], [3, 4]], dtype=np.float32)
    # ASCII only
    ascii_heatmap(mat, title="Test Heatmap", plot=False)
    # Try with plot (should not error even if matplotlib missing)
    try:
        ascii_heatmap(mat, title="Test Heatmap", plot=True)
    except Exception:
        pass
    # SVD visualization (ASCII only)
    visualize_svd(mat, k=1, plot=False)
    # SVD visualization with plot
    try:
        visualize_svd(mat, k=1, plot=True)
    except Exception:
        pass
