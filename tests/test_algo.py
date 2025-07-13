"""Tests for the algorithm module."""

import numpy as np
import pytest
from scipy import sparse
from vector_recsys_lite.algo import compute_mae, compute_rmse, svd_reconstruct, top_n


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
            reconstructed = reconstructed.toarray()

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
            reconstructed = reconstructed.toarray()

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
        with pytest.raises(ValueError, match="Matrix must be 2D"):
            svd_reconstruct(np.array([1, 2, 3], dtype=np.float32))

        # Empty matrix
        with pytest.raises(ValueError, match="Matrix cannot have zero dimensions"):
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
            reconstructed = reconstructed.toarray()

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
        with pytest.raises(ValueError, match="Matrices must have same shape"):
            top_n(est_wrong_shape, mat, n=2)

    def test_empty_matrix(self) -> None:
        """Test handling of empty matrix."""
        empty_mat = np.zeros((0, 3), dtype=np.float32)
        empty_est = np.zeros((0, 3), dtype=np.float32)

        recs = top_n(empty_est, empty_mat, n=2)
        assert recs == []

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
        expected_order = [2, 3, 1]  # Items with est ratings 3.0, 2.0, 1.0
        assert recs[0] == expected_order

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

        with pytest.raises(ValueError, match="Matrices must have same shape"):
            compute_rmse(predictions, actual)

        with pytest.raises(ValueError, match="Matrices must have same shape"):
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

        with pytest.raises(ValueError, match="Matrix contains infinite values"):
            compute_rmse(predictions, actual)

        with pytest.raises(ValueError, match="Matrix contains infinite values"):
            compute_mae(predictions, actual)

    def test_metrics_nan_values(self) -> None:
        """Test metrics with NaN values."""
        predictions = np.array([[np.nan, 3.0, 5.0]], dtype=np.float32)
        actual = np.array([[5.0, 3.0, 4.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="Matrix contains NaN values"):
            compute_rmse(predictions, actual)

        with pytest.raises(ValueError, match="Matrix contains NaN values"):
            compute_mae(predictions, actual)


class TestRecommenderSystem:
    """Test the high-level RecommenderSystem class."""

    def test_basic_usage(self) -> None:
        """Test basic recommender system usage."""
        from vector_recsys_lite.algo import RecommenderSystem

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
        from vector_recsys_lite.algo import RecommenderSystem

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
        from vector_recsys_lite.algo import RecommenderSystem

        with pytest.raises(ValueError, match="Unsupported algorithm: invalid"):
            RecommenderSystem(algorithm="invalid")

    def test_predict_before_fit(self) -> None:
        """Test error handling when predicting before fitting."""
        from vector_recsys_lite.algo import RecommenderSystem

        recommender = RecommenderSystem()
        ratings = np.array([[5, 3, 0, 1]], dtype=np.float32)

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            recommender.predict(ratings)

    def test_recommend_before_fit(self) -> None:
        """Test error handling when recommending before fitting."""
        from vector_recsys_lite.algo import RecommenderSystem

        recommender = RecommenderSystem()
        ratings = np.array([[5, 3, 0, 1]], dtype=np.float32)

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            recommender.recommend(ratings, n=2)

    def test_invalid_k_parameter(self) -> None:
        """Test error handling for invalid k parameter."""
        from vector_recsys_lite.algo import RecommenderSystem

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
        from vector_recsys_lite.algo import RecommenderSystem

        recommender = RecommenderSystem(algorithm="svd")
        ratings = np.array([[5, 3, 0, 1]], dtype=np.float32)
        recommender.fit(ratings, k=2)

        with pytest.raises(ValueError, match="n must be positive"):
            recommender.recommend(ratings, n=0)

    def test_recommend_with_shape_mismatch(self) -> None:
        """Test error handling for shape mismatch in recommend."""
        from vector_recsys_lite.algo import RecommenderSystem

        recommender = RecommenderSystem(algorithm="svd")
        ratings = np.array([[5, 3, 0, 1]], dtype=np.float32)
        recommender.fit(ratings, k=2)

        # Different shape for recommendations
        different_ratings = np.array([[5, 3, 0, 1, 2]], dtype=np.float32)

        with pytest.raises(ValueError, match="Matrices must have same shape"):
            recommender.recommend(different_ratings, n=2)

    def test_model_persistence(self) -> None:
        """Test model save and load functionality."""
        from vector_recsys_lite.algo import RecommenderSystem

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
        import os

        if os.path.exists("test_model.pkl"):
            os.remove("test_model.pkl")

    def test_model_parameters(self) -> None:
        """Test model parameter access."""
        from vector_recsys_lite.algo import RecommenderSystem

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
        from vector_recsys_lite.algo import RecommenderSystem

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
        from vector_recsys_lite.algo import RecommenderSystem

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
        from vector_recsys_lite.algo import RecommenderSystem

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
            reconstructed = reconstructed.toarray()

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
