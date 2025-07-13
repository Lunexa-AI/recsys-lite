"""Tests for the IO module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from vector_recsys_lite.io import (
    create_sample_ratings,
    load_ratings,
    save_ratings,
)


class TestLoadRatings:
    """Test rating matrix loading functionality."""

    def test_load_basic_csv(self) -> None:
        """Test loading a basic CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("5,3,0,1\n")
            f.write("0,0,4,5\n")
            f.write("1,1,0,0\n")
            temp_path = f.name

        try:
            matrix = load_ratings(temp_path)

            assert matrix.shape == (3, 4)
            assert matrix.dtype == np.float32
            assert matrix[0, 0] == 5.0
            assert matrix[0, 2] == 0.0  # Missing value
        finally:
            Path(temp_path).unlink()

    def test_load_with_different_delimiters(self) -> None:
        """Test loading with different delimiters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("5;3;0;1\n")
            f.write("0;0;4;5\n")
            temp_path = f.name

        try:
            matrix = load_ratings(temp_path, delimiter=";")

            assert matrix.shape == (2, 4)
            assert matrix[0, 0] == 5.0
        finally:
            Path(temp_path).unlink()

    def test_load_with_missing_values(self) -> None:
        """Test loading with various missing value formats."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("5,3,,1\n")
            f.write("0,0,4,5\n")
            f.write("1,1,0,0\n")
            temp_path = f.name

        try:
            matrix = load_ratings(temp_path)

            assert matrix.shape == (3, 4)
            assert matrix[0, 2] == 0.0  # Empty cell becomes 0
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self) -> None:
        """Test error handling for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_ratings("nonexistent.csv")

    def test_load_invalid_file(self) -> None:
        """Test error handling for invalid file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("5,3,invalid,1\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid value at row 1, col 3"):
                load_ratings(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_empty_file(self) -> None:
        """Test handling of empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Empty file
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="CSV must contain a 2D matrix"):
                load_ratings(temp_path)
        finally:
            Path(temp_path).unlink()


class TestSaveRatings:
    """Test rating matrix saving functionality."""

    def test_save_basic_matrix(self) -> None:
        """Test saving a basic matrix."""
        matrix = np.array(
            [[5.0, 3.0, 0.0, 1.0], [0.0, 0.0, 4.0, 5.0]], dtype=np.float32
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            save_ratings(matrix, temp_path)

            # Verify the file was created and has correct content
            assert Path(temp_path).exists()

            # Load it back and compare
            loaded = load_ratings(temp_path)
            np.testing.assert_array_equal(matrix, loaded)
        finally:
            Path(temp_path).unlink()

    def test_save_with_custom_format(self) -> None:
        """Test saving with custom format."""
        matrix = np.array([[1.23456, 2.34567]], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            save_ratings(matrix, temp_path, fmt="%.2f")

            # Check that values are rounded to 2 decimal places
            with open(temp_path) as f:
                content = f.read()
                assert "1.23" in content
                assert "2.35" in content
        finally:
            Path(temp_path).unlink()

    def test_save_invalid_matrix(self) -> None:
        """Test error handling for invalid matrix."""
        # 1D array
        matrix = np.array([1, 2, 3], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Matrix must be 2D"):
                save_ratings(matrix, temp_path)
        finally:
            Path(temp_path).unlink()


class TestCreateSampleRatings:
    """Test sample rating matrix generation."""

    def test_basic_sample_creation(self) -> None:
        """Test basic sample matrix creation."""
        matrix = create_sample_ratings(n_users=10, n_items=5, sparsity=0.5)

        assert matrix.shape == (10, 5)
        assert matrix.dtype == np.float32
        assert not np.any(np.isnan(matrix))
        assert not np.any(np.isinf(matrix))

    def test_sparsity_control(self) -> None:
        """Test that sparsity is properly controlled."""
        # Test dense matrix
        dense = create_sample_ratings(n_users=100, n_items=50, sparsity=0.0)
        dense_sparsity = 1.0 - (np.count_nonzero(dense) / dense.size)
        assert dense_sparsity < 0.1  # Should be very dense

        # Test sparse matrix
        sparse = create_sample_ratings(n_users=100, n_items=50, sparsity=0.9)
        sparse_sparsity = 1.0 - (np.count_nonzero(sparse) / sparse.size)
        assert sparse_sparsity > 0.8  # Should be very sparse

    def test_rating_range(self) -> None:
        """Test that ratings are within specified range."""
        matrix = create_sample_ratings(n_users=10, n_items=5, rating_range=(1.0, 5.0))

        # Only check non-zero ratings
        non_zero = matrix[matrix > 0]
        assert np.all(non_zero >= 1.0)
        assert np.all(non_zero <= 5.0)

    def test_reproducibility(self) -> None:
        """Test that random seed produces reproducible results."""
        matrix1 = create_sample_ratings(n_users=10, n_items=5, random_state=42)
        matrix2 = create_sample_ratings(n_users=10, n_items=5, random_state=42)

        np.testing.assert_array_equal(matrix1, matrix2)

    def test_invalid_parameters(self) -> None:
        """Test error handling for invalid parameters."""
        # Negative dimensions
        with pytest.raises(ValueError, match="n_users and n_items must be positive"):
            create_sample_ratings(n_users=-1, n_items=5)

        # Invalid sparsity
        with pytest.raises(ValueError, match="Sparsity must be between 0 and 1"):
            create_sample_ratings(n_users=10, n_items=5, sparsity=1.5)

        # Invalid rating range
        with pytest.raises(ValueError, match="Invalid rating range"):
            create_sample_ratings(n_users=10, n_items=5, rating_range=(5.0, 1.0))


class TestIntegration:
    """Integration tests for IO operations."""

    def test_load_save_roundtrip(self) -> None:
        """Test that loading and saving preserves data."""
        # Create sample matrix
        original = create_sample_ratings(n_users=20, n_items=10, sparsity=0.7)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            # Save and reload
            save_ratings(original, temp_path)
            loaded = load_ratings(temp_path)

            # Should be approximately equal (within float32 precision)
            np.testing.assert_array_almost_equal(original, loaded, decimal=3)
        finally:
            Path(temp_path).unlink()

    def test_large_matrix_handling(self) -> None:
        """Test handling of larger matrices."""
        # Create a larger matrix
        matrix = create_sample_ratings(n_users=1000, n_items=500, sparsity=0.8)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            # Should handle large matrices efficiently
            save_ratings(matrix, temp_path)
            loaded = load_ratings(temp_path)

            assert loaded.shape == matrix.shape
            # Should be approximately equal (within float32 precision)
            np.testing.assert_array_almost_equal(matrix, loaded, decimal=3)
        finally:
            Path(temp_path).unlink()

    def test_load_ratings_with_custom_delimiter(self) -> None:
        """Test loading with custom delimiter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("5;3;0;1\n0;0;4;5\n")
            temp_path = f.name

        try:
            matrix = load_ratings(temp_path, delimiter=";")

            assert matrix.shape == (2, 4)
            assert matrix[0, 0] == 5.0
            assert matrix[1, 2] == 4.0
        finally:
            Path(temp_path).unlink()

    def test_load_ratings_with_custom_missing_value(self) -> None:
        """Test loading with custom missing value."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("5,3,-1,1\n0,0,4,5\n")
            temp_path = f.name

        try:
            matrix = load_ratings(temp_path, missing_value=-1.0)

            assert matrix.shape == (2, 4)
            assert matrix[0, 2] == -1.0  # Custom missing value
        finally:
            Path(temp_path).unlink()

    def test_save_ratings_with_custom_format(self) -> None:
        """Test saving with custom format."""
        matrix = np.array([[1.23456, 2.34567]], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            save_ratings(matrix, temp_path, fmt="%.1f")

            # Check that values are rounded to 1 decimal place
            with open(temp_path) as f:
                content = f.read()
                assert "1.2" in content
                assert "2.3" in content
        finally:
            Path(temp_path).unlink()

    def test_save_ratings_with_custom_delimiter(self) -> None:
        """Test saving with custom delimiter."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            save_ratings(matrix, temp_path, delimiter=";")

            # Check that semicolon delimiter is used
            with open(temp_path) as f:
                content = f.read()
                assert ";" in content
                assert "," not in content
        finally:
            Path(temp_path).unlink()

    def test_load_ratings_sparse_format(self) -> None:
        """Test loading in sparse format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("5,3,0,1\n0,0,4,5\n")
            temp_path = f.name

        try:
            matrix = load_ratings(temp_path, sparse_format=True)

            from scipy import sparse

            assert sparse.issparse(matrix)
            assert matrix.shape == (2, 4)
        finally:
            Path(temp_path).unlink()

    def test_save_ratings_sparse_matrix(self) -> None:
        """Test saving sparse matrix."""
        from scipy import sparse

        matrix = sparse.csr_matrix(
            np.array([[5, 3, 0, 1], [0, 0, 4, 5]], dtype=np.float32)
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            save_ratings(matrix, temp_path)

            # Load back and compare
            loaded = load_ratings(temp_path)
            np.testing.assert_array_equal(matrix.toarray(), loaded)
        finally:
            Path(temp_path).unlink()

    def test_load_ratings_json_format(self) -> None:
        """Test loading JSON format."""
        import json

        matrix_data = {
            "matrix": [[5.0, 3.0, 0.0, 1.0], [0.0, 0.0, 4.0, 5.0]],
            "shape": [2, 4],
            "dtype": "float32",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(matrix_data, f)
            temp_path = f.name

        try:
            matrix = load_ratings(temp_path)

            assert matrix.shape == (2, 4)
            assert matrix[0, 0] == 5.0
            assert matrix[1, 2] == 4.0
        finally:
            Path(temp_path).unlink()

    def test_save_ratings_json_format(self) -> None:
        """Test saving JSON format."""
        matrix = np.array(
            [[5.0, 3.0, 0.0, 1.0], [0.0, 0.0, 4.0, 5.0]], dtype=np.float32
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_ratings(matrix, temp_path)

            # Load back and compare
            loaded = load_ratings(temp_path)
            np.testing.assert_array_equal(matrix, loaded)
        finally:
            Path(temp_path).unlink()

    def test_load_ratings_parquet_format(self) -> None:
        """Test loading Parquet format."""
        try:
            import pandas as pd

            matrix = np.array(
                [[5.0, 3.0, 0.0, 1.0], [0.0, 0.0, 4.0, 5.0]], dtype=np.float32
            )
            df = pd.DataFrame(matrix)

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                temp_path = f.name

            try:
                df.to_parquet(temp_path, index=False)

                loaded = load_ratings(temp_path)

                assert loaded.shape == (2, 4)
                assert loaded[0, 0] == 5.0
                assert loaded[1, 2] == 4.0
            finally:
                Path(temp_path).unlink()
        except ImportError:
            pytest.skip("pandas not available")

    def test_save_ratings_parquet_format(self) -> None:
        """Test saving Parquet format."""
        try:
            matrix = np.array(
                [[5.0, 3.0, 0.0, 1.0], [0.0, 0.0, 4.0, 5.0]], dtype=np.float32
            )

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                temp_path = f.name

            try:
                save_ratings(matrix, temp_path)

                # Load back and compare
                loaded = load_ratings(temp_path)
                np.testing.assert_array_equal(matrix, loaded)
            finally:
                Path(temp_path).unlink()
        except ImportError:
            pytest.skip("pandas not available")

    def test_load_ratings_npz_format(self) -> None:
        """Test loading NPZ format."""
        matrix = np.array(
            [[5.0, 3.0, 0.0, 1.0], [0.0, 0.0, 4.0, 5.0]], dtype=np.float32
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name

        try:
            from scipy import sparse

            sparse.save_npz(temp_path, sparse.csr_matrix(matrix))

            loaded = load_ratings(temp_path)

            assert loaded.shape == (2, 4)
            assert loaded[0, 0] == 5.0
            assert loaded[1, 2] == 4.0
        finally:
            Path(temp_path).unlink()

    def test_save_ratings_npz_format(self) -> None:
        """Test saving NPZ format."""
        from scipy import sparse

        matrix = sparse.csr_matrix(
            np.array([[5.0, 3.0, 0.0, 1.0], [0.0, 0.0, 4.0, 5.0]], dtype=np.float32)
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name

        try:
            save_ratings(matrix, temp_path)

            # Load back and compare
            loaded = load_ratings(temp_path)
            np.testing.assert_array_equal(matrix.toarray(), loaded)
        finally:
            Path(temp_path).unlink()

    def test_load_ratings_unsupported_format(self) -> None:
        """Test loading unsupported format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("5,3,0,1\n0,0,4,5\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                load_ratings(temp_path, format="unsupported")
        finally:
            Path(temp_path).unlink()

    def test_save_ratings_unsupported_format(self) -> None:
        """Test saving unsupported format."""
        matrix = np.array([[5.0, 3.0, 0.0, 1.0]], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                save_ratings(matrix, temp_path, format="unsupported")
        finally:
            Path(temp_path).unlink()

    def test_create_sample_ratings_sparse_format(self) -> None:
        """Test creating sample ratings in sparse format."""
        matrix = create_sample_ratings(
            n_users=10, n_items=5, sparsity=0.8, sparse_format=True
        )

        from scipy import sparse

        assert sparse.issparse(matrix)
        assert matrix.shape == (10, 5)

    def test_create_sample_ratings_custom_range(self) -> None:
        """Test creating sample ratings with custom range."""
        matrix = create_sample_ratings(
            n_users=10, n_items=5, rating_range=(0.0, 10.0), sparsity=0.5
        )

        # Only check non-zero ratings
        non_zero = matrix[matrix > 0]
        assert np.all(non_zero >= 0.0)
        assert np.all(non_zero <= 10.0)

    def test_create_sample_ratings_reproducibility(self) -> None:
        """Test that sample ratings are reproducible with same seed."""
        matrix1 = create_sample_ratings(n_users=10, n_items=5, random_state=42)
        matrix2 = create_sample_ratings(n_users=10, n_items=5, random_state=42)

        np.testing.assert_array_equal(matrix1, matrix2)

    def test_create_sample_ratings_different_seeds(self) -> None:
        """Test that different seeds produce different results."""
        matrix1 = create_sample_ratings(n_users=10, n_items=5, random_state=42)
        matrix2 = create_sample_ratings(n_users=10, n_items=5, random_state=43)

        # Should be different (very unlikely to be identical)
        assert not np.array_equal(matrix1, matrix2)

    def test_load_ratings_with_progress_bar(self) -> None:
        """Test loading with progress bar."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("5,3,0,1\n0,0,4,5\n")
            temp_path = f.name

        try:
            matrix = load_ratings(temp_path, show_progress=True)

            assert matrix.shape == (2, 4)
        finally:
            Path(temp_path).unlink()

    def test_save_ratings_with_progress_bar(self) -> None:
        """Test saving with progress bar."""
        matrix = np.array([[5.0, 3.0, 0.0, 1.0]], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            save_ratings(matrix, temp_path, show_progress=True)

            # Check that file was created
            assert Path(temp_path).exists()
        finally:
            Path(temp_path).unlink()

    def test_load_ratings_without_progress_bar(self) -> None:
        """Test loading without progress bar."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("5,3,0,1\n0,0,4,5\n")
            temp_path = f.name

        try:
            matrix = load_ratings(temp_path, show_progress=False)

            assert matrix.shape == (2, 4)
        finally:
            Path(temp_path).unlink()

    def test_save_ratings_without_progress_bar(self) -> None:
        """Test saving without progress bar."""
        matrix = np.array([[5.0, 3.0, 0.0, 1.0]], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            save_ratings(matrix, temp_path, show_progress=False)

            # Check that file was created
            assert Path(temp_path).exists()
        finally:
            Path(temp_path).unlink()
