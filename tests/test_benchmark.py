"""Tests for the benchmark module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from rich.table import Table

from vector_recsys_lite.benchmark import (
    BenchmarkSuite,
    create_benchmark_dataset,
    quick_benchmark,
)


class TestBenchmarkSuite:
    """Test BenchmarkSuite functionality."""

    def test_benchmark_suite_creation(self) -> None:
        """Test BenchmarkSuite creation."""
        suite = BenchmarkSuite()

        assert suite.output_dir is not None
        assert isinstance(suite.results, list)

    def test_benchmark_suite_with_custom_output_dir(self) -> None:
        """Test BenchmarkSuite with custom output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = BenchmarkSuite(output_dir=temp_dir)

            assert suite.output_dir == Path(temp_dir)

    def test_run_algorithm_benchmark_basic(self) -> None:
        """Test basic algorithm benchmarking."""
        suite = BenchmarkSuite()
        matrix = np.random.rand(10, 8).astype(np.float32)

        results = suite.run_algorithm_benchmark(
            ratings=matrix,
            algorithms=["svd"],
            k_values=[2, 4],
            n_runs=1,
            save_results=False,
        )

        assert isinstance(results, dict)
        assert "matrix_info" in results
        assert "configurations" in results
        assert "summary" in results
        assert len(results["configurations"]) > 0

    def test_run_dataset_benchmark_basic(self) -> None:
        """Test basic dataset benchmarking."""
        suite = BenchmarkSuite()

        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("5,3,0,1\n0,0,4,5\n")
            temp_path = f.name

        try:
            results = suite.run_dataset_benchmark(
                data_paths=[temp_path], algorithms=["svd"], k=2, n_runs=1
            )

            assert isinstance(results, dict)
            assert "datasets" in results
            assert "algorithms" in results
            assert "summary" in results
        finally:
            Path(temp_path).unlink()

    def test_run_memory_profiling(self) -> None:
        """Test memory profiling."""
        suite = BenchmarkSuite()
        matrix = np.random.rand(10, 8).astype(np.float32)

        results = suite.run_memory_profiling(ratings=matrix, k=2, n_runs=1)

        assert isinstance(results, dict)
        assert "matrix_info" in results
        assert "profiles" in results

    def test_generate_report(self) -> None:
        """Test report generation."""
        suite = BenchmarkSuite()

        # Mock results
        results = {
            "matrix_info": {"shape": (10, 8), "sparsity": 0.5},
            "configurations": [
                {
                    "algorithm": "svd",
                    "k": 2,
                    "avg_time": 0.1,
                    "avg_rmse": 0.5,
                    "avg_memory_mb": 0.5,
                    "avg_mae": 0.4,
                }
            ],
            "summary": {"best_algorithm": "svd"},
        }

        report = suite.generate_report(results)

        # generate_report returns a Table object
        assert isinstance(report, Table)

        # Just check that it has some content
        assert len(str(report)) > 0


class TestQuickBenchmark:
    """Test quick_benchmark function."""

    def test_quick_benchmark_basic(self) -> None:
        """Test basic quick benchmark."""
        matrix = np.random.rand(10, 8).astype(np.float32)

        results = quick_benchmark(matrix, k=2, n_runs=1)

        assert isinstance(results, dict)
        assert "time" in results
        assert "rmse" in results
        assert "mae" in results

    def test_quick_benchmark_sparse_matrix(self) -> None:
        """Test quick benchmark with sparse matrix."""
        from scipy import sparse

        matrix = sparse.csr_matrix(np.random.rand(10, 8).astype(np.float32))

        results = quick_benchmark(matrix, k=2, n_runs=1)

        assert isinstance(results, dict)
        assert "time" in results

    def test_quick_benchmark_invalid_k(self) -> None:
        """Test quick benchmark with invalid k."""
        matrix = np.random.rand(3, 2).astype(np.float32)

        # Should handle the error gracefully and return error info
        results = quick_benchmark(matrix, k=10, n_runs=1)

        assert isinstance(results, dict)
        assert "error" in results or "time" in results


class TestCreateBenchmarkDataset:
    """Test create_benchmark_dataset function."""

    def test_create_benchmark_dataset_basic(self) -> None:
        """Test basic benchmark dataset creation."""
        matrix = create_benchmark_dataset(n_users=10, n_items=5, sparsity=0.5)

        assert matrix.shape == (10, 5)
        assert matrix.dtype == np.float32

    def test_create_benchmark_dataset_sparse(self) -> None:
        """Test benchmark dataset creation with high sparsity."""
        matrix = create_benchmark_dataset(n_users=100, n_items=50, sparsity=0.9)

        # Should be sparse
        if hasattr(matrix, "nnz"):  # sparse matrix
            sparsity_actual = 1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        else:  # dense matrix
            sparsity_actual = 1.0 - (np.count_nonzero(matrix) / matrix.size)

        assert sparsity_actual > 0.8

    def test_create_benchmark_dataset_save(self) -> None:
        """Test benchmark dataset creation with saving."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            matrix = create_benchmark_dataset(
                n_users=10, n_items=5, sparsity=0.5, save_path=temp_path
            )

            assert matrix.shape == (10, 5)
            assert Path(temp_path).exists()
        finally:
            Path(temp_path).unlink()

    def test_create_benchmark_dataset_invalid_params(self) -> None:
        """Test benchmark dataset creation with invalid parameters."""
        # Negative dimensions
        with pytest.raises(ValueError):
            create_benchmark_dataset(n_users=-1, n_items=5)

        # Invalid sparsity
        with pytest.raises(ValueError):
            create_benchmark_dataset(n_users=10, n_items=5, sparsity=1.5)


class TestBenchmarkIntegration:
    """Integration tests for benchmarking."""

    def test_full_benchmark_pipeline(self) -> None:
        """Test complete benchmark pipeline."""
        # Create test matrix
        matrix = np.random.rand(20, 15).astype(np.float32)

        # Run quick benchmark
        results = quick_benchmark(matrix, k=5, n_runs=2)

        # Verify results
        assert isinstance(results, dict)
        assert "time" in results
        assert "rmse" in results
        assert "mae" in results

        # Check that times are positive
        assert results["time"] > 0

        # Check that metrics are reasonable
        assert results["rmse"] >= 0
        assert results["mae"] >= 0

    def test_benchmark_suite_full_pipeline(self) -> None:
        """Test complete benchmark suite pipeline."""
        # Create test matrix
        matrix = np.random.rand(20, 15).astype(np.float32)

        # Create benchmark suite
        suite = BenchmarkSuite()

        # Run algorithm benchmark
        results = suite.run_algorithm_benchmark(
            ratings=matrix,
            algorithms=["svd"],
            k_values=[5, 10],
            n_runs=1,
            save_results=False,
        )

        # Verify results structure
        assert isinstance(results, dict)
        assert "matrix_info" in results
        assert "configurations" in results
        assert "summary" in results

        # Check that we have results for each configuration
        assert len(results["configurations"]) == 2  # 2 k values

        # Check that all configurations have required fields
        for config in results["configurations"]:
            assert "algorithm" in config
            assert "k" in config
            assert "avg_time" in config

    def test_benchmark_performance_characteristics(self) -> None:
        """Test benchmark performance characteristics."""
        # Create larger test matrix
        matrix = np.random.rand(100, 50).astype(np.float32)

        import time

        start_time = time.time()

        # Run quick benchmark
        results = quick_benchmark(matrix, k=10, n_runs=2)

        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 5.0  # Should complete in under 5 seconds

        # Verify results
        assert isinstance(results, dict)
        assert results["time"] > 0
        assert results["rmse"] >= 0
        assert results["mae"] >= 0
