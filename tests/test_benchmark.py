"""Tests for the benchmark module."""

import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from rich.table import Table
from scipy import sparse

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


def test_run_algorithm_benchmark_invalid_algorithm() -> None:
    suite = BenchmarkSuite()
    matrix = np.random.rand(5, 5).astype(np.float32)
    results = suite.run_algorithm_benchmark(
        ratings=matrix,
        algorithms=["invalid"],
        k_values=[2],
        n_runs=1,
        save_results=False,
    )
    assert any(
        "error" in c and "Unknown algorithm" in c["error"]
        for c in results["configurations"]
    )


def test_run_dataset_benchmark_missing_file() -> None:
    suite = BenchmarkSuite()
    results = suite.run_dataset_benchmark(
        data_paths=["nonexistent.csv"], algorithms=["svd"], k=2, n_runs=1
    )
    assert isinstance(results, dict)
    assert "datasets" in results
    assert any("error" in d["info"] for d in results["datasets"])


def test_generate_report_with_empty_results() -> None:
    suite = BenchmarkSuite()
    results: dict[str, Any] = {}
    table = suite.generate_report(results)
    assert hasattr(table, "add_row")
    # Check that the table has a column named 'Error'
    assert any(col.header == "Error" for col in table.columns)


def test_run_memory_profiling_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    suite = BenchmarkSuite()
    matrix = np.random.rand(5, 5).astype(np.float32)
    monkeypatch.setitem(__import__("sys").modules, "psutil", None)
    monkeypatch.setitem(__import__("sys").modules, "tracemalloc", None)
    results = suite.run_memory_profiling(ratings=matrix, k=2, n_runs=1)
    assert results == {}


def test_generate_summary_no_success() -> None:
    suite = BenchmarkSuite()
    summary = suite._generate_summary([{"algorithm": "svd", "k": 2, "error": "fail"}])
    assert "error" in summary


def test_generate_summary_best_selection() -> None:
    suite = BenchmarkSuite()
    configs = [
        {"algorithm": "svd", "k": 2, "avg_time": 1.0, "avg_rmse": 0.5, "avg_mae": 0.4},
        {"algorithm": "svd", "k": 3, "avg_time": 0.5, "avg_rmse": 0.6, "avg_mae": 0.3},
    ]
    summary = suite._generate_summary(configs)
    assert summary["best_time"]["k"] == 3
    assert summary["best_rmse"]["k"] == 2
    assert summary["best_mae"]["k"] == 3


def test_generate_cross_dataset_summary_no_success() -> None:
    suite = BenchmarkSuite()
    results = {"datasets": [{"info": {"error": "fail"}, "results": []}]}
    summary = suite._generate_cross_dataset_summary(results)
    assert "error" in summary


def test_generate_cross_dataset_summary_aggregate() -> None:
    suite = BenchmarkSuite()
    results = {
        "datasets": [
            {
                "info": {},
                "results": [
                    {
                        "algorithm": "svd",
                        "avg_time": 1.0,
                        "avg_rmse": 0.5,
                        "avg_mae": 0.4,
                    },
                    {
                        "algorithm": "svd",
                        "avg_time": 2.0,
                        "avg_rmse": 0.7,
                        "avg_mae": 0.6,
                    },
                ],
            }
        ]
    }
    summary = suite._generate_cross_dataset_summary(results)
    assert "svd" in summary
    assert abs(summary["svd"]["avg_time"] - 1.5) < 1e-6


def test_save_results_file_error(tmp_path: Path) -> None:
    suite = BenchmarkSuite(output_dir=tmp_path)
    # Patch open to raise
    with mock.patch("builtins.open", side_effect=IOError), pytest.raises(IOError):
        suite._save_results({"foo": "bar"})


def test_generate_algorithm_report_all_errors() -> None:
    suite = BenchmarkSuite()
    results = {
        "configurations": [
            {"algorithm": "svd", "k": 2, "error": "fail"},
            {"algorithm": "svd", "k": 3, "error": "fail"},
        ]
    }
    table = suite._generate_algorithm_report(results)
    assert table is not None
    assert len(list(table.rows)) == 2


def test_generate_algorithm_report_mixed() -> None:
    suite = BenchmarkSuite()
    results = {
        "configurations": [
            {
                "algorithm": "svd",
                "k": 2,
                "avg_time": 1.0,
                "avg_memory_mb": 1.0,
                "avg_rmse": 0.5,
                "avg_mae": 0.4,
            },
            {"algorithm": "svd", "k": 3, "error": "fail"},
        ]
    }
    table = suite._generate_algorithm_report(results)
    assert table is not None
    assert len(list(table.rows)) == 2


def test_generate_dataset_report_all_errors() -> None:
    suite = BenchmarkSuite()
    results = {
        "datasets": [
            {"info": {"error": "fail", "path": "foo.csv"}, "results": []},
            {"info": {"error": "fail", "path": "bar.csv"}, "results": []},
        ]
    }
    table = suite._generate_dataset_report(results)
    assert table is not None
    assert len(list(table.rows)) == 2


def test_generate_dataset_report_mixed() -> None:
    suite = BenchmarkSuite()
    results = {
        "datasets": [
            {
                "info": {"shape": (2, 2), "path": "foo.csv"},
                "results": [
                    {
                        "algorithm": "svd",
                        "avg_time": 1.0,
                        "avg_rmse": 0.5,
                        "avg_mae": 0.4,
                    },
                    {"algorithm": "svd", "error": "fail"},
                ],
            }
        ]
    }
    table = suite._generate_dataset_report(results)
    assert table is not None
    assert len(list(table.rows)) == 2


def test_run_algorithm_benchmark_unknown_algorithm():
    from vector_recsys_lite.benchmark import BenchmarkSuite

    suite = BenchmarkSuite()
    matrix = np.random.rand(5, 5).astype(np.float32)
    results = suite.run_algorithm_benchmark(
        ratings=matrix,
        algorithms=["unknown"],
        k_values=[2],
        n_runs=1,
        save_results=False,
    )
    # Should contain an error in the configurations
    assert any(
        "error" in config and "Unknown algorithm" in config["error"]
        for config in results["configurations"]
    )


def test_benchmark_suite_save_results_file_error(tmp_path):
    from vector_recsys_lite.benchmark import BenchmarkSuite

    suite = BenchmarkSuite(output_dir=tmp_path)
    # Simulate a directory that cannot be written to
    results = {"matrix_info": {}, "configurations": [], "summary": {}}
    # Remove write permissions
    import os

    os.chmod(tmp_path, 0o400)
    try:
        with pytest.raises(Exception):
            suite._save_results(results)
    finally:
        os.chmod(tmp_path, 0o700)
