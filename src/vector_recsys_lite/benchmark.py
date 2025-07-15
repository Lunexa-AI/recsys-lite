"""Performance benchmarking and profiling for vector_recsys_lite.

This module provides comprehensive benchmarking tools for evaluating algorithm
performance, memory usage, and accuracy across different datasets and configurations.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy import sparse

from .algo import benchmark_algorithm, compute_mae, compute_rmse, svd_reconstruct
from .io import create_sample_ratings, load_ratings, save_ratings
from .utils import as_dense

console = Console()

__all__ = [
    "BenchmarkSuite",
    "quick_benchmark",
]


class BenchmarkSuite:
    """Comprehensive benchmarking suite for recommender systems."""

    def __init__(self, output_dir: str | Path | None = None):
        """
        Initialize benchmark suite.

        Parameters
        ----------
        output_dir : Optional[Union[str, Path]], default=None
            Directory to save benchmark results. If None, uses current directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict[str, Any]] = []

    def run_algorithm_benchmark(
        self,
        ratings: np.ndarray | sparse.spmatrix,
        algorithms: list[str] | None = None,
        k_values: list[int] | None = None,
        n_runs: int = 3,
        save_results: bool = True,
    ) -> dict[str, Any]:
        """
        Benchmark multiple algorithms and configurations.

        Parameters
        ----------
        ratings : Union[np.ndarray, sparse.spmatrix]
            Rating matrix to benchmark on.
        algorithms : List[str], optional
            List of algorithms to test. Default: ["svd", "svd_sparse"].
        k_values : List[int], optional
            List of rank values to test. Default: [5, 10, 20].
        n_runs : int, default=3
            Number of runs per configuration for averaging.
        save_results : bool, default=True
            Whether to save results to file.

        Returns
        -------
        Dict[str, Any]
            Comprehensive benchmark results.
        """
        # Ensure algorithms is a list for type safety
        if algorithms is None:
            algorithms = ["svd", "svd_sparse"]
        algorithms = list(algorithms)  # Ensure it's a list
        if k_values is None:
            k_values = [5, 10, 20]
        k_values = list(k_values)  # Ensure it's a list

        console.print(
            f"🚀 Starting benchmark with {len(algorithms)} algorithms, {len(k_values)} k-values"
        )

        configurations: list[dict[str, Any]] = []
        results = {
            "matrix_info": self._get_matrix_info(ratings),
            "configurations": configurations,
            "summary": {},
        }

        total_configs = len(algorithms) * len(k_values)
        current_config = 0

        for algorithm in algorithms:
            for k in k_values:
                current_config += 1
                console.print(
                    f"📊 Testing {algorithm} with k={k} ({current_config}/{total_configs})"
                )

                try:
                    # Run benchmark
                    benchmark_result = benchmark_algorithm(
                        algorithm=algorithm, ratings=ratings, k=k, n_runs=n_runs
                    )

                    # Add configuration info
                    benchmark_result.update(
                        {"algorithm": algorithm, "k": k, "n_runs": n_runs}
                    )

                    configurations.append(benchmark_result)

                    # Print progress
                    console.print(
                        f"✅ {algorithm} k={k}: {benchmark_result['avg_time']:.3f}s, "
                        f"RMSE={benchmark_result['avg_rmse']:.4f}"
                    )

                except Exception as e:
                    console.print(f"❌ {algorithm} k={k} failed: {e}")
                    configurations.append(
                        {"algorithm": algorithm, "k": k, "error": str(e)}
                    )

        # Generate summary
        results["summary"] = self._generate_summary(configurations)

        if save_results:
            self._save_results(results)

        return results

    def run_dataset_benchmark(
        self,
        data_paths: list[str | Path],
        algorithms: list[str] | None = None,
        k: int = 10,
        n_runs: int = 3,
    ) -> dict[str, Any]:
        """
        Benchmark algorithms across multiple datasets.

        Parameters
        ----------
        data_paths : List[Union[str, Path]]
            List of paths to rating matrices.
        algorithms : List[str], optional
            List of algorithms to test.
        k : int, default=10
            Rank for matrix factorization.
        n_runs : int, default=3
            Number of runs per dataset.

        Returns
        -------
        Dict[str, Any]
            Cross-dataset benchmark results.
        """
        if algorithms is None:
            algorithms = ["svd", "svd_sparse"]

        console.print(
            f"🌍 Starting cross-dataset benchmark with {len(data_paths)} datasets"
        )

        # Ensure parameters are lists for type safety
        data_paths = list(data_paths)
        algorithms_list = list(algorithms)
        datasets: list[dict[str, Any]] = []
        results = {"datasets": datasets, "algorithms": algorithms_list, "summary": {}}

        for i, data_path in enumerate(data_paths):
            console.print(f"📁 Loading dataset {i + 1}/{len(data_paths)}: {data_path}")

            try:
                # Load dataset
                ratings = load_ratings(data_path, show_progress=False)

                # Get dataset info
                dataset_info = self._get_matrix_info(ratings)
                dataset_info["path"] = str(data_path)

                # Benchmark each algorithm
                dataset_results: list[dict[str, Any]] = []
                for algorithm in algorithms_list:
                    try:
                        benchmark_result = benchmark_algorithm(
                            algorithm=algorithm, ratings=ratings, k=k, n_runs=n_runs
                        )
                        benchmark_result["algorithm"] = algorithm
                        dataset_results.append(benchmark_result)

                    except Exception as e:
                        console.print(f"❌ {algorithm} failed on {data_path}: {e}")
                        dataset_results.append(
                            {"algorithm": algorithm, "error": str(e)}
                        )

                datasets.append({"info": dataset_info, "results": dataset_results})

            except Exception as e:
                console.print(f"❌ Failed to load {data_path}: {e}")
                datasets.append(
                    {"info": {"path": str(data_path), "error": str(e)}, "results": []}
                )

        # Generate summary
        results["summary"] = self._generate_cross_dataset_summary(results)

        return results

    def run_memory_profiling(
        self, ratings: np.ndarray | sparse.spmatrix, k: int = 10, n_runs: int = 5
    ) -> dict[str, Any]:
        """
        Detailed memory profiling of algorithms.

        Parameters
        ----------
        ratings : Union[np.ndarray, sparse.spmatrix]
            Rating matrix to profile.
        k : int, default=10
            Rank for matrix factorization.
        n_runs : int, default=5
            Number of profiling runs.

        Returns
        -------
        Dict[str, Any]
            Detailed memory profiling results.
        """
        try:
            import tracemalloc

            import psutil
        except ImportError:
            console.print("❌ psutil and tracemalloc required for memory profiling")
            return {}

        console.print("🧠 Starting memory profiling")

        profiles: list[dict[str, Any]] = []
        results = {"matrix_info": self._get_matrix_info(ratings), "profiles": profiles}

        algorithms = ["svd", "svd_sparse"]

        for algorithm in algorithms:
            console.print(f"📊 Profiling {algorithm}")

            profile_data: dict[str, Any] = {"algorithm": algorithm, "runs": []}
            runs: list[dict[str, Any]] = []

            for run in range(n_runs):
                # Force garbage collection
                gc.collect()

                # Start memory tracking
                tracemalloc.start()
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB

                start_time = time.time()

                try:
                    # Run algorithm
                    if algorithm == "svd":
                        reconstructed = svd_reconstruct(ratings, k=k)
                    else:
                        raise ValueError(f"Unknown algorithm: {algorithm}")

                    end_time = time.time()

                    # Get memory info
                    current, peak = tracemalloc.get_traced_memory()
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB

                    tracemalloc.stop()

                    # Compute accuracy
                    actual = as_dense(ratings)
                    predictions = as_dense(reconstructed)

                    rmse = compute_rmse(predictions, actual)
                    mae = compute_mae(predictions, actual)

                    runs.append(
                        {
                            "run": run + 1,
                            "time": end_time - start_time,
                            "memory_start_mb": start_memory,
                            "memory_end_mb": end_memory,
                            "memory_peak_mb": peak / 1024 / 1024,
                            "memory_current_mb": current / 1024 / 1024,
                            "rmse": rmse,
                            "mae": mae,
                        }
                    )

                except Exception as e:
                    tracemalloc.stop()
                    runs.append({"run": run + 1, "error": str(e)})
            profile_data["runs"] = runs
            profiles.append(profile_data)
        results = {"matrix_info": self._get_matrix_info(ratings), "profiles": profiles}

        return results

    def generate_report(self, results: dict[str, Any]) -> Table:
        """
        Generate a comprehensive benchmark report.

        Parameters
        ----------
        results : Dict[str, Any]
            Benchmark results from run_algorithm_benchmark.

        Returns
        -------
        Table
            Formatted report as table.
        """
        if "configurations" in results:
            return self._generate_algorithm_report(results)
        elif "datasets" in results:
            return self._generate_dataset_report(results)
        else:
            # Return empty table for unknown format
            table = Table(title="Unknown Results Format")
            table.add_column("Error", style="red")
            table.add_row("Unknown results format")
            return table

    def _get_matrix_info(
        self, ratings: np.ndarray | sparse.csr_matrix
    ) -> dict[str, Any]:
        """Extract information about the rating matrix."""
        shape = ratings.shape
        if sparse.issparse(ratings):
            # Type assertion to help mypy understand this is a sparse matrix
            sparse_ratings = ratings  # type: sparse.csr_matrix
            nnz = sparse_ratings.nnz
            sparsity = 1 - (nnz / (shape[0] * shape[1]))
            memory_mb = sparse_ratings.data.nbytes / 1024 / 1024
        else:
            nnz = np.count_nonzero(ratings)
            sparsity = 1 - (nnz / ratings.size)
            memory_mb = ratings.nbytes / 1024 / 1024

        return {
            "shape": shape,
            "non_zero_elements": nnz,
            "sparsity": sparsity,
            "memory_mb": memory_mb,
            "is_sparse": sparse.issparse(ratings),
        }

    def _generate_summary(self, configurations: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        successful_configs = [c for c in configurations if "error" not in c]

        if not successful_configs:
            return {"error": "No successful configurations"}

        # Find best configurations
        best_time = min(successful_configs, key=lambda x: x["avg_time"])
        best_rmse = min(successful_configs, key=lambda x: x["avg_rmse"])
        best_mae = min(successful_configs, key=lambda x: x["avg_mae"])

        return {
            "total_configurations": len(configurations),
            "successful_configurations": len(successful_configs),
            "best_time": {
                "algorithm": best_time["algorithm"],
                "k": best_time["k"],
                "time": best_time["avg_time"],
            },
            "best_rmse": {
                "algorithm": best_rmse["algorithm"],
                "k": best_rmse["k"],
                "rmse": best_rmse["avg_rmse"],
            },
            "best_mae": {
                "algorithm": best_mae["algorithm"],
                "k": best_mae["k"],
                "mae": best_mae["avg_mae"],
            },
        }

    def _generate_cross_dataset_summary(
        self, results: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate summary for cross-dataset benchmark."""
        successful_datasets = [
            d for d in results["datasets"] if "error" not in d["info"]
        ]

        if not successful_datasets:
            return {"error": "No successful datasets"}

        # Aggregate results across datasets
        algorithm_stats: dict[str, dict[str, list[float]]] = {}
        for dataset in successful_datasets:
            for result in dataset["results"]:
                if "error" not in result:
                    algo = result["algorithm"]
                    if algo not in algorithm_stats:
                        algorithm_stats[algo] = {"times": [], "rmses": [], "maes": []}

                    algorithm_stats[algo]["times"].append(result["avg_time"])
                    algorithm_stats[algo]["rmses"].append(result["avg_rmse"])
                    algorithm_stats[algo]["maes"].append(result["avg_mae"])

        # Compute averages
        summary = {}
        for algo, stats in algorithm_stats.items():
            summary[algo] = {
                "avg_time": np.mean(stats["times"]),
                "avg_rmse": np.mean(stats["rmses"]),
                "avg_mae": np.mean(stats["maes"]),
                "std_time": np.std(stats["times"]),
                "std_rmse": np.std(stats["rmses"]),
                "std_mae": np.std(stats["maes"]),
            }

        return summary

    def _save_results(self, results: dict[str, Any]) -> None:
        """Save benchmark results to file."""
        import json
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"benchmark_results_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        console.print(f"💾 Results saved to {filename}")

    def _generate_algorithm_report(self, results: dict[str, Any]) -> Table:
        """Generate report for algorithm benchmark."""
        table = Table(title="Algorithm Benchmark Results")
        table.add_column("Algorithm", style="cyan")
        table.add_column("k", style="magenta")
        table.add_column("Time (s)", style="green")
        table.add_column("Memory (MB)", style="yellow")
        table.add_column("RMSE", style="red")
        table.add_column("MAE", style="blue")

        for config in results["configurations"]:
            if "error" not in config:
                table.add_row(
                    config["algorithm"],
                    str(config["k"]),
                    f"{config['avg_time']:.3f}",
                    f"{config['avg_memory_mb']:.1f}",
                    f"{config['avg_rmse']:.4f}",
                    f"{config['avg_mae']:.4f}",
                )
            else:
                table.add_row(
                    config["algorithm"],
                    str(config["k"]),
                    "FAILED",
                    "FAILED",
                    "FAILED",
                    "FAILED",
                )

        return table

    def _generate_dataset_report(self, results: dict[str, Any]) -> Table:
        """Generate report for cross-dataset benchmark."""
        table = Table(title="Cross-Dataset Benchmark Results")
        table.add_column("Dataset", style="cyan")
        table.add_column("Shape", style="magenta")
        table.add_column("Algorithm", style="green")
        table.add_column("Time (s)", style="yellow")
        table.add_column("RMSE", style="red")
        table.add_column("MAE", style="blue")

        for dataset in results["datasets"]:
            if "error" not in dataset["info"]:
                shape_str = (
                    f"{dataset['info']['shape'][0]}x{dataset['info']['shape'][1]}"
                )
                for result in dataset["results"]:
                    if "error" not in result:
                        table.add_row(
                            Path(dataset["info"]["path"]).name,
                            shape_str,
                            result["algorithm"],
                            f"{result['avg_time']:.3f}",
                            f"{result['avg_rmse']:.4f}",
                            f"{result['avg_mae']:.4f}",
                        )
                    else:
                        table.add_row(
                            Path(dataset["info"]["path"]).name,
                            shape_str,
                            result["algorithm"],
                            "FAILED",
                            "FAILED",
                            "FAILED",
                        )
            else:
                table.add_row(
                    Path(dataset["info"]["path"]).name,
                    "FAILED",
                    "FAILED",
                    "FAILED",
                    "FAILED",
                    "FAILED",
                )

        return table


def quick_benchmark(
    ratings: np.ndarray | sparse.spmatrix, k: int = 10, n_runs: int = 3
) -> dict[str, Any]:
    """Quick benchmark for SVD and recommendations."""
    try:
        suite = BenchmarkSuite()
        results = suite.run_algorithm_benchmark(
            ratings=ratings,
            algorithms=["svd"],
            k_values=[k],
            n_runs=n_runs,
            save_results=False,
        )
        # Extract timing and metrics if available
        configs = results.get("configurations", [])
        if configs and "avg_time" in configs[0]:
            return {
                "time": configs[0]["avg_time"],
                "rmse": configs[0].get("avg_rmse", 0.0),
                "mae": configs[0].get("avg_mae", 0.0),
                "configurations": configs,
            }
        else:
            return {"time": None, "rmse": None, "mae": None, "configurations": configs}
    except Exception as e:
        return {
            "time": None,
            "rmse": None,
            "mae": None,
            "error": str(e),
            "configurations": [],
        }


def create_benchmark_dataset(
    n_users: int = 1000,
    n_items: int = 500,
    sparsity: float = 0.9,
    save_path: str | Path | None = None,
) -> np.ndarray | sparse.csr_matrix:
    """
    Create a realistic benchmark dataset.

    Parameters
    ----------
    n_users : int, default=1000
        Number of users.
    n_items : int, default=500
        Number of items.
    sparsity : float, default=0.9
        Sparsity level (fraction of missing ratings).
    save_path : Optional[Union[str, Path]], default=None
        Path to save the dataset.

    Returns
    -------
    Union[np.ndarray, sparse.csr_matrix]
        Benchmark dataset.
    """
    console.print(
        f"🎲 Creating benchmark dataset: {n_users}x{n_items}, sparsity={sparsity}"
    )

    # Create sparse dataset
    ratings = create_sample_ratings(
        n_users=n_users, n_items=n_items, sparsity=sparsity, sparse_format=True
    )

    if save_path:
        save_ratings(ratings, save_path)
        console.print(f"💾 Dataset saved to {save_path}")

    return ratings
