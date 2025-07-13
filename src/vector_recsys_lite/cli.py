# mypy: disable-error-code=unreachable
"""Command-line interface for vector_recsys_lite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .algo import compute_mae, compute_rmse, svd_reconstruct, top_n
from .benchmark import quick_benchmark
from .io import create_sample_ratings, load_ratings, save_ratings

console = Console()

cli = typer.Typer(
    help="🧊 Fast SVD-based recommender system with optional Numba acceleration",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@cli.command()
def predict(
    csv: Path = typer.Argument(
        ..., exists=True, help="Path to ratings CSV file (users × items)"
    ),
    k: int = typer.Option(
        50, "--rank", "-k", help="SVD rank (number of singular values to use)"
    ),
    n: int = typer.Option(
        10, "--top-n", "-n", help="Number of top recommendations per user"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for recommendations (CSV format)"
    ),
    show_metrics: bool = typer.Option(
        False, "--metrics", "-m", help="Show prediction quality metrics"
    ),
    max_users: int = typer.Option(
        5, "--max-users", help="Maximum number of users to show recommendations for"
    ),
) -> None:
    """
    Generate top-N recommendations for users.

    Loads a rating matrix from CSV and generates personalized recommendations
    using truncated SVD matrix factorization.

    Examples:
        vector-recsys predict ratings.csv
        vector-recsys predict ratings.csv --rank 20 --top-n 5
        vector-recsys predict ratings.csv --output recs.csv --metrics
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading ratings...", total=None)

            # Load ratings
            ratings = load_ratings(csv)
            # Type assertion to help mypy understand ratings has shape attribute
            if not hasattr(ratings, "shape"):
                raise ValueError("Ratings must have shape attribute")
            progress.update(
                task,
                description=f"Loaded {ratings.shape[0]} users, {ratings.shape[1]} items",
            )

            # Compute SVD reconstruction
            progress.update(task, description="Computing SVD reconstruction...")
            reconstructed = svd_reconstruct(ratings, k=k)

            # Generate recommendations
            progress.update(task, description="Generating recommendations...")
            recommendations = top_n(reconstructed, ratings, n=n)

            progress.update(task, description="Done!")

        # Display results
        console.print(
            f"\n[bold green]✓[/] Generated recommendations for {len(recommendations)} users"
        )

        # Show sample recommendations
        if len(recommendations) > 0:
            table = Table(
                title=f"Top-{n} Recommendations (first {min(max_users, len(recommendations))} users)"
            )
            table.add_column("User", style="cyan")
            table.add_column("Recommendations", style="green")

            for i, user_recs in enumerate(recommendations[:max_users]):
                table.add_row(f"User {i}", ", ".join(map(str, user_recs)))

            console.print(table)

        # Show metrics if requested
        if show_metrics:
            rmse = compute_rmse(reconstructed, ratings)
            mae = compute_mae(reconstructed, ratings)

            metrics_table = Table(title="Prediction Quality Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")

            metrics_table.add_row("RMSE", f"{rmse:.4f}")
            metrics_table.add_row("MAE", f"{mae:.4f}")

            console.print(metrics_table)

        # Save to file if requested
        if output:
            # Convert recommendations to matrix format for saving
            rec_matrix = np.zeros((len(recommendations), n), dtype=np.float32)
            for i, recs in enumerate(recommendations):
                rec_matrix[i, : len(recs)] = recs

            save_ratings(rec_matrix, output)
            console.print(f"[bold green]✓[/] Saved recommendations to {output}")

    except Exception as e:
        console.print(f"[bold red]✗[/] Error: {e}")
        raise SystemExit(1) from None


def bench(csv: Path, k: int = 50, n: int = 10, iterations: int = 1) -> None:
    """Minimal bench function for CLI benchmarking."""
    for i in range(iterations):
        console.print(f"\n[dim]Iteration {i + 1}/{iterations}[/]")
        ratings = load_ratings(csv)
        result = quick_benchmark(ratings, k=k, n_runs=1)

        if result is None:
            console.print(f"[red]Benchmark failed for k={k}[/]")
            continue

        console.print(
            f"[green]Time:[/] {result['time']:.4f}s | [cyan]RMSE:[/] {result['rmse']:.4f} | [magenta]MAE:[/] {result['mae']:.4f}"
        )


@cli.command()
def benchmark(
    csv: Path = typer.Argument(..., exists=True, help="Path to ratings CSV file"),
    k: int = typer.Option(50, "--rank", "-k", help="SVD rank for benchmarking"),
    n: int = typer.Option(
        10, "--top-n", "-n", help="Number of recommendations per user"
    ),
    iterations: int = typer.Option(
        1, "--iterations", "-i", help="Number of benchmark iterations"
    ),
) -> None:
    """
    Run performance benchmark on the recommender system.

    Measures the time taken for loading data, SVD computation,
    and recommendation generation.

    Examples:
        vector-recsys benchmark ratings.csv
        vector-recsys benchmark ratings.csv --rank 100 --iterations 3
    """
    try:
        console.print(
            f"[bold cyan]Running benchmark with {iterations} iteration(s)...[/]"
        )
        bench(csv, k=k, n=n, iterations=iterations)
        if iterations > 1:
            console.print(
                f"\n[bold green]✓[/] Completed {iterations} benchmark iterations"
            )
    except Exception as e:
        console.print(f"[bold red]✗[/] Benchmark error: {e}")
        raise SystemExit(1) from None


@cli.command()
def sample(
    output: Path = typer.Argument(..., help="Output CSV file path"),
    users: int = typer.Option(100, "--users", "-u", help="Number of users"),
    items: int = typer.Option(50, "--items", "-i", help="Number of items"),
    sparsity: float = typer.Option(
        0.8, "--sparsity", "-s", help="Sparsity level (0.0 = dense, 1.0 = sparse)"
    ),
    rating_min: float = typer.Option(1.0, "--min-rating", help="Minimum rating value"),
    rating_max: float = typer.Option(5.0, "--max-rating", help="Maximum rating value"),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for reproducible results"
    ),
) -> None:
    """
    Generate a sample rating matrix for testing.

    Creates a synthetic rating matrix with specified dimensions and properties.
    Useful for testing and development.

    Examples:
        vector-recsys sample test_ratings.csv
        vector-recsys sample large_ratings.csv --users 1000 --items 200 --sparsity 0.9
    """
    try:
        console.print("Generating sample rating matrix...")

        ratings = create_sample_ratings(
            n_users=users,
            n_items=items,
            sparsity=sparsity,
            rating_range=(rating_min, rating_max),
            random_state=seed,
        )

        save_ratings(ratings, output)

        # Show statistics
        non_zero = np.count_nonzero(ratings)
        total = ratings.size
        actual_sparsity = 1.0 - (non_zero / total)

        stats_table = Table(title="Sample Matrix Statistics")
        stats_table.add_column("Property", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Shape", f"{ratings.shape[0]} × {ratings.shape[1]}")
        stats_table.add_row("Non-zero ratings", str(non_zero))
        stats_table.add_row("Actual sparsity", f"{actual_sparsity:.3f}")
        stats_table.add_row("Rating range", f"{rating_min:.1f} - {rating_max:.1f}")

        console.print(stats_table)
        console.print(f"[bold green]✓[/] Saved sample matrix to {output}")

    except Exception as e:
        console.print(f"[bold red]✗[/] Error generating sample: {e}")
        raise SystemExit(1) from None


@cli.command()
def info(
    file: Path = typer.Argument(..., exists=True, help="Path to ratings file"),
) -> None:
    """Show info about a ratings file (stub)."""
    console.print(f"[bold cyan]Info:[/] {file}")
    sys.exit(0)


@cli.command()
def convert(
    input_file: Path = typer.Argument(..., exists=True, help="Input file"),
    output_file: Path = typer.Argument(..., help="Output file"),
) -> None:
    """Convert between supported formats (stub)."""
    console.print(f"[bold cyan]Convert:[/] {input_file} -> {output_file}")
    sys.exit(0)


@cli.command()
def evaluate(
    actual_file: Path = typer.Argument(..., exists=True, help="Actual ratings file"),
    pred_file: Path = typer.Argument(..., exists=True, help="Predicted ratings file"),
) -> None:
    """Evaluate predictions (stub)."""
    console.print(f"[bold cyan]Evaluate:[/] {actual_file} vs {pred_file}")
    sys.exit(0)


if __name__ == "__main__":
    cli()
