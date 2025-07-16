# mypy: disable-error-code=unreachable
"""Command-line interface for vector_recsys_lite."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .algo import compute_mae, compute_rmse, svd_reconstruct, top_n
from .benchmark import quick_benchmark
from .explain import visualize_svd
from .io import create_sample_ratings, load_ratings, save_ratings
from .utils import as_dense

console = Console()

cli = typer.Typer(
    help="🧊 Fast SVD-based recommender system with optional Numba acceleration",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

__all__ = ["cli"]


@cli.command()
def predict(
    csv: Path = typer.Argument(..., exists=True),
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
    explain: bool = typer.Option(
        False, "--explain", "-e", help="Explain SVD steps for educational purposes"
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

            if explain:
                console.print(
                    "[bold]Step 1: Input Matrix Sample (first 5 users x 5 items)[/]"
                )
                console.print(str(as_dense(ratings)[:5, :5]))

            # Compute SVD reconstruction
            progress.update(task, description="Computing SVD reconstruction...")
            reconstructed = svd_reconstruct(ratings, k=k)
            # Ensure dense for downstream metrics and recommendations
            reconstructed = as_dense(reconstructed)

            if explain:
                visualize_svd(ratings, k)  # Assuming visualize_svd handles the printing

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
            rmse = compute_rmse(reconstructed, as_dense(ratings))
            mae = compute_mae(reconstructed, as_dense(ratings))

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
    try:
        has_psutil = True
    except ImportError:
        has_psutil = False
    for i in range(iterations):
        console.print(f"\n[dim]Iteration {i + 1}/{iterations}[/]")
        if has_psutil:
            start_mem = psutil.Process().memory_info().rss / 1024 / 1024
            start_cpu = psutil.cpu_percent()
        ratings = load_ratings(csv)
        result = quick_benchmark(ratings, k=k, n_runs=1)
        if has_psutil:
            end_mem = psutil.Process().memory_info().rss / 1024 / 1024
            end_cpu = psutil.cpu_percent()
            console.print(
                f"CPU: {end_cpu - start_cpu}% | RAM used: {end_mem - start_mem:.2f} MB"
            )
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

        stats_table.add_row("Shape", f"{ratings.shape[0]} x {ratings.shape[1]}")
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


@cli.command()
def deploy(
    model: Path = typer.Argument(..., exists=True, help="Path to saved model file"),
    port: int = typer.Option(8000, "--port", "-p", help="Port for the API server"),
) -> None:
    """
    Generate a ready-to-run FastAPI deployment script.
    """

    script = f"""
from fastapi import FastAPI
import numpy as np
from vector_recsys_lite import RecommenderSystem

app = FastAPI()
rec = RecommenderSystem.load('{model}')

@app.get('/recommend/{{user_id}}')
def recommend(user_id: int, n: int = 5):
    # Dummy ratings for demo; replace with real logic
    ratings = np.zeros((1, 10))  # Adjust shape
    recs = rec.recommend(ratings, n=n)[0]
    return {{'recommendations': recs.tolist()}}
"""
    output_file = "deploy_app.py"
    with open(output_file, "w") as f:
        f.write(script)
    console.print(
        f"[green]Generated {output_file}. Run: uvicorn {output_file}:app --port {port}"
    )


@cli.command()
def teach(
    concept: str = typer.Option(
        "svd",
        "--concept",
        "-c",
        help="Concept to teach: svd, matrix, factors, recommendations",
    ),
) -> None:
    """
    Interactive teaching mode for learning recommender system concepts.

    Examples:
        vector-recsys teach --concept svd
        vector-recsys teach --concept matrix
    """
    console.print(f"[bold cyan]Teaching Mode: {concept.upper()}[/bold cyan]\n")

    if concept == "svd":
        console.print("[yellow]Singular Value Decomposition (SVD) Explained:[/yellow]")
        console.print("SVD breaks down a matrix into 3 components:")
        console.print("• U: User features (what users like)")
        console.print("• S: Importance values (how strong each pattern is)")
        console.print("• V: Item features (what makes items similar)\n")

        if typer.confirm("Would you like to see a demo?"):
            console.print("\nCreating a small example...")
            ratings = create_sample_ratings(n_users=3, n_items=4, sparsity=0.5)
            console.print(f"Sample ratings:\n{ratings}\n")

            console.print("Running SVD...")
            visualize_svd(ratings, k=2)

    elif concept == "matrix":
        console.print("[yellow]User-Item Rating Matrix:[/yellow]")
        console.print("• Rows = Users")
        console.print("• Columns = Items")
        console.print("• Values = Ratings (0 means unrated)")
        console.print("• Goal: Predict the 0s!\n")

        if typer.confirm("Generate example matrix?"):
            mat = create_sample_ratings(n_users=4, n_items=5, sparsity=0.6)
            console.print(f"\nExample matrix:\n{mat}")
            console.print(f"\nShape: {mat.shape[0]} users × {mat.shape[1]} items")
            console.print(
                f"Sparsity: {np.count_nonzero(mat == 0) / mat.size:.1%} unrated"
            )

    elif concept == "factors":
        console.print("[yellow]Latent Factors:[/yellow]")
        console.print("Think of factors as hidden preferences:")
        console.print("• Factor 1 might be 'Action vs Romance'")
        console.print("• Factor 2 might be 'Old vs New'")
        console.print("• Users and items are positioned in this space\n")
        console.print("SVD automatically discovers these patterns!")

    elif concept == "recommendations":
        console.print("[yellow]Making Recommendations:[/yellow]")
        console.print("1. Start with incomplete ratings")
        console.print("2. Use SVD to find patterns")
        console.print("3. Reconstruct full matrix")
        console.print("4. Recommend highest predicted ratings\n")

        if typer.confirm("Run recommendation demo?"):
            ratings = create_sample_ratings(n_users=5, n_items=10, sparsity=0.7)
            reconstructed = svd_reconstruct(ratings, k=3)
            recs = top_n(reconstructed, ratings, n=3)

            console.print("\nTop-3 recommendations per user:")
            for i, user_recs in enumerate(recs[:3]):
                console.print(f"User {i}: Items {user_recs}")
    else:
        console.print(f"[red]Unknown concept: {concept}[/red]")
        console.print("Available: svd, matrix, factors, recommendations")


if __name__ == "__main__":
    cli()
