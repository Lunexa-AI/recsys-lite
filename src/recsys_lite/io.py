# mypy: disable-error-code=operator

"""
I/O utilities for loading and saving rating matrices.

This module provides flexible I/O operations for rating matrices
in various formats including CSV, JSON, Parquet, HDF5, SQLite, and NPZ.
"""

import csv
import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from scipy import sparse
from scipy.sparse import csr_matrix

from .utils import as_dense

# Type aliases
FloatMatrix = Union[np.ndarray, csr_matrix]
FloatMatrixOptional = Optional[FloatMatrix]

# Console for rich output
console = Console()

# Try to import optional dependencies
try:
    import h5py

    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False


class DataLoader:
    """Smart data loader with format auto-detection and optimization."""

    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress
        self._supported_formats = {
            ".csv": self._load_csv,
            ".json": self._load_json,
            ".parquet": self._load_parquet,
            ".h5": self._load_hdf5,
            ".hdf5": self._load_hdf5,
            ".db": self._load_sqlite,
            ".sqlite": self._load_sqlite,
        }

    def load(
        self,
        path: Union[str, Path],
        *,
        format: Optional[str] = None,
        sparse_format: bool = False,
        **kwargs: Any,
    ) -> FloatMatrix:
        """Load matrix from file with format detection."""
        path = Path(path)
        if format is None:
            format = self._detect_format(path)
        if format == "csv":
            return self._load_csv(path, **kwargs)
        if format == "json":
            return self._load_json(path, **kwargs)
        if format == "parquet":
            return self._load_parquet(path, **kwargs)
        if format == "hdf5":
            return self._load_hdf5(path, **kwargs)
        if format == "sqlite":
            return self._load_sqlite(path, **kwargs)
        if format == "npz":
            # Load sparse matrix from NPZ
            return sparse.load_npz(path)
        raise ValueError(f"Unsupported format: {format}")

    def _detect_format(self, path: Union[str, Path]) -> str:
        """Detect file format from extension or content."""
        path = Path(path)
        ext = path.suffix.lower()
        if ext in {".csv"}:
            return "csv"
        if ext in {".json"}:
            return "json"
        if ext in {".parquet"}:
            return "parquet"
        if ext in {".h5", ".hdf5"}:
            return "hdf5"
        if ext in {".npz"}:
            return "npz"
        if ext in {".db", ".sqlite"}:
            return "sqlite"
        # Fallback: try to guess from content
        with open(path, "rb") as f:
            start = f.read(4)
            if start == b"PK\x03\x04":
                return "npz"
        # Default to CSV
        return "csv"

    def _load_csv(
        self,
        path: Union[str, Path],
        delimiter: str = ",",
        missing_value: float = 0.0,
        sparse_format: bool = False,
        **kwargs: Any,
    ) -> FloatMatrix:
        """Load matrix from CSV file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}. Please check the file path and try again."
            )

        try:
            # Try pandas first for better performance and missing value handling
            import pandas as pd

            df = pd.read_csv(
                path,
                delimiter=delimiter,
                header=None,
                na_values=["", "nan", "na", "null", "NULL"],
                keep_default_na=True,
            )
            matrix = df.fillna(missing_value).values.astype(np.float32)
        except Exception:
            # Fallback to manual CSV parsing for more control
            rows = []
            expected_len = None
            with open(path, encoding="utf-8") as f:
                for row_num, line in enumerate(f):
                    row = line.strip().split(delimiter)
                    if not row or all(cell.strip() == "" for cell in row):
                        continue
                    float_row = []
                    for col_num, cell in enumerate(row):
                        cell = cell.strip()
                        if cell == "" or cell.lower() in ("nan", "na", "null"):
                            float_row.append(missing_value)
                        else:
                            try:
                                float_row.append(float(cell))
                            except ValueError:
                                raise ValueError(
                                    f"Invalid value at row {row_num + 1}, col {col_num + 1}: {cell}"
                                ) from None
                    if expected_len is None:
                        expected_len = len(float_row)
                    elif len(float_row) != expected_len:
                        raise ValueError(
                            f"CSV file '{path}' has inconsistent row lengths at row {row_num + 1}."
                        )
                    rows.append(float_row)
            if not rows:
                raise ValueError(f"CSV file '{path}' must contain a 2D matrix.")
            matrix = np.array(rows, dtype=np.float32)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)

        if matrix.ndim != 2:
            raise ValueError(
                f"CSV file '{path}' must contain a 2D matrix. Got shape {matrix.shape}."
            )

        if sparse_format:
            from scipy import sparse

            return sparse.csr_matrix(matrix)
        return matrix

    def _load_csv_manual(
        self, path: Path, delimiter: str, missing_value: float
    ) -> np.ndarray:
        """Manual CSV loading for complex formats."""
        rows = []
        expected_len = None
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)

            for row_num, row in enumerate(reader):
                if not row:  # Skip empty rows
                    continue

                # Convert row to floats, handling missing values
                float_row = []
                for col_num, cell in enumerate(row):
                    cell = cell.strip()
                    if cell == "" or cell.lower() in ("nan", "na", "null"):
                        float_row.append(missing_value)
                    else:
                        try:
                            float_row.append(float(cell))
                        except ValueError as e:
                            raise ValueError(
                                f"Invalid value at row {row_num + 1}, col {col_num + 1}: {cell}"
                            ) from e
                if expected_len is None:
                    expected_len = len(float_row)
                elif len(float_row) != expected_len:
                    raise ValueError(
                        f"CSV file '{path}' has inconsistent row lengths at row {row_num + 1}."
                    )
                rows.append(float_row)
        if not rows:
            raise ValueError(f"CSV file '{path}' is empty or contains no valid data.")
        # Convert to numpy array
        matrix = np.array(rows, dtype=np.float32)
        # Ensure all rows have the same number of columns
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return matrix

    def _load_json(self, path: Union[str, Path], **kwargs: Any) -> FloatMatrix:
        """Load matrix from JSON file."""
        path = Path(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            matrix = np.array(data, dtype=np.float32)
        elif isinstance(data, dict):
            if "matrix" in data:
                matrix = np.array(data["matrix"], dtype=np.float32)
            elif "data" in data:
                matrix = np.array(data["data"], dtype=np.float32)
            else:
                raise ValueError("JSON must contain 'matrix' or 'data' key")
        else:
            raise ValueError("JSON must contain a list or dict with matrix data")

        if matrix.ndim != 2:
            raise ValueError("JSON must contain a 2D matrix")

        return matrix

    def _load_parquet(self, path: Union[str, Path], **kwargs: Any) -> FloatMatrix:
        """Load matrix from Parquet file."""
        try:
            df = pd.read_parquet(path)
            matrix = df.values.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to load Parquet file: {e}") from e

        if matrix.ndim != 2:
            raise ValueError("Parquet must contain a 2D matrix")

        return matrix

    def _load_hdf5(
        self, path: Union[str, Path], key: str = "matrix", **kwargs: Any
    ) -> FloatMatrix:
        """Load matrix from HDF5 file."""
        try:
            import h5py

            with h5py.File(path, "r") as f:
                if key in f:
                    matrix = f[key][:].astype(np.float32)
                else:
                    available_keys = list(f.keys())
                    raise ValueError(
                        f"Key '{key}' not found. Available keys: {available_keys}"
                    )
        except ImportError as e:
            raise ImportError(
                "h5py is required for HDF5 support. Install with: pip install h5py"
            ) from e
        except Exception as e:
            raise ValueError(f"Failed to load HDF5 file: {e}") from e

        if matrix.ndim != 2:
            raise ValueError(
                f"HDF5 file '{path}' must contain a 2D matrix. Got shape {matrix.shape}."
            )

        return matrix

    def _load_sqlite(
        self,
        path: Union[str, Path],
        table: str = "ratings",
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
        **kwargs: Any,
    ) -> FloatMatrix:
        """Load matrix from SQLite database."""
        db_url = f"sqlite:///{path}" if isinstance(path, Path) else str(path)

        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(db_url)
            with engine.connect() as conn:
                # Get table info
                result = conn.execute(text(f"PRAGMA table_info({table})"))
                columns = [row[1] for row in result.fetchall()]

                if (
                    user_col not in columns
                    or item_col not in columns
                    or rating_col not in columns
                ):
                    raise ValueError(
                        f"Table must contain columns: {user_col}, {item_col}, {rating_col}"
                    )

                # Validate identifiers
                user_col = _validate_identifier(user_col)
                item_col = _validate_identifier(item_col)
                rating_col = _validate_identifier(rating_col)
                table = _validate_identifier(table)
                # Create pivot table
                query = f"""
                SELECT {user_col}, {item_col}, {rating_col}
                FROM {table}
                ORDER BY {user_col}, {item_col}
                """  # nosec
                df = pd.read_sql(query, conn)

                # Create pivot matrix
                matrix = df.pivot_table(
                    index=user_col, columns=item_col, values=rating_col, fill_value=0.0
                ).values.astype(np.float32)

        except ImportError as e:
            raise ImportError(
                "sqlalchemy is required for SQLite support. Install with: pip install sqlalchemy"
            ) from e
        except Exception as e:
            raise ValueError(f"Failed to load from database: {e}") from e

        return matrix


class DataSaver:
    """Smart data saver with format optimization."""

    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress
        self._supported_formats = {
            ".csv": self._save_csv,
            ".json": self._save_json,
            ".parquet": self._save_parquet,
            ".h5": self._save_hdf5,
            ".hdf5": self._save_hdf5,
            ".npz": self._save_sparse,
        }

    def save(
        self,
        matrix: Union[FloatMatrix, sparse.csr_matrix],
        path: Union[str, Path],
        *,
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Save rating matrix with automatic format detection.

        Parameters
        ----------
        matrix : Union[FloatMatrix, sparse.spmatrix]
            Rating matrix to save.
        path : Union[str, Path]
            Output file path.
        format : Optional[str], default=None
            Explicit format override. If None, auto-detected from file extension.
        **kwargs : Any
            Additional format-specific parameters.
        """
        path = Path(path)

        if format is None:
            format = path.suffix.lower()
        else:
            # Normalize: ensure format starts with a dot
            if not format.startswith("."):
                format = f".{format.lower()}"
            else:
                format = format.lower()

        if format not in self._supported_formats:
            raise ValueError(f"Unsupported format: {format}")

        saver_func = self._supported_formats[format]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console if self.show_progress else None,
            disable=not self.show_progress,
        ) as progress:
            task = progress.add_task(f"Saving {format.upper()} file...", total=None)

            try:
                saver_func(matrix, path, **kwargs)
                progress.update(task, description=f"✅ Saved {format.upper()} file")

            except Exception as e:
                progress.update(
                    task, description=f"❌ Failed to save {format.upper()} file"
                )
                raise e

    def _save_csv(
        self,
        matrix: Union[FloatMatrix, sparse.csr_matrix],
        path: Path,
        delimiter: str = ",",
        fmt: str = "%.3f",
        **kwargs: Any,
    ) -> None:
        """Save matrix to CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        matrix = as_dense(matrix)

        # Validate matrix dimensions
        if matrix.ndim != 2:
            raise ValueError(f"Matrix must be 2D, got {matrix.ndim}D")

        np.savetxt(path, matrix, delimiter=delimiter, fmt=fmt, encoding="utf-8")

    def _save_json(
        self, matrix: Union[FloatMatrix, sparse.csr_matrix], path: Path, **kwargs: Any
    ) -> None:
        """Save matrix to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        matrix = as_dense(matrix)

        data = {
            "matrix": matrix.tolist(),
            "shape": matrix.shape,
            "dtype": str(matrix.dtype),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _save_parquet(
        self, matrix: Union[FloatMatrix, sparse.csr_matrix], path: Path, **kwargs: Any
    ) -> None:
        """Save matrix to Parquet file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        matrix = as_dense(matrix)

        df = pd.DataFrame(matrix)
        df.to_parquet(path, index=False)

    def _save_hdf5(
        self,
        matrix: Union[FloatMatrix, sparse.csr_matrix],
        path: Path,
        key: str = "matrix",
        **kwargs: Any,
    ) -> None:
        """Save matrix to HDF5 file."""
        try:
            import h5py

            path.parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(path, "w") as f:
                matrix = as_dense(matrix)
                f.create_dataset(key, data=matrix, compression="gzip")

        except ImportError as e:
            raise ImportError(
                "h5py is required for HDF5 support. Install with: pip install h5py"
            ) from e

    def _save_sparse(
        self, matrix: Union[FloatMatrix, sparse.csr_matrix], path: Path, **kwargs: Any
    ) -> None:
        """Save sparse matrix to NPZ file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        if sparse.issparse(matrix):
            sparse.save_npz(path, matrix)
        else:
            # Convert to sparse for storage
            sparse_matrix = sparse.csr_matrix(matrix)
            sparse.save_npz(path, sparse_matrix)


def _validate_identifier(identifier: str) -> str:
    """Validate that a SQL identifier (table/column name) is safe."""
    if not identifier.replace("_", "").isalnum():
        raise ValueError(f"Invalid SQL identifier: {identifier}")
    return identifier


def _detect_format(path: Path) -> str:
    """Detect file format from extension."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return "csv"
    elif ext == ".json":
        return "json"
    elif ext == ".parquet":
        return "parquet"
    elif ext == ".h5" or ext == ".hdf5":
        return "hdf5"
    elif ext == ".db" or ext == ".sqlite":
        return "sqlite"
    elif ext == ".npz":
        return "npz"
    else:
        return "csv"  # Default to CSV


def _load_csv(path: Path, **kwargs: Any) -> FloatMatrix:
    """Load matrix from CSV file."""
    delimiter = kwargs.get("delimiter", ",")
    missing_value = kwargs.get("missing_value", 0.0)

    try:
        matrix = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)
        # Replace missing values
        matrix = np.where(matrix == missing_value, 0.0, matrix)
        return matrix
    except Exception as e:
        raise ValueError(f"Failed to load CSV file '{path}': {e}") from e


def _load_json(path: Path, **kwargs: Any) -> FloatMatrix:
    """Load matrix from JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, dict) and "matrix" in data:
            matrix = np.array(data["matrix"], dtype=np.float32)
        else:
            matrix = np.array(data, dtype=np.float32)

        return matrix
    except Exception as e:
        raise ValueError(f"Failed to load JSON: {e}") from e


def _load_parquet(path: Path, **kwargs: Any) -> FloatMatrix:
    """Load matrix from Parquet file."""
    try:
        df = pd.read_parquet(path)
        matrix = df.values.astype(np.float32)
        return matrix
    except ImportError as e:
        raise ImportError("pyarrow is required for Parquet support") from e
    except Exception as e:
        raise ValueError(f"Failed to load Parquet: {e}") from e


def _load_hdf5(path: Path, **kwargs: Any) -> FloatMatrix:
    """Load matrix from HDF5 file."""
    if not HAS_HDF5:
        raise ImportError("h5py is required for HDF5 support")

    key = kwargs.get("key", "matrix")
    try:
        with h5py.File(path, "r") as f:
            matrix = f[key][:].astype(np.float32)
        return matrix
    except Exception as e:
        raise ValueError(f"Failed to load HDF5: {e}") from e


def _load_sqlite(path: Path, **kwargs: Any) -> FloatMatrix:
    """Load matrix from SQLite database."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError as e:
        raise ImportError("sqlalchemy is required for SQLite support") from e
    table = kwargs.get("table", "ratings")
    table = _validate_identifier(table)
    try:
        engine = create_engine(f"sqlite:///{path}")
        with engine.connect() as conn:
            # Use validated table name
            result = conn.execute(text(f"SELECT * FROM {table}"))  # nosec
            data = result.fetchall()
            matrix = np.array(data, dtype=np.float32)
        return matrix
    except Exception as e:
        raise ValueError(f"Failed to load SQLite: {e}") from e


def load_ratings(
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    sparse_format: bool = False,
    **kwargs: Any,
) -> FloatMatrix:
    """Load matrix from file with format detection."""
    path = Path(path)

    # Auto-detect format if not specified
    if format is None:
        format = _detect_format(path)

    # Load based on format
    if format == "csv":
        return DataLoader()._load_csv(path, sparse_format=sparse_format, **kwargs)
    elif format == "json":
        return _load_json(path, **kwargs)
    elif format == "parquet":
        return _load_parquet(path, **kwargs)
    elif format == "hdf5":
        return _load_hdf5(path, **kwargs)
    elif format == "sqlite":
        return _load_sqlite(path, **kwargs)
    elif format == "npz":
        # Load sparse matrix from NPZ
        try:
            from scipy.sparse import load_npz

            return load_npz(path)
        except Exception as e:
            raise ValueError(f"Failed to load NPZ: {e}") from e
    raise ValueError(f"Unsupported format: {format}")


def save_ratings(
    matrix: Union[FloatMatrix, sparse.csr_matrix],
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    show_progress: bool = True,
    **kwargs: Any,
) -> None:
    """
    Save rating matrix with automatic format detection.

    Supports CSV, JSON, Parquet, HDF5, and NPZ formats with optimization.

    Parameters
    ----------
    matrix : Union[FloatMatrix, sparse.spmatrix]
        Rating matrix to save.
    path : Union[str, Path]
        Output file path.
    format : Optional[str], default=None
        Explicit format override. If None, auto-detected from file extension.
    show_progress : bool, default=True
        Whether to show progress bar during saving.
    **kwargs : Any
        Additional format-specific parameters.

    Examples
    --------
    >>> from vector_recsys_lite import save_ratings
    >>>
    >>> # Save to CSV
    >>> save_ratings(matrix, "ratings.csv")
    >>>
    >>> # Save to JSON
    >>> save_ratings(matrix, "ratings.json")
    >>>
    >>> # Save sparse matrix to NPZ
    >>> save_ratings(sparse_matrix, "ratings.npz")
    """
    path = Path(path)
    ext = path.suffix.lower()
    if format is None:
        format = ext[1:] if ext.startswith(".") else ext
    if format == "npz":
        if not sparse.issparse(matrix):
            matrix = sparse.csr_matrix(matrix)
        sparse.save_npz(path, matrix)
        return
    saver = DataSaver(show_progress=show_progress)
    saver.save(matrix, path, format=format, **kwargs)


def create_sample_ratings(
    n_users: int = 100,
    n_items: int = 50,
    sparsity: float = 0.8,
    rating_range: tuple[float, float] = (1.0, 5.0),
    random_state: Optional[int] = None,
    sparse_format: bool = False,
) -> Union[FloatMatrix, sparse.csr_matrix]:
    """
    Create a sample rating matrix for testing.

    Parameters
    ----------
    n_users : int, default=100
        Number of users.
    n_items : int, default=50
        Number of items.
    sparsity : float, default=0.8
        Fraction of missing ratings (0.0 = fully dense, 1.0 = fully sparse).
    rating_range : tuple[float, float], default=(1.0, 5.0)
        Range for rating values (min, max).
    random_state : Optional[int], default=None
        Random seed for reproducible results.
    sparse_format : bool, default=False
        Whether to return a sparse matrix.

    Returns
    -------
    Union[FloatMatrix, sparse.csr_matrix]
        Sample rating matrix.

    Examples
    --------
    >>> from vector_recsys_lite import create_sample_ratings
    >>>
    >>> # Create a small sample matrix
    >>> ratings = create_sample_ratings(10, 5, sparsity=0.5)
    >>> print(f"Shape: {ratings.shape}")
    >>> print(f"Non-zero elements: {np.count_nonzero(ratings)}")
    >>>
    >>> # Create sparse matrix
    >>> sparse_ratings = create_sample_ratings(100, 50, sparse_format=True)
    """
    if not 0 <= sparsity <= 1:
        raise ValueError(f"Sparsity must be between 0 and 1, got {sparsity}")

    if rating_range[0] >= rating_range[1]:
        raise ValueError(f"Invalid rating range: {rating_range}")

    if n_users <= 0 or n_items <= 0:
        raise ValueError("n_users and n_items must be positive")

    rng = np.random.RandomState(random_state)

    # Create dense matrix with random ratings
    matrix = rng.uniform(
        low=rating_range[0], high=rating_range[1], size=(n_users, n_items)
    ).astype(np.float32)

    # Apply sparsity by setting some ratings to 0
    if sparsity > 0:
        mask = rng.random((n_users, n_items)) < sparsity
        matrix[mask] = 0.0

    if sparse_format:
        return sparse.csr_matrix(matrix)
    else:
        return matrix


__all__ = [
    "load_ratings",
    "save_ratings",
    "create_sample_ratings",
]
