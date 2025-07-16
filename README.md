# vector_recsys_lite 🧊

[![CI](https://github.com/Lunexa-AI/vector-recsys-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/Lunexa-AI/vector-recsys-lite/actions)
[![PyPI](https://img.shields.io/pypi/v/vector_recsys_lite)](https://pypi.org/project/vector_recsys_lite/)
[![Python](https://img.shields.io/badge/python->=3.9-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/vector_recsys_lite)](https://pypi.org/project/vector_recsys_lite/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **The lightweight recommender system for teaching and small-scale production**

Built for educators and students in resource-constrained environments, vector-recsys-lite makes recommendation systems accessible to everyone. Whether you're teaching SVD math on a 10-year-old laptop in a university with unreliable internet, or deploying a small recommendation service for a local library, this library has you covered.

## 🎯 Who This Is For

- **Educators** teaching recommendation systems with limited lab resources
- **Students** learning on old hardware or with poor internet connectivity
- **Developers** building small-scale recommendation features (< 10k users/items)
- **Researchers** needing a simple, hackable SVD implementation

## 🌍 Why We Built This

In many parts of the world, especially in African universities and developing regions, students and educators face:
- Old computers with limited RAM (2-4GB)
- Unreliable or no internet connectivity
- Need for practical, hands-on learning tools
- Requirement for production-ready code that works at small scale

vector-recsys-lite solves these problems by being truly lightweight, offline-capable, and educationally focused while maintaining production quality.

## ✨ Key Features

- **🎓 Educational First** - Built-in teaching mode, step-by-step explanations, Jupyter examples
- **💾 Truly Lightweight** - Core runs on 2GB RAM, minimal dependencies (NumPy + SciPy only)
- **📡 Offline Capable** - Works without internet, includes offline install guide
- **⚡ Fast on Old Hardware** - Optimized for older CPUs, optional Numba acceleration
- **🔧 Simple Yet Extensible** - Clean API for learning, powerful enough for production
- **📊 Multi-format I/O** - CSV, JSON, Parquet support for various data sources
- **🚀 Quick Deployment** - One-command API generation for small-scale production

Built for both rapid prototyping and production deployment, vector-recsys-lite delivers enterprise-grade recommendation capabilities without the complexity of heavyweight frameworks.

## 🚀 Quick Start

### Installation

```bash
# From PyPI
pip install vector_recsys_lite

# From source
git clone https://github.com/Lunexa-AI/lunexa-labs.git
cd lunexa-labs/packages/vector_recsys_lite
poetry install
```

#### Offline Installation (for low-network environments)
To install without internet:
1. On a machine with internet: `pip download vector_recsys_lite --no-deps`
2. Transfer the .whl file via USB.
3. On target machine: `pip install vector_recsys_lite-<version>-py3-none-any.whl`
This works on old laptops without network.

### Basic Usage

```python
import numpy as np
from vector_recsys_lite import svd_reconstruct, top_n

# Create sample ratings matrix (users × items)
ratings = np.array([
    [5, 3, 0, 1],  # User 1: rated items 0,1,3
    [0, 0, 4, 5],  # User 2: rated items 2,3
    [1, 0, 0, 4],  # User 3: rated items 0,3
], dtype=np.float32)

# Generate recommendations
reconstructed = svd_reconstruct(ratings, k=2)
recommendations = top_n(reconstructed, ratings, n=3)

print(f"Top-3 recommendations for {len(recommendations)} users:")
for i, recs in enumerate(recommendations):
    print(f"User {i}: {recs}")
```

### CLI Usage

```bash
# Generate sample data
vector-recsys sample ratings.csv --users 100 --items 50 --sparsity 0.8

# Generate recommendations
vector-recsys predict ratings.csv --rank 20 --top-n 5 --metrics

# Run benchmarks
vector-recsys benchmark ratings.csv --rank 50 --iterations 3
```

## 🎓 Educational Usage

### Interactive Teaching Mode
Perfect for classrooms without Jupyter:

```bash
# Teach SVD concepts interactively
vector-recsys teach --concept svd

# Explain matrices and ratings
vector-recsys teach --concept matrix

# Show how recommendations work
vector-recsys teach --concept recommendations
```

### Jupyter Notebooks
For labs with Jupyter (optional):

```bash
# Check out our examples
cd examples/
jupyter notebook svd_math_demo.ipynb
```

### Explaining Results
Use `--explain` flag for step-by-step breakdowns:

```bash
# See the math behind predictions
vector-recsys predict ratings.csv --explain
```

## 📊 Features

### 🎯 **Core Algorithms**
- **Truncated SVD**: Matrix factorization for collaborative filtering
- **Top-N Recommendations**: Exclude known items, rank by predicted ratings
- **Sparse Matrix Support**: Memory-efficient for large datasets
- **Numba Acceleration**: Optional JIT compilation for performance

### 🔒 **Security & Reliability**
- **Secure Serialization**: Uses `joblib` instead of `pickle` for model saving
- **SQL Injection Protection**: Parameterized queries with identifier validation
- **Robust Error Handling**: Comprehensive validation and error messages
- **Type Safety**: Full type annotations with mypy compliance

### 📁 **Multi-Format I/O**
- **CSV**: Standard comma/tab-separated files
- **JSON**: Structured data with metadata
- **Parquet**: Columnar format for large datasets
- **HDF5**: Hierarchical data format
- **NPZ**: NumPy compressed format for sparse matrices
- **SQLite**: Database storage with pivot table support

### 🎨 **Beautiful CLI**
- **Rich Interface**: Progress bars, tables, and colored output
- **Auto-detection**: Smart format detection from file extensions
- **Comprehensive Help**: Detailed command documentation
- **Benchmarking**: Performance testing with timing and metrics

## 🛠️ CLI Commands

### `vector-recsys sample`
Generate synthetic rating matrices for testing.

```bash
# Basic usage
vector-recsys sample ratings.csv

# Custom parameters
vector-recsys sample large_ratings.csv \
  --users 1000 \
  --items 200 \
  --sparsity 0.9 \
  --min-rating 1.0 \
  --max-rating 5.0 \
  --seed 42
```

### `vector-recsys predict`
Generate top-N recommendations for users.

```bash
# Basic usage
vector-recsys predict ratings.csv

# With custom parameters
vector-recsys predict ratings.csv \
  --rank 50 \
  --top-n 10 \
  --output recommendations.csv \
  --metrics \
  --max-users 5
```

### `vector-recsys benchmark`
Run performance benchmarks.

```bash
# Single benchmark
vector-recsys benchmark ratings.csv

# Multiple iterations
vector-recsys benchmark ratings.csv \
  --rank 100 \
  --iterations 5
```

### `vector-recsys info`
Show information about a ratings file.

```bash
vector-recsys info ratings.csv
```

### `vector-recsys convert`
Convert between supported formats.

```bash
vector-recsys convert ratings.csv ratings.json
vector-recsys convert ratings.csv ratings.parquet
```

### `vector-recsys evaluate`
Evaluate prediction quality.

```bash
vector-recsys evaluate actual.csv predicted.csv
```

## 📈 Performance

### Resource Usage

Designed for resource-constrained environments:

| Dataset Size | RAM Usage | Time (old laptop) | Time (modern PC) |
|--------------|-----------|-------------------|------------------|
| 100 × 50     | < 10 MB   | < 0.1s           | < 0.01s         |
| 1K × 1K      | < 50 MB   | < 1s             | < 0.1s          |
| 10K × 5K     | < 500 MB  | < 10s            | < 2s            |

### Tested On
- 10-year-old laptops (Core i3, 2GB RAM)
- Raspberry Pi 4
- Modern workstations
- Cloud containers (minimal resources)

### Memory Efficiency
- **Sparse matrix support**: Handles 90% sparse data efficiently
- **Chunked processing**: Works with limited RAM
- **Minimal dependencies**: ~50MB total install size

## 🔧 API Reference

### Core Functions

```python
def svd_reconstruct(
    mat: FloatMatrix,
    *,
    k: Optional[int] = None,
    random_state: Optional[int] = None,
    use_sparse: bool = True,
) -> FloatMatrix:
    """Truncated SVD reconstruction for collaborative filtering."""

def top_n(
    est: FloatMatrix,
    known: FloatMatrix,
    *,
    n: int = 10
) -> np.ndarray:
    """Get top-N items for each user, excluding known items."""

def compute_rmse(predictions: FloatMatrix, actual: FloatMatrix) -> float:
    """Compute Root Mean Square Error between predictions and actual ratings."""

def compute_mae(predictions: FloatMatrix, actual: FloatMatrix) -> float:
    """Compute Mean Absolute Error between predictions and actual ratings."""
```

### I/O Functions

```python
def load_ratings(
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    sparse_format: bool = False,
    **kwargs: Any,
) -> FloatMatrix:
    """Load matrix from file with format detection."""

def save_ratings(
    matrix: Union[FloatMatrix, sparse.csr_matrix],
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    show_progress: bool = True,
    **kwargs: Any,
) -> None:
    """Save rating matrix with automatic format detection."""

def create_sample_ratings(
    n_users: int = 100,
    n_items: int = 50,
    sparsity: float = 0.8,
    rating_range: tuple[float, float] = (1.0, 5.0),
    random_state: Optional[int] = None,
    sparse_format: bool = False,
) -> Union[FloatMatrix, sparse.csr_matrix]:
    """Create a sample rating matrix for testing."""
```

### RecommenderSystem Class

```python
class RecommenderSystem:
    """Production-ready recommender system with model persistence."""

    def fit(self, ratings: FloatMatrix, k: Optional[int] = None) -> "RecommenderSystem":
        """Fit the model to training data."""

    def predict(self, ratings: FloatMatrix) -> FloatMatrix:
        """Generate predictions for the input matrix."""

    def recommend(self, ratings: FloatMatrix, n: int = 10, exclude_rated: bool = True) -> np.ndarray:
        """Generate top-N recommendations for each user."""

    def save(self, path: str) -> None:
        """Save model to file using secure joblib serialization."""

    @classmethod
    def load(cls, path: str) -> "RecommenderSystem":
        """Load model from file."""
```

## 🧩 Real-World Usage Examples

### Using with Pandas

```python
import pandas as pd
from vector_recsys_lite import svd_reconstruct, top_n

# Load ratings from a DataFrame
ratings_df = pd.read_csv('ratings.csv', index_col=0)
ratings = ratings_df.values.astype(float)

# SVD recommendations
reconstructed = svd_reconstruct(ratings, k=20)
recommendations = top_n(reconstructed, ratings, n=5)
print(recommendations)
```

### Integrating in a Web App (FastAPI Example)

```python
from fastapi import FastAPI
from vector_recsys_lite import svd_reconstruct, top_n
import numpy as np

app = FastAPI()
ratings = np.load('ratings.npy')
reconstructed = svd_reconstruct(ratings, k=10)

@app.get('/recommend/{user_id}')
def recommend(user_id: int, n: int = 5):
    recs = top_n(reconstructed, ratings, n=n)
    return {"user_id": user_id, "recommendations": recs[user_id].tolist()}
```

## 📚 Use Cases

### Education
- **University courses**: Teach recommendation systems without expensive infrastructure
- **Self-learning**: Students can run everything on personal laptops
- **Workshops**: Quick demos that work offline
- **Research**: Simple baseline implementation for papers

### Small-Scale Production
- **School library**: Recommend books to students (500 books, 200 students)
- **Local business**: Product recommendations for small e-commerce
- **Community app**: Match local services to residents
- **Personal projects**: Add recommendations to your blog or app

### Development
- **Prototyping**: Test recommendation ideas quickly
- **Learning**: Understand SVD by reading clean, documented code
- **Benchmarking**: Compare against simple, fast baseline
- **Integration**: Easy to embed in larger systems

---

## ❓ FAQ

**Q: Why do I get 'package or version not found' on PyPI badges?**
A: The package must be published to PyPI for these badges to work. See the PyPI section above.

**Q: How do I enable Parquet or HDF5 support?**
A: Install with the appropriate extras: `poetry install --with parquet,hdf5` or `pip install vector_recsys_lite[parquet,hdf5]`.

**Q: Can I use this with very large datasets?**
A: Yes! Use the sparse matrix support and optional Numba acceleration for best performance.

**Q: How do I get reproducible results?**
A: Set the `random_state` parameter in SVD functions or CLI commands.

---

## ⚠️ Common Pitfalls

- **Missing optional dependencies:** If you see errors about `pyarrow`, `h5py`, or `sqlalchemy`, install the relevant extras.
- **Shape mismatches:** Ensure your ratings matrix is 2D (users × items).
- **Sparse vs. dense confusion:** Use `as_dense` to convert sparse matrices for downstream processing.
- **CLI not found:** Make sure you installed with `poetry install` or `pip install .` to get the `vector-recsys` command.

---

## 💬 How to Get Help

- **GitHub Issues:** [Open an issue](https://github.com/Lunexa-AI/vector-recsys-lite/issues) for bugs, questions, or feature requests.
- **Discussions:** Use the GitHub Discussions tab for general Q&A.
- **Email:** Contact the maintainer at stimire92@gmail.com for urgent matters.

---

## 🛠️ CLI Commands

### `vector-recsys sample`
Generate synthetic rating matrices for testing.

```bash
# Basic usage
vector-recsys sample ratings.csv

# Custom parameters
vector-recsys sample large_ratings.csv \
  --users 1000 \
  --items 200 \
  --sparsity 0.9 \
  --min-rating 1.0 \
  --max-rating 5.0 \
  --seed 42
```

### `vector-recsys predict`
Generate top-N recommendations for users.

```bash
# Basic usage
vector-recsys predict ratings.csv

# With custom parameters
vector-recsys predict ratings.csv \
  --rank 50 \
  --top-n 10 \
  --output recommendations.csv \
  --metrics \
  --max-users 5
```

### `vector-recsys benchmark`
Run performance benchmarks.

```bash
# Single benchmark
vector-recsys benchmark ratings.csv

# Multiple iterations
vector-recsys benchmark ratings.csv \
  --rank 100 \
  --iterations 5
```

### `vector-recsys info`
Show information about a ratings file.

```bash
vector-recsys info ratings.csv
```

### `vector-recsys convert`
Convert between supported formats.

```bash
vector-recsys convert ratings.csv ratings.json
vector-recsys convert ratings.csv ratings.parquet
```

### `vector-recsys evaluate`
Evaluate prediction quality.

```bash
vector-recsys evaluate actual.csv predicted.csv
```

## 📈 Performance

### Resource Usage

Designed for resource-constrained environments:

| Dataset Size | RAM Usage | Time (old laptop) | Time (modern PC) |
|--------------|-----------|-------------------|------------------|
| 100 × 50     | < 10 MB   | < 0.1s           | < 0.01s         |
| 1K × 1K      | < 50 MB   | < 1s             | < 0.1s          |
| 10K × 5K     | < 500 MB  | < 10s            | < 2s            |

### Tested On
- 10-year-old laptops (Core i3, 2GB RAM)
- Raspberry Pi 4
- Modern workstations
- Cloud containers (minimal resources)

### Memory Efficiency
- **Sparse matrix support**: Handles 90% sparse data efficiently
- **Chunked processing**: Works with limited RAM
- **Minimal dependencies**: ~50MB total install size

## 🔧 API Reference

### Core Functions

```python
def svd_reconstruct(
    mat: FloatMatrix,
    *,
    k: Optional[int] = None,
    random_state: Optional[int] = None,
    use_sparse: bool = True,
) -> FloatMatrix:
    """Truncated SVD reconstruction for collaborative filtering."""

def top_n(
    est: FloatMatrix,
    known: FloatMatrix,
    *,
    n: int = 10
) -> np.ndarray:
    """Get top-N items for each user, excluding known items."""

def compute_rmse(predictions: FloatMatrix, actual: FloatMatrix) -> float:
    """Compute Root Mean Square Error between predictions and actual ratings."""

def compute_mae(predictions: FloatMatrix, actual: FloatMatrix) -> float:
    """Compute Mean Absolute Error between predictions and actual ratings."""
```

### I/O Functions

```python
def load_ratings(
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    sparse_format: bool = False,
    **kwargs: Any,
) -> FloatMatrix:
    """Load matrix from file with format detection."""

def save_ratings(
    matrix: Union[FloatMatrix, sparse.csr_matrix],
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    show_progress: bool = True,
    **kwargs: Any,
) -> None:
    """Save rating matrix with automatic format detection."""

def create_sample_ratings(
    n_users: int = 100,
    n_items: int = 50,
    sparsity: float = 0.8,
    rating_range: tuple[float, float] = (1.0, 5.0),
    random_state: Optional[int] = None,
    sparse_format: bool = False,
) -> Union[FloatMatrix, sparse.csr_matrix]:
    """Create a sample rating matrix for testing."""
```

### RecommenderSystem Class

```python
class RecommenderSystem:
    """Production-ready recommender system with model persistence."""

    def fit(self, ratings: FloatMatrix, k: Optional[int] = None) -> "RecommenderSystem":
        """Fit the model to training data."""

    def predict(self, ratings: FloatMatrix) -> FloatMatrix:
        """Generate predictions for the input matrix."""

    def recommend(self, ratings: FloatMatrix, n: int = 10, exclude_rated: bool = True) -> np.ndarray:
        """Generate top-N recommendations for each user."""

    def save(self, path: str) -> None:
        """Save model to file using secure joblib serialization."""

    @classmethod
    def load(cls, path: str) -> "RecommenderSystem":
        """Load model from file."""
```

## 🐳 Docker Support

```bash
# Build Docker image
docker build -t vector-recsys-lite .

# Run CLI commands
docker run vector-recsys-lite vector-recsys --help
docker run vector-recsys-lite vector-recsys sample test.csv
```

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=vector_recsys_lite

# Run specific test categories
poetry run pytest tests/test_algo.py
poetry run pytest tests/test_io.py
poetry run pytest tests/test_cli.py
```

## 📦 Dependencies

### Required
- `numpy>=1.21.0`: Numerical computing
- `scipy>=1.7.0`: Sparse matrices and SVD
- `joblib>=1.1.0`: Secure model serialization
- `typer>=0.9.0`: CLI framework
- `rich>=13.0.0`: Beautiful terminal output

### Optional
- `numba>=0.56.0`: JIT compilation for performance
- `pandas>=1.3.0`: Data manipulation
- `pyarrow>=7.0.0`: Parquet file support
- `h5py>=3.7.0`: HDF5 file support
- `sqlalchemy>=1.4.0`: SQLite database support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and run: `make ci`
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Development Standards

- **Type hints**: All functions have type annotations
- **Docstrings**: All public functions have docstrings
- **Tests**: New features include tests
- **Linting**: Code passes all linting checks
- **Coverage**: Maintain >80% test coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- **NumPy/SciPy**: Core numerical computing
- **Numba**: JIT compilation for performance
- **Rich**: Beautiful terminal interface
- **Typer**: Modern CLI framework

---

> **"Fast, secure, and production-ready recommender systems for everyone."**

## 🗓️ Deprecation Policy

We strive to maintain backward compatibility and provide clear deprecation warnings. Deprecated features will:
- Be marked in the documentation and code with a warning.
- Remain available for at least one minor release cycle.
- Be removed only after clear notice in the changelog and release notes.

If you rely on a feature that is marked for deprecation, please open an issue to discuss migration strategies.

## 👩‍💻 Developer Quickstart

To get started as a contributor or to simulate CI locally:

```sh
# 1. Set up your full dev environment (Poetry, pre-commit, all dev deps)
make dev

# 2. Run all linters
make lint

# 3. Run all tests with coverage
make test

# 4. Run all pre-commit hooks
make precommit

# 5. Build the Docker image
make docker-build

# 6. Run tests inside Docker (as CI does)
make docker-test

# 7. Simulate the full CI pipeline (lint, test, coverage, Docker)
make ci
```

- All commands work on Linux, macOS, and CI.
- See CONTRIBUTING.md for more details.

<!-- Trigger CI: workflow file updated -->
