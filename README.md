# vector_recsys_lite 🧊

[![CI](https://github.com/Lunexa-AI/vector-recsys-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/Lunexa-AI/vector-recsys-lite/actions)
[![Python](https://img.shields.io/badge/python->=3.9-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/vector-recsys-lite.svg)](https://pypi.org/project/vector-recsys-lite/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

**Lightweight recommenders for teaching and small-scale production.**

## 🌟 Why Use vector-recsys-lite?

- **Purpose-built for teaching and learning:**
  Interactive CLI, educator guides, and Jupyter demos—no need to reinvent the wheel for every class or workshop.
- **Robust, tested, and production-ready:**
  Handles edge cases, bad input, and large/sparse data gracefully—unlike most one-off scripts.
- **Resource-light and offline-ready:**
  <50MB install, runs on 2GB RAM, and works without internet after install.
- **Simple, reproducible production:**
  One-command API deployment, clean CLI, and Docker support.
- **Multiple algorithms, modern tooling:**
  SVD, ALS (implicit), KNN (cosine), bias handling, chunked processing, and more.
- **Extensible and community-focused:**
  Easy to add new datasets, metrics, or algorithms. Contributor-friendly with templates and guides.
- **World-class documentation and examples:**
  Notebooks, educator guides, and API docs for fast onboarding.

---

## 🚀 Getting Started

### Installation

```bash
pip install vector_recsys_lite
```

Offline: See guide below.

### Quick Example

```python
import numpy as np
from vector_recsys_lite import svd_reconstruct, top_n

ratings = np.array([[5, 3, 0, 1], [0, 0, 4, 5]], dtype=np.float32)
reconstructed = svd_reconstruct(ratings, k=2)
recommendations = top_n(reconstructed, ratings, n=2)
print(recommendations)
```

CLI: `vector-recsys --help`

## 🎓 For Educators & Students

Teach recommendation systems without fancy hardware.

- **Interactive Mode**: `vector-recsys teach --concept svd` (prompts, examples)
- **Notebooks**: `examples/svd_math_demo.ipynb` (math breakdowns, plots)
- **Low-Resource Demos**: Generate data and run in <1s

Full guide: [docs/educator_guide.md](docs/educator_guide.md)

## 🚀 For Small App Deployers

Build production recommenders for <10k users/items.

- **Deploy API**: `vector-recsys deploy model.pkl`
- **Offline Install**: Download wheel, install via USB
- **Resource Efficient**: Sparse matrices for low memory

Examples: Library book recommender, local e-commerce.

Full guide: [docs/deployment_guide.md](docs/deployment_guide.md)

## 🛠️ For Developers

Extend or contribute easily.

- **API**: Clean, typed functions (svd_reconstruct, RecommenderSystem)
- **Contributing**: `make dev` setup, tests, linting
- **Benchmarks**: `vector-recsys benchmark`

See [CONTRIBUTING.md](CONTRIBUTING.md) and [API Reference](#🔧-api-reference).

### ML Tooling Examples

**ALS (Implicit Feedback)**:
```python
from vector_recsys_lite import RecommenderSystem
ratings = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)  # binary implicit
model = RecommenderSystem(algorithm="als")
model.fit(ratings, k=2)
preds = model.predict(ratings)
```

**KNN (Cosine Similarity)**:
```python
from vector_recsys_lite import RecommenderSystem
ratings = np.array([[5, 3, 0], [0, 0, 4]], dtype=np.float32)
model = RecommenderSystem(algorithm="knn")
model.fit(ratings, k=2)
preds = model.predict(ratings)
```

**Bias Handling (SVD)**:
```python
from vector_recsys_lite import RecommenderSystem
ratings = np.array([[5, 3, 0], [0, 0, 4]], dtype=np.float32)
model = RecommenderSystem(algorithm="svd")
model.fit(ratings, k=2)
preds = model.predict(ratings)  # Includes global/user/item bias
```

**Chunked SVD**:
```python
from vector_recsys_lite import svd_reconstruct
large_mat = np.random.rand(10000, 500)
reconstructed = svd_reconstruct(large_mat, k=10, use_sparse=True)
```

**Metrics**:
```python
recs = [[1,2,3], [4,5]]
actual = [{1,3}, {4,6}]
print(precision_at_k(recs, actual, 3))
print(recall_at_k(recs, actual, 3))
print(ndcg_at_k(recs, actual, 3))
```

**CV Split**:
```python
train, test = train_test_split_ratings(mat, test_size=0.2)
```

**Pipeline**:
```python
from vector_recsys_lite import RecsysPipeline, RecommenderSystem
pipe = RecsysPipeline([('model', RecommenderSystem())])
pipe.fit(mat, k=2)
recs = pipe.recommend(mat, n=3)
```

**Grid Search**:
```python
result = grid_search_k(mat, [2,4], 'rmse')
print(result['best_k'])
```

Full details in API Reference.

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
    """Truncated SVD reconstruction for collaborative filtering. Supports chunked processing for large matrices."""

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
    """Production-ready recommender system with model persistence. Supports SVD, ALS (implicit), and KNN algorithms, with bias handling."""

    def __init__(self, algorithm: str = "svd", ...):
        """algorithm: 'svd', 'als', or 'knn'"""
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
