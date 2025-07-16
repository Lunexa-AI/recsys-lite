# Educator & Student Guide

Welcome! This guide helps you use vector-recsys-lite for teaching recommendation systems in classrooms with limited resources.

## Quick Setup

1. **Install** (one-time, ~5 min):
   - With internet: `pip install vector_recsys_lite`
   - Offline: Download wheel on good connection, transfer via USB, `pip install <wheel>`

2. **Test**: `vector-recsys --help` (should show commands)

3. **For classrooms**: Share installed package via USB/external drive.

## Teaching Workflows

### 1. Basic Concepts (10 min demo)
Use interactive mode:

```bash
vector-recsys teach --concept matrix  # Explains user-item matrix
vector-recsys teach --concept svd     # SVD breakdown with example
vector-recsys teach --concept als     # ALS (implicit feedback) demo
vector-recsys teach --concept knn     # KNN (similarity) demo
```

Follow prompts to generate examples.

### 2. Hands-on Lab (30 min)
- Generate data: `vector-recsys sample class_data.csv --users 5 --items 4`
- Predict (SVD): `vector-recsys predict class_data.csv --explain`
- Predict (ALS): `vector-recsys predict class_data.csv --algorithm als --explain`
- Predict (KNN): `vector-recsys predict class_data.csv --algorithm knn --explain`
- Discuss output: Shows matrix samples, math, and bias terms.

### 3. Notebook Session (if Jupyter available)
```bash
cd examples/
jupyter notebook svd_math_demo.ipynb
```
- Walk through sections, run cells - uses <1MB RAM.
- Try ALS and KNN in Python:

```python
from vector_recsys_lite import RecommenderSystem
ratings = ...  # your matrix
als = RecommenderSystem(algorithm="als").fit(ratings, k=2)
knn = RecommenderSystem(algorithm="knn").fit(ratings, k=2)
```

## Low-Resource Tips
- **Old hardware**: Use sparse mode: `vector-recsys predict --use-sparse`
- **Large matrices**: Use chunked SVD: `vector-recsys predict --use-sparse` or in Python with `svd_reconstruct(..., use_sparse=True)`
- **No internet**: All features work offline after install
- **Large classes**: Pre-install on lab computers
- **Extensions**: Have students modify `RecommenderSystem` class, try ALS/KNN/bias options

## Sample Lesson Plan
1. Intro (5 min): Explain matrix concept
2. Demo (10 min): Run teach command (SVD, ALS, KNN)
3. Exercise (15 min): Students generate data and predict with all algorithms
4. Discuss (10 min): Compare predictions, discuss bias and algorithm differences

Questions? Open an issue on GitHub.
