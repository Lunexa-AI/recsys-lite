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
```

Follow prompts to generate examples.

### 2. Hands-on Lab (30 min)
- Generate data: `vector-recsys sample class_data.csv --users 5 --items 4`
- Predict: `vector-recsys predict class_data.csv --explain`
- Discuss output: Shows matrix samples and math.

### 3. Notebook Session (if Jupyter available)
```bash
cd examples/
jupyter notebook svd_math_demo.ipynb
```
- Walk through sections, run cells - uses <1MB RAM.

## Low-Resource Tips
- **Old hardware**: Use sparse mode: `vector-recsys predict --use-sparse`
- **No internet**: All features work offline after install
- **Large classes**: Pre-install on lab computers
- **Extensions**: Have students modify `RecommenderSystem` class

## Sample Lesson Plan
1. Intro (5 min): Explain matrix concept
2. Demo (10 min): Run teach command
3. Exercise (15 min): Students generate data and predict
4. Discuss (10 min): Compare predictions to math

Questions? Open an issue on GitHub.
