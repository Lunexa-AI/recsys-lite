# Developer Guide

This guide is for developers extending or contributing to recsys-lite. It assumes Python knowledge.

## 🏗️ Project Structure

- `src/recsys_lite/`: Core code
  - `algo.py`: SVD and recommendation logic
  - `cli.py`: Command-line interface
  - `explain.py`: Educational tools
  - `io.py`: Data loading/saving
- `tests/`: Unit tests
- `examples/`: Jupyter notebooks
- `docs/`: Guides like this one

## 🔧 Extending the Library

### Adding a New Algorithm
1. Subclass `RecommenderSystem` in `algo.py`:

```python
class MyAlgorithm(RecommenderSystem):
    def fit(self, ratings, **kwargs):
        # Your implementation
        pass
```

2. Add to CLI if needed.

### Custom I/O Formats
Extend `load_ratings` in `io.py` with new cases.

Full examples in [API Reference](../README.md#🔧-api-reference).

## 🤝 Contributing

Follow [CONTRIBUTING.md](../CONTRIBUTING.md):
1. Fork and branch: `git checkout -b feat/my-feature`
2. Develop and test: `make test`
3. Lint: `make lint`
4. PR with clear description

We welcome education-focused contributions!

## ⚙️ Advanced Topics

- **Performance Tuning**: Use Numba for JIT
- **Testing**: `pytest --cov`
- **Building Docs**: `cd docs; make html`

See scikit-learn docs for inspiration on structure.
