# Contributing to vector_recsys_lite

Thank you for your interest in contributing! 🎉

## How to Contribute

- Fork the repository and create your branch from `main`.
- Make your changes with clear, descriptive commit messages.
- Ensure your code is well-documented and type-annotated.
- Add or update tests for any new features or bug fixes.
- Run all tests and linting before submitting a pull request.
- Open a pull request and describe your changes.

## Code Style

- Use [Black](https://github.com/psf/black) for formatting.
- Use [isort](https://pycqa.github.io/isort/) for import sorting.
- Use [ruff](https://github.com/astral-sh/ruff) for linting.
- Type annotations are required for all public functions.
- Docstrings are required for all public functions and classes.

## Running Tests and Linting

```bash
make lint
make test
```

Or, with Poetry:

```bash
poetry run pytest
poetry run black --check src/ tests/
poetry run isort --check src/ tests/
poetry run ruff src/ tests/
```

## Pull Request Checklist

- [ ] All tests pass
- [ ] Linting passes
- [ ] Code is documented and type-annotated
- [ ] PR description is clear and complete

Thank you for helping make this project better! 🚀 