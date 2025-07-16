# Changelog

All notable changes to this project will be documented in this file.

## [0.1.5] - 2024-07-14
### Changed
- Major robustness and production-readiness improvements across all modules
- All imports are now at the top of each file (except for truly optional dependencies)
- Optional dependencies (e.g., matplotlib) are handled gracefully and do not block import or test runs
- Hardened all metric, grid search, and SVD logic for edge cases and empty input
- Suppressed or fixed all test and runtime warnings (numpy, SVD, grid search, etc.)
- Improved test coverage and added edge-case tests for all core features
- Updated README with "Why This Over a Script?" section
- Updated CONTRIBUTING.md with "Good First Issues" and community guidance
- Cleaned up Poetry environment and ensured all dependencies are managed via pyproject.toml
- CI, lint, pre-commit, and Docker builds all pass

## [0.1.4] - 2024-07-14
### Added
- Initial standalone release
- All core algorithms, CLI, I/O, benchmarking, and Docker support
- Modern README with badges, real-world examples, FAQ, and help sections
- Pre-commit, Makefile, CI, and full test suite
- Optional support for Parquet, HDF5, SQLite, and Numba
- CONTRIBUTING.md and CODE_OF_CONDUCT.md for open source best practices
