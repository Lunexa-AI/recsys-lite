# Contributing to recsys-lite

Thanks for your interest! We welcome contributions that improve education or usability.

## Good First Issues

- **Add a new toy dataset**: Create a small, interesting dataset for demos or tests (e.g., books, music, local products).
- **Improve educator guides**: Add examples, clarify concepts, or translate docs.
- **Add edge-case tests**: Help us reach 90%+ coverage by testing rare or tricky scenarios.
- **CLI polish**: Improve help messages, error handling, or add new CLI options.
- **Docs and templates**: Refine README, CONTRIBUTING, or GitHub issue/PR templates.

See [issues](https://github.com/Lunexa-AI/recsys-lite/issues) for more ideas!

## Reporting Issues

1. Check existing issues
2. Open new issue with template
3. Include: description, steps to reproduce, environment

## Contributing Code

1. Fork repo
2. Create branch: `git checkout -b feat/my-feature`
3. Develop with `make dev`
4. Test: `make test`
5. Lint: `make lint`
6. Commit: `git commit -m 'feat: add my feature'`
7. Push: `git push origin feat/my-feature`
8. Open PR with clear title/description

## Code Standards
- Python 3.9+
- Black formatting
- Type hints
- Tests for new features (>80% coverage)
- Docstrings in Google style

See [developer_guide.md](docs/developer_guide.md) for more.
