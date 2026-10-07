# Contributing to Slate

Thanks for your interest in contributing! Bug reports, feature requests and pull requests are all welcome.

## Reporting Issues

If you find a bug or have an idea for a new feature, please [open an issue](https://github.com/Matt-Ord/slate/issues). For bugs, a short snippet which reproduces the problem is very helpful.

## Development Setup

Slate uses [uv](https://docs.astral.sh/uv/) to manage dependencies. To get set up:

```bash
git clone https://github.com/Matt-Ord/slate.git
cd slate
uv sync --all-extras
```

Alternatively, the repository includes a dev container configuration, which can be used with VS Code or GitHub Codespaces.

## Making Changes

1. Fork the repository and create a branch for your change.
2. Make your changes, adding tests in the `tests` folder where appropriate.
3. Check that everything passes before opening a pull request:

   ```bash
   uv run ruff check
   uv run ruff format --check
   uv run ty check
   uv run pytest
   ```

4. Open a pull request against `main`, with a short description of what has changed and why.

All public functions should have type annotations and a NumPy style docstring. These checks are also run by CI on every pull request.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).
