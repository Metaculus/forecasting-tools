# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
`forecasting-tools` is a Python library for AI-powered forecasting, with a Streamlit front-end. No database is required. See `README.md` for full documentation.

### Running commands
- Use `poetry run <command>` to run anything in the virtualenv (e.g. `poetry run pytest`, `poetry run streamlit run ...`).
- Ensure `$HOME/.local/bin` is on `PATH` so `poetry` resolves correctly.

### Linting
- Linting is managed via pre-commit hooks (Black, isort, ruff, typos): `poetry run pre-commit run --all-files`
- The individual tools (ruff, black) are installed inside the pre-commit hook environments, not as standalone Poetry dependencies.

### Testing
- Unit tests: `poetry run pytest code_tests/unit_tests` — no API keys needed, all mocked.
- Integration tests: `poetry run pytest code_tests/integration_tests` — require live API keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, etc.) and incur costs.
- `pytest.ini` enables `asyncio_mode = auto` and parallel execution (`-nauto`).

### Running the Streamlit front-end
- `poetry run streamlit run front_end/Home.py --server.port 8501 --server.headless true`
- The app works without API keys for browsing the UI; submitting queries requires appropriate API keys set in a `.env` file (see `.env.template`).

### Environment variables
- Copy `.env.template` to `.env` and fill in any needed keys. The app loads `.env` automatically via `python-dotenv`.
- `METACULUS_TOKEN` is required for Metaculus API interactions. `OPENROUTER_API_KEY` and `ASKNEWS_API_KEY` are the most heavily used providers.

### Gotchas
- The `--timeout` flag is not available for pytest in this project (no `pytest-timeout` dependency). Do not pass it.
- Streamlit warnings about `METACULUS_TOKEN` and `MemoryCacheStorageManager` during tests are benign and expected.
