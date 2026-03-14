# Repository Improvement Suggestions

A comprehensive review of the `forecasting-tools` repository with actionable improvement recommendations, organized by priority and effort.

---

## 1. Testing & Code Quality Infrastructure

### 1.1 Add Test Coverage Reporting (High Priority, Low Effort)

There is no test coverage configuration anywhere in the repo. `pytest-cov` is not in dev dependencies, and CI does not report coverage. This means there is no visibility into which parts of the codebase are tested.

**Recommendations:**
- Add `pytest-cov` to dev dependencies
- Add `[tool.coverage]` configuration in `pyproject.toml`
- Add coverage reporting to the `unit-tests.yaml` CI workflow (e.g., upload to Codecov or Coveralls)
- Consider enforcing a minimum coverage threshold to prevent regressions

```toml
# pyproject.toml
[tool.coverage.run]
source = ["forecasting_tools"]
omit = ["forecasting_tools/front_end/*", "*/deprecated/*"]

[tool.coverage.report]
fail_under = 60
show_missing = true
```

### 1.2 Centralize Linting & Formatting Configuration (Medium Priority, Low Effort)

Black, isort, and ruff settings live only in `.pre-commit-config.yaml`. This means IDE integrations, manual CLI runs, and CI may use different settings than pre-commit. Centralizing into `pyproject.toml` gives a single source of truth.

**Recommendations:**
- Add `[tool.black]`, `[tool.isort]`, and `[tool.ruff]` sections to `pyproject.toml`
- Pre-commit hooks can then just inherit those settings without duplicating args

```toml
# pyproject.toml
[tool.black]
line-length = 88

[tool.isort]
profile = "black"
line_length = 88

[tool.ruff]
line-length = 88
ignore = ["E741", "E731", "E402", "E711", "E712", "E721"]
```

### 1.3 Add a Makefile or Task Runner (Medium Priority, Low Effort)

There is no `Makefile`, `tox.ini` (despite `tox` being in dev deps), or other task runner. This means contributors must memorize or look up commands for common tasks.

**Recommendations:**
- Add a `Makefile` with targets for common workflows:

```makefile
.PHONY: test lint format install

install:
	poetry install

test:
	poetry run pytest ./code_tests/unit_tests

test-cov:
	poetry run pytest ./code_tests/unit_tests --cov=forecasting_tools --cov-report=html

lint:
	poetry run ruff check forecasting_tools/

format:
	poetry run black forecasting_tools/ code_tests/
	poetry run isort forecasting_tools/ code_tests/
```

- Alternatively, configure `tox` since it is already a dev dependency but unused.

---

## 2. Error Handling & Robustness

### 2.1 Narrow Broad `except Exception` Catches (High Priority, Medium Effort)

There are **~85 instances** of `except Exception` across the codebase. While some are intentional fallbacks, many silently swallow errors or catch too broadly, making debugging difficult.

**Key areas to address:**
- `forecasting_tools/cp_benchmarking/benchmark_for_bot.py` — 8 occurrences
- `forecasting_tools/forecast_bots/forecast_bot.py` — 6 occurrences
- `forecasting_tools/agents_and_tools/ai_congress_v2/congress_orchestrator.py` — 6 occurrences
- `forecasting_tools/data_models/questions.py` — 5 occurrences

**Recommendations:**
- Replace with specific exception types where possible (e.g., `ValueError`, `KeyError`, `httpx.HTTPError`)
- Where broad catches are intentional, add a comment explaining why
- Consider using `except Exception as e: logger.exception(...)` instead of silent catches

### 2.2 Validate Environment Variables Before Use (Medium Priority, Low Effort)

Several environment variables are used without validation. When they are `None`, this produces confusing runtime errors (e.g., `Authorization: Token None` headers).

**Files affected:**
- `forecasting_tools/ai_models/general_llm.py` — `METACULUS_TOKEN` used in auth header without validation
- `forecasting_tools/util/coda_utils.py` — `CODA_API_KEY` used at class level, can be `None`

**Recommendation:** Add a helper function for required env var access:

```python
def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Required environment variable '{name}' is not set")
    return value
```

---

## 3. Code Organization & Style

### 3.1 Fix Import Ordering Violations (Low Priority, Low Effort)

`forecasting_tools/util/coda_utils.py` has code (`logger = logging.getLogger(__name__)`) placed between import blocks, violating PEP 8. All imports should precede module-level code.

### 3.2 Add Missing Type Hints (Medium Priority, Medium Effort)

Several public functions are missing return type hints:
- `metaculus_api.py:94` — `delete_question_link`
- `metaculus_client.py:544` — `delete_question_link`
- `coda_utils.py:62,75` — `add_row_to_table`, `check_that_row_matches_columns`
- `app_page.py:54,58` — `header`, `footer`
- `questions.py:792` — `preprocess`
- `base_rate_researcher.py:528` — `count_must_be_non_negative`

**Recommendation:** Add return type annotations to all public methods. Consider enabling `ruff` rules for type hint enforcement (e.g., `ANN` rules).

### 3.3 Replace `print()` with `logging` in Non-Test Code (Low Priority, Low Effort)

At least 26 files use `print()` instead of `logging`. While acceptable in tests, production and script code should use the existing logging infrastructure.

**Key offenders:**
- `scripts/generate_bot_costs_csv.py` — 5+ print calls
- `scripts/run_benchmarker.py` — mix of print and logger
- Various front-end app pages

### 3.4 Clean Up `typing` Imports (Low Priority, Low Effort)

The `conftest.py` imports `Generator` from `typing`, but Python 3.11+ supports `collections.abc.Generator` or just `Generator` from `collections.abc`. Several files import from `typing` where built-in generics work.

---

## 4. Dependency Management

### 4.1 Pin Pre-Commit Hook Versions More Aggressively (Low Priority, Low Effort)

Pre-commit hooks are pinned to specific revisions (good), but some are getting stale:
- Black: `24.8.0` (current latest is significantly newer)
- Ruff: `v0.6.7` (current latest is significantly newer)
- isort: `5.13.2`

**Recommendation:** Set up Dependabot or Renovate to auto-update pre-commit hooks.

### 4.2 Consider Migrating from Poetry to uv (Low Priority, High Effort)

`uv` is significantly faster for dependency resolution and installation. Poetry lock resolution can be slow for large dependency trees. This is a lower-priority, higher-effort change to consider for the future.

### 4.3 Remove Unused Dev Dependencies (Low Priority, Low Effort)

`tox` is listed as a dev dependency but has no `tox.ini` or `[tool.tox]` configuration. Either configure it or remove it.

---

## 5. CI/CD Improvements

### 5.1 Add Integration Test CI Job (High Priority, Medium Effort)

Integration tests exist in `code_tests/integration_tests/` but there is no CI workflow running them. They are only run manually.

**Recommendation:** Add a scheduled CI workflow (e.g., nightly or weekly) that runs integration tests. Use GitHub Actions scheduled triggers and limit API costs with sampling.

### 5.2 Add Coverage Reporting to CI (Medium Priority, Low Effort)

The unit test workflow should report coverage alongside pass/fail. This provides ongoing visibility.

```yaml
- name: "Run unit tests with coverage"
  run: poetry run pytest ./code_tests/unit_tests --cov=forecasting_tools --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v4
  with:
    file: ./coverage.xml
```

### 5.3 Run Unit Tests on PRs to `develop` Too (Low Priority, Low Effort)

The `unit-tests.yaml` workflow only triggers on PRs to `main`. If the team uses a `develop` branch (referenced in `pre-commit.yaml`), unit tests should also run on PRs targeting it.

### 5.4 Add Dependency Caching for Pre-Commit in CI (Low Priority, Low Effort)

The `pre-commit.yaml` workflow doesn't cache pre-commit environments. Adding caching can speed up CI:

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/pre-commit
    key: pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}
```

---

## 6. Documentation

### 6.1 Add Contributing Guidelines (Medium Priority, Low Effort)

There is no `CONTRIBUTING.md` file. This makes it harder for external contributors to understand:
- How to set up the development environment
- How to run tests
- Code style and conventions
- PR review process

### 6.2 Add Architecture Documentation (Low Priority, Medium Effort)

The README covers installation and usage well but lacks an architecture overview. A diagram or document explaining how modules interact (bots -> AI models -> APIs, benchmarking pipeline, etc.) would help new contributors ramp up faster.

### 6.3 Address Outstanding TODOs (Medium Priority, Varies)

There are **~35 TODO comments** scattered across the codebase. Some notable ones:

| File | TODO |
|------|------|
| `forecast_database_manager.py:30` | Move to more robust database |
| `metaculus_client.py:1183` | Timezone for strftime |
| `hosted_file.py:47` | Handle zip bombs (security) |
| `benchmark_displayer.py:580` | Refactor file |
| `general_llm.py:352,380,445` | Citation, reasoning, token tracking |
| `data_models/questions.py:85` | Move `unit_of_measure` to numeric questions |

**Recommendation:** Triage these into GitHub Issues so they are visible and can be prioritized. The zip bomb handling TODO in particular has security implications.

---

## 7. Security

### 7.1 Handle Zip Bombs in File Processing (High Priority, Medium Effort)

There is an explicit TODO at `forecasting_tools/agents_and_tools/other/hosted_file.py:47` about handling zip bombs. This is a security risk if user-supplied files are processed.

### 7.2 Add API Key Validation at Startup (Medium Priority, Low Effort)

Rather than failing deep in a call stack when an API key is missing, validate required keys at application startup (or at least at the point of first use with a clear error message). The existing pattern in some files (e.g., `adjacent_news_api.py`) should be applied consistently.

---

## 8. Performance & Scalability

### 8.1 Use `asyncio` More Consistently (Low Priority, High Effort)

The codebase uses both sync (`requests`) and async (`aiohttp`) HTTP patterns. Standardizing on async where possible would improve throughput for parallel API calls, which is common in forecasting workflows.

### 8.2 Add Request Timeouts Everywhere (Medium Priority, Low Effort)

Some HTTP requests (e.g., in `coda_utils.py`) don't specify explicit timeouts. This can cause the application to hang indefinitely.

```python
response = requests.post(uri, headers=headers, json=full_payload, timeout=30)
```

---

## Summary: Prioritized Action Items

| # | Item | Priority | Effort |
|---|------|----------|--------|
| 1 | Add test coverage reporting | High | Low |
| 2 | Narrow `except Exception` catches | High | Medium |
| 3 | Handle zip bomb TODO (security) | High | Medium |
| 4 | Add integration test CI job | High | Medium |
| 5 | Centralize lint/format config in pyproject.toml | Medium | Low |
| 6 | Validate env vars before use | Medium | Low |
| 7 | Add Makefile / task runner | Medium | Low |
| 8 | Add CONTRIBUTING.md | Medium | Low |
| 9 | Triage TODOs into GitHub Issues | Medium | Low |
| 10 | Add missing type hints | Medium | Medium |
| 11 | Add coverage to CI | Medium | Low |
| 12 | Add request timeouts | Medium | Low |
| 13 | Fix import ordering | Low | Low |
| 14 | Replace print with logging | Low | Low |
| 15 | Update pre-commit hook versions | Low | Low |
| 16 | Remove/configure tox | Low | Low |
| 17 | Run tests on develop branch too | Low | Low |
| 18 | Add architecture docs | Low | Medium |
| 19 | Standardize on async HTTP | Low | High |
| 20 | Consider uv migration | Low | High |
