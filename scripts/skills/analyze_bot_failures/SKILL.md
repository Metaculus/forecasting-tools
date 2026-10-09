---
name: analyze-bot-failures
description: Analyze failures from the Metaculus bot forecasting GitHub Actions workflow. Pulls failed job logs, aggregates failure reasons, and investigates whether real bugs need fixing. Use when the user asks why bot runs/workflows are failing, wants a summary of forecasting errors, or mentions analyzing bot failure logs from GitHub Actions.
---

# Analyze Bot Forecasting Failures

Workflow for diagnosing failures in the `run-bot-aib-tournament.yaml` GitHub Actions workflow, which runs ~30 bot jobs every 30 minutes via `run_bots.py`.

## Running this automatically (cron + Slack report)

This skill is wired up to run unattended on a daily schedule via the
`Analyze Bot Forecasting Failures` GitHub Actions workflow
(`.github/workflows/analyze-bot-failures.yaml`). That workflow calls
`scripts/skills/analyze_bot_failures/run_scheduled_analysis.py`, which runs the
parser below, hands the report to a headless Cursor agent that performs steps
2-5, and posts the agent's prioritized summary to Slack.

Required repo secret: `CURSOR_API_KEY` (for the agent). Optional: `SLACK_BOT_TOKEN`
and `SLACK_CHANNEL_ID` for Slack delivery. Optional repo variable `CURSOR_AGENT_MODEL`
(defaults to `auto`). The default `GITHUB_TOKEN` already has the `actions: read` scope
the parser needs.

The same orchestrator script can be run from a local crontab if preferred (set
`GITHUB_TOKEN`, `CURSOR_API_KEY`, `SLACK_BOT_TOKEN`, and `SLACK_CHANNEL_ID` in the environment).
The steps below describe the manual/interactive version of the workflow.

## Step 1: Pull and aggregate the failure logs

A GitHub token is required (the job-log API rejects unauthenticated requests). Check `GITHUB_TOKEN` env var or `gh auth token`; if neither works, ask the user to run `gh auth login`.

```bash
poetry run python scripts/skills/analyze_bot_failures/analyze_bot_run_failures.py
```

Useful options:
- `--since 12h|2d|1w|4w` or an ISO datetime (default `3d`). Any reasonable period works; the whole window is analyzed by default (note the workflow runs every 30 min, so wide windows pull many runs and take longer).
- `--run-id <id>` to analyze one specific run
- `--max-runs <n>` optional cap on runs fetched (default: no cap, so the full `--since` window is covered). A warning is logged if the cap truncates the window.

Output goes to `logs/workflow_failure_analysis/<timestamp>/`:
- `report.md` — counts by category/bot/question, plus failure groups (deduped by normalized signature) with an example message, traceback, and the deepest repo code frame
- `failures.json` — every parsed failure with full messages and signatures (machine-readable)
- `raw_logs/run<id>_<bot>.log` — complete raw log of each failed job

## Step 2: Read the report, then verify against raw logs

Read `report.md` first. The parser extracts full tracebacks when present (falling back to `❌ Exception: ... | Message: ...` summary lines, then to the raw log tail), so each failure group usually includes a "deepest repo frame" — the last traceback frame inside this repo rather than a dependency — which is the first place to look in the code. Still **spot-check 2-3 raw logs** for the most frequent signatures to see surrounding context (retry attempts, which LLM call failed). Failures of type `UnparsedFailure` mean nothing parseable was found — the job likely failed at the infrastructure level (poetry install failures, token resolution, job-level timeout at 55 min, cancellation) — read their raw logs directly.

## Step 3: Triage — transient noise vs real bug

Likely transient (usually not worth fixing, just note frequency):
- Provider 5xx / overloaded / rate-limit errors that hit one bot in one run
- Occasional timeouts that succeeded on the in-run retry (`run_bots.py` retries reports whose error contains "TimeoutError")

Likely real bugs (investigate the code):
- The same bot failing across most runs (config/model-name/credentials issue in its `RunBotConfig` in `run_bots.py`)
- Structured-output or validation errors clustering on one question type (binary/MC/numeric/group) — points at parsing or prompt code
- Prediction validation errors (probabilities out of bounds, CDF/percentile issues) — points at report data models
- The same question ID recurring across bots/runs — candidate for `POST_IDS_TO_SKIP` or `POST_IDS_TO_NOT_RAISE_ERRORS_FOR` in `run_bots.py`
- Tracebacks ending inside `forecasting_tools/` rather than a provider SDK

## Step 4: Map errors to code

| Symptom | Where to look |
|---|---|
| Bot config, skip lists, question selection, final RuntimeError | `run_bots.py` |
| Forecast orchestration, retries, report summary format | `forecasting_tools/forecast_bots/forecast_bot.py` |
| Structured output parsing failures | `forecasting_tools/helpers/structure_output.py` |
| LLM call errors, model names, retries, token limits | `forecasting_tools/ai_models/` (esp. `general_llm.py`) |
| AskNews/research errors, cache misses | `forecasting_tools/helpers/` and `.github/scripts/precache_asknews.py` |
| Metaculus API / publish errors | `forecasting_tools/helpers/metaculus_client.py` |
| Job setup/infra failures, secrets, timeouts | `.github/workflows/run-bot-launcher.yaml`, `run-bot-aib-tournament.yaml` |

It may not be in any of the above, so be willing to search the codebase.

## Step 5: Summarize findings

Report back with:
1. Failure counts by category and bot (from `report.md`)
2. **Real bugs found**: each with evidence (log excerpt + code location) and a proposed fix
3. **Transient noise**: what to ignore and why
4. **Question skip candidates**: recurring question IDs with failure counts
5. Do not change code or skip lists without confirming the proposed fixes with the user first
