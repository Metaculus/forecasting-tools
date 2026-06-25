"""
Pulls logs of failed bot jobs from the GitHub Actions tournament workflow,
parses out the failure reasons, and writes an aggregated markdown/json report.

Requires a GitHub token with repo read access (the job-log endpoint rejects
unauthenticated requests even on public repos). The token is resolved from the
GITHUB_TOKEN env var, then from `gh auth token`.

Example usage:
    poetry run python scripts/skills/analyze_bot_failures/analyze_bot_run_failures.py --since 2d
    poetry run python scripts/skills/analyze_bot_failures/analyze_bot_run_failures.py --run-id 27362307940
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "Metaculus/forecasting-tools"
DEFAULT_WORKFLOW = "run-bot-aib-tournament.yaml"
DEFAULT_OUTPUT_DIR = Path("logs/workflow_failure_analysis")
LOG_TAIL_LINES_FOR_UNPARSED_FAILURES = 60

ERROR_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "timeout": [r"TimeoutError", r"timed?[\s_-]?out", r"Timeout"],
    "rate_limit": [r"429", r"rate[\s_-]?limit", r"RateLimitError", r"quota"],
    "provider_5xx": [
        r"\b50[0-49]\b",
        r"InternalServerError",
        r"ServiceUnavailable",
        r"upstream",
        r"overloaded",
    ],
    "auth_or_credits": [
        r"\b401\b",
        r"\b402\b",
        r"\b403\b",
        r"AuthenticationError",
        r"insufficient[\s_-]?credits",
        r"API key",
    ],
    "bad_request_or_model_refusal": [
        r"\b400\b",
        r"BadRequestError",
        r"InvalidRequestError",
        r"refus",
        r"content[\s_-]?policy",
    ],
    "structured_output_parsing": [
        r"structure[d]?[\s_-]?output",
        r"ValidationError",
        r"JSONDecodeError",
        r"Failed to parse",
        r"pydantic",
    ],
    "prediction_validation": [
        r"probabilit(y|ies).{0,80}(sum|bound|range|between)",
        r"out of bounds",
        r"cdf",
        r"percentile",
        r"AssertionError",
        r"bad probability",
    ],
    "asknews_or_research": [r"asknews", r"AskNews", r"news[\s_-]?result", r"exa\.ai"],
    "metaculus_api": [r"metaculus.{0,80}(api|post|publish)", r"Failed to post"],
    "connection": [
        r"ConnectionError",
        r"ConnectTimeout",
        r"SSLError",
        r"Connection reset",
        r"APIConnectionError",
    ],
}

QUESTION_URL_PATTERN = re.compile(r"https://www\.metaculus\.com/questions/\d+/?")
GH_TIMESTAMP_PREFIX_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s?"
)
PYTHON_LOG_RECORD_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ")
FAILED_FORECAST_LINE_PATTERN = re.compile(
    r"❌ Exception: (?P<exception_type>\w+) \| Message: (?P<message>.*)"
)
TRACEBACK_FRAME_PATTERN = re.compile(
    r'File "(?P<file_path>[^"]+)", line (?P<line_number>\d+)'
)
EXCEPTION_LINE_PATTERN = re.compile(
    r"^(?P<exception_type>[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)?): (?P<message>.+)"
)
GITHUB_RUNNER_REPO_ROOT_PATTERN = re.compile(r"^/home/runner/work/[^/]+/[^/]+/")


@dataclass
class RepoFrame:
    file_path: str
    line_number: int


@dataclass
class FailureEvent:
    bot_name: str
    run_id: int
    job_name: str
    job_url: str
    exception_type: str
    message: str
    categories: list[str]
    question_url: str | None
    traceback_text: str | None
    deepest_repo_frame: RepoFrame | None


@dataclass
class FailureGroup:
    signature: str
    events: list[FailureEvent]


@dataclass
class JobAnalysis:
    bot_name: str
    run_id: int
    job_url: str
    log_path: str
    events: list[FailureEvent]


def resolve_github_token() -> str:
    env_token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if env_token:
        return env_token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "No GitHub token found. Set GITHUB_TOKEN or run `gh auth login`. "
            "A token is required to download job logs (even for public repos)."
        )


def github_get(
    path: str, token: str, params: dict | None = None, as_text: bool = False
) -> dict | str:
    response = requests.get(
        f"{GITHUB_API}{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    return response.text if as_text else response.json()


def parse_since_to_datetime(since: str) -> datetime:
    match = re.fullmatch(r"(\d+)([hdw])", since)
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        unit_to_delta = {
            "h": timedelta(hours=amount),
            "d": timedelta(days=amount),
            "w": timedelta(weeks=amount),
        }
        return datetime.now(timezone.utc) - unit_to_delta[unit]
    return datetime.fromisoformat(since).astimezone(timezone.utc)


def list_workflow_runs(
    repo: str, workflow: str, token: str, since: datetime, max_runs: int | None
) -> list[dict]:
    runs: list[dict] = []
    page = 1
    while max_runs is None or len(runs) < max_runs:
        data = github_get(
            f"/repos/{repo}/actions/workflows/{workflow}/runs",
            token,
            params={
                "per_page": 100,
                "page": page,
                "created": f">={since.isoformat()}",
            },
        )
        assert isinstance(data, dict)
        page_runs = data.get("workflow_runs", [])
        if not page_runs:
            break
        runs.extend(page_runs)
        page += 1
    if max_runs is not None and len(runs) > max_runs:
        logger.warning(
            f"--max-runs={max_runs} truncated the {len(runs)} runs found in the "
            f"--since window; some of the time period is not covered. Raise or drop "
            f"--max-runs to analyze the full window."
        )
        return runs[:max_runs]
    return runs


def list_failed_jobs(repo: str, run_id: int, token: str) -> list[dict]:
    jobs: list[dict] = []
    page = 1
    while True:
        data = github_get(
            f"/repos/{repo}/actions/runs/{run_id}/jobs",
            token,
            params={"per_page": 100, "page": page, "filter": "latest"},
        )
        assert isinstance(data, dict)
        page_jobs = data.get("jobs", [])
        if not page_jobs:
            break
        jobs.extend(page_jobs)
        page += 1
    return [job for job in jobs if job.get("conclusion") == "failure"]


def download_job_log(repo: str, job_id: int, token: str) -> str:
    log_text = github_get(
        f"/repos/{repo}/actions/jobs/{job_id}/logs", token, as_text=True
    )
    assert isinstance(log_text, str)
    return log_text


def strip_github_timestamps(log_text: str) -> str:
    return "\n".join(
        GH_TIMESTAMP_PREFIX_PATTERN.sub("", line) for line in log_text.splitlines()
    )


def categorize_error(error_text: str) -> list[str]:
    categories = [
        category
        for category, patterns in ERROR_CATEGORY_PATTERNS.items()
        if any(re.search(pattern, error_text, re.IGNORECASE) for pattern in patterns)
    ]
    return categories or ["uncategorized"]


def extract_bot_name(job_name: str) -> str:
    return job_name.split("/")[0].strip()


def extract_traceback_blocks(log_lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current_block: list[str] | None = None
    for line in log_lines:
        if line.startswith("Traceback (most recent call last)"):
            if current_block is not None:
                blocks.append(current_block)
            current_block = [line]
        elif current_block is not None:
            if PYTHON_LOG_RECORD_PATTERN.match(line) or line.startswith("##["):
                blocks.append(current_block)
                current_block = None
            else:
                current_block.append(line)
    if current_block is not None:
        blocks.append(current_block)
    return blocks


def extract_deepest_repo_frame(traceback_lines: list[str]) -> RepoFrame | None:
    repo_frames = []
    for line in traceback_lines:
        frame_match = TRACEBACK_FRAME_PATTERN.search(line)
        if not frame_match:
            continue
        file_path = frame_match.group("file_path")
        is_dependency_frame = ".venv" in file_path or "site-packages" in file_path
        if is_dependency_frame:
            continue
        relative_path = GITHUB_RUNNER_REPO_ROOT_PATTERN.sub("", file_path)
        repo_frames.append(
            RepoFrame(
                file_path=relative_path,
                line_number=int(frame_match.group("line_number")),
            )
        )
    return repo_frames[-1] if repo_frames else None


def extract_exception_from_traceback(traceback_lines: list[str]) -> tuple[str, str]:
    for line_index in range(len(traceback_lines) - 1, -1, -1):
        exception_match = EXCEPTION_LINE_PATTERN.match(traceback_lines[line_index])
        if exception_match:
            message_continuation = traceback_lines[line_index + 1 :]
            full_message = "\n".join(
                [exception_match.group("message")] + message_continuation
            ).strip()
            exception_type = exception_match.group("exception_type").split(".")[-1]
            return exception_type, full_message
    return "UnknownException", traceback_lines[-1] if traceback_lines else ""


def extract_question_url(text: str) -> str | None:
    url_match = QUESTION_URL_PATTERN.search(text)
    return url_match.group(0) if url_match else None


def parse_traceback_failure_events(
    log_lines: list[str], bot_name: str, run_id: int, job_name: str, job_url: str
) -> list[FailureEvent]:
    events = []
    for block_lines in extract_traceback_blocks(log_lines):
        exception_type, message = extract_exception_from_traceback(block_lines)
        traceback_text = "\n".join(block_lines)
        events.append(
            FailureEvent(
                bot_name=bot_name,
                run_id=run_id,
                job_name=job_name,
                job_url=job_url,
                exception_type=exception_type,
                message=message,
                categories=categorize_error(f"{exception_type} {message}"),
                question_url=extract_question_url(message)
                or extract_question_url(traceback_text),
                traceback_text=traceback_text,
                deepest_repo_frame=extract_deepest_repo_frame(block_lines),
            )
        )
    return events


def parse_short_summary_failure_events(
    log_lines: list[str], bot_name: str, run_id: int, job_name: str, job_url: str
) -> list[FailureEvent]:
    events = []
    for line_index, line in enumerate(log_lines):
        line_match = FAILED_FORECAST_LINE_PATTERN.search(line)
        if not line_match:
            continue
        message_lines = [line_match.group("message")]
        for continuation_line in log_lines[line_index + 1 :]:
            is_new_section = (
                "✅" in continuation_line
                or "❌" in continuation_line
                or continuation_line.startswith("Stats for passing reports")
                or continuation_line.startswith("----")
                or PYTHON_LOG_RECORD_PATTERN.match(continuation_line)
            )
            if is_new_section:
                break
            message_lines.append(continuation_line)
        message = "\n".join(message_lines).strip()
        exception_type = line_match.group("exception_type")
        events.append(
            FailureEvent(
                bot_name=bot_name,
                run_id=run_id,
                job_name=job_name,
                job_url=job_url,
                exception_type=exception_type,
                message=message,
                categories=categorize_error(f"{exception_type} {message}"),
                question_url=extract_question_url(message),
                traceback_text=None,
                deepest_repo_frame=None,
            )
        )
    return events


def make_unparsed_failure_event(
    log_lines: list[str], bot_name: str, run_id: int, job_name: str, job_url: str
) -> FailureEvent:
    log_tail = "\n".join(log_lines[-LOG_TAIL_LINES_FOR_UNPARSED_FAILURES:])
    return FailureEvent(
        bot_name=bot_name,
        run_id=run_id,
        job_name=job_name,
        job_url=job_url,
        exception_type="UnparsedFailure",
        message=(
            "No traceback or per-question failure line found. The job likely failed "
            "at the infrastructure level (setup, token resolution, job cancellation/"
            "timeout) or the log format changed. Log tail attached as traceback_text."
        ),
        categories=categorize_error(log_tail),
        question_url=None,
        traceback_text=log_tail,
        deepest_repo_frame=None,
    )


def parse_failures_from_log(
    log_text: str, bot_name: str, run_id: int, job_name: str, job_url: str
) -> list[FailureEvent]:
    log_lines = strip_github_timestamps(log_text).splitlines()
    events = parse_traceback_failure_events(
        log_lines, bot_name, run_id, job_name, job_url
    )
    if not events:
        events = parse_short_summary_failure_events(
            log_lines, bot_name, run_id, job_name, job_url
        )
    if not events:
        events = [
            make_unparsed_failure_event(log_lines, bot_name, run_id, job_name, job_url)
        ]
    return events


def normalize_message_to_signature(exception_type: str, message: str) -> str:
    no_urls = QUESTION_URL_PATTERN.sub("<QUESTION_URL>", message)
    no_request_ids = re.sub(r"(req|chatcmpl|gen)-[\w-]+", "<ID>", no_urls)
    no_numbers = re.sub(r"\d+", "<N>", no_request_ids)
    return f"{exception_type}: {no_numbers[:300]}"


def group_failures(events: list[FailureEvent]) -> list[FailureGroup]:
    groups_by_signature: dict[str, FailureGroup] = {}
    for event in events:
        signature = normalize_message_to_signature(event.exception_type, event.message)
        if signature not in groups_by_signature:
            groups_by_signature[signature] = FailureGroup(
                signature=signature, events=[]
            )
        groups_by_signature[signature].events.append(event)
    return sorted(groups_by_signature.values(), key=lambda group: -len(group.events))


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


def count_by(items: list[str]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return sorted(counts.items(), key=lambda pair: -pair[1])


def truncate_text(text: str, max_length: int) -> str:
    return text if len(text) < max_length else text[:max_length] + "...(truncated)"


def build_failure_group_section(group: FailureGroup) -> list[str]:
    example = group.events[0]
    bots_affected = sorted({event.bot_name for event in group.events})
    section_lines = [
        f"### [{len(group.events)}x] {example.exception_type} | "
        f"categories={','.join(example.categories)}",
        f"Bots affected: {', '.join(bots_affected)}",
    ]
    if example.deepest_repo_frame:
        section_lines.append(
            f"Deepest repo frame: `{example.deepest_repo_frame.file_path}:"
            f"{example.deepest_repo_frame.line_number}`"
        )
    section_lines.append(f"\n```\n{truncate_text(example.message, 2000)}\n```\n")
    if example.traceback_text:
        section_lines.append(
            f"<details><summary>Example traceback / log tail</summary>\n\n"
            f"```\n{truncate_text(example.traceback_text, 3000)}\n```\n\n</details>\n"
        )
    return section_lines


def build_question_failure_section(all_events: list[FailureEvent]) -> list[str]:
    events_with_question = [event for event in all_events if event.question_url]
    if not events_with_question:
        return []

    stats_by_question: dict[str, dict] = {}
    for event in events_with_question:
        question_stats = stats_by_question.setdefault(
            event.question_url,
            {"total": 0, "run_ids": set(), "bot_names": set(), "categories": {}},
        )
        question_stats["total"] += 1
        question_stats["run_ids"].add(event.run_id)
        question_stats["bot_names"].add(event.bot_name)
        for category in event.categories:
            question_stats["categories"][category] = (
                question_stats["categories"].get(category, 0) + 1
            )

    questions_ranked_by_consistency = sorted(
        stats_by_question.items(),
        key=lambda item: (-len(item[1]["run_ids"]), -item[1]["total"]),
    )

    section_lines = [
        "\n## Failures by question (most distinct runs first)\n",
        "A question failing across many *distinct runs* is far more likely to be "
        "genuinely broken than one that failed many times within a single run. "
        "Questions recurring across several runs are candidates for "
        "`POST_IDS_TO_SKIP` or `POST_IDS_TO_NOT_RAISE_ERRORS_FOR` in `run_bots.py`.\n",
        "| Question | Failures | Distinct runs | Distinct bots | Top categories |",
        "| --- | --- | --- | --- | --- |",
    ]
    for question_url, question_stats in questions_ranked_by_consistency:
        top_categories = ", ".join(
            f"{category} ({count})"
            for category, count in sorted(
                question_stats["categories"].items(), key=lambda pair: -pair[1]
            )[:3]
        )
        section_lines.append(
            f"| {question_url} | {question_stats['total']} | "
            f"{len(question_stats['run_ids'])} | {len(question_stats['bot_names'])} | "
            f"{top_categories} |"
        )
    return section_lines


def build_report(job_analyses: list[JobAnalysis], output_dir: Path) -> str:
    all_events = [event for analysis in job_analyses for event in analysis.events]
    groups = group_failures(all_events)

    report_lines = [
        "# Bot Workflow Failure Report",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
        f"\nFailed jobs analyzed: {len(job_analyses)}",
        f"Individual failures parsed: {len(all_events)}",
        f"Unique failure signatures: {len(groups)}",
        "\n## Failures by category\n",
    ]
    category_occurrences = [
        category for event in all_events for category in event.categories
    ]
    for category, count in count_by(category_occurrences):
        report_lines.append(f"- {category}: {count}")

    report_lines.append("\n## Failures by bot\n")
    for bot_name, count in count_by([event.bot_name for event in all_events]):
        report_lines.append(f"- {bot_name}: {count}")

    report_lines.extend(build_question_failure_section(all_events))

    report_lines.append("\n## Failure groups (most frequent first)\n")
    for group in groups:
        report_lines.extend(build_failure_group_section(group))

    report_lines.append("\n## Per-job details\n")
    for analysis in job_analyses:
        unparsed_note = (
            " | NOTE: only an unparsed log tail was extracted, read the raw log"
            if any(
                event.exception_type == "UnparsedFailure" for event in analysis.events
            )
            else ""
        )
        report_lines.append(
            f"- **{analysis.bot_name}** (run {analysis.run_id}): "
            f"{len(analysis.events)} failures parsed | "
            f"[job link]({analysis.job_url}) | raw log: `{analysis.log_path}`"
            f"{unparsed_note}"
        )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines))

    json_path = output_dir / "failures.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "bot_name": event.bot_name,
                    "run_id": event.run_id,
                    "job_name": event.job_name,
                    "job_url": event.job_url,
                    "exception_type": event.exception_type,
                    "categories": event.categories,
                    "question_url": event.question_url,
                    "signature": normalize_message_to_signature(
                        event.exception_type, event.message
                    ),
                    "deepest_repo_frame": (
                        f"{event.deepest_repo_frame.file_path}:"
                        f"{event.deepest_repo_frame.line_number}"
                        if event.deepest_repo_frame
                        else None
                    ),
                    "message": event.message,
                }
                for event in all_events
            ],
            indent=2,
        )
    )
    return str(report_path)


def analyze_failed_jobs_for_run(
    repo: str, run: dict, token: str, raw_logs_dir: Path
) -> list[JobAnalysis]:
    failed_jobs = list_failed_jobs(repo, run["id"], token)
    logger.info(
        f"Run {run['id']} ({run['created_at']}): {len(failed_jobs)} failed jobs"
    )
    job_analyses: list[JobAnalysis] = []
    for job in failed_jobs:
        bot_name = extract_bot_name(job["name"])
        log_path = raw_logs_dir / f"run{run['id']}_{sanitize_filename(bot_name)}.log"
        try:
            log_text = download_job_log(repo, job["id"], token)
        except requests.HTTPError as http_error:
            logger.warning(f"Could not download log for job {job['id']}: {http_error}")
            continue
        log_path.write_text(log_text)
        events = parse_failures_from_log(
            log_text, bot_name, run["id"], job["name"], job["html_url"]
        )
        job_analyses.append(
            JobAnalysis(
                bot_name=bot_name,
                run_id=run["id"],
                job_url=job["html_url"],
                log_path=str(log_path),
                events=events,
            )
        )
    return job_analyses


def analyze_runs(
    repo: str,
    workflow: str,
    since: str,
    max_runs: int | None,
    output_dir: Path,
    run_id: int | None = None,
) -> str:
    token = resolve_github_token()
    if run_id is not None:
        runs = [github_get(f"/repos/{repo}/actions/runs/{run_id}", token)]
    else:
        since_datetime = parse_since_to_datetime(since)
        all_runs = list_workflow_runs(repo, workflow, token, since_datetime, max_runs)
        runs = [
            run
            for run in all_runs
            if run.get("conclusion") not in ("success", None)
            or run.get("status") != "completed"
        ]
        logger.info(
            f"Found {len(all_runs)} runs since {since_datetime.isoformat()}, "
            f"{len(runs)} with failures"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    analysis_dir = output_dir / timestamp
    raw_logs_dir = analysis_dir / "raw_logs"
    raw_logs_dir.mkdir(parents=True, exist_ok=True)

    job_analyses: list[JobAnalysis] = []
    for run in runs:
        assert isinstance(run, dict)
        job_analyses.extend(analyze_failed_jobs_for_run(repo, run, token, raw_logs_dir))

    if not job_analyses:
        logger.info("No failed jobs found in the selected window.")

    report_path = build_report(job_analyses, analysis_dir)
    logger.info(f"Report written to {report_path}")
    logger.info(f"Raw logs saved under {raw_logs_dir}")
    return report_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Analyze failed bot forecasting jobs from GitHub Actions"
    )
    parser.add_argument(
        "--since",
        default="1d",
        help=(
            "Time window to analyze. Accepts <N>h / <N>d / <N>w (e.g. 12h, 2d, 1w, "
            "4w) or an ISO datetime. The whole window is analyzed by default "
            "(default: 1d)"
        ),
    )
    parser.add_argument(
        "--run-id", type=int, default=None, help="Analyze a single specific run id"
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of workflow runs fetched. Default is no cap, "
            "so the full --since window is analyzed. Only set this to limit work on "
            "very large windows (a warning is logged if it truncates the window)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    try:
        analyze_runs(
            repo=args.repo,
            workflow=args.workflow,
            since=args.since,
            max_runs=args.max_runs,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    except RuntimeError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
