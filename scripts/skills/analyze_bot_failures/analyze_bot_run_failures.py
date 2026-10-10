"""
Pulls logs of failed bot jobs from the GitHub Actions tournament workflow,
parses out the failure reasons, and writes an aggregated markdown/json report.

Requires a GitHub token with repo read access (the job-log endpoint rejects
unauthenticated requests even on public repos). The token is resolved from the
GITHUB_TOKEN env var, then from `gh auth token`.

Raw logs are cached in <output-dir>/raw_logs/ keyed by job id, so rerunning with
the same --output-dir reuses logs that were already downloaded.

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
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "Metaculus/forecasting-tools"
DEFAULT_WORKFLOW = "run-bot-aib-tournament.yaml"
DEFAULT_OUTPUT_DIR = Path("logs/workflow_failure_analysis")
LOG_TAIL_LINES_FOR_UNPARSED_FAILURES = 60
REPORT_SUMMARY_STATS_LINE = "Stats for passing reports"
GITHUB_RETRY_DELAYS_SECONDS = [5, 15, 45]

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
    "model_not_found": [r"NotFoundError", r"No endpoints found", r"not_found_error"],
    "bad_request_or_model_refusal": [
        r"\b400\b",
        r"BadRequestError",
        r"InvalidRequestError",
        r"refus",
        r"content[\s_-]?policy",
    ],
    "content_filter": [r"content[\s_-]?filter", r"considered high risk"],
    "structured_output_parsing": [
        r"structure[d]?[\s_-]?output",
        r"ValidationError",
        r"JSONDecodeError",
        r"Failed to parse",
        r"pydantic",
    ],
    "empty_llm_response": [r"Answer is not a string"],
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
QUESTION_ERROR_LINE_PATTERN = re.compile(
    r" - ERROR - Error while processing question url: '(?P<question_url>[^']+)': "
    r"(?P<nested_message>.*)"
)
WRAPPED_QUESTION_EXCEPTION_PATTERN = re.compile(
    r"Error while processing question url: '(?P<question_url>[^']+)': "
    r"(?P<exception_type>\w+) - "
)
FAILED_QUESTION_SUMMARY_LINE_PATTERN = re.compile(
    r"❌ Exception: \w+ \| Message: .*?"
    r"Error while processing question url: '(?P<question_url>[^']+)'"
)
QUESTIONS_ATTEMPTED_LINE_PATTERN = re.compile(
    r"Running bot \S+ with (?P<question_count>\d+) questions"
)
NESTED_ERROR_ITEM_START_PATTERN = re.compile(
    r"(?:\[|, )(?P<quote>\\*['\"])"
    r"(?P<exception_type>(?=[A-Z])\w*(?:Error|Exception|Timeout|ExceptionGroup)): "
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
class LeafError:
    exception_type: str
    message: str


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
    context_text: str | None
    deepest_repo_frame: RepoFrame | None
    is_question_failure: bool = False
    leaf_error_counts: dict[str, int] = field(default_factory=dict)


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
    questions_attempted: int | None

    @property
    def failed_question_count(self) -> int:
        return sum(1 for event in self.events if event.is_question_failure)


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


def get_with_network_retries(
    url: str, headers: dict, params: dict | None
) -> requests.Response:
    for retry_delay_seconds in GITHUB_RETRY_DELAYS_SECONDS:
        try:
            return requests.get(url, headers=headers, params=params, timeout=60)
        except (requests.ConnectionError, requests.Timeout) as network_error:
            logger.warning(
                f"Network error requesting {url}: {network_error}. "
                f"Retrying in {retry_delay_seconds}s"
            )
            time.sleep(retry_delay_seconds)
    return requests.get(url, headers=headers, params=params, timeout=60)


def github_get(
    path: str, token: str, params: dict | None = None, as_text: bool = False
) -> dict | str:
    response = get_with_network_retries(
        f"{GITHUB_API}{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        params=params,
    )
    response.raise_for_status()
    if not as_text:
        return response.json()
    # Log downloads come back without a charset, which requests decodes as latin-1
    response.encoding = "utf-8"
    return response.text


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
    repo: str, workflow: str, token: str, since: datetime, max_runs: int
) -> list[dict]:
    runs: list[dict] = []
    page = 1
    while len(runs) < max_runs:
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
    return runs[:max_runs]


def list_run_jobs(repo: str, run_id: int, token: str) -> list[dict]:
    jobs: list[dict] = []
    page = 1
    page_size = 100
    while True:
        data = github_get(
            f"/repos/{repo}/actions/runs/{run_id}/jobs",
            token,
            params={"per_page": page_size, "page": page, "filter": "latest"},
        )
        assert isinstance(data, dict)
        page_jobs = data.get("jobs", [])
        jobs.extend(page_jobs)
        if len(page_jobs) < page_size:
            break
        page += 1
    return jobs


def download_job_log(repo: str, job_id: int, token: str) -> str:
    log_text = github_get(
        f"/repos/{repo}/actions/jobs/{job_id}/logs", token, as_text=True
    )
    assert isinstance(log_text, str)
    return log_text


def load_or_download_job_log(repo: str, job_id: int, token: str, log_path: Path) -> str:
    if log_path.exists():
        return log_path.read_text(encoding="utf-8")
    log_text = download_job_log(repo, job_id, token)
    partial_path = log_path.with_suffix(".partial")
    partial_path.write_text(log_text, encoding="utf-8")
    partial_path.replace(log_path)
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


def unescape_nested_repr(text: str) -> str:
    with_newlines = re.sub(r"\\+n", "\n", text)
    return re.sub(r"\\+(?=['\"])", "", with_newlines)


def extract_leaf_errors(nested_message: str) -> list[LeafError]:
    """
    Errors are nested as `'ErrorType: message'` items of repr'd lists, with one
    more level of quote escaping per level of nesting. A leaf ends at the next
    quote of its own escaping level that closes a list item.
    """
    leaf_errors: list[LeafError] = []
    last_leaf_end = 0
    for item_match in NESTED_ERROR_ITEM_START_PATTERN.finditer(nested_message):
        exception_type = item_match.group("exception_type")
        if item_match.start() < last_leaf_end or exception_type.endswith(
            "ExceptionGroup"
        ):
            continue
        closing_quote_pattern = re.compile(
            r"(?<!\\)" + re.escape(item_match.group("quote")) + r"(?=, |\])"
        )
        closing_match = closing_quote_pattern.search(nested_message, item_match.end())
        last_leaf_end = closing_match.start() if closing_match else len(nested_message)
        leaf_errors.append(
            LeafError(
                exception_type=exception_type,
                message=unescape_nested_repr(
                    nested_message[item_match.end() : last_leaf_end]
                ),
            )
        )
    return leaf_errors


def pick_representative_leaf(leaf_errors: list[LeafError]) -> LeafError:
    signatures = [
        normalize_message_to_signature(leaf.exception_type, leaf.message)
        for leaf in leaf_errors
    ]
    most_common_signature = Counter(signatures).most_common(1)[0][0]
    return leaf_errors[signatures.index(most_common_signature)]


def find_matches(log_lines: list[str], pattern: re.Pattern[str]) -> list[re.Match[str]]:
    matches = []
    for line in log_lines:
        line_match = pattern.search(line)
        if line_match:
            matches.append(line_match)
    return matches


def select_final_question_error_matches(
    error_matches: list[re.Match[str]], final_failure_count_by_url: Counter[str]
) -> list[re.Match[str]]:
    remaining_failures_by_url = Counter(final_failure_count_by_url)
    selected_matches = []
    for error_match in reversed(error_matches):
        question_url = error_match.group("question_url")
        if remaining_failures_by_url[question_url] > 0:
            selected_matches.append(error_match)
            remaining_failures_by_url[question_url] -= 1
    return list(reversed(selected_matches))


def parse_question_failure_events(
    log_lines: list[str], bot_name: str, run_id: int, job_name: str, job_url: str
) -> list[FailureEvent]:
    """
    One event per question report that ended in failure. Error lines alone can
    over-count (run_bots retries timeouts), so when the report summary was logged
    its ❌ lines set how many failures each URL had, and the latest that many error
    lines are kept. Counting is per URL occurrence because group subquestions share
    a URL. Without a summary (e.g. job cancelled) every error line counts.
    """
    error_matches = find_matches(log_lines, QUESTION_ERROR_LINE_PATTERN)
    final_failure_count_by_url = Counter(
        summary_match.group("question_url")
        for summary_match in find_matches(
            log_lines, FAILED_QUESTION_SUMMARY_LINE_PATTERN
        )
    )
    report_summary_was_logged = any(
        line.startswith(REPORT_SUMMARY_STATS_LINE) for line in log_lines
    )
    if report_summary_was_logged:
        error_matches = select_final_question_error_matches(
            error_matches, final_failure_count_by_url
        )
    wrapped_exception_type_by_url = {
        wrapped_match.group("question_url"): wrapped_match.group("exception_type")
        for wrapped_match in find_matches(log_lines, WRAPPED_QUESTION_EXCEPTION_PATTERN)
    }
    events = []
    for error_match in error_matches:
        question_url = error_match.group("question_url")
        nested_message = error_match.group("nested_message")
        leaf_errors = extract_leaf_errors(nested_message)
        if leaf_errors:
            representative_leaf = pick_representative_leaf(leaf_errors)
            exception_type = representative_leaf.exception_type
            message = representative_leaf.message
        else:
            exception_type = wrapped_exception_type_by_url.get(
                question_url, "UnknownException"
            )
            message = nested_message
        events.append(
            FailureEvent(
                bot_name=bot_name,
                run_id=run_id,
                job_name=job_name,
                job_url=job_url,
                exception_type=exception_type,
                message=message,
                categories=categorize_error(f"{exception_type} {message}"),
                question_url=question_url,
                context_text=nested_message,
                deepest_repo_frame=None,
                is_question_failure=True,
                leaf_error_counts=dict(
                    Counter(leaf.exception_type for leaf in leaf_errors)
                ),
            )
        )
    return events


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
                context_text=traceback_text,
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
                or continuation_line.startswith(REPORT_SUMMARY_STATS_LINE)
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
                context_text=None,
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
            "timeout) or the log format changed. Log tail attached as context_text."
        ),
        categories=categorize_error(log_tail),
        question_url=None,
        context_text=log_tail,
        deepest_repo_frame=None,
    )


def parse_failures_from_log(
    log_text: str, bot_name: str, run_id: int, job_name: str, job_url: str
) -> list[FailureEvent]:
    log_lines = strip_github_timestamps(log_text).splitlines()
    events = parse_question_failure_events(
        log_lines, bot_name, run_id, job_name, job_url
    )
    if not events:
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


def parse_questions_attempted(log_text: str) -> int | None:
    line_match = QUESTIONS_ATTEMPTED_LINE_PATTERN.search(log_text)
    return int(line_match.group("question_count")) if line_match else None


def normalize_message_to_signature(exception_type: str, message: str) -> str:
    no_pydantic_input = re.sub(
        r"\[type=\w+, input_value=.*?, input_type=\w+\]", "", message
    )
    no_pydantic_links = re.sub(
        r"For further information visit \S+", "", no_pydantic_input
    )
    no_urls = QUESTION_URL_PATTERN.sub("<QUESTION_URL>", no_pydantic_links)
    no_request_ids = re.sub(r"\b(req|chatcmpl|gen)[-_][\w-]+", "<ID>", no_urls)
    no_numbers = re.sub(r"\d+", "<N>", no_request_ids)
    no_list_contents = re.sub(r"\[[^\[\]]*\]", "[...]", no_numbers)
    collapsed_whitespace = re.sub(r"\s+", " ", no_list_contents).strip()
    return f"{exception_type}: {collapsed_whitespace[:300]}"


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
    bot_counts = count_by([event.bot_name for event in group.events])
    section_lines = [
        f"### [{len(group.events)}x] {example.exception_type} | "
        f"categories={','.join(example.categories)}",
        "Bots affected: "
        + ", ".join(f"{bot_name} ({count})" for bot_name, count in bot_counts),
    ]
    leaf_error_counts: Counter[str] = Counter()
    for event in group.events:
        leaf_error_counts.update(event.leaf_error_counts)
    if leaf_error_counts:
        section_lines.append(
            "Leaf errors across these questions: "
            + ", ".join(
                f"{count}x {exception_type}"
                for exception_type, count in leaf_error_counts.most_common()
            )
        )
    if example.deepest_repo_frame:
        section_lines.append(
            f"Deepest repo frame: `{example.deepest_repo_frame.file_path}:"
            f"{example.deepest_repo_frame.line_number}`"
        )
    section_lines.append(f"\n```\n{truncate_text(example.message, 2000)}\n```\n")
    if example.context_text:
        section_lines.append(
            f"<details><summary>Example full error / traceback / log tail</summary>\n\n"
            f"```\n{truncate_text(example.context_text, 3000)}\n```\n\n</details>\n"
        )
    return section_lines


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator}/unknown"
    return f"{numerator}/{denominator} ({numerator / denominator:.0%})"


def build_bot_summary_lines(
    job_analyses: list[JobAnalysis], total_jobs_by_bot: dict[str, int]
) -> list[str]:
    analyses_by_bot: dict[str, list[JobAnalysis]] = {}
    for analysis in job_analyses:
        analyses_by_bot.setdefault(analysis.bot_name, []).append(analysis)

    summary_lines = [
        "\n## Failures by bot\n",
        "- **Failed jobs / jobs in analyzed runs**: analyzed runs are those with at "
        "least one failed job, so runs where every bot passed are not counted.",
        "- **Failed questions / questions attempted**: only failed jobs are "
        "downloaded, so this is the share of questions that failed *within failed "
        "jobs* (low = one bad question, high = bot broken), not the bot's overall "
        "question failure rate. Attempts come from the `Running bot X with N "
        "questions` log line.\n",
        "| Bot | Failed jobs / jobs in analyzed runs | Failed questions / questions "
        "attempted | Failed jobs without question-level errors |",
        "|---|---|---|---|",
    ]
    sorted_bots = sorted(
        analyses_by_bot.items(), key=lambda bot_item: (-len(bot_item[1]), bot_item[0])
    )
    for bot_name, bot_analyses in sorted_bots:
        analyses_with_known_attempts = [
            analysis
            for analysis in bot_analyses
            if analysis.questions_attempted is not None
        ]
        failed_questions = sum(
            analysis.failed_question_count for analysis in analyses_with_known_attempts
        )
        questions_attempted = sum(
            analysis.questions_attempted or 0
            for analysis in analyses_with_known_attempts
        )
        jobs_without_question_errors = sum(
            1 for analysis in bot_analyses if analysis.failed_question_count == 0
        )
        summary_lines.append(
            f"| {bot_name} "
            f"| {format_ratio(len(bot_analyses), total_jobs_by_bot.get(bot_name, 0))} "
            f"| {format_ratio(failed_questions, questions_attempted)} "
            f"| {jobs_without_question_errors} |"
        )
    return summary_lines


def build_report(
    job_analyses: list[JobAnalysis],
    total_jobs_by_bot: dict[str, int],
    report_dir: Path,
) -> str:
    all_events = [event for analysis in job_analyses for event in analysis.events]
    question_events = [event for event in all_events if event.is_question_failure]
    groups = group_failures(all_events)

    report_lines = [
        "# Bot Workflow Failure Report",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
        f"\nFailed jobs analyzed: {len(job_analyses)}",
        f"Failed questions (one event each, grouped by leaf error): "
        f"{len(question_events)}",
        f"Other failure events (from jobs without question-level errors): "
        f"{len(all_events) - len(question_events)}",
        f"Unique failure signatures: {len(groups)}",
        "\n## Failures by category\n",
    ]
    category_occurrences = [
        category for event in all_events for category in event.categories
    ]
    for category, count in count_by(category_occurrences):
        report_lines.append(f"- {category}: {count}")

    report_lines.extend(build_bot_summary_lines(job_analyses, total_jobs_by_bot))

    question_urls = [event.question_url for event in all_events if event.question_url]
    if question_urls:
        report_lines.append(
            "\n## Questions appearing in failures "
            "(recurring ones are POST_IDS_TO_SKIP candidates)\n"
        )
        for question_url, count in count_by(question_urls):
            report_lines.append(f"- {question_url} : {count}")

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
        questions_attempted_text = (
            analysis.questions_attempted
            if analysis.questions_attempted is not None
            else "unknown"
        )
        report_lines.append(
            f"- **{analysis.bot_name}** (run {analysis.run_id}): "
            f"{analysis.failed_question_count}/{questions_attempted_text} questions "
            f"failed, {len(analysis.events)} failure events | "
            f"[job link]({analysis.job_url}) | raw log: `{analysis.log_path}`"
            f"{unparsed_note}"
        )

    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    json_path = report_dir / "failures.json"
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
                    "is_question_failure": event.is_question_failure,
                    "leaf_error_counts": event.leaf_error_counts,
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
        ),
        encoding="utf-8",
    )
    return str(report_path)


def analyze_failed_jobs(
    repo: str, run_id: int, failed_jobs: list[dict], token: str, raw_logs_dir: Path
) -> list[JobAnalysis]:
    job_analyses: list[JobAnalysis] = []
    for job in failed_jobs:
        bot_name = extract_bot_name(job["name"])
        log_path = (
            raw_logs_dir
            / f"run{run_id}_job{job['id']}_{sanitize_filename(bot_name)}.log"
        )
        try:
            log_text = load_or_download_job_log(repo, job["id"], token, log_path)
        except requests.HTTPError as http_error:
            logger.warning(f"Could not download log for job {job['id']}: {http_error}")
            continue
        events = parse_failures_from_log(
            log_text, bot_name, run_id, job["name"], job["html_url"]
        )
        questions_attempted = parse_questions_attempted(log_text)
        if questions_attempted is None and any(
            event.is_question_failure for event in events
        ):
            logger.warning(
                f"No 'Running bot ... with N questions' line in {log_path}, so its "
                "failed questions are left out of the per-bot question ratio"
            )
        job_analyses.append(
            JobAnalysis(
                bot_name=bot_name,
                run_id=run_id,
                job_url=job["html_url"],
                log_path=str(log_path),
                events=events,
                questions_attempted=questions_attempted,
            )
        )
    return job_analyses


def analyze_runs(
    repo: str,
    workflow: str,
    since: str,
    max_runs: int,
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
    report_dir = output_dir / timestamp
    raw_logs_dir = output_dir / "raw_logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_logs_dir.mkdir(parents=True, exist_ok=True)

    job_analyses: list[JobAnalysis] = []
    total_jobs_by_bot: Counter[str] = Counter()
    for run_index, run in enumerate(runs, start=1):
        assert isinstance(run, dict)
        jobs = list_run_jobs(repo, run["id"], token)
        total_jobs_by_bot.update(extract_bot_name(job["name"]) for job in jobs)
        failed_jobs = [job for job in jobs if job.get("conclusion") == "failure"]
        logger.info(
            f"[{run_index}/{len(runs)}] Run {run['id']} ({run['created_at']}): "
            f"{len(failed_jobs)} of {len(jobs)} jobs failed"
        )
        job_analyses.extend(
            analyze_failed_jobs(repo, run["id"], failed_jobs, token, raw_logs_dir)
        )

    if not job_analyses:
        logger.info("No failed jobs found in the selected window.")

    report_path = build_report(job_analyses, total_jobs_by_bot, report_dir)
    logger.info(f"Report written to {report_path}")
    logger.info(f"Raw logs cached under {raw_logs_dir}")
    return report_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Analyze failed bot forecasting jobs from GitHub Actions"
    )
    parser.add_argument(
        "--since",
        default="1d",
        help="Time window like 12h, 2d, 1w, or an ISO datetime (default: 1d)",
    )
    parser.add_argument(
        "--run-id", type=int, default=None, help="Analyze a single specific run id"
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--max-runs", type=int, default=50)
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
