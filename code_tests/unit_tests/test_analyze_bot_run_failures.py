from pathlib import Path

import pytest
import requests

from scripts.skills.analyze_bot_failures import analyze_bot_run_failures
from scripts.skills.analyze_bot_failures.analyze_bot_run_failures import (
    GITHUB_RETRY_DELAYS_SECONDS,
    JobAnalysis,
    build_report,
    extract_leaf_errors,
    github_get,
    group_failures,
    load_or_download_job_log,
    normalize_message_to_signature,
    parse_failures_from_log,
    parse_questions_attempted,
)

GH_PREFIX = "2026-06-11T17:13:45.1234567Z "
PYTHON_ERROR_PREFIX = (
    "2026-06-11 17:13:45,123 - forecasting_tools.forecast_bots.forecast_bot - ERROR - "
)
QUESTION_URL = "https://www.metaculus.com/questions/44722"
OTHER_QUESTION_URL = "https://www.metaculus.com/questions/44999"
SUMMARY_STATS_LINE = "Stats for passing reports:"

TRACEBACK_LOG = (
    f"{GH_PREFIX}2026-06-11 17:13:40,100 - root - INFO - Running bot METAC_GPT_5_5_HIGH with 3 questions\n"
    f"{GH_PREFIX}2026-06-11 17:13:45,123 - forecasting_tools.forecast_bots.forecast_bot - ERROR - Exception occurred during forecasting:\n"
    f"{GH_PREFIX}Traceback (most recent call last):\n"
    f'{GH_PREFIX}  File "/home/runner/work/forecasting-tools/forecasting-tools/forecasting_tools/forecast_bots/forecast_bot.py", line 341, in _run_individual_question_with_error_propagation\n'
    f"{GH_PREFIX}    return await self._run_individual_question(question)\n"
    f'{GH_PREFIX}  File "/home/runner/work/forecasting-tools/forecasting-tools/.venv/lib/python3.11/site-packages/litellm/main.py", line 99, in completion\n'
    f"{GH_PREFIX}    raise error\n"
    f"{GH_PREFIX}RuntimeError: Error while processing question url: 'https://www.metaculus.com/questions/12345/': Rate limit hit for model gpt-5.5\n"
    f"{GH_PREFIX}2026-06-11 17:14:00,500 - root - INFO - done\n"
)

SHORT_SUMMARY_ONLY_LOG = f"{GH_PREFIX}2026-06-11 17:13:45,123 - root - INFO - ❌ Exception: ValueError | Message: Error while processing question url: 'https://www.metaculus.com/questions/678/': bad probability\n"

UNPARSABLE_LOG = (
    f"{GH_PREFIX}2026-06-11 17:13:45,123 - root - INFO - starting up\n"
    f"{GH_PREFIX}The operation was canceled.\n"
)

ALL_PREDICTIONS_FAILED_NESTED_MESSAGE = r"""1 sub-exceptions -> All 1 research reports/predictions failed: Errors: ['ExceptionGroup: 2 sub-exceptions -> Error while running research and predictions: Errors: [\'BadRequestError: litellm.BadRequestError: OpenrouterException - {"error":{"code":400,"metadata":{"raw":"{\\\\"message\\\\":\\\\"The request was rejected because it was considered high risk\\\\"}"}},"request_id":"req_011AAA"}\', \'BadRequestError: litellm.BadRequestError: OpenrouterException - {"error":{"code":400,"metadata":{"raw":"{\\\\"message\\\\":\\\\"The request was rejected because it was considered high risk\\\\"}"}},"request_id":"req_011BBB"}\'] (2 sub-exceptions)'] (1 sub-exception)"""

PARTIAL_FAILURE_NESTED_MESSAGE = r"""Expected at least 2.5 successful predictions, but only got 1. Errors encountered: ['Timeout: litellm.Timeout: AnthropicException - litellm.Timeout: Connection timed out. Timeout passed=60.0, time taken=60.1 seconds', "AssertionError: Answer is not a string and is of type: <class 'NoneType'>. Answer: None. Finish reason: content_filter", 'Timeout: litellm.Timeout: AnthropicException - litellm.Timeout: Connection timed out. Timeout passed=60.0, time taken=61.3 seconds']"""

VALIDATION_LEAF_AT_FIRST_NESTING_LEVEL = r"""Expected at least 2.5 successful predictions, but only got 2. Errors encountered: ["ValidationError: 1 validation error for NumericDistribution\n  Value error, No declared percentiles are within the range of the question +/- 25.0%. Lower bound: 0.0 [type=value_error, input_value={'declared_percentiles': [1, 2]}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.13/v/value_error"]"""

VALIDATION_LEAF_AT_SECOND_NESTING_LEVEL = r"""1 sub-exceptions -> All 1 research reports/predictions failed: Errors: ['ExceptionGroup: 1 sub-exceptions -> Error while running research and predictions: Errors: ["ValidationError: 1 validation error for NumericDistribution\\n  Value error, No declared percentiles are within the range of the question +/- 25.0%. Lower bound: 5.0 [type=value_error, input_value={\'declared_percentiles\': [3]}, input_type=dict]\\n    For further information visit https://errors.pydantic.dev/2.13/v/value_error"] (1 sub-exception)'] (1 sub-exception)"""


def make_log(*lines: str) -> str:
    return "".join(f"{GH_PREFIX}{line}\n" for line in lines)


def make_question_error_line(question_url: str, nested_message: str) -> str:
    return f"{PYTHON_ERROR_PREFIX}Error while processing question url: '{question_url}': {nested_message}"


def make_failed_summary_line(question_url: str, wrapped_message: str) -> str:
    return (
        f"❌ Exception: RuntimeError | Message: Error while processing question url: "
        f"'{question_url}': {wrapped_message}"
    )


def make_succeeded_summary_line(question_url: str) -> str:
    return f"✅ URL: {question_url} | Minor Errors: 0"


def leaf_list_message(*leaf_errors: str) -> str:
    return f"Expected at least 2.5 successful predictions, but only got 0. Errors encountered: {list(leaf_errors)}"


def parse_job_log(log_text: str) -> list[analyze_bot_run_failures.FailureEvent]:
    return parse_failures_from_log(
        log_text, "bot_a", 111, "bot_a / run_bot", "http://job-url"
    )


def make_response(status_code: int, body: bytes) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = body
    response.headers["Content-Type"] = "text/plain"
    response.url = "https://api.github.com/repos/owner/repo/actions/jobs/1/logs"
    return response


def test_parses_traceback_failure_with_question_url_and_repo_frame() -> None:
    events = parse_failures_from_log(
        TRACEBACK_LOG,
        "METAC_GPT_5_5_HIGH",
        111,
        "bot_gpt_5_5_high / run_bot",
        "http://job-url",
    )
    assert len(events) == 1
    event = events[0]
    assert event.exception_type == "RuntimeError"
    assert "Rate limit hit" in event.message
    assert event.question_url == "https://www.metaculus.com/questions/12345/"
    assert event.deepest_repo_frame is not None
    assert (
        event.deepest_repo_frame.file_path
        == "forecasting_tools/forecast_bots/forecast_bot.py"
    )
    assert event.deepest_repo_frame.line_number == 341
    assert "Traceback (most recent call last):" in (event.context_text or "")
    assert not event.is_question_failure


def test_falls_back_to_short_summary_lines() -> None:
    events = parse_failures_from_log(
        SHORT_SUMMARY_ONLY_LOG, "BOT_A", 222, "bot_a / run_bot", "http://job-url"
    )
    assert len(events) == 1
    assert events[0].exception_type == "ValueError"
    assert events[0].question_url == "https://www.metaculus.com/questions/678/"


def test_falls_back_to_log_tail_when_nothing_parseable() -> None:
    events = parse_failures_from_log(
        UNPARSABLE_LOG, "BOT_B", 333, "bot_b / run_bot", "http://job-url"
    )
    assert len(events) == 1
    assert events[0].exception_type == "UnparsedFailure"
    assert "The operation was canceled." in (events[0].context_text or "")


def test_grouping_normalizes_urls_and_numbers() -> None:
    signature_one = normalize_message_to_signature(
        "RuntimeError",
        "Error while processing question url: 'https://www.metaculus.com/questions/12345/': Rate limit hit after 30 seconds",
    )
    signature_two = normalize_message_to_signature(
        "RuntimeError",
        "Error while processing question url: 'https://www.metaculus.com/questions/99999/': Rate limit hit after 61 seconds",
    )
    assert signature_one == signature_two

    events_one = parse_failures_from_log(
        TRACEBACK_LOG, "BOT_A", 111, "bot_a / run_bot", "http://job-a"
    )
    events_two = parse_failures_from_log(
        TRACEBACK_LOG, "BOT_B", 112, "bot_b / run_bot", "http://job-b"
    )
    groups = group_failures(events_one + events_two)
    assert len(groups) == 1
    assert len(groups[0].events) == 2
    assert {event.bot_name for event in groups[0].events} == {"BOT_A", "BOT_B"}


def test_question_error_line_yields_leaf_error_from_nested_exception_groups() -> None:
    events = parse_job_log(
        make_log(
            make_question_error_line(
                QUESTION_URL, ALL_PREDICTIONS_FAILED_NESTED_MESSAGE
            )
        )
    )
    assert len(events) == 1
    event = events[0]
    assert event.is_question_failure
    assert event.question_url == QUESTION_URL
    assert event.exception_type == "BadRequestError"
    assert event.leaf_error_counts == {"BadRequestError": 2}
    assert event.message.startswith("litellm.BadRequestError: OpenrouterException")
    assert event.message.endswith('"request_id":"req_011AAA"}')
    assert '"message":"The request was rejected' in event.message
    assert "content_filter" in event.categories


def test_representative_leaf_is_the_most_common_one() -> None:
    events = parse_job_log(
        make_log(make_question_error_line(QUESTION_URL, PARTIAL_FAILURE_NESTED_MESSAGE))
    )
    assert len(events) == 1
    assert events[0].exception_type == "Timeout"
    assert events[0].leaf_error_counts == {"Timeout": 2, "AssertionError": 1}
    assert "timeout" in events[0].categories


def test_leaf_message_stops_at_its_closing_quote() -> None:
    leaf_errors = extract_leaf_errors(PARTIAL_FAILURE_NESTED_MESSAGE)
    assert [leaf.exception_type for leaf in leaf_errors] == [
        "Timeout",
        "AssertionError",
        "Timeout",
    ]
    assert leaf_errors[1].message == (
        "Answer is not a string and is of type: <class 'NoneType'>. "
        "Answer: None. Finish reason: content_filter"
    )
    assert leaf_errors[2].message.endswith("time taken=61.3 seconds")


def test_same_leaf_at_different_nesting_levels_shares_a_signature() -> None:
    first_level_leaf = extract_leaf_errors(VALIDATION_LEAF_AT_FIRST_NESTING_LEVEL)[0]
    second_level_leaf = extract_leaf_errors(VALIDATION_LEAF_AT_SECOND_NESTING_LEVEL)[0]
    first_signature = normalize_message_to_signature(
        first_level_leaf.exception_type, first_level_leaf.message
    )
    second_signature = normalize_message_to_signature(
        second_level_leaf.exception_type, second_level_leaf.message
    )
    assert first_signature == second_signature
    assert "No declared percentiles are within the range" in first_signature
    assert "input_value" not in first_signature


def test_wrapper_tracebacks_are_not_counted_when_question_errors_exist() -> None:
    log_text = make_log(
        "2026-06-11 17:13:40,100 - __main__ - INFO - Running bot METAC_BOT_A with 2 questions",
        make_question_error_line(QUESTION_URL, PARTIAL_FAILURE_NESTED_MESSAGE),
        make_succeeded_summary_line(OTHER_QUESTION_URL),
        make_failed_summary_line(QUESTION_URL, "ValueError - Expected at least 2.5"),
        SUMMARY_STATS_LINE,
        f"{PYTHON_ERROR_PREFIX}Exception occurred during forecasting:",
        "Traceback (most recent call last):",
        '  File "/home/runner/work/forecasting-tools/forecasting-tools/forecasting_tools/forecast_bots/forecast_bot.py", line 440, in _handle_errors',
        "ValueError: Expected at least 2.5 successful predictions, but only got 1.",
        "Traceback (most recent call last):",
        '  File "/home/runner/work/forecasting-tools/forecasting-tools/run_bots.py", line 197, in configure_and_run_bot',
        "RuntimeError: 1 errors occurred while forecasting: [RuntimeError('...')]",
    )
    events = parse_job_log(log_text)
    assert len(events) == 1
    assert events[0].is_question_failure
    assert parse_questions_attempted(log_text) == 2


def test_retried_question_keeps_only_its_final_error() -> None:
    timeout_message = leaf_list_message("Timeout: litellm.Timeout: timed out")
    validation_message = leaf_list_message("ValidationError: bad percentiles")
    events = parse_job_log(
        make_log(
            make_question_error_line(QUESTION_URL, timeout_message),
            make_question_error_line(QUESTION_URL, validation_message),
            make_failed_summary_line(QUESTION_URL, "ValueError - Expected"),
            SUMMARY_STATS_LINE,
        )
    )
    assert [event.exception_type for event in events] == ["ValidationError"]


def test_question_that_succeeded_on_retry_is_not_a_failure() -> None:
    events = parse_job_log(
        make_log(
            make_question_error_line(
                QUESTION_URL, leaf_list_message("Timeout: litellm.Timeout: timed out")
            ),
            make_question_error_line(
                OTHER_QUESTION_URL, leaf_list_message("NotFoundError: No endpoints")
            ),
            make_succeeded_summary_line(QUESTION_URL),
            make_failed_summary_line(OTHER_QUESTION_URL, "ValueError - Expected"),
            SUMMARY_STATS_LINE,
        )
    )
    assert [event.question_url for event in events] == [OTHER_QUESTION_URL]
    assert events[0].categories == ["model_not_found"]


def test_no_question_failures_when_summary_shows_every_retry_succeeded() -> None:
    events = parse_job_log(
        make_log(
            make_question_error_line(
                QUESTION_URL, leaf_list_message("Timeout: litellm.Timeout: timed out")
            ),
            make_succeeded_summary_line(QUESTION_URL),
            SUMMARY_STATS_LINE,
        )
    )
    assert not any(event.is_question_failure for event in events)


def test_group_subquestions_sharing_a_url_count_separately() -> None:
    events = parse_job_log(
        make_log(
            make_question_error_line(
                QUESTION_URL, leaf_list_message("NotFoundError: No endpoints")
            ),
            make_question_error_line(
                QUESTION_URL, leaf_list_message("NotFoundError: No endpoints")
            ),
            make_failed_summary_line(QUESTION_URL, "ValueError - Expected"),
            make_failed_summary_line(QUESTION_URL, "ValueError - Expected"),
            SUMMARY_STATS_LINE,
        )
    )
    assert len(events) == 2


def test_keeps_every_question_error_when_summary_is_missing() -> None:
    events = parse_job_log(
        make_log(
            make_question_error_line(
                QUESTION_URL, leaf_list_message("NotFoundError: No endpoints")
            ),
            make_question_error_line(
                OTHER_QUESTION_URL, leaf_list_message("NotFoundError: No endpoints")
            ),
            "##[error]The operation was canceled.",
        )
    )
    assert len(events) == 2


def test_question_error_without_leaves_uses_the_wrapped_exception_type() -> None:
    message = "All predictions must have the same option names, but {'A'} != ['A', 'B']"
    events = parse_job_log(
        make_log(
            make_question_error_line(QUESTION_URL, message),
            make_failed_summary_line(QUESTION_URL, f"ValueError - {message}"),
            SUMMARY_STATS_LINE,
        )
    )
    assert len(events) == 1
    assert events[0].exception_type == "ValueError"
    assert events[0].message == message
    assert events[0].leaf_error_counts == {}


def test_questions_attempted_is_none_without_running_bot_line() -> None:
    assert parse_questions_attempted(UNPARSABLE_LOG) is None


def test_bot_summary_reports_job_and_question_ratios(tmp_path: Path) -> None:
    job_with_two_failures = JobAnalysis(
        bot_name="bot_a",
        run_id=1,
        job_url="http://job-1",
        log_path="run1.log",
        events=parse_job_log(
            make_log(
                make_question_error_line(
                    QUESTION_URL, leaf_list_message("NotFoundError: No endpoints")
                ),
                make_question_error_line(
                    OTHER_QUESTION_URL, leaf_list_message("NotFoundError: No endpoints")
                ),
            )
        ),
        questions_attempted=5,
    )
    job_with_one_failure = JobAnalysis(
        bot_name="bot_a",
        run_id=2,
        job_url="http://job-2",
        log_path="run2.log",
        events=parse_job_log(
            make_log(
                make_question_error_line(
                    QUESTION_URL, leaf_list_message("NotFoundError: No endpoints")
                )
            )
        ),
        questions_attempted=3,
    )
    report_path = build_report(
        [job_with_two_failures, job_with_one_failure],
        {"bot_a": 2},
        {"bot_a": 4},
        tmp_path,
    )
    report_text = Path(report_path).read_text(encoding="utf-8")
    assert "| bot_a | 2/4 (50%) | 3/8 (38%) | 0 |" in report_text
    assert "[3x] NotFoundError" in report_text


def test_analyze_runs_counts_jobs_without_logs_and_skips_unfinished_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_by_run_id = {
        1: [
            {
                "id": 11,
                "name": "bot_a / run_bot",
                "status": "completed",
                "conclusion": "failure",
                "html_url": "http://job-11",
            },
            {
                "id": 12,
                "name": "bot_b / run_bot",
                "status": "completed",
                "conclusion": "failure",
                "html_url": "http://job-12",
            },
            {
                "id": 13,
                "name": "bot_c / run_bot",
                "status": "completed",
                "conclusion": "failure",
                "html_url": "http://job-13",
            },
        ],
        2: [
            {
                "id": 21,
                "name": "bot_a / run_bot",
                "status": "completed",
                "conclusion": "success",
                "html_url": "http://job-21",
            },
            {
                "id": 22,
                "name": "bot_b / run_bot",
                "status": "in_progress",
                "conclusion": None,
                "html_url": "http://job-22",
            },
            {
                "id": 23,
                "name": "bot_c / run_bot",
                "status": "completed",
                "conclusion": "failure",
                "html_url": "http://job-23",
            },
        ],
    }
    downloaded_log_paths: list[Path] = []

    def fake_load_or_download(
        repo: str, job_id: int, token: str, log_path: Path
    ) -> str:
        if job_id != 11:
            raise requests.HTTPError(
                "404 Client Error: The specified blob does not exist."
            )
        downloaded_log_paths.append(log_path)
        return make_log(
            "2026-06-11 17:13:40,100 - __main__ - INFO - Running bot METAC_BOT_A with 3 questions",
            make_question_error_line(
                QUESTION_URL, leaf_list_message("NotFoundError: No endpoints")
            ),
        )

    monkeypatch.setattr(
        analyze_bot_run_failures, "resolve_github_token", lambda: "token"
    )
    monkeypatch.setattr(
        analyze_bot_run_failures,
        "list_workflow_runs",
        lambda repo, workflow, token, since, max_runs: [
            {
                "id": 1,
                "created_at": "t1",
                "status": "completed",
                "conclusion": "failure",
            },
            {
                "id": 2,
                "created_at": "t2",
                "status": "completed",
                "conclusion": "failure",
            },
        ],
    )
    monkeypatch.setattr(
        analyze_bot_run_failures,
        "list_run_jobs",
        lambda repo, run_id, token: jobs_by_run_id[run_id],
    )
    monkeypatch.setattr(
        analyze_bot_run_failures, "load_or_download_job_log", fake_load_or_download
    )

    report_path = analyze_bot_run_failures.analyze_runs(
        repo="owner/repo",
        workflow="workflow.yaml",
        since="1d",
        max_runs=10,
        output_dir=tmp_path,
    )

    report_text = Path(report_path).read_text(encoding="utf-8")
    assert downloaded_log_paths == [tmp_path / "raw_logs" / "run1_job11_bot_a.log"]
    assert "| bot_c | 2/2 (100%) | 0/unknown | 0 |" in report_text
    assert "| bot_a | 1/2 (50%) | 1/3 (33%) | 0 |" in report_text
    assert "| bot_b | 1/1 (100%) | 0/unknown | 0 |" in report_text


def test_github_get_retries_network_errors_and_decodes_logs_as_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued_outcomes: list[Exception | requests.Response] = [
        requests.ConnectionError("No route to host"),
        requests.Timeout("Read timed out"),
        make_response(200, "❌ Exception: ValueError".encode("utf-8")),
    ]
    request_count = 0

    def fake_get(*args: object, **kwargs: object) -> requests.Response:
        nonlocal request_count
        request_count += 1
        outcome = queued_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    sleep_durations: list[float] = []
    monkeypatch.setattr(analyze_bot_run_failures.requests, "get", fake_get)
    monkeypatch.setattr(analyze_bot_run_failures.time, "sleep", sleep_durations.append)

    log_text = github_get(
        "/repos/owner/repo/actions/jobs/1/logs", "token", as_text=True
    )

    assert log_text == "❌ Exception: ValueError"
    assert request_count == 3
    assert sleep_durations == GITHUB_RETRY_DELAYS_SECONDS[:2]


def test_github_get_raises_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_count = 0

    def always_unreachable(*args: object, **kwargs: object) -> requests.Response:
        nonlocal request_count
        request_count += 1
        raise requests.ConnectionError("No route to host")

    monkeypatch.setattr(analyze_bot_run_failures.requests, "get", always_unreachable)
    monkeypatch.setattr(analyze_bot_run_failures.time, "sleep", lambda seconds: None)

    with pytest.raises(requests.ConnectionError):
        github_get("/repos/owner/repo/actions/runs/1/jobs", "token")
    assert request_count == len(GITHUB_RETRY_DELAYS_SECONDS) + 1


def test_github_get_does_not_retry_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_count = 0

    def not_found(*args: object, **kwargs: object) -> requests.Response:
        nonlocal request_count
        request_count += 1
        return make_response(404, b"The specified blob does not exist.")

    monkeypatch.setattr(analyze_bot_run_failures.requests, "get", not_found)
    monkeypatch.setattr(analyze_bot_run_failures.time, "sleep", lambda seconds: None)

    with pytest.raises(requests.HTTPError):
        github_get("/repos/owner/repo/actions/jobs/1/logs", "token", as_text=True)
    assert request_count == 1


def test_reuses_cached_log_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "run1_job2_bot_a.log"
    log_path.write_text("cached ❌ log", encoding="utf-8")

    def fail_if_downloaded(repo: str, job_id: int, token: str) -> str:
        raise AssertionError("cached log should not be downloaded again")

    monkeypatch.setattr(
        analyze_bot_run_failures, "download_job_log", fail_if_downloaded
    )
    assert load_or_download_job_log("owner/repo", 2, "token", log_path) == (
        "cached ❌ log"
    )


def test_downloads_and_caches_missing_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        analyze_bot_run_failures,
        "download_job_log",
        lambda repo, job_id, token: "fresh ❌ log",
    )
    log_path = tmp_path / "run1_job2_bot_a.log"

    assert load_or_download_job_log("owner/repo", 2, "token", log_path) == (
        "fresh ❌ log"
    )
    assert log_path.read_text(encoding="utf-8") == "fresh ❌ log"
    assert list(tmp_path.iterdir()) == [log_path]
