from scripts.skills.analyze_bot_failures.analyze_bot_run_failures import (
    group_failures,
    normalize_message_to_signature,
    parse_failures_from_log,
)

GH_PREFIX = "2026-06-11T17:13:45.1234567Z "

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
    assert "Traceback (most recent call last):" in (event.traceback_text or "")


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
    assert "The operation was canceled." in (events[0].traceback_text or "")


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
