from __future__ import annotations

import json
from pathlib import Path

from forecasting_tools.agents_and_tools.source_archive.ingest.trace_extraction import (
    extract_records_from_events,
    extract_records_from_question_dir,
    extract_records_from_trace_file,
    harvest_run,
    trace_label,
)


def test_trace_label_strips_prefix_and_suffix():
    assert trace_label("/x/traces_forecast_1_attempt_1.jsonl") == "forecast_1_attempt_1"
    assert trace_label("traces_summarize.jsonl") == "summarize"


def test_tool_call_carries_query_and_tool_args():
    events = [
        {
            "type": "tool_call",
            "call_id": "c1",
            "name": "search_online",
            "args": {"query": "uk election polls", "max_results": 5},
        }
    ]
    records = extract_records_from_events(events, trace="forecast_1")
    # No URL in the args -> nothing emitted from the tool_call itself.
    assert records == []


def test_tool_result_attributed_to_originating_call():
    events = [
        {
            "type": "tool_call",
            "call_id": "c1",
            "name": "search_online",
            "args": {"query": "uk election polls"},
        },
        {
            "type": "tool_result",
            "call_id": "c1",
            "content": "Top hit: [poll](https://a.test/poll) and https://b.test/x",
            "timestamp": "2026-05-12T12:00:00+00:00",
        },
    ]
    records = extract_records_from_events(events, trace="forecast_1", bot="template")
    assert [r.url for r in records] == ["https://a.test/poll", "https://b.test/x"]
    rec = records[0]
    assert rec.origin == "tool_result"
    assert rec.tool_name == "search_online"
    assert rec.query == "uk election polls"
    assert rec.tool_args == {"query": "uk election polls"}
    assert rec.trace == "forecast_1"
    assert rec.bot == "template"
    assert rec.first_seen == "2026-05-12T12:00:00+00:00"


def test_query_from_list_args():
    events = [
        {
            "type": "tool_call",
            "call_id": "c1",
            "name": "s",
            "args": {"queries": ["a", "b"]},
        },
        {"type": "tool_result", "call_id": "c1", "content": "https://a.test/x"},
    ]
    records = extract_records_from_events(events, trace="t")
    assert records[0].query == "a b"


def test_url_directly_in_tool_call_args():
    events = [
        {
            "type": "tool_call",
            "call_id": "c1",
            "name": "fetch_page",
            "args": {"url": "https://a.test/page"},
        }
    ]
    records = extract_records_from_events(events, trace="t")
    assert len(records) == 1
    assert records[0].url == "https://a.test/page"
    assert records[0].origin == "tool_call"
    assert records[0].tool_name == "fetch_page"
    assert records[0].tool_args == {"url": "https://a.test/page"}


def test_initial_prompt_only_scanned_when_enabled():
    events = [
        {"type": "initial_prompt", "prompt": "background: https://a.test/bg"},
    ]
    assert extract_records_from_events(events, trace="forecast_1") == []
    records = extract_records_from_events(
        events, trace="summarize", include_initial_prompt=True
    )
    assert [r.url for r in records] == ["https://a.test/bg"]
    assert records[0].origin == "initial_prompt"
    assert records[0].tool_name == ""


def test_non_dict_events_skipped():
    events = ["garbage", None, {"type": "tool_result", "content": "https://a.test/x"}]
    records = extract_records_from_events(events, trace="t")
    assert [r.url for r in records] == ["https://a.test/x"]


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


def test_trace_file_uses_summarize_rule(tmp_path: Path):
    f = tmp_path / "traces_summarize.jsonl"
    _write_jsonl(f, [{"type": "initial_prompt", "prompt": "see https://a.test/r"}])
    records = extract_records_from_trace_file(str(f), run_id="run1", bot="template")
    assert [r.url for r in records] == ["https://a.test/r"]
    assert records[0].trace == "summarize"
    assert records[0].run_id == "run1"


def test_trace_file_skips_blank_and_bad_lines(tmp_path: Path):
    f = tmp_path / "traces_forecast_1.jsonl"
    f.write_text(
        '\n{"type": "tool_result", "content": "https://a.test/x"}\nnot json\n',
        encoding="utf-8",
    )
    records = extract_records_from_trace_file(str(f))
    assert [r.url for r in records] == ["https://a.test/x"]


def test_question_dir_reads_metadata_and_builds_url(tmp_path: Path):
    qdir = tmp_path / "q_123"
    qdir.mkdir()
    (qdir / "question.json").write_text(
        json.dumps({"question_id": "metac_123", "metaculus_id": 123}),
        encoding="utf-8",
    )
    _write_jsonl(
        qdir / "traces_forecast_1.jsonl",
        [{"type": "tool_result", "content": "https://a.test/x"}],
    )
    records = extract_records_from_question_dir(
        str(qdir), run_id="run1", bot="template"
    )
    assert len(records) == 1
    rec = records[0]
    assert rec.question_id == "metac_123"
    assert rec.metaculus_id == "123"
    assert rec.question_url == "https://www.metaculus.com/questions/123/"


def test_question_dir_without_metadata_still_emits(tmp_path: Path):
    qdir = tmp_path / "q_x"
    qdir.mkdir()
    _write_jsonl(
        qdir / "traces_forecast_1.jsonl",
        [{"type": "tool_result", "content": "https://a.test/x"}],
    )
    records = extract_records_from_question_dir(str(qdir))
    assert [r.url for r in records] == ["https://a.test/x"]
    assert records[0].question_id is None
    assert records[0].question_url is None


def test_harvest_run_walks_bot_and_question_dirs(tmp_path: Path):
    run = tmp_path / "run_demo"
    qdir = run / "bot_template" / "q_1"
    qdir.mkdir(parents=True)
    (qdir / "question.json").write_text(
        json.dumps({"metaculus_id": 1}), encoding="utf-8"
    )
    _write_jsonl(
        qdir / "traces_forecast_1.jsonl",
        [{"type": "tool_result", "content": "https://a.test/x"}],
    )
    records = harvest_run(str(run))
    assert len(records) == 1
    rec = records[0]
    assert rec.run_id == "run_demo"
    assert rec.bot == "template"
    assert rec.metaculus_id == "1"


def test_harvest_run_flat_layout_without_bot_dirs(tmp_path: Path):
    # Flat layout: <run>/<question>/traces_*.jsonl with no bot_* grouping.
    run = tmp_path / "s3_backfill"
    qdir = run / "2026-05-20_metac_43538"
    qdir.mkdir(parents=True)
    (qdir / "question.json").write_text(
        json.dumps({"metaculus_id": 43538}), encoding="utf-8"
    )
    _write_jsonl(
        qdir / "traces_forecast_1.jsonl",
        [{"type": "tool_result", "content": "https://a.test/x"}],
    )
    records = harvest_run(str(run), bot="mybot")
    assert len(records) == 1
    rec = records[0]
    assert rec.bot == "mybot"  # the flat-layout bot override
    assert rec.metaculus_id == "43538"  # still read from question.json


def test_harvest_run_flat_layout_defaults_bot_to_run_name(tmp_path: Path):
    run = tmp_path / "myrun"
    qdir = run / "q_only"
    qdir.mkdir(parents=True)
    _write_jsonl(
        qdir / "traces_x.jsonl",
        [{"type": "tool_result", "content": "https://a.test/y"}],
    )
    records = harvest_run(str(run))  # no bot= -> defaults to run dir name
    assert [r.bot for r in records] == ["myrun"]
