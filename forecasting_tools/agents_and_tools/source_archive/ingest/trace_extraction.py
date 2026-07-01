"""Build a citation manifest from a bot's run traces.

When the template bot is run with tracing enabled it writes one JSONL trace per
forecast attempt, recording the agent loop step by step. Those traces are the
*fullest* record of what the bot actually looked at — richer than the reasoning
comment it posts, which is length-truncated.

This module walks those traces and pulls out every external URL the bot touched,
turning each into a :class:`CitationRecord` with provenance (which trace, which
tool, the search query that surfaced it). That manifest is the input to the
capture pipeline, exactly like the comment-harvested one.

Trace layout
------------
A traced run is a directory tree::

    <run_dir>/
        bot_<name>/
            q_<question_id>/
                question.json
                traces_forecast_1_attempt_1.jsonl
                traces_summarize.jsonl
                ...

Each ``traces_*.jsonl`` file is a stream of newline-delimited event objects. The
events that can carry external links are:

- ``tool_call``  — the arguments the bot passed to a tool (e.g. a search query,
  or a ``url`` handed to a page fetcher). Carries ``name`` and ``call_id``.
- ``tool_result`` — what the tool returned. Search tools inline their citations
  here as ``[n](url)`` or as a list of result URLs. Carries ``call_id`` so the
  result can be attributed back to the originating ``tool_call``.
- ``initial_prompt`` — the first prompt of a trace. Only scanned for the
  ``summarize`` trace: the template bot runs research *outside* the agent loop
  and pastes the research blob verbatim into the summarizer's first prompt, so
  that is the one place those URLs are recoverable. Other traces' initial
  prompts just echo the question text (background, resolution criteria), whose
  URLs aren't research, so they're skipped.

Search provenance (``query`` / ``tool_args``) only exists in these instrumented
traces — it is populated here from each ``tool_call`` and carried onto the URLs
that the matching ``tool_result`` returned.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

from forecasting_tools.agents_and_tools.source_archive.ingest.url_extraction import (
    extract_urls,
)
from forecasting_tools.agents_and_tools.source_archive.models import CitationRecord

METACULUS_QUESTION_URL_FMT = "https://www.metaculus.com/questions/{}/"

# Event type -> the field on that event that carries the URL-bearing payload.
_SCANNABLE_FIELDS: dict[str, str] = {
    "tool_call": "args",
    "tool_result": "content",
    "initial_prompt": "prompt",
}
# The trace whose initial prompt holds pasted-in research (see module docstring).
_SUMMARIZE_TRACE_LABEL = "summarize"
# Keys a tool's input commonly uses for the search string, best-effort.
_QUERY_KEYS = ("query", "q", "search_query", "search", "queries", "question")


def _urls_in(value: Any) -> list[str]:
    """Return URLs found anywhere in a string / dict / list, in first-seen order.

    Tool args are structured (a dict) and tool results may be either a blob of
    text or a structured payload, so we walk the whole value and run the shared
    :func:`extract_urls` over every string we reach — keeping markdown-link and
    trailing-punctuation handling identical to the comment path.
    """
    urls: list[str] = []

    def walk(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            urls.extend(extract_urls(v))
            return
        if isinstance(v, dict):
            for key, val in v.items():
                walk(key)
                walk(val)
            return
        if isinstance(v, (list, tuple, set, frozenset)):
            for item in v:
                walk(item)
            return
        walk(str(v))

    walk(value)
    return urls


def _query_from_args(args: Any) -> str | None:
    """Pull the search string out of a tool's arguments, if recognisable."""
    if not isinstance(args, dict):
        return None
    for key in _QUERY_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, (list, tuple)) and val:
            joined = " ".join(str(item) for item in val if item)
            if joined.strip():
                return joined
    return None


def trace_label(trace_path: str) -> str:
    """``traces_forecast_1_attempt_1.jsonl`` -> ``forecast_1_attempt_1``."""
    name = os.path.basename(trace_path)
    if name.startswith("traces_"):
        name = name[len("traces_") :]
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    return name


def extract_records_from_events(
    events: Any,
    *,
    trace: str | None = None,
    include_initial_prompt: bool = False,
    run_id: str | None = None,
    bot: str | None = None,
    question_id: str | None = None,
    metaculus_id: str | None = None,
    question_url: str | None = None,
) -> list[CitationRecord]:
    """Turn one trace's event stream into CitationRecords.

    ``events`` is any iterable of event dicts (already parsed from JSONL). The
    given provenance is stamped onto every record; per-event provenance
    (``trace``, ``tool_name``, ``origin``, ``query``, ``tool_args``,
    ``first_seen``) is filled in here.

    Set ``include_initial_prompt`` to scan ``initial_prompt`` events — callers
    should only do this for the ``summarize`` trace (see module docstring).
    """
    records: list[CitationRecord] = []
    # Attribute tool_result events (which only carry call_id) back to the
    # originating tool_call's name and arguments.
    tool_name_by_call_id: dict[str, str] = {}
    tool_args_by_call_id: dict[str, Any] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")

        if event_type == "tool_call":
            call_id = str(event.get("call_id") or "").strip()
            name = event.get("name") or ""
            if call_id:
                if name:
                    tool_name_by_call_id[call_id] = name
                if "args" in event:
                    tool_args_by_call_id[call_id] = event.get("args")

        field = _SCANNABLE_FIELDS.get(event_type or "")
        if field is None:
            continue
        if event_type == "initial_prompt" and not include_initial_prompt:
            continue

        urls = _urls_in(event.get(field))
        if not urls:
            continue

        if event_type == "tool_call":
            tool_name = event.get("name") or ""
            origin = "tool_call"
            tool_args = (
                event.get("args") if isinstance(event.get("args"), dict) else None
            )
        elif event_type == "tool_result":
            call_id = str(event.get("call_id") or "").strip()
            tool_name = tool_name_by_call_id.get(call_id, "")
            origin = "tool_result"
            originating_args = tool_args_by_call_id.get(call_id)
            tool_args = originating_args if isinstance(originating_args, dict) else None
        else:  # initial_prompt
            tool_name = ""
            origin = event_type or ""
            tool_args = None

        query = _query_from_args(tool_args)
        timestamp = event.get("timestamp")
        for url in urls:
            record = CitationRecord(
                url=url,
                run_id=run_id,
                bot=bot,
                question_id=question_id,
                metaculus_id=metaculus_id,
                question_url=question_url,
                trace=trace,
                tool_name=tool_name,
                origin=origin,
                query=query,
                tool_args=tool_args,
            )
            if timestamp:
                record.first_seen = str(timestamp)
            records.append(record)

    return records


def _read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file, skipping blank or unparsable lines."""
    events: list[dict] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def extract_records_from_trace_file(
    trace_path: str,
    *,
    run_id: str | None = None,
    bot: str | None = None,
    question_id: str | None = None,
    metaculus_id: str | None = None,
    question_url: str | None = None,
) -> list[CitationRecord]:
    """Extract CitationRecords from one ``traces_*.jsonl`` file."""
    label = trace_label(trace_path)
    return extract_records_from_events(
        _read_jsonl(trace_path),
        trace=label,
        include_initial_prompt=(label == _SUMMARIZE_TRACE_LABEL),
        run_id=run_id,
        bot=bot,
        question_id=question_id,
        metaculus_id=metaculus_id,
        question_url=question_url,
    )


def _read_question_metadata(question_dir: str) -> tuple[str | None, str | None]:
    """Return ``(question_id, metaculus_id)`` from ``question.json`` in the dir.

    Read as a plain dict with flexible keys so the ingest stays decoupled from
    any particular question model. Missing/unparsable metadata is non-fatal —
    records are still emitted, just with empty question provenance.
    """
    question_path = os.path.join(question_dir, "question.json")
    if not os.path.exists(question_path):
        return None, None
    try:
        data = json.loads(Path(question_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(data, dict):
        return None, None

    def _str_or_none(*keys: str) -> str | None:
        for key in keys:
            val = data.get(key)
            if val is not None:
                return str(val)
        return None

    question_id = _str_or_none("question_id", "id", "post_id")
    metaculus_id = _str_or_none("metaculus_id", "post_id", "id")
    return question_id, metaculus_id


def extract_records_from_question_dir(
    question_dir: str,
    *,
    run_id: str | None = None,
    bot: str | None = None,
    question_id: str | None = None,
    metaculus_id: str | None = None,
    question_url: str | None = None,
) -> list[CitationRecord]:
    """Aggregate CitationRecords across every trace in one ``q_*`` dir.

    Question provenance is read from ``question.json`` in the dir; pass any of
    ``question_id`` / ``metaculus_id`` / ``question_url`` to override what's
    found there (or to supply it when the file is absent).
    """
    found_qid, found_mid = _read_question_metadata(question_dir)
    question_id = question_id or found_qid
    metaculus_id = metaculus_id or found_mid
    if question_url is None and metaculus_id is not None:
        question_url = METACULUS_QUESTION_URL_FMT.format(metaculus_id)

    records: list[CitationRecord] = []
    for trace_path in sorted(glob.glob(os.path.join(question_dir, "traces_*.jsonl"))):
        records.extend(
            extract_records_from_trace_file(
                trace_path,
                run_id=run_id,
                bot=bot,
                question_id=question_id,
                metaculus_id=metaculus_id,
                question_url=question_url,
            )
        )
    return records


def _bot_name_from_dir(bot_dir: str) -> str:
    """``.../bot_complex`` -> ``complex``."""
    name = os.path.basename(bot_dir)
    return name[len("bot_") :] if name.startswith("bot_") else name


def _question_dirs_flat(run_dir: str) -> list[str]:
    """Question dirs directly under ``run_dir`` (no ``bot_*`` level).

    A "question dir" is any immediate subdirectory that actually contains
    ``traces_*.jsonl``. This handles flatter layouts (e.g. a backfill of one
    bot's runs as ``<run_dir>/<question>/traces_*.jsonl``) where the ``bot_*``
    grouping is absent.
    """
    dirs = []
    for entry in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if os.path.isdir(entry) and glob.glob(os.path.join(entry, "traces_*.jsonl")):
            dirs.append(entry)
    return dirs


def harvest_run(
    run_dir: str, *, run_id: str | None = None, bot: str | None = None
) -> list[CitationRecord]:
    """Build a citation manifest from a whole traced run directory.

    Primary layout is ``<run_dir>/bot_*/q_*/traces_*.jsonl``, deriving ``run_id``
    from the run dir's name and ``bot`` from each ``bot_*`` subdir. If no
    ``bot_*`` subdirs exist, falls back to a **flat layout** —
    ``<run_dir>/<question>/traces_*.jsonl`` — attributing every question to a
    single bot (the ``bot`` argument, else the run dir's name). Question
    provenance still comes from each dir's ``question.json``.

    Returns the flat list of CitationRecords (one per URL occurrence); feed it
    through :func:`url_extraction.dedupe_records` before capture for one row per
    URL.
    """
    run_id = run_id or os.path.basename(os.path.normpath(run_dir))
    records: list[CitationRecord] = []

    bot_dirs = sorted(glob.glob(os.path.join(run_dir, "bot_*")))
    if bot_dirs:
        for bot_dir in bot_dirs:
            bot_name = _bot_name_from_dir(bot_dir)
            for question_dir in sorted(glob.glob(os.path.join(bot_dir, "q_*"))):
                records.extend(
                    extract_records_from_question_dir(
                        question_dir, run_id=run_id, bot=bot_name
                    )
                )
        return records

    # Flat fallback: no bot_* grouping. One bot, question dirs directly below.
    bot_name = bot or os.path.basename(os.path.normpath(run_dir))
    for question_dir in _question_dirs_flat(run_dir):
        records.extend(
            extract_records_from_question_dir(question_dir, run_id=run_id, bot=bot_name)
        )
    return records
