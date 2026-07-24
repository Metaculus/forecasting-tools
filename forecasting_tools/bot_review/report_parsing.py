"""Pull structure back out of a report explanation.

``ForecastBot._create_comment`` assembles every report into the same shape: a
SUMMARY listing each forecaster's prediction, the RESEARCH behind them, and one
FORECASTS subsection per forecaster. That string is what gets saved to disk and
what gets posted to Metaculus as a comment, so the same functions read either.

Forecasters are keyed ``R<research report>:F<forecaster>``, since a bot running
more than one research report per question has a forecaster 1 in each.
"""

from __future__ import annotations

import re

SECTIONS = ("summary", "research", "forecasts")
REPORT_PATTERN = re.compile(r"^## Report (\d+) Summary\s*$", re.MULTILINE)
FORECASTER_PATTERN = re.compile(
    r"^\*Forecaster (\d+)(?: \(([^)]*)\))?\*:", re.MULTILINE
)
RATIONALE_PATTERN = re.compile(
    r"^## R(\d+): Forecaster (\d+) Reasoning\s*$", re.MULTILINE
)
FINAL_PREDICTION_PATTERN = re.compile(
    r"^\*Final Prediction\*:(.*?)(?=^\*[A-Z]|^## |\Z)", re.MULTILINE | re.DOTALL
)


def split_sections(explanation: str) -> dict[str, str]:
    """The SUMMARY / RESEARCH / FORECASTS blocks, by lowercased name."""
    marks = []
    for name in SECTIONS:
        match = re.search(rf"^# {name.upper()}\s*$", explanation, re.MULTILINE)
        if match:
            marks.append((match.start(), match.end(), name))
    marks.sort()
    sections = {}
    for i, (_, end, name) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(explanation)
        sections[name] = explanation[end:stop].strip()
    return sections


def _bodies_between(text: str, marks: list[tuple[int, int]]) -> list[str]:
    return [
        text[end : (marks[i + 1][0] if i + 1 < len(marks) else len(text))].strip()
        for i, (_, end) in enumerate(marks)
    ]


def forecasts_block(summary: str) -> str:
    """The ``### Forecasts`` block of one report's summary."""
    match = re.search(r"^### Forecasts\s*$", summary, re.MULTILINE)
    if not match:
        return ""
    rest = summary[match.end() :]
    following = re.search(r"^### ", rest, re.MULTILINE)
    return rest[: following.start()] if following else rest


def parse_forecasters(summary: str) -> list[dict]:
    """Each forecaster's prediction as it was written in the summary.

    ``model`` is None unless the bot annotates its own bullets with model names;
    stock forecasting-tools does not.
    """
    reports = list(REPORT_PATTERN.finditer(summary))
    blocks = _bodies_between(summary, [(m.start(), m.end()) for m in reports])
    forecasters = []
    for report, block in zip(reports, blocks):
        bullets = forecasts_block(block)
        matches = list(FORECASTER_PATTERN.finditer(bullets))
        bodies = _bodies_between(bullets, [(m.start(), m.end()) for m in matches])
        forecasters += [
            {
                "key": f"R{report.group(1)}:F{match.group(1)}",
                "model": match.group(2),
                "prediction": body,
            }
            for match, body in zip(matches, bodies)
        ]
    return forecasters


def forecaster_rationales(explanation: str) -> dict[str, str]:
    """Full reasoning text for each forecaster."""
    block = split_sections(explanation).get("forecasts", "")
    matches = list(RATIONALE_PATTERN.finditer(block))
    bodies = _bodies_between(block, [(m.start(), m.end()) for m in matches])
    return {
        f"R{match.group(1)}:F{match.group(2)}": body
        for match, body in zip(matches, bodies)
    }


def final_prediction(summary: str) -> str:
    match = FINAL_PREDICTION_PATTERN.search(summary)
    return match.group(1).strip() if match else ""


def was_truncated(explanation: str) -> bool:
    """Whether the explanation hit the comment size cap and lost its tail."""
    return "has been truncated" in explanation
