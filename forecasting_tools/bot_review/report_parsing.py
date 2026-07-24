"""Pull structure back out of a report explanation.

``ForecastBot._create_comment`` assembles every report into the same shape: a
SUMMARY listing each forecaster's prediction, the RESEARCH behind them, and one
FORECASTS subsection per forecaster. That string is what gets saved to disk and
what gets posted to Metaculus as a comment, so the same functions read either.

Sections come from ``MarkdownTree``, the same splitter ``ForecastReport`` uses.
The summary bullets are not headings, so they are read directly.

Forecasters are keyed ``R<research report>:F<forecaster>``, since a bot running
more than one research report per question has a forecaster 1 in each.
"""

from __future__ import annotations

import re

from forecasting_tools.data_models.markdown_tree import MarkdownTree

SECTIONS = ("summary", "research", "forecasts")
REPORT_PATTERN = re.compile(r"^## Report (\d+) Summary\s*$", re.MULTILINE)
FORECASTER_PATTERN = re.compile(
    r"^\*Forecaster (\d+)(?: \(([^)]*)\))?\*:", re.MULTILINE
)
RATIONALE_TITLE_PATTERN = re.compile(r"^R(\d+): Forecaster (\d+) Reasoning$")
FINAL_PREDICTION_PATTERN = re.compile(
    r"^\*Final Prediction\*:(.*?)(?=^\*[A-Z]|^## |\Z)", re.MULTILINE | re.DOTALL
)


def split_sections(explanation: str) -> dict[str, str]:
    """The SUMMARY / RESEARCH / FORECASTS blocks, by lowercased name."""
    return {
        section.title.strip().lower(): section.text_of_section_and_subsections.strip()
        for section in MarkdownTree.turn_markdown_into_report_sections(explanation)
        if section.title and section.title.strip().lower() in SECTIONS
    }


def forecaster_rationales(explanation: str) -> dict[str, str]:
    """Full reasoning text for each forecaster."""
    rationales = {}
    for section in MarkdownTree.turn_markdown_into_report_sections(explanation):
        if not section.title or section.title.strip().lower() != "forecasts":
            continue
        for subsection in section.sub_sections:
            match = RATIONALE_TITLE_PATTERN.match((subsection.title or "").strip())
            if match:
                key = f"R{match.group(1)}:F{match.group(2)}"
                rationales[key] = subsection.text_of_section_and_subsections.strip()
    return rationales


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

    ``model`` is None unless the bot annotates its own bullets with model names,
    which the framework does not do.
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


def final_prediction(summary: str) -> str:
    match = FINAL_PREDICTION_PATTERN.search(summary)
    return match.group(1).strip() if match else ""


def was_truncated(explanation: str) -> bool:
    """Whether the explanation hit the comment size cap and lost its tail."""
    return "has been truncated" in explanation
