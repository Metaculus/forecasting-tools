"""Attach the bot's own comments to an outcome table as run traces.

The comment a bot posts is its report, so it is the trace of that run. A
question can have several, one per run; the one that earned the score is the
run standing when the question was spot scored.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from forecasting_tools.bot_review.outcomes import OutcomeTable, RunTrace
from forecasting_tools.bot_review.report_parsing import (
    forecaster_rationales,
    parse_forecasters,
    split_sections,
    was_truncated,
)
from forecasting_tools.data_models.comment import Comment
from forecasting_tools.helpers.metaculus_client import MetaculusClient

logger = logging.getLogger(__name__)

COST_PATTERN = re.compile(r"^\*Total Cost\*:\s*\$([\d.]+)", re.MULTILINE)
MINUTES_PATTERN = re.compile(r"^\*Time Spent\*:\s*([\d.]+) minutes", re.MULTILINE)
QUESTION_PATTERN = re.compile(r"^\*Question\*:(.*)$", re.MULTILINE)


def reduce_comment(comment: Comment) -> RunTrace:
    """Reduce a posted report to the fields a review needs."""
    summary = split_sections(comment.text).get("summary", "")
    question = QUESTION_PATTERN.search(comment.text)
    cost = COST_PATTERN.search(comment.text)
    minutes = MINUTES_PATTERN.search(comment.text)
    return RunTrace(
        comment_id=comment.id,
        post_id=comment.on_post,
        run_time=comment.created_at,
        question_text=question.group(1).strip() if question else None,
        forecasters=parse_forecasters(summary),
        cost=float(cost.group(1)) if cost else None,
        minutes=float(minutes.group(1)) if minutes else None,
        truncated=was_truncated(comment.text),
    )


def attach_traces(table: OutcomeTable, client: MetaculusClient | None = None) -> None:
    """
    Fill in ``traces`` on every question in the table, one request per post.

    :param table: the table to attach to, modified in place
    :param client: client to fetch with, created from the environment if not given
    """
    client = client or MetaculusClient()
    rows_by_post = defaultdict(list)
    for outcome in table.questions:
        rows_by_post[outcome.post_id].append(outcome)
    logger.info(f"Fetching comments for {len(rows_by_post)} posts")
    for post_id, rows in rows_by_post.items():
        runs = [reduce_comment(c) for c in client.get_own_comments(post_id=post_id)]
        for outcome in rows:
            outcome.traces = (
                runs
                if len(rows) == 1
                else [run for run in runs if run.question_text == outcome.title]
            )


def get_trace(
    post_id: int,
    section: str = "research",
    forecaster: str | None = None,
    client: MetaculusClient | None = None,
) -> str:
    """
    One part of the bot's latest report on a post.

    :param post_id: post id, as it appears in question urls
    :param section: summary, research or forecasts
    :param forecaster: a key like ``R1:F2``, to read that rationale instead
    :param client: client to fetch with, created from the environment if not given
    """
    client = client or MetaculusClient()
    comments = client.get_own_comments(post_id=post_id)
    if not comments:
        return ""
    text = max(comments, key=lambda comment: comment.created_at).text
    if forecaster:
        return forecaster_rationales(text).get(forecaster, "")
    return split_sections(text).get(section, "")
