"""Async worker that runs the streaming AgenticResolver and posts events to the TUI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from textual.message import Message
from textual.widget import Widget

from forecasting_tools.agents_and_tools.auto_resolver.agentic import AgenticResolver
from forecasting_tools.agents_and_tools.auto_resolver.tui.models import QuestionItem
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

if TYPE_CHECKING:
    from forecasting_tools.data_models.questions import MetaculusQuestion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Textual Messages posted by the worker so the App can react
# ---------------------------------------------------------------------------

class FeedEvent(Message):
    """A single line of agent feed output for a question."""

    def __init__(self, post_id: int, event_type: str, text: str) -> None:
        super().__init__()
        self.post_id = post_id
        self.event_type = event_type
        self.text = text


class ResolutionComplete(Message):
    """Emitted when a question's resolution finishes (success or error)."""

    def __init__(self, post_id: int) -> None:
        super().__init__()
        self.post_id = post_id


# ---------------------------------------------------------------------------
# Worker coroutine
# ---------------------------------------------------------------------------

async def run_resolution(
    item: QuestionItem,
    resolver: AgenticResolver,
    poster: Widget,
) -> None:
    """Run streaming resolution on a single question, posting messages to the TUI.

    This is intended to be called via Textual's Worker system
    (e.g. ``self.run_worker(run_resolution(...))``) so it executes
    on a background asyncio task while the UI remains responsive.

    Args:
        item: The QuestionItem to resolve -- mutated in place.
        resolver: The AgenticResolver instance (shared, but each call is independent).
        poster: A Textual Widget whose ``.post_message()`` method will be used
            to deliver events back to the app's message loop.
    """
    item.status = "running"
    item.feed_lines.clear()
    item.cost = 0.0
    post_id = item.post_id

    cost_manager = MonetaryCostManager(hard_limit=0)

    try:
        with cost_manager:
            async for event_type, text in resolver.resolve_question_streamed(item.question):
                # Accumulate non-delta lines (deltas are single tokens, too noisy individually)
                if event_type == "text":
                    # For text deltas we still post them but accumulate separately
                    poster.post_message(FeedEvent(post_id, event_type, text))
                elif event_type == "error":
                    item.feed_lines.append(f"[ERROR] {text}")
                    item.status = "error"
                    item.error_message = text
                    item.cost = cost_manager.current_usage
                    poster.post_message(FeedEvent(post_id, event_type, text))
                    poster.post_message(ResolutionComplete(post_id))
                    return
                elif event_type == "result":
                    item.feed_lines.append(text)
                    # Extract resolution info directly from the result text
                    # to avoid race conditions with shared resolver state.
                    lines = text.splitlines()
                    for line in lines:
                        if line.startswith("Resolution: "):
                            item.resolution_status_str = line.removeprefix("Resolution: ").strip()
                        elif line.startswith("Reasoning: "):
                            reasoning = line.removeprefix("Reasoning: ").strip()
                            item.resolution_metadata = item.resolution_metadata or {}
                            item.resolution_metadata["reasoning"] = reasoning
                    # Gather key evidence lines
                    evidence: list[str] = []
                    in_evidence = False
                    for line in lines:
                        if line.strip() == "Key Evidence:":
                            in_evidence = True
                            continue
                        if in_evidence and line.strip().startswith("- "):
                            evidence.append(line.strip().removeprefix("- "))
                    if evidence:
                        item.resolution_metadata = item.resolution_metadata or {}
                        item.resolution_metadata["key_evidence"] = evidence

                    item.status = "completed"
                    poster.post_message(FeedEvent(post_id, event_type, text))
                else:
                    # "status" or "tool" events
                    item.feed_lines.append(text)
                    poster.post_message(FeedEvent(post_id, event_type, text))

    except Exception as exc:
        logger.exception("Resolution failed for question %s", post_id)
        item.status = "error"
        item.error_message = str(exc)
        item.feed_lines.append(f"[EXCEPTION] {exc}")
        poster.post_message(FeedEvent(post_id, "error", str(exc)))

    item.cost = cost_manager.current_usage
    poster.post_message(ResolutionComplete(post_id))
