"""Data models for TUI question state tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from textual.message import Message

from forecasting_tools.data_models.questions import MetaculusQuestion, ResolutionType


# Sentinel post_id used by the "Home" sidebar entry.
HOME_SENTINEL_ID: int = -1


class QuestionSelected(Message):
    """Posted when the user selects a question in the sidebar list or home table."""

    def __init__(self, post_id: int) -> None:
        super().__init__()
        self.post_id = post_id


QuestionStatus = Literal["pending", "running", "completed", "error"]


@dataclass
class QuestionItem:
    """Tracks the state of a single question through the resolution pipeline.

    Attributes:
        question: The Metaculus question being resolved.
        status: Current resolution status.
        resolution: The typed resolution result, or None if not yet resolved.
        resolution_status_str: Raw status string from the resolver
            (e.g. "TRUE", "FALSE", "NOT_YET_RESOLVABLE").
        resolution_metadata: Reasoning and key evidence from the resolver.
        feed_lines: Accumulated agent feed messages for the live log.
        error_message: Error details if status is "error".
    """

    question: MetaculusQuestion
    status: QuestionStatus = "pending"
    resolution: Optional[ResolutionType] = None
    resolution_status_str: Optional[str] = None
    resolution_metadata: Optional[dict] = None
    feed_lines: list[str] = field(default_factory=list)
    error_message: Optional[str] = None
    cost: float = 0.0

    @property
    def post_id(self) -> int:
        return self.question.id_of_post  # type: ignore[return-value]

    @property
    def title(self) -> str:
        text = self.question.question_text or "Untitled"
        return text[:80] + ("..." if len(text) > 80 else "")

    @property
    def status_icon(self) -> str:
        return {
            "pending": " -- ",
            "running": " >> ",
            "completed": " OK ",
            "error": " !! ",
        }[self.status]

    @property
    def resolution_display(self) -> str:
        if self.status == "error":
            return f"Error: {self.error_message or 'unknown'}"
        if self.resolution_status_str is not None:
            return self.resolution_status_str
        if self.status == "running":
            return "Resolving..."
        return "Pending"
