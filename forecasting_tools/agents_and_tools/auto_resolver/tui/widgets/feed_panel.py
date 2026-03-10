"""Main content panel showing resolution status and live agent feed."""

from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import RichLog, Static

from forecasting_tools.agents_and_tools.auto_resolver.tui.models import QuestionItem


# ---------------------------------------------------------------------------
# Helpers for normalising and colour-coding resolution values
# ---------------------------------------------------------------------------

def _resolution_color(value: str) -> str:
    """Return a Rich colour name for a resolution string."""
    v = value.strip().upper()
    if v == "TRUE":
        return "green"
    elif v == "FALSE":
        return "red"
    elif v in ("NOT_YET_RESOLVABLE", "NOT YET RESOLVABLE", "RESOLVING...", "PENDING"):
        return "yellow"
    else:
        return "cyan"  # AMBIGUOUS, ANNULLED, errors, etc.


def _normalize_ground_truth(raw: str | None) -> str:
    """Convert Metaculus ground-truth strings to True/False style."""
    if raw is None:
        return "Not Yet Resolvable"
    lowered = raw.strip().lower()
    if lowered == "yes":
        return "True"
    elif lowered == "no":
        return "False"
    elif lowered == "ambiguous":
        return "Ambiguous"
    elif lowered == "annulled":
        return "Annulled"
    else:
        return raw


class ResolutionHeader(Static):
    """Displays the resolution status and metadata for the selected question."""

    DEFAULT_CSS = """
    ResolutionHeader {
        dock: top;
        height: auto;
        max-height: 14;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $primary;
        color: $text;
    }
    """

    def render_item(self, item: Optional[QuestionItem]) -> None:
        if item is None:
            self.update("No question selected. Press [bold]a[/bold] to add questions.")
            return

        parts: list[str] = []
        parts.append(f"[bold]{item.question.question_text or 'Untitled'}[/bold]")

        if item.question.page_url:
            parts.append(f"[dim]{item.question.page_url}[/dim]")

        # Predicted resolution line (colour-coded)
        parts.append("")
        resolution_str = item.resolution_display
        res_color = _resolution_color(resolution_str)
        parts.append(
            f"Predicted: [{res_color}]{resolution_str}[/{res_color}]"
        )

        # Ground truth (normalised and colour-coded)
        gt_normalized = _normalize_ground_truth(item.question.resolution_string)
        gt_color = _resolution_color(gt_normalized)
        parts.append(
            f"Ground Truth: [{gt_color}]{gt_normalized}[/{gt_color}]"
        )

        # Cost
        if item.cost > 0:
            parts.append(f"Cost: [bold]${item.cost:.4f}[/bold]")

        # Metadata (reasoning + evidence) once available
        if item.resolution_metadata:
            reasoning = item.resolution_metadata.get("reasoning", "")
            if reasoning:
                parts.append("")
                parts.append(f"[italic]Reasoning:[/italic] {reasoning}")
            evidence = item.resolution_metadata.get("key_evidence", [])
            if evidence:
                parts.append("[italic]Evidence:[/italic]")
                for ev in evidence:
                    parts.append(f"  - {ev}")

        self.update("\n".join(parts))


class AgentFeedLog(RichLog):
    """Scrollable live log of agent events."""

    DEFAULT_CSS = """
    AgentFeedLog {
        height: 1fr;
        border-top: solid $accent;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(highlight=True, markup=True, wrap=True, **kwargs)
        self._current_text_block: str = ""

    def append_event(self, event_type: str, text: str) -> None:
        """Append a single event to the log with type-appropriate styling."""
        if event_type == "text":
            # Accumulate text deltas into a running block rather than one-per-token
            self._current_text_block += text
            return
        else:
            # Flush any accumulated text block first
            self._flush_text_block()

        if event_type == "status":
            self.write(f"[bold blue]>>> {text}[/bold blue]")
        elif event_type == "tool":
            self.write(f"[yellow]{text}[/yellow]")
        elif event_type == "result":
            self.write(f"[bold green]--- RESULT ---[/bold green]")
            self.write(f"[green]{text}[/green]")
        elif event_type == "error":
            self.write(f"[bold red]ERROR: {text}[/bold red]")
        else:
            self.write(text)

    def flush_text(self) -> None:
        """Flush any buffered text delta block to the log."""
        self._flush_text_block()

    def _flush_text_block(self) -> None:
        if self._current_text_block:
            self.write(self._current_text_block)
            self._current_text_block = ""

    def clear_feed(self) -> None:
        """Clear the log and reset state."""
        self._current_text_block = ""
        self.clear()


class FeedPanel(Vertical):
    """Composite widget: resolution header + scrollable agent feed log."""

    DEFAULT_CSS = """
    FeedPanel {
        width: 1fr;
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield ResolutionHeader(id="resolution-header")
        yield AgentFeedLog(id="agent-feed-log")

    @property
    def header(self) -> ResolutionHeader:
        return self.query_one("#resolution-header", ResolutionHeader)

    @property
    def feed_log(self) -> AgentFeedLog:
        return self.query_one("#agent-feed-log", AgentFeedLog)

    def show_question(self, item: Optional[QuestionItem]) -> None:
        """Switch the panel to display a specific question's data."""
        self.header.render_item(item)
        self.feed_log.clear_feed()
        if item is not None:
            # Replay existing feed lines
            for line in item.feed_lines:
                self.feed_log.write(line)
