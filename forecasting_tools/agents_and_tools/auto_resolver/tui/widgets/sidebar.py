"""Sidebar widget: action buttons and navigable question list."""

from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, ListItem, ListView, Static

from forecasting_tools.agents_and_tools.auto_resolver.tui.models import (
    HOME_SENTINEL_ID,
    QuestionSelected,
)


class SidebarListItem(ListItem):
    """A single entry in the sidebar question list."""

    DEFAULT_CSS = """
    SidebarListItem {
        height: auto;
        padding: 0 1;
    }
    SidebarListItem:hover {
        background: $boost;
    }
    SidebarListItem.-highlight {
        background: $accent 30%;
    }
    """

    def __init__(self, post_id: int, display_text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.post_id = post_id
        self._label = Static(display_text)

    def compose(self) -> ComposeResult:
        yield self._label

    def update_text(self, text: str) -> None:
        self._label.update(text)


class QuestionListView(ListView):
    """Navigable list of questions with keyboard support."""

    DEFAULT_CSS = """
    QuestionListView {
        height: 1fr;
        border-top: solid $primary;
    }
    """

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, SidebarListItem):
            self.post_message(QuestionSelected(item.post_id))

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        item = event.item
        if isinstance(item, SidebarListItem):
            self.post_message(QuestionSelected(item.post_id))


class Sidebar(Vertical):
    """Left sidebar with action buttons and the question list."""

    DEFAULT_CSS = """
    Sidebar {
        width: 40;
        min-width: 30;
        max-width: 60;
        height: 1fr;
        border-right: solid $primary;
        background: $surface;
    }
    Sidebar > .sidebar-header {
        dock: top;
        height: auto;
        padding: 1 1;
        text-align: center;
        text-style: bold;
        color: $text;
        background: $primary 20%;
    }
    Sidebar > .sidebar-buttons {
        dock: top;
        height: auto;
        padding: 1 1;
    }
    Sidebar > .sidebar-buttons Button {
        width: 100%;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Auto Resolver", classes="sidebar-header")
        with Vertical(classes="sidebar-buttons"):
            yield Button("Add Tournament [t]", id="btn-add-tournament", variant="primary")
            yield Button("Add Question  [a]", id="btn-add-question", variant="default")
        yield QuestionListView(id="question-list")

    def on_mount(self) -> None:
        """Insert the Home entry as the first (and initially highlighted) item."""
        home_item = SidebarListItem(
            HOME_SENTINEL_ID, "[bold]Home[/bold]  Overview", id="q-home"
        )
        self.question_list.append(home_item)

    @property
    def question_list(self) -> QuestionListView:
        return self.query_one("#question-list", QuestionListView)

    def add_question_entry(self, post_id: int, display_text: str) -> None:
        """Append a question to the list."""
        list_item = SidebarListItem(post_id, display_text, id=f"q-{post_id}")
        self.question_list.append(list_item)

    def update_question_entry(self, post_id: int, display_text: str) -> None:
        """Update an existing question entry's display text."""
        try:
            item = self.question_list.query_one(f"#q-{post_id}", SidebarListItem)
            item.update_text(display_text)
        except Exception:
            pass  # Item may not exist yet

    def get_selected_post_id(self) -> Optional[int]:
        """Return the post_id of the currently highlighted item, or None."""
        highlighted = self.question_list.highlighted_child
        if isinstance(highlighted, SidebarListItem):
            return highlighted.post_id
        return None
