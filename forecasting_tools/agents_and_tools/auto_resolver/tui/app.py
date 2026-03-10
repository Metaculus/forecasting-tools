"""Main Textual application for the Auto Resolver TUI."""

from __future__ import annotations

import asyncio
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, cast

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, Footer, Header

from forecasting_tools import MetaculusClient, ApiFilter
from forecasting_tools.data_models.questions import MetaculusQuestion, QuestionBasicType
from forecasting_tools.helpers.metaculus_client import QuestionStateAsString
from forecasting_tools.agents_and_tools.auto_resolver.agentic import AgenticResolver
from forecasting_tools.agents_and_tools.auto_resolver.tui.models import (
    HOME_SENTINEL_ID,
    QuestionItem,
    QuestionSelected,
)
from forecasting_tools.agents_and_tools.auto_resolver.tui.resolver_worker import (
    FeedEvent,
    ResolutionComplete,
    run_resolution,
)
from forecasting_tools.agents_and_tools.auto_resolver.tui.widgets.sidebar import (
    Sidebar,
)
from forecasting_tools.agents_and_tools.auto_resolver.tui.widgets.feed_panel import (
    FeedPanel,
)
from forecasting_tools.agents_and_tools.auto_resolver.tui.widgets.home_panel import (
    HomePanel,
)
from forecasting_tools.agents_and_tools.auto_resolver.tui.report import (
    generate_markdown_report,
)
from forecasting_tools.agents_and_tools.auto_resolver.tui.widgets.input_modal import (
    AddIdRequested,
    InputModal,
)
class AutoResolverApp(App):
    """TUI application for interactive agentic question resolution.

    Layout:
        [Sidebar (question list)]  |  [Home Panel / Feed Panel]

    Keybindings:
        a  -- Add a question by post ID
        t  -- Add questions from a tournament
        r  -- Re-run resolution on the selected question
        e  -- Export report to markdown
        q  -- Quit
    """

    TITLE = "Auto Resolver TUI"
    CSS = """
    Screen {
        layout: horizontal;
    }
    #main-area {
        width: 1fr;
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("a", "add_question", "Add Question", show=True),
        Binding("t", "add_tournament", "Add Tournament", show=True),
        Binding("r", "rerun", "Re-run Selected", show=True),
        Binding("e", "export_report", "Export Report", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        max_concurrency: int = 3,
        initial_tournaments: list[int | str] | None = None,
        initial_questions: list[int] | None = None,
      ) -> None:
        super().__init__()
        self._resolver = AgenticResolver()
        self._client = MetaculusClient()
        self._items: dict[int, QuestionItem] = {}
        self._selected_post_id: Optional[int] = HOME_SENTINEL_ID
        self._concurrency_sem = asyncio.Semaphore(max_concurrency)
        self._initial_tournaments = initial_tournaments or []
        self._initial_questions = initial_questions or []
        self._original_stderr = sys.stderr

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-area"):
            yield Sidebar(id="sidebar")
            yield HomePanel(id="home-panel")
            yield FeedPanel(id="feed-panel")
        yield Footer()

    @property
    def sidebar(self) -> Sidebar:
        return self.query_one("#sidebar", Sidebar)

    @property
    def feed_panel(self) -> FeedPanel:
        return self.query_one("#feed-panel", FeedPanel)

    @property
    def home_panel(self) -> HomePanel:
        return self.query_one("#home-panel", HomePanel)


    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_mount(self) -> None:
        """Load any questions / tournaments passed via CLI."""
        # Start with home panel visible, feed/log panels hidden
        self.feed_panel.display = False
        self.home_panel.display = True

        # Redirect sys.stderr so that non-logging stderr writes (e.g.
        # from C extensions, Python warnings, litellm debug output) are
        # also captured instead of corrupting the TUI display.
        self._original_stderr = sys.stderr
    
        for tid in self._initial_tournaments:
            self._schedule_load_tournament(tid)
        for qid in self._initial_questions:
            self._schedule_load_question(qid)

    # ------------------------------------------------------------------
    # View switching
    # ------------------------------------------------------------------

    def _show_home(self) -> None:
        """Switch to the home overview panel."""
        self.home_panel.display = True
        self.feed_panel.display = False
        self.home_panel.refresh_table(self._items)

    def _show_feed(self, item: QuestionItem | None) -> None:
        """Switch to the feed panel for a specific question."""
        self.home_panel.display = False
        self.feed_panel.display = True
        self.feed_panel.show_question(item)

    # ------------------------------------------------------------------
    # Actions (keybindings)
    # ------------------------------------------------------------------

    def action_add_question(self) -> None:
        self.push_screen(InputModal(default_type="question"))

    def action_add_tournament(self) -> None:
        self.push_screen(InputModal(default_type="tournament"))

    def action_rerun(self) -> None:
        pid = self.sidebar.get_selected_post_id()
        if pid and pid in self._items:
            item = self._items[pid]
            if item.status not in ("running",):
                self._start_resolution(item)

    def action_export_report(self) -> None:
        """Export the current results to a markdown file."""
        if not self._items:
            self.notify("No questions to export.", severity="warning")
            return

        report = self._generate_markdown_report()

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = reports_dir / f"tui_report_{timestamp}.md"

        filepath.write_text(report, encoding="utf-8")
        self.notify(f"Report exported to {filepath}")


    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle sidebar button clicks."""
        if event.button.id == "btn-add-tournament":
            self.action_add_tournament()
        elif event.button.id == "btn-add-question":
            self.action_add_question()

    def on_add_id_requested(self, message: AddIdRequested) -> None:
        if message.id_type == "tournament":
            self._schedule_load_tournament(
                message.id_value,
                allowed_types=message.allowed_types,
                max_questions=message.max_questions,
                exclude_unresolved=message.exclude_unresolved,
            )
        else:
            self._schedule_load_question(int(message.id_value))

    def on_question_selected(self, message: QuestionSelected) -> None:
        self._selected_post_id = message.post_id

        if message.post_id == HOME_SENTINEL_ID:
            self._show_home()
        else:
            item = self._items.get(message.post_id)
            self._show_feed(item)

    def on_feed_event(self, message: FeedEvent) -> None:
        """Route a live feed event to the feed panel if it's for the selected question."""
        if message.post_id == self._selected_post_id:
            self.feed_panel.feed_log.append_event(message.event_type, message.text)
        # Also update the sidebar entry
        item = self._items.get(message.post_id)
        if item:
            self._refresh_sidebar_entry(item)

    def on_resolution_complete(self, message: ResolutionComplete) -> None:
        """Handle completion: refresh sidebar + header if selected, refresh home table."""
        item = self._items.get(message.post_id)
        if item:
            self._refresh_sidebar_entry(item)
            # Flush any remaining text in the feed log
            if message.post_id == self._selected_post_id:
                self.feed_panel.feed_log.flush_text()
                self.feed_panel.header.render_item(item)

        # Always refresh home table so it stays up to date
        if self.home_panel.display:
            self.home_panel.refresh_table(self._items)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _schedule_load_question(self, post_id: int) -> None:
        """Kick off an async worker to load a single question from Metaculus."""
        self.run_worker(self._load_question(post_id), exclusive=False)

    def _schedule_load_tournament(
        self,
        tournament_id: int | str,
        allowed_types: list[QuestionBasicType] | None = None,
        max_questions: int | None = None,
        exclude_unresolved: bool = False,
    ) -> None:
        """Kick off an async worker to load all questions from a tournament."""
        self.run_worker(
            self._load_tournament(
                tournament_id,
                allowed_types=allowed_types,
                max_questions=max_questions,
                exclude_unresolved=exclude_unresolved,
            ),
            exclusive=False,
        )

    async def _load_question(self, post_id: int) -> None:
        """Fetch a single question by post ID and add it to the list."""
        if post_id in self._items:
            self.notify(f"Question {post_id} already loaded.", severity="warning")
            return

        self.notify(f"Loading question {post_id}...")
        try:
            question = await asyncio.to_thread(
                self._client.get_question_by_post_id, post_id
            )
        except Exception as e:
            self.notify(f"Failed to load question {post_id}: {e}", severity="error")
            return

        if isinstance(question, list):
            self.notify(
                f"Question {post_id} is a group question (not supported). Skipping.",
                severity="warning",
            )
            return

        self._add_question(question)

    async def _load_tournament(
        self,
        tournament_id: int | str,
        allowed_types: list[QuestionBasicType] | None = None,
        max_questions: int | None = None,
        exclude_unresolved: bool = False,
    ) -> None:
        """Fetch questions from a tournament and add them.

        Pages are fetched one at a time on a background thread so the
        event loop stays free and the UI remains responsive.  When no
        random sampling is needed (``max_questions is None``), questions
        are added to the sidebar as each page arrives.  When random
        sampling *is* requested, all pages are collected first, then a
        random subset is sampled and added.
        """
        self.notify(f"Loading tournament {tournament_id}...")
        try:
            allowed_statuses: list[QuestionStateAsString] | None = None
            if exclude_unresolved:
                allowed_statuses = cast(
                    list[QuestionStateAsString], ["resolved"]
                )

            api_filter = ApiFilter(
                allowed_tournaments=[tournament_id],
                allowed_types=allowed_types or ["binary"],
                allowed_statuses=allowed_statuses,
                group_question_mode="exclude",
                order_by="-published_time",
            )

            # Fetch page-by-page on a thread so we don't block the
            # event loop (the underlying client uses time.sleep +
            # requests.get which are synchronous).
            page_size = self._client.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
            all_questions: list[MetaculusQuestion] = []
            added = 0
            page_num = 0
            more_available = True
            needs_sampling = max_questions is not None

            while more_available:
                offset = page_num * page_size
                new_questions, continue_searching = await asyncio.to_thread(
                    self._client._grab_filtered_questions_with_offset,
                    api_filter,
                    offset,
                )

                if needs_sampling:
                    # Collect for later random sampling
                    all_questions.extend(new_questions)
                else:
                    # Add to sidebar immediately — no sampling needed
                    for q in new_questions:
                        if q.id_of_post and q.id_of_post not in self._items:
                            self._add_question(q)
                            added += 1

                if not continue_searching:
                    more_available = False
                page_num += 1

            # Apply random sampling if max_questions was requested
            if needs_sampling:
                if max_questions is not None and len(all_questions) > max_questions:
                    all_questions = random.sample(all_questions, max_questions)
                for q in all_questions:
                    if q.id_of_post and q.id_of_post not in self._items:
                        self._add_question(q)
                        added += 1

        except Exception as e:
            self.notify(
                f"Failed to load tournament {tournament_id}: {e}", severity="error"
            )
            return

        self.notify(f"Loaded {added} questions from tournament {tournament_id}.")

        # Refresh home table after batch load
        if self.home_panel.display:
            self.home_panel.refresh_table(self._items)

    def _add_question(self, question: MetaculusQuestion) -> None:
        """Register a question and add it to the sidebar, then start resolution."""
        post_id = question.id_of_post
        if post_id is None:
            return
        item = QuestionItem(question=question)
        self._items[post_id] = item

        display = f"{item.status_icon} [{post_id}] {item.title}"
        self.sidebar.add_question_entry(post_id, display)

        # Refresh home table if visible
        if self.home_panel.display:
            self.home_panel.refresh_table(self._items)

        # Start resolution immediately
        self._start_resolution(item)

    def _start_resolution(self, item: QuestionItem) -> None:
        """Launch a bounded async worker for resolution."""

        async def _bounded_resolve() -> None:
            async with self._concurrency_sem:
                await run_resolution(item, self._resolver, self.sidebar)

        self.run_worker(_bounded_resolve(), exclusive=False)

    def _refresh_sidebar_entry(self, item: QuestionItem) -> None:
        """Update the sidebar text for a question."""
        display = f"{item.status_icon} [{item.post_id}] {item.title}"
        self.sidebar.update_question_entry(item.post_id, display)

    # ------------------------------------------------------------------
    # Markdown report generation
    # ------------------------------------------------------------------

    def _generate_markdown_report(self) -> str:
        """Build a markdown report matching the assess.py format."""
        return generate_markdown_report(self._items)
