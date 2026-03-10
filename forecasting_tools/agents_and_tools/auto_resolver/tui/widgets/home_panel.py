"""Home panel showing a live-updating confusion matrix and question table."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from forecasting_tools.agents_and_tools.auto_resolver.tui.models import (
    HOME_SENTINEL_ID,
    QuestionItem,
    QuestionSelected,
)
from forecasting_tools.agents_and_tools.auto_resolver.tui.report import (
    build_report_from_items,
)
from forecasting_tools.agents_and_tools.auto_resolver.tui.widgets.feed_panel import (
    _normalize_ground_truth,
)


def _normalize_for_compare(s: str) -> str:
    """Normalize a resolution string for comparison (uppercase, underscores for spaces)."""
    return s.strip().upper().replace(" ", "_")


def _match_str(predicted: str | None, ground_truth: str | None) -> str:
    """Return a match indicator comparing predicted and ground truth."""
    if predicted is None or ground_truth is None:
        return "-"
    p = _normalize_for_compare(predicted)
    g = _normalize_for_compare(ground_truth)
    if p in ("PENDING", "RESOLVING..."):
        return "-"
    if g in ("", "NONE", "-"):
        return "-"
    return "Y" if p == g else "N"


class HomePanel(Vertical):
    """Overview panel with confusion matrix, question table, and summary."""

    DEFAULT_CSS = """
    HomePanel {
        width: 1fr;
        height: 1fr;
    }
    HomePanel > #home-title {
        dock: top;
        text-align: center;
        text-style: bold;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $primary;
    }
    HomePanel > #home-matrix {
        dock: top;
        height: auto;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $accent;
    }
    HomePanel > DataTable {
        height: 1fr;
    }
    HomePanel > #home-summary {
        dock: bottom;
        height: auto;
        padding: 1 2;
        background: $surface;
        border-top: solid $primary;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Auto Resolver - Overview", id="home-title")
        yield Static("", id="home-matrix")
        table: DataTable = DataTable(id="home-table")
        table.cursor_type = "row"
        table.zebra_stripes = True
        yield table
        yield Static("No questions loaded.", id="home-summary")

    def on_mount(self) -> None:
        table = self.query_one("#home-table", DataTable)
        table.add_columns(
            "ID", "Question", "Status", "Predicted", "Ground Truth", "Match", "Cost"
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Navigate to the selected question's feed view."""
        try:
            post_id = int(str(event.row_key.value))
        except (ValueError, TypeError):
            return
        self.post_message(QuestionSelected(post_id))

    def refresh_table(self, items: dict[int, QuestionItem]) -> None:
        """Rebuild the confusion matrix, table rows, and summary."""
        table = self.query_one("#home-table", DataTable)
        table.clear()

        total = 0
        completed = 0
        correct = 0
        total_cost = 0.0

        for post_id, item in items.items():
            total += 1
            total_cost += item.cost

            predicted = item.resolution_display
            gt_raw = item.question.resolution_string
            gt = _normalize_ground_truth(gt_raw) if gt_raw else "Not Yet Resolvable"
            match = _match_str(predicted, gt)

            if item.status == "completed":
                completed += 1
                if match == "Y":
                    correct += 1

            cost_str = f"${item.cost:.4f}" if item.cost > 0 else "-"

            table.add_row(
                str(post_id),
                item.title,
                item.status.title(),
                predicted,
                gt,
                match,
                cost_str,
                key=str(post_id),
            )

        # --- Confusion matrix ---
        report = build_report_from_items(items)
        matrix_text = report.binary_results_table()

        matrix_total = (
            report.n_pp + report.n_pn + report.n_pc
            + report.n_np + report.n_nn + report.n_nc
            + report.n_cp + report.n_cn + report.n_cc
            + report.n_xp + report.n_xn + report.n_xc
            + report.n_nyr_p + report.n_nyr_n + report.n_nyr_c
            + report.n_nyr_nyr
        )
        matrix_correct = report.n_pp + report.n_nn + report.n_cc + report.n_nyr_nyr
        matrix_accuracy = (matrix_correct / matrix_total * 100) if matrix_total > 0 else 0

        matrix_display = (
            f"{matrix_text}\n\n"
            f"Correct: {matrix_correct}/{matrix_total}  "
            f"Accuracy: {matrix_accuracy:.1f}%"
        )
        self.query_one("#home-matrix", Static).update(matrix_display)

        # --- Summary footer ---
        accuracy = (correct / completed * 100) if completed > 0 else 0
        summary_parts = [
            f"Total: {total}",
            f"Completed: {completed}",
            f"Correct: {correct}",
            f"Accuracy: {accuracy:.1f}%",
            f"Total Cost: ${total_cost:.4f}",
        ]
        self.query_one("#home-summary", Static).update("  |  ".join(summary_parts))
