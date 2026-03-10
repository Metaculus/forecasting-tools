"""Modal dialog for adding tournament or question IDs."""

from __future__ import annotations

from typing import Literal

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Static,
)

from forecasting_tools.data_models.questions import QuestionBasicType


QUESTION_TYPES: list[QuestionBasicType] = [
    "binary",
    "numeric",
    "multiple_choice",
    "date",
    "discrete",
    "conditional",
]


class AddIdRequested(Message):
    """Posted when the user submits an ID from the modal."""

    def __init__(
        self,
        id_type: Literal["tournament", "question"],
        id_value: int | str,
        allowed_types: list[QuestionBasicType] | None = None,
        max_questions: int | None = None,
        exclude_unresolved: bool = False,
    ) -> None:
        super().__init__()
        self.id_type = id_type
        self.id_value = id_value
        self.allowed_types = allowed_types or ["binary"]
        self.max_questions = max_questions
        self.exclude_unresolved = exclude_unresolved


class InputModal(ModalScreen[None]):
    """A modal screen for entering a tournament or question ID."""

    DEFAULT_CSS = """
    InputModal {
        align: center middle;
    }
    InputModal > #modal-container {
        width: 60;
        height: auto;
        max-height: 40;
        overflow-y: auto;
        background: $surface;
        border: thick $primary;
        padding: 2 3;
    }
    InputModal > #modal-container > #modal-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    InputModal > #modal-container > RadioSet {
        height: auto;
        margin-bottom: 1;
    }
    InputModal > #modal-container > Input {
        margin-bottom: 1;
    }
    InputModal > #modal-container > #tournament-options {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        background: $boost;
    }
    InputModal > #modal-container > #tournament-options > .option-label {
        text-style: bold;
        margin-bottom: 1;
    }
    InputModal > #modal-container > #tournament-options > #type-checkboxes {
        height: auto;
        margin-bottom: 1;
    }
    InputModal > #modal-container > #tournament-options > #type-checkboxes Checkbox {
        height: auto;
        padding: 0;
        margin: 0;
    }
    InputModal > #modal-container > #tournament-options > #filters-section {
        height: auto;
        margin-bottom: 1;
    }
    InputModal > #modal-container > #tournament-options > #filters-section Checkbox {
        height: auto;
        padding: 0;
        margin: 0;
    }
    InputModal > #modal-container > #modal-buttons {
        height: auto;
        align: center middle;
    }
    InputModal > #modal-container > #modal-buttons > Button {
        margin: 0 2;
    }
    InputModal > #modal-container > #modal-error {
        color: $error;
        height: auto;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    _is_tournament: reactive[bool] = reactive(False)

    def __init__(self, default_type: Literal["tournament", "question"] = "question") -> None:
        super().__init__()
        self._default_type = default_type

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label("Add Questions", id="modal-title")
            yield RadioSet(
                RadioButton("Tournament ID", value=self._default_type == "tournament"),
                RadioButton("Question ID", value=self._default_type == "question"),
                id="id-type-radio",
            )
            yield Input(
                placeholder="Enter ID (numeric or slug for tournaments)...",
                id="id-input",
            )
            with Vertical(id="tournament-options"):
                yield Static("Question Types:", classes="option-label")
                with Vertical(id="type-checkboxes"):
                    for qtype in QUESTION_TYPES:
                        yield Checkbox(
                            qtype.replace("_", " ").title(),
                            value=(qtype == "binary"),
                            id=f"cb-{qtype}",
                        )
                with Vertical(id="filters-section"):
                    yield Static("Filters:", classes="option-label")
                    yield Checkbox(
                        "Exclude unresolved questions",
                        value=False,
                        id="cb-exclude-unresolved",
                    )
                yield Static("Random Sample (optional):", classes="option-label")
                yield Input(
                    placeholder="Max number of questions to load (leave empty for all)",
                    id="max-questions-input",
                )
            yield Label("", id="modal-error")
            with Horizontal(id="modal-buttons"):
                yield Button("Submit", id="btn-submit", variant="primary")
                yield Button("Cancel", id="btn-cancel", variant="default")

    def on_mount(self) -> None:
        self.query_one("#id-input", Input).focus()
        radio_set = self.query_one("#id-type-radio", RadioSet)
        self._is_tournament = radio_set.pressed_index == 0
        self._update_tournament_options_visibility()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "id-type-radio":
            self._is_tournament = event.radio_set.pressed_index == 0
            self._update_tournament_options_visibility()

    def _update_tournament_options_visibility(self) -> None:
        try:
            tournament_options = self.query_one("#tournament-options", Vertical)
            tournament_options.display = self._is_tournament
        except Exception:
            pass

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-submit":
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        error_label = self.query_one("#modal-error", Label)
        raw_value = self.query_one("#id-input", Input).value.strip()

        if not raw_value:
            error_label.update("Please enter an ID.")
            return

        radio_set = self.query_one("#id-type-radio", RadioSet)
        id_type: Literal["tournament", "question"] = (
            "tournament" if radio_set.pressed_index == 0 else "question"
        )

        id_value: int | str
        try:
            id_value = int(raw_value)
        except ValueError:
            if id_type == "question":
                error_label.update("Question ID must be a numeric integer.")
                return
            id_value = raw_value

        allowed_types: list[QuestionBasicType] = []
        max_questions: int | None = None
        exclude_unresolved = False

        if id_type == "tournament":
            for qtype in QUESTION_TYPES:
                try:
                    cb = self.query_one(f"#cb-{qtype}", Checkbox)
                    if cb.value:
                        allowed_types.append(qtype)
                except Exception:
                    pass

            if not allowed_types:
                error_label.update("Please select at least one question type.")
                return

            max_input = self.query_one("#max-questions-input", Input).value.strip()
            if max_input:
                try:
                    max_questions = int(max_input)
                    if max_questions <= 0:
                        error_label.update("Max questions must be a positive number.")
                        return
                except ValueError:
                    error_label.update("Max questions must be a number.")
                    return

            try:
                exclude_unresolved = self.query_one("#cb-exclude-unresolved", Checkbox).value
            except Exception:
                pass

        self.app.post_message(
            AddIdRequested(
                id_type=id_type,
                id_value=id_value,
                allowed_types=allowed_types if id_type == "tournament" else None,
                max_questions=max_questions,
                exclude_unresolved=exclude_unresolved,
            )
        )
        self.dismiss(None)
