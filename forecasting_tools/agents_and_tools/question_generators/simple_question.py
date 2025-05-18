from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.util.jsonable import Jsonable


class SimpleQuestion(BaseModel, Jsonable):
    question_text: str
    resolution_criteria: str
    fine_print: str
    background_information: str
    expected_resolution_date: datetime
    question_type: Literal["binary", "numeric", "multiple_choice"] = "binary"
    options: list[str] = Field(
        default_factory=list,
        description="Options for multiple choice question. Empty if numeric or binary",
    )
    open_upper_bound: bool | None = Field(
        default=None,
        description="Whether there can be a value higher than upper bound. Only used for numeric questions.",
    )
    open_lower_bound: bool | None = Field(
        default=None,
        description="Whether there can be a value lower than lower bound. Only used for numeric questions.",
    )
    max_value: float | None = Field(
        default=None,
        description="The max value that the answer to the question can be. Only used for numeric questions.",
    )
    min_value: float | None = Field(
        default=None,
        description="The min value that the answer to the question can be. Only used for numeric questions.",
    )

    @classmethod
    def get_field_descriptions(cls) -> str:
        return clean_indents(
            """
            - question_text: A clear question about a future event
            - resolution_criteria: Specific criteria for how the question will resolve. If possible include a link to a status page (e.g. a website with a live number or condition that is easy to resolve). Mention the units/scale expected (give an example like "a value of $1.2 million of income will resolve as '1.2'")
            - fine_print: Additional information covering *every* edge case that could happen. There should be no chance of an ambiguous resolution. Resolution criteria + fine print should pass the clairvoyance test such that after the event happens there is no debate about whether it happened or not no matter how it resolves.
            - background_information: Relevant context and historical information to help understand the question
            - expected_resolution_date: The date when the question is expected to resolve
            - question_type: The type of question, either binary, numeric, or multiple_choice based on how the forecaster should answer (with yes/no, a number, or a choice from a list)
            - options: The options for the question, only used for multiple_choice questions. Empty list for other question types.
            - open_upper_bound: Whether there can be a value higher than upper bound (e.g. if the value is a percentag, 100 is the max the bound is closed, but number of certifications in a population has an open upper bound), only used for numeric questions.
            - open_lower_bound: Whether there can be a value lower than lower bound (e.g. distances can't be negative the bound is closed at 0, but profit margins can be negative so the bound is open), only used for numeric questions.
            - max_value: The max value that the answer to the question can be. If bound is closed then choose the max number. If bound is open then pick a really really big number. Only used for numeric questions. (e.g. 100 for a percentage, 1000 for a number of certifications from an small org, 100000 for a number of new houses built in a large city in a year)
            - min_value: The min value that the answer to the question can be. If bound is closed then choose the min number. If bound is open then pick a really really negative number. Only used for numeric questions. (e.g. 0 for a percentage, 0 for a number of certifications from a small org, -10000000 for a medium company net profit)
            """
        )

    @field_validator("expected_resolution_date", mode="after")
    @classmethod
    def ensure_utc_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @model_validator(
        mode="after",
    )
    def validate_question_type_fields(self: SimpleQuestion) -> SimpleQuestion:
        if self.question_type == "numeric":
            assert (
                self.max_value is not None
            ), "Upper bound must be provided for numeric questions"
            assert (
                self.min_value is not None
            ), "Lower bound must be provided for numeric questions"
            assert (
                self.open_upper_bound is not None
            ), "Open upper bound must be provided for numeric questions"
            assert (
                self.open_lower_bound is not None
            ), "Open lower bound must be provided for numeric questions"
        else:
            assert (
                self.max_value is None
            ), "Upper bound must not be provided for non-numeric questions"
            assert (
                self.min_value is None
            ), "Lower bound must not be provided for non-numeric questions"
            assert (
                self.open_upper_bound is None
            ), "Open upper bound must not be provided for non-numeric questions"
            assert (
                self.open_lower_bound is None
            ), "Open lower bound must not be provided for non-numeric questions"

        if self.question_type == "multiple_choice":
            assert (
                len(self.options) > 0
            ), "Options must be provided for multiple choice questions"
        else:
            assert (
                len(self.options) == 0
            ), "Options must not be provided for non-multiple choice questions"
        return self

    @classmethod
    def full_questions_to_simple_questions(
        cls, full_questions: list[MetaculusQuestion]
    ) -> list[SimpleQuestion]:
        simple_questions = []
        for question in full_questions:
            if isinstance(question, DateQuestion):
                # TODO: Give more direct support for date questions
                continue

            assert question.question_text is not None
            assert question.resolution_criteria is not None
            assert question.background_info is not None
            assert question.scheduled_resolution_time is not None
            assert question.fine_print is not None

            if isinstance(question, NumericQuestion):
                # TODO: Give more direct support for date questions
                question_type = "numeric"
                options = []
                upper_bound = question.upper_bound
                lower_bound = question.lower_bound
                open_upper_bound = question.open_upper_bound
                open_lower_bound = question.open_lower_bound
            elif isinstance(question, BinaryQuestion):
                question_type = "binary"
                options = []
                upper_bound = None
                lower_bound = None
                open_upper_bound = None
                open_lower_bound = None
            elif isinstance(question, MultipleChoiceQuestion):
                question_type = "multiple_choice"
                options = question.options
                upper_bound = None
                lower_bound = None
                open_upper_bound = None
                open_lower_bound = None
            else:
                raise ValueError(f"Unknown question type: {type(question)}")

            simple_question = SimpleQuestion(
                question_text=question.question_text,
                resolution_criteria=question.resolution_criteria,
                fine_print=question.fine_print,
                background_information=question.background_info,
                expected_resolution_date=question.scheduled_resolution_time,
                question_type=question_type,
                options=options,
                max_value=upper_bound,
                min_value=lower_bound,
                open_upper_bound=open_upper_bound,
                open_lower_bound=open_lower_bound,
            )
            simple_questions.append(simple_question)
        return simple_questions

    @classmethod
    def simple_questions_to_metaculus_questions(
        cls, simple_questions: list[SimpleQuestion]
    ) -> list[MetaculusQuestion]:
        full_questions = []
        for question in simple_questions:
            if question.question_type == "binary":
                full_question = BinaryQuestion(
                    question_text=question.question_text,
                    background_info=question.background_information,
                    resolution_criteria=question.resolution_criteria,
                    fine_print=question.fine_print,
                    scheduled_resolution_time=question.expected_resolution_date,
                )
            elif question.question_type == "numeric":
                assert question.max_value is not None
                assert question.min_value is not None
                assert question.open_upper_bound is not None
                assert question.open_lower_bound is not None
                full_question = NumericQuestion(
                    question_text=question.question_text,
                    background_info=question.background_information,
                    resolution_criteria=question.resolution_criteria,
                    fine_print=question.fine_print,
                    upper_bound=question.max_value,
                    lower_bound=question.min_value,
                    open_upper_bound=question.open_upper_bound,
                    open_lower_bound=question.open_lower_bound,
                    scheduled_resolution_time=question.expected_resolution_date,
                )
            elif question.question_type == "multiple_choice":
                full_question = MultipleChoiceQuestion(
                    question_text=question.question_text,
                    background_info=question.background_information,
                    resolution_criteria=question.resolution_criteria,
                    fine_print=question.fine_print,
                    options=question.options,
                    scheduled_resolution_time=question.expected_resolution_date,
                )
            else:
                raise ValueError(
                    f"Unknown question type: {question.question_type}"
                )
            full_questions.append(full_question)
        return full_questions

    def is_within_date_range(
        self, resolve_before_date: datetime, resolve_after_date: datetime
    ) -> bool:

        return (
            resolve_before_date.astimezone(timezone.utc)
            >= self.expected_resolution_date.astimezone(timezone.utc)
            >= resolve_after_date.astimezone(timezone.utc)
        )
