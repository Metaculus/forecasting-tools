from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    DiscreteQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.util.misc import clean_indents


class SimpleQuestion(BaseModel, Jsonable):
    question_text: str
    resolution_criteria: str
    fine_print: str | None = None
    background_information: str | None = None
    expected_resolution_date: datetime
    question_type: Literal["binary", "numeric", "multiple_choice", "discrete"] = (
        "binary"
    )
    options: list[str] = Field(
        default_factory=list,
        description="Options are for multiple choice question. Empty if numeric, discrete, or binary. Must be defined for multiple choice questions.",
    )
    open_upper_bound: bool | None = Field(
        default=None,
        description="Open upper bound defines whether there can be a value higher than upper bound. Must be defined for numeric and discrete questions and None for other question types.",
    )
    open_lower_bound: bool | None = Field(
        default=None,
        description="Open lower bound defines whether there can be a value lower than lower bound. Must be defined for numeric and discrete questions and None for other question types.",
    )
    max_value: float | None = Field(
        default=None,
        description="Max value defines the largest reasonable value that the answer to the question can be. Must be defined for numeric and discrete questions and None for other question types.",
    )
    min_value: float | None = Field(
        default=None,
        description="Min value defines the smallest reasonable value that the answer to the question can be. Must be defined for numeric and discrete questions and None for other question types.",
    )
    step: float | None = Field(
        default=None,
        description="Spacing between consecutive outcomes for discrete questions. Required for discrete questions; must be None for all other question types.",
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
            - question_type: The type of question — binary, numeric, multiple_choice, or discrete — based on how the forecaster should answer (yes/no, a continuous number, a choice from a list, or a value from a small fixed set of evenly-spaced outcomes).
            - options: The options for the question, only used for multiple_choice questions. Empty list for other question types.
            - open_upper_bound: Whether there can be a value higher than upper bound (e.g. if the value is a percentage, 100 is the max the bound is closed, but number of certifications in a population has an open upper bound), only used for numeric and discrete questions.
            - open_lower_bound: Whether there can be a value lower than lower bound (e.g. distances can't be negative the bound is closed at 0, but profit margins can be negative so the bound is open), only used for numeric and discrete questions.
            - max_value: The max value that the answer to the question can be. If bound is closed then choose the max number. If bound is open then pick a really really big number. For discrete questions, this is the largest nominal outcome. Only used for numeric and discrete questions. (e.g. 100 for a percentage, 1000 for a number of certifications from an small org, 100000 for a number of new houses built in a large city in a year)
            - min_value: The min value that the answer to the question can be. If bound is closed then choose the min number. If bound is open then pick a really really negative number. For discrete questions, this is the smallest nominal outcome. Only used for numeric and discrete questions. (e.g. 0 for a percentage, 0 for a number of certifications from a small org, -10000000 for a medium company net profit)
            - step: Spacing between consecutive outcomes for discrete questions only (must be None for other types). `(max_value - min_value) / step` must be an integer in [2, 199]. Example: integer counts 0..10 use step=1.
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
        if self.question_type in ("numeric", "discrete"):
            assert (
                self.max_value is not None
            ), "Upper bound must be provided for continuous questions"
            assert (
                self.min_value is not None
            ), "Lower bound must be provided for continuous questions"
            assert (
                self.open_upper_bound is not None
            ), "Open upper bound must be provided for continuous questions"
            assert (
                self.open_lower_bound is not None
            ), "Open lower bound must be provided for continuous questions"
        else:
            assert (
                self.max_value is None
            ), "Upper bound must not be provided for non-numeric/discrete questions"
            assert (
                self.min_value is None
            ), "Lower bound must not be provided for non-numeric/discrete questions"
            assert (
                self.open_upper_bound is None
            ), "Open upper bound must not be provided for non-numeric/discrete questions"
            assert (
                self.open_lower_bound is None
            ), "Open lower bound must not be provided for non-numeric/discrete questions"

        if self.question_type == "multiple_choice":
            assert (
                len(self.options) > 0
            ), "Options must be provided for multiple choice questions"
        else:
            assert (
                len(self.options) == 0
            ), "Options must not be provided for non-multiple choice questions"

        if self.question_type == "discrete":
            assert self.step is not None, "step must be provided for discrete questions"
            assert self.step > 0, f"step must be positive, got {self.step}"
            assert self.max_value is not None and self.min_value is not None
            range_size = self.max_value - self.min_value
            assert range_size > 0, "max_value must be greater than min_value"
            assert (
                self.step <= range_size / 2
            ), "step too large: must be <= (max_value - min_value) / 2"
            assert (
                self.step >= range_size / 199
            ), f"step too small: must be >= (max_value - min_value) / 199"
            quotient = range_size / self.step
            assert abs(round(quotient) - quotient) < 1e-6, (
                "range / step must be an integer; "
                f"(max_value - min_value) = {range_size} is not a multiple of "
                f"step = {self.step}"
            )
            inbound_outcome_count = round(quotient) + 1
            assert 3 <= inbound_outcome_count <= 200, (
                f"derived inbound_outcome_count={inbound_outcome_count} outside "
                "the platform's [3, 200] range"
            )
        else:
            assert (
                self.step is None
            ), f"step must not be provided for {self.question_type} questions"

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

            step = None
            if isinstance(question, DiscreteQuestion):
                question_type = "discrete"
                options = []
                open_upper_bound = question.open_upper_bound
                open_lower_bound = question.open_lower_bound
                inbound_outcome_count = question.cdf_size - 1
                if (
                    question.nominal_lower_bound is not None
                    and question.nominal_upper_bound is not None
                ):
                    upper_bound = question.nominal_upper_bound
                    lower_bound = question.nominal_lower_bound
                    step = (
                        (upper_bound - lower_bound) / (inbound_outcome_count - 1)
                        if inbound_outcome_count > 1
                        else None
                    )
                else:
                    # Back-compute nominal bounds from actual bounds + cdf_size.
                    # Actual range spans inbound_outcome_count * step (since the
                    # actual bounds are ±0.5*step beyond the nominal bounds), so
                    # step = actual_range / inbound_outcome_count, and the
                    # nominal bounds sit half a step inside the actual bounds.
                    step = (
                        (question.upper_bound - question.lower_bound)
                        / inbound_outcome_count
                    )
                    lower_bound = question.lower_bound + step / 2
                    upper_bound = question.upper_bound - step / 2
            elif isinstance(question, NumericQuestion):
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
                step=step,
            )
            simple_questions.append(simple_question)
        return simple_questions

    @classmethod
    def simple_questions_to_metaculus_questions(
        cls, simple_questions: list[SimpleQuestion]
    ) -> list[MetaculusQuestion]:
        full_questions = []
        for question in simple_questions:
            base_attrs = {
                "question_text": question.question_text,
                "background_info": question.background_information,
                "resolution_criteria": question.resolution_criteria,
                "fine_print": question.fine_print,
                "scheduled_resolution_time": question.expected_resolution_date,
            }

            if question.question_type == "binary":
                full_question = BinaryQuestion(**base_attrs)
            elif question.question_type == "numeric":
                assert question.max_value is not None
                assert question.min_value is not None
                assert question.open_upper_bound is not None
                assert question.open_lower_bound is not None
                full_question = NumericQuestion(
                    **base_attrs,
                    upper_bound=question.max_value,
                    lower_bound=question.min_value,
                    open_upper_bound=question.open_upper_bound,
                    open_lower_bound=question.open_lower_bound,
                )
            elif question.question_type == "discrete":
                assert question.max_value is not None
                assert question.min_value is not None
                assert question.step is not None and question.step > 0
                assert question.open_upper_bound is not None
                assert question.open_lower_bound is not None
                half_step = question.step / 2
                inbound_outcome_count = (
                    round((question.max_value - question.min_value) / question.step) + 1
                )
                full_question = DiscreteQuestion(
                    **base_attrs,
                    nominal_lower_bound=question.min_value,
                    nominal_upper_bound=question.max_value,
                    lower_bound=question.min_value - half_step,
                    upper_bound=question.max_value + half_step,
                    open_upper_bound=question.open_upper_bound,
                    open_lower_bound=question.open_lower_bound,
                    cdf_size=inbound_outcome_count + 1,
                )
            elif question.question_type == "multiple_choice":
                full_question = MultipleChoiceQuestion(
                    **base_attrs,
                    options=question.options,
                )
            else:
                raise ValueError(f"Unknown question type: {question.question_type}")
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
