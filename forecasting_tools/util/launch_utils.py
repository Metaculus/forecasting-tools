from __future__ import annotations

import asyncio
import logging
import math
import random
from datetime import date, datetime, timedelta
from typing import Any, Coroutine, Literal

from pydantic import BaseModel, field_validator, model_validator

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.util.custom_logger import CustomLogger
from forecasting_tools.util.file_manipulation import (
    append_to_text_file,
    load_csv_file,
    load_text_file,
    write_csv_file,
)
from forecasting_tools.util.jsonable import Jsonable

full_datetime_format = "%m/%d/%Y %H:%M:%S"
sheet_date_format1 = "%m/%d/%Y"
sheet_date_format2 = "%m/%d/%y"

logger = logging.getLogger(__name__)


class LaunchQuestion(BaseModel, Jsonable):
    parent_url: str
    author: str
    title: str
    type: Literal["binary", "numeric", "multiple_choice"]
    resolution_criteria: str
    fine_print: str
    description: str
    question_weight: float | None = None
    open_time: datetime | None = None
    scheduled_close_time: datetime | None = None
    scheduled_resolve_time: datetime | None = None
    range_min: float | int | None = None
    range_max: float | int | None = None
    zero_point: float | int | None = None
    open_lower_bound: bool | None = None
    open_upper_bound: bool | None = None
    unit: str | None = None
    group_variable: str | None = None
    options: list[str] | None = None
    tournament: str | None = None
    original_order: int = 0

    @field_validator(
        "open_time",
        "scheduled_close_time",
        "scheduled_resolve_time",
        mode="before",
    )
    @classmethod
    def parse_datetime(cls, value: Any) -> datetime | None:
        if isinstance(value, datetime) or value is None:
            return value
        elif isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
            for format in [
                full_datetime_format,
                sheet_date_format1,
                sheet_date_format2,
            ]:
                try:
                    return datetime.strptime(value, format)
                except ValueError:
                    continue
        raise ValueError(f"Invalid datetime format: {value}")

    @field_validator("range_min", "range_max", "zero_point", mode="before")
    @classmethod
    def parse_numeric_fields(cls, value: Any) -> int | float | None:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return None
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            for format in [int, float]:
                try:
                    return format(value)
                except ValueError:
                    continue
        raise ValueError(f"Invalid numeric value type: {type(value)}")

    @field_validator("open_lower_bound", "open_upper_bound", mode="before")
    @classmethod
    def parse_boolean_fields(cls, value: Any) -> bool | None:  # NOSONAR
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value = value.strip().upper()
            if value == "TRUE":
                return True
            if value == "FALSE":
                return False
        raise ValueError(f"Invalid boolean value: {value}")

    @field_validator("options", mode="before")
    @classmethod
    def parse_options(cls, value: Any) -> list[str] | None:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [opt.strip() for opt in value.split("|") if opt.strip()]
        raise ValueError(f"Invalid options format: {value}")

    @field_validator("question_weight", mode="before")
    @classmethod
    def parse_question_weight(cls, value: Any) -> float | None:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise ValueError(f"Invalid question weight value: {value}")

    @model_validator(mode="after")
    def validate_times(self: LaunchQuestion) -> LaunchQuestion:
        open_time = self.open_time
        close_time = self.scheduled_close_time
        resolve_date = self.scheduled_resolve_time

        if open_time and close_time:
            assert open_time <= close_time
        if close_time and resolve_date:
            assert close_time <= resolve_date
        if open_time and resolve_date:
            assert open_time <= resolve_date
        return self

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "parent_url": self.parent_url,
            "author": self.author,
            "title": self.title,
            "type": self.type,
            "resolution_criteria": self.resolution_criteria,
            "fine_print": self.fine_print,
            "description": self.description,
            "question_weight": self.question_weight,
            "open_time": (
                self.open_time.strftime(full_datetime_format)
                if self.open_time
                else ""
            ),
            "scheduled_close_time": (
                self.scheduled_close_time.strftime(full_datetime_format)
                if self.scheduled_close_time
                else ""
            ),
            "scheduled_resolve_time": (
                self.scheduled_resolve_time.strftime(sheet_date_format1)
                if self.scheduled_resolve_time
                else ""
            ),
            "range_min": self.range_min,
            "range_max": self.range_max,
            "zero_point": self.zero_point,
            "open_lower_bound": self.open_lower_bound,
            "open_upper_bound": self.open_upper_bound,
            "unit": self.unit,
            "group_variable": self.group_variable,
            "options": "|".join(self.options) if self.options else "",
        }

    @classmethod
    def from_csv_row(cls, row: dict, original_order: int) -> LaunchQuestion:
        # Create a new dictionary with cleaned keys
        cleaned_row = {k.replace("\ufeff", ""): v for k, v in row.items()}
        cleaned_row["original_order"] = original_order
        return cls(**cleaned_row)


class LaunchWarning(BaseModel, Jsonable):
    warning: str
    relevant_question: LaunchQuestion | None = None


class SheetOrganizer:

    @classmethod
    async def schedule_questions_from_file(
        cls,
        input_file_path: str,
        bot_output_file_path: str,
        start_date: datetime,
        end_date: datetime,
        chance_to_skip_slot: float,
        pro_output_file_path: str | None = None,
        num_pro_questions: int | None = None,
    ) -> list[LaunchQuestion]:
        input_questions = cls.load_questions_from_csv(input_file_path)
        random.shuffle(input_questions)
        bot_questions = cls.schedule_questions(
            input_questions, start_date, chance_to_skip_slot
        )
        cls.save_questions_to_csv(bot_questions, bot_output_file_path)
        warnings = await cls.find_processing_errors(
            input_questions,
            bot_questions,
            start_date,
            end_date,
            question_type="bots",
        )
        if pro_output_file_path and num_pro_questions:
            pro_questions = cls.make_pro_questions_from_bot_questions(
                bot_questions,
                num_pro_questions,
                start_date,
            )
            cls.save_questions_to_csv(pro_questions, pro_output_file_path)
            additional_warnings = await cls.find_processing_errors(
                bot_questions,
                pro_questions,
                start_date,
                end_date,
                question_type="pros",
            )
            warnings.extend(additional_warnings)
        for warning in warnings:
            logger.warning(warning)
        return bot_questions

    @classmethod
    def schedule_questions(
        cls,
        questions: list[LaunchQuestion],
        start_date: datetime,
        chance_to_skip_slot: float,
    ) -> list[LaunchQuestion]:
        if chance_to_skip_slot >= 0.95:
            raise ValueError(
                "Chance to skip slot is too high and could run indefinitely"
            )

        copied_input_questions = [
            question.model_copy(deep=True) for question in questions
        ]
        prescheduled_questions = [
            q
            for q in copied_input_questions
            if q.open_time is not None and q.scheduled_close_time is not None
        ]
        questions_to_schedule = [
            q
            for q in copied_input_questions
            if q not in prescheduled_questions
        ]
        # May 2 2025: Consider not sorting by scheduled resolve time since it can put too many numerics back to back which s hard for pros.
        questions_to_schedule.sort(
            key=lambda q: (
                (
                    q.scheduled_resolve_time
                    if q.scheduled_resolve_time
                    else datetime.max
                ),
            )
        )

        proposed_open_time = start_date.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        newly_scheduled_questions = []

        # Handle the case when there are no questions to schedule
        if not questions_to_schedule:
            all_questions = prescheduled_questions
            all_questions.sort(
                key=lambda q: (q.open_time if q.open_time else datetime.max,)
            )
            return all_questions

        current_question = questions_to_schedule.pop(0)
        while True:
            proposed_open_time += timedelta(hours=2)
            proposed_closed_time = proposed_open_time + timedelta(hours=2)
            current_question.open_time = proposed_open_time
            current_question.scheduled_close_time = proposed_closed_time

            prescheduled_question_overlaps = any(
                cls._open_window_overlaps_for_questions(
                    prescheduled_question, current_question
                )
                for prescheduled_question in prescheduled_questions
            )

            if not prescheduled_question_overlaps:
                if (
                    current_question.scheduled_resolve_time is not None
                    and current_question.scheduled_resolve_time
                    < current_question.scheduled_close_time
                ):
                    raise RuntimeError(
                        f"Question {current_question.title} has a scheduled resolve time that can't find a valid close time"
                    )

                if random.random() < chance_to_skip_slot:
                    continue
                new_question = LaunchQuestion(
                    **current_question.model_dump(),
                )  # For model validation purposes
                newly_scheduled_questions.append(new_question)

                # Break out of the loop if there are no more questions to schedule
                if not questions_to_schedule:
                    break

                current_question = questions_to_schedule.pop(0)

        all_questions = prescheduled_questions + newly_scheduled_questions
        all_questions.sort(
            key=lambda q: (q.open_time if q.open_time else datetime.max,)
        )

        return all_questions

    @staticmethod
    def compute_upcoming_day(
        day_of_week: Literal["monday", "saturday", "friday"],
    ) -> datetime:
        day_number = {"monday": 0, "saturday": 5, "friday": 4}
        today = datetime.now().date()
        today_weekday = today.weekday()
        target_weekday = day_number[day_of_week]

        if today_weekday == target_weekday:
            # If today is the target day, return next week's day
            days_to_add = 7
        elif today_weekday < target_weekday:
            # If target day is later this week
            days_to_add = target_weekday - today_weekday
        else:
            # If target day is in next week
            days_to_add = 7 - today_weekday + target_weekday

        target_date = today + timedelta(days=days_to_add)
        return datetime(target_date.year, target_date.month, target_date.day)

    @classmethod
    def make_pro_questions_from_bot_questions(
        cls,
        bot_launch_questions: list[LaunchQuestion],
        num_pro_questions: int,
        start_date: datetime,
    ) -> list[LaunchQuestion]:
        if num_pro_questions > len(bot_launch_questions):
            raise ValueError(
                "Number of pro questions is greater than the number of bot questions"
            )

        date_to_question_map: dict[date, list[LaunchQuestion]] = {}
        for question in bot_launch_questions:
            if (
                question.scheduled_close_time
                and question.scheduled_close_time > start_date
            ):
                if (
                    question.scheduled_close_time.date()
                    not in date_to_question_map
                ):
                    date_to_question_map[
                        question.scheduled_close_time.date()
                    ] = []
                date_to_question_map[
                    question.scheduled_close_time.date()
                ].append(question)

        questions_left_to_sample = num_pro_questions
        pro_launch_questions: list[LaunchQuestion] = []
        available_days = list(date_to_question_map.keys())
        iterations = 0
        while questions_left_to_sample > 0:
            iterations += 1
            if iterations > 1000:
                raise RuntimeError(
                    "Number of iterations is greater than 1000 and could run indefinitely"
                )

            for day in available_days:
                if questions_left_to_sample == 0:
                    break

                questions_for_day = date_to_question_map[day]
                if len(questions_for_day) == 0:
                    continue

                sampled_bot_question = random.sample(questions_for_day, 1)[0]
                date_to_question_map[day].remove(sampled_bot_question)

                if cls._weighted_pair_is_already_in_list(
                    sampled_bot_question, pro_launch_questions
                ):
                    continue

                pro_question = cls._make_pro_question_from_bot_question(
                    sampled_bot_question
                )
                pro_launch_questions.append(pro_question)
                questions_left_to_sample -= 1

        pro_launch_questions.sort(
            key=lambda q: ((q.open_time if q.open_time else datetime.max),)
        )
        return pro_launch_questions

    @classmethod
    def _weighted_pair_is_already_in_list(
        cls,
        question: LaunchQuestion,
        chosen_questions: list[LaunchQuestion],
    ) -> bool:
        # TODO: This is incorrectly triggered when the adjacent question had a weight of 1
        #       (or a different weight overall) and thus was not a weighted pair.
        if (
            question.question_weight is not None
            and question.question_weight < 1
        ):
            question_index_one_above = question.original_order + 1
            question_index_one_below = question.original_order - 1
            all_question_indexes = [q.original_order for q in chosen_questions]
            if (
                question_index_one_above in all_question_indexes
                or question_index_one_below in all_question_indexes
            ):
                return True
        return False

    @classmethod
    def _make_pro_question_from_bot_question(
        cls,
        bot_question: LaunchQuestion,
    ) -> LaunchQuestion:
        question_copy = bot_question.model_copy(deep=True)
        assert question_copy.open_time is not None
        question_copy.open_time -= timedelta(days=2.5)
        question_copy.question_weight = 1
        return question_copy

    @classmethod
    def load_questions_from_csv(cls, file_path: str) -> list[LaunchQuestion]:
        questions = load_csv_file(file_path)
        loaded_questions = [
            LaunchQuestion.from_csv_row(row, i)
            for i, row in enumerate(questions)
        ]
        return loaded_questions

    @classmethod
    def save_questions_to_csv(
        cls, questions: list[LaunchQuestion], file_path: str
    ) -> None:
        write_csv_file(
            file_path, [question.to_csv_row() for question in questions]
        )

    @classmethod
    async def find_processing_errors(  # NOSONAR
        cls,
        original_questions: list[LaunchQuestion],
        new_questions: list[LaunchQuestion],
        start_date: datetime,
        end_date: datetime,
        question_type: Literal["bots", "pros"],
    ) -> list[LaunchWarning]:
        final_warnings = []

        # Some questions will already have a open and close time. These must be respected and stay the same
        def _check_existing_times_preserved(
            question_type: Literal["bots", "pros"],
        ) -> list[LaunchWarning]:
            warnings = []
            for orig_q in original_questions:
                if (
                    orig_q.open_time is not None
                    and orig_q.scheduled_close_time is not None
                ):
                    for new_q in new_questions:
                        if not new_q.title == orig_q.title:
                            continue
                        if (
                            new_q.open_time != orig_q.open_time
                            and question_type == "bots"
                        ):
                            warnings.append(
                                LaunchWarning(
                                    relevant_question=new_q,
                                    warning=f"Existing open times must be preserved. Original: {orig_q.open_time} to {orig_q.scheduled_close_time}",
                                )
                            )
                        if (
                            new_q.scheduled_close_time
                            != orig_q.scheduled_close_time
                        ):
                            warnings.append(
                                LaunchWarning(
                                    relevant_question=new_q,
                                    warning=f"Existing close times must be preserved. Original: {orig_q.open_time} to {orig_q.scheduled_close_time}",
                                )
                            )
            return warnings

        # No overlapping windows except for questions originally with a open/close time
        def _check_no_new_overlapping_windows() -> list[LaunchWarning]:
            warnings = []
            overlapping_pairs = cls._find_overlapping_windows(new_questions)

            for q1, q2 in overlapping_pairs:
                # Check if either question had preexisting times in original questions
                q1_had_times = any(
                    orig_q.title == q1.title
                    and orig_q.open_time is not None
                    and orig_q.scheduled_close_time is not None
                    for orig_q in original_questions
                )
                q2_had_times = any(
                    orig_q.title == q2.title
                    and orig_q.open_time is not None
                    and orig_q.scheduled_close_time is not None
                    for orig_q in original_questions
                )

                if not (q1_had_times and q2_had_times):
                    warnings.append(
                        LaunchWarning(
                            relevant_question=q1,
                            warning=f"Overlapping time window with question: {q2.title}",
                        )
                    )
            return warnings

        # If bots, The open time must be 2hr before scheduled close time unless. If pros it must be 2 days
        def _check_window_duration(
            question_type: Literal["bots", "pros"],
        ) -> list[LaunchWarning]:
            warnings = []
            if question_type == "bots":
                required_duration = timedelta(hours=2)
            elif question_type == "pros":
                required_duration = timedelta(days=2, hours=14)
            else:
                raise ValueError(f"Invalid question type: {question_type}")

            for question in new_questions:
                if question.open_time and question.scheduled_close_time:
                    actual_duration = (
                        question.scheduled_close_time - question.open_time
                    )
                    if actual_duration != required_duration:
                        warnings.append(
                            LaunchWarning(
                                relevant_question=question,
                                warning=f"Incorrect time window duration. Required: {required_duration}, Actual: {actual_duration}",
                            )
                        )
            return warnings

        # No open/close windows are exactly the same
        def _check_unique_windows() -> list[LaunchWarning]:
            warnings = []
            window_map: dict[tuple[datetime, datetime], str] = {}

            for question in new_questions:
                if question.open_time and question.scheduled_close_time:
                    window = (
                        question.open_time,
                        question.scheduled_close_time,
                    )
                    if window in window_map:
                        warnings.append(
                            LaunchWarning(
                                relevant_question=question,
                                warning=f"Identical time window with question: {window_map[window]}",
                            )
                        )
                    else:
                        window_map[window] = question.title
            return warnings

        # For each question All fields exist (i.e. are not none or empty)
        def _check_required_fields() -> list[LaunchWarning]:
            warnings = []
            for question in new_questions:
                try:
                    assert question.author, "author is required"
                    assert question.title, "title is required"
                    assert question.type, "type is required"
                    assert (
                        question.resolution_criteria
                    ), "resolution_criteria is required"
                    assert question.description, "description is required"
                    assert (
                        question.question_weight is not None
                        and 1 >= question.question_weight >= 0
                    ), "question_weight must be between 0 and 1"
                    assert question.open_time, "open_time is required"
                    assert (
                        question.scheduled_close_time
                    ), "scheduled_close_time is required"
                    assert (
                        question.scheduled_resolve_time
                    ), "scheduled_resolve_time is required"
                except AssertionError as e:
                    warnings.append(
                        LaunchWarning(
                            relevant_question=question,
                            warning=f"Missing at least one required field. Error: {e}",
                        )
                    )
            return warnings

        # If numeric should include range_min and range_max and upper and lower bounds
        def _check_numeric_fields() -> list[LaunchWarning]:
            warnings = []

            for question in new_questions:
                if question.type == "numeric":
                    try:
                        assert (
                            question.range_min is not None
                        ), "range_min is required"
                        assert (
                            question.range_max is not None
                        ), "range_max is required"
                        assert (
                            question.open_lower_bound is not None
                        ), "open_lower_bound is required"
                        assert (
                            question.open_upper_bound is not None
                        ), "open_upper_bound is required"
                        assert (
                            question.unit is not None and question.unit.strip()
                        ), "unit is required for numeric questions"
                    except AssertionError as e:
                        warnings.append(
                            LaunchWarning(
                                relevant_question=question,
                                warning=f"Missing at least one numeric field. Error: {e}",
                            )
                        )
                else:
                    # Check that non-numeric questions don't have unit
                    if question.unit is not None and question.unit.strip():
                        warnings.append(
                            LaunchWarning(
                                relevant_question=question,
                                warning="Unit should only be specified for numeric questions",
                            )
                        )
            return warnings

        # If MC should include options and group_variable
        def _check_mc_fields() -> list[LaunchWarning]:
            warnings = []

            for question in new_questions:
                if question.type == "multiple_choice":
                    if not question.options:
                        warnings.append(
                            LaunchWarning(
                                relevant_question=question,
                                warning="Multiple choice question missing options",
                            )
                        )
                    if not question.group_variable:
                        warnings.append(
                            LaunchWarning(
                                relevant_question=question,
                                warning="Multiple choice question missing group_variable",
                            )
                        )
            return warnings

        # No fields changed between original and new question other than open/close time
        def _check_no_field_changes(
            question_type: Literal["bots", "pros"],
        ) -> list[LaunchWarning]:
            warnings = []

            duplicate_title_warnings = _check_duplicate_titles()
            if duplicate_title_warnings:
                warnings.append(
                    LaunchWarning(
                        relevant_question=None,
                        warning=(
                            "Cannot check if persistent fields changed"
                            " between original and new questions because"
                            " duplicate titles were found (titles needed to match"
                            " between original and new questions)"
                        ),
                    )
                )
                return warnings

            for orig_q in original_questions:
                for new_q in new_questions:
                    if orig_q.title == new_q.title:
                        for field in LaunchQuestion.model_fields.keys():
                            if field not in [
                                "open_time",
                                "scheduled_close_time",
                            ]:
                                if (
                                    question_type == "pros"
                                    and field == "question_weight"
                                ):
                                    continue

                                if getattr(orig_q, field) != getattr(
                                    new_q, field
                                ):
                                    warnings.append(
                                        LaunchWarning(
                                            relevant_question=new_q,
                                            warning=f"Field {field} was changed from {getattr(orig_q, field)} to {getattr(new_q, field)}",
                                        )
                                    )
            return warnings

        # If there is a parent url, then the description, resolution criteria, and fine print should be ".p"
        def _check_parent_url_fields() -> list[LaunchWarning]:
            warnings = []

            for question in new_questions:
                if question.parent_url and question.parent_url.strip():
                    try:
                        assert (
                            question.description == ".p"
                        ), "description should be '.p'"
                        assert (
                            question.resolution_criteria == ".p"
                        ), "resolution_criteria should be '.p'"
                        assert (
                            question.fine_print == ".p"
                        ), "fine_print should be '.p'"
                    except AssertionError as e:
                        warnings.append(
                            LaunchWarning(
                                relevant_question=question,
                                warning=f"Question with parent_url should have description, resolution_criteria, and fine_print set to '.p'. Error: {e}",
                            )
                        )
            return warnings

        # The earliest open time is on start_date
        def _check_earliest_open_time(
            question_type: Literal["bots", "pros"],
        ) -> list[LaunchWarning]:
            warnings = []

            earliest_open = min(
                (
                    q.open_time
                    for q in new_questions
                    if q.open_time is not None
                ),
                default=None,
            )
            target_date = (
                start_date
                if question_type == "bots"
                else start_date - timedelta(days=2)
            )
            if earliest_open and earliest_open.date() != target_date.date():
                # Find the question with the earliest open time
                earliest_question = next(
                    q for q in new_questions if q.open_time == earliest_open
                )
                warnings.append(
                    LaunchWarning(
                        relevant_question=earliest_question,
                        warning=f"Earliest open time should be on {target_date.date()}, not {earliest_open.date()}",
                    )
                )
            return warnings

        # open times are between start_date and the questions' resolve date
        def _check_adherence_to_end_and_start_time(
            question_type: Literal["bots", "pros"],
        ) -> list[LaunchWarning]:
            warnings = []
            for question in new_questions:
                warning_messages = []
                if question.open_time:
                    if question.open_time > end_date + timedelta(days=1):
                        warning_messages.append(
                            f"Question opens at {question.open_time} after end date {end_date}"
                        )
                    pro_adjustment = (
                        timedelta(days=2)
                        if question_type == "pros"
                        else timedelta(days=0)
                    )
                    if question.open_time < start_date - pro_adjustment:
                        warning_messages.append(
                            f"Question opens at {question.open_time} before start date {start_date}"
                        )
                if question.scheduled_close_time:
                    if question.scheduled_close_time > end_date + timedelta(
                        days=1
                    ):
                        warning_messages.append(
                            f"Question closes at {question.scheduled_close_time} after end date {end_date}"
                        )
                    if question.scheduled_close_time < start_date:
                        warning_messages.append(
                            f"Question closes at {question.scheduled_close_time} before start date {start_date}"
                        )

                for warning_message in warning_messages:
                    warnings.append(
                        LaunchWarning(
                            relevant_question=question,
                            warning=warning_message,
                        )
                    )
            return warnings

        # None of the questions have "Bridgewater" in tournament name if pros
        def _check_bridgewater_tournament() -> list[LaunchWarning]:
            warnings = []

            if question_type == "pros":
                for question in new_questions:
                    if (
                        question.tournament
                        and "Bridgewater" in question.tournament
                    ):
                        warnings.append(
                            LaunchWarning(
                                relevant_question=question,
                                warning="Pros questions cannot have 'Bridgewater' in tournament name",
                            )
                        )
            return warnings

        # All numeric questions have their range_min and range_max larger than 100 difference
        def _check_numeric_range() -> list[LaunchWarning]:
            warnings = []

            for question in new_questions:
                if (
                    question.type == "numeric"
                    and question.range_min is not None
                    and question.range_max is not None
                    and question.range_max - question.range_min < 100
                ):
                    warnings.append(
                        LaunchWarning(
                            relevant_question=question,
                            warning=f"Numeric range difference ({question.range_max - question.range_min}) is less than 100 and might be discrete",
                        )
                    )
            return warnings

        # None of the questions have duplicate titles
        def _check_duplicate_titles(
            use_stored_titles: bool = False,
            title_file: str = "temp/question_name_list.txt",
        ) -> list[LaunchWarning]:
            warnings = []
            title_count = {}

            new_question_titles = [
                question.title for question in new_questions
            ]
            if use_stored_titles:
                stored_question_titles = load_text_file(title_file)
                assert (
                    len(stored_question_titles) > 0
                ), "Stored question titles file is empty"
                stored_question_titles = [
                    title.strip()
                    for title in stored_question_titles.split("\n")
                    if title.strip()
                ]
                combined_question_titles = (
                    stored_question_titles + new_question_titles
                )
            else:
                combined_question_titles = new_question_titles

            duplicate_titles = set()
            for title in combined_question_titles:
                title_count[title] = title_count.get(title, 0) + 1
                if (
                    title_count[title] == 2
                ):  # Only add warning first time duplicate is found
                    duplicate_titles.add(title)
                    warnings.append(
                        LaunchWarning(
                            warning=f"Duplicate title found: {title}",
                        )
                    )

            if use_stored_titles:
                non_duplicate_titles = [
                    title
                    for title in new_question_titles
                    if title not in duplicate_titles
                ]
                append_to_text_file(
                    title_file, "\n".join(non_duplicate_titles)
                )

            if len(warnings) > 8:
                warnings = [
                    LaunchWarning(
                        warning=f"There are {len(warnings)} duplicate titles. You probably just reran the script.",
                    )
                ]
            return warnings

        # If bot questions
        # 16-24% of questions are numeric
        # 16-24% of questions are MC
        # 56-64% of questions are binary
        def _check_question_type_distribution() -> list[LaunchWarning]:
            warnings = []

            if question_type == "bots":
                total = len(new_questions)
                if total == 0:
                    return warnings

                numeric_count = sum(
                    1 for q in new_questions if q.type == "numeric"
                )
                mc_count = sum(
                    1 for q in new_questions if q.type == "multiple_choice"
                )
                binary_count = sum(
                    1 for q in new_questions if q.type == "binary"
                )

                numeric_percent = numeric_count / total * 100
                mc_percent = mc_count / total * 100
                binary_percent = binary_count / total * 100

                if not (16 <= numeric_percent <= 24):
                    warnings.append(
                        LaunchWarning(
                            warning=f"Numeric questions ({numeric_percent:.1f}%) outside 16-24% range",
                        )
                    )

                if not (16 <= mc_percent <= 24):
                    warnings.append(
                        LaunchWarning(
                            warning=f"Multiple choice questions ({mc_percent:.1f}%) outside 16-24% range",
                        )
                    )

                if not (56 <= binary_percent <= 64):
                    warnings.append(
                        LaunchWarning(
                            warning=f"Binary questions ({binary_percent:.1f}%) outside 56-64% range",
                        )
                    )
            return warnings

        # The average question_weight is greater than 0.8
        def _check_average_weight() -> list[LaunchWarning]:
            warnings = []

            if new_questions:
                avg_weight = sum(
                    q.question_weight
                    for q in new_questions
                    if q.question_weight is not None
                ) / len(new_questions)
                if avg_weight <= 0.8:
                    warnings.append(
                        LaunchWarning(
                            warning=f"Average question weight ({avg_weight:.2f}) is not greater than 0.8",
                        )
                    )
            return warnings

        # The original order is different than the new ordering
        def _check_order_changed() -> list[LaunchWarning]:
            warnings = []
            same_order = True
            for i in range(len(original_questions)):
                if original_questions[i].title != new_questions[i].title:
                    same_order = False
                    break

            if same_order:
                warnings.append(
                    LaunchWarning(
                        relevant_question=None,
                        warning="Question order has not changed from original",
                    )
                )
            return warnings

        # Questions are ordered by open time
        def _check_ordered_by_open_time() -> list[LaunchWarning]:
            warnings = []

            for i in range(1, len(new_questions)):
                previous_question = new_questions[i - 1]
                current_question = new_questions[i]
                if (
                    previous_question.open_time
                    and current_question.open_time
                    and previous_question.open_time
                    > current_question.open_time
                ):
                    warnings.append(
                        LaunchWarning(
                            relevant_question=current_question,
                            warning=f"Questions not ordered by open time. {previous_question.title} opens after {current_question.title}",
                        )
                    )
            return warnings

        # Same number of questions as original
        def _check_same_number_of_questions() -> list[LaunchWarning]:
            warnings = []
            if len(original_questions) != len(new_questions):
                # Find missing questions by comparing titles
                original_titles = {q.title for q in original_questions}
                new_titles = {q.title for q in new_questions}

                missing_from_new = original_titles - new_titles
                missing_from_original = new_titles - original_titles

                warning_msg = f"Number of questions is different from original -> Original: {len(original_questions)} != New: {len(new_questions)}"

                if missing_from_new:
                    warning_msg += f"\nMissing from new: {', '.join(sorted(missing_from_new))}"
                if missing_from_original:
                    warning_msg += f"\nNew questions not in original: {', '.join(sorted(missing_from_original))}"

                warnings.append(LaunchWarning(warning=warning_msg))
            return warnings

        # There are no glaring errors in the text of the question
        async def _no_inconsistencies_causing_annulment_gpt() -> (
            list[LaunchWarning]
        ):
            tasks: list[Coroutine[Any, Any, dict[str, Any]]] = []
            for question in new_questions:
                model = GeneralLlm(
                    model="openrouter/anthropic/claude-3.7-sonnet",  # NOSONAR
                    temperature=1,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 32000,
                    },
                    max_tokens=40000,
                    timeout=40,
                )
                prompt = clean_indents(
                    f"""
                    You are an expert proofreader for Metaculus and Good Judgement Open tasked with evaluating the quality of a forecasting question.
                    Please review the following question for internal inconsistencies or obvious errors that would result in annulment/ambiguous resolution.

                    Question Data:
                    ```json
                    {question.model_dump_json(indent=2)}
                    ```

                    Based on your review, provide a rating and a brief reason.

                    Rating criteria:
                    - "typo": There is a typo in the question.
                    - "explicit_contradiction": There is an explicit contradiction in the question. For example the question title says one month while the resolution criteria says another.
                    - "bad_timing": The timing of the question is bad. For example the events of interest will have already happened by the time the question opens.
                    - "other": There is a general inconsistency in the question.
                    - "good": The question is well-written, clear, and free of significant errors.

                    Remember:
                    - DO NOT give a bad rating for a field having the question is only open for a short time (this is a testing spot forecasting)
                    - DO NOT give a bad rating for a field having ".p". This will use a parent url to get the question.
                    - DO NOT give a bad rating for the close date being long before the resolution date. The close date is just when the forecasters stop forecasting.
                    - DO NOT give a bad rating for a question saying "as of <date in last 2 months> the state of <x> was <y>" even if the question will be published in the next week
                    - 90% of questions are good. If in doubt, say "good"
                    - Today is {datetime.now().strftime("%Y-%m-%d")} and these questions were created in the last 1-4 weeks

                    Respond ONLY with a JSON object in the following format:
                    {{
                        "rating": "good" | "typo" | "explicit_contradiction" | "bad_timing" | "other",
                        "reason": "A brief explanation for your rating, highlighting any specific issues found."
                    }}

                    Example Output for a bad question:
                    {{
                        "rating": "explicit_contradiction",
                        "reason": "The resolution criteria and question title contain different months of for when the question is resolved, which would result in annulment."
                    }}

                    Your JSON Response:
                    """
                )
                tasks.append(
                    model.invoke_and_return_verified_type(
                        prompt,
                        dict,
                    )
                )

            with MonetaryCostManager() as cost_manager:
                responses = await asyncio.gather(*tasks)
                cost = cost_manager.current_usage
            warnings = []
            warnings.append(
                LaunchWarning(
                    warning=f"Cost of checking question for inconsistencies: ${cost:.6f}",
                )
            )
            for question, response in zip(new_questions, responses):
                if (
                    response["rating"] == "typo"
                    or response["rating"] == "explicit_contradiction"
                ):
                    warnings.append(
                        LaunchWarning(
                            relevant_question=question,
                            warning=response["reason"],
                        )
                    )
            return warnings

        # Each weighted question is weighted in a group that is originally next to each other. Make sure only one from any group is in the final list.
        def _weighted_pairs_are_not_in_list() -> list[LaunchWarning]:
            warnings = []
            for new_question in new_questions:
                question_above_index = new_question.original_order + 1
                question_below_index = new_question.original_order - 1
                for question in original_questions:
                    if question.original_order == question_above_index:
                        original_question_above = question
                    if question.original_order == question_below_index:
                        original_question_below = question
                    if question.original_order == new_question.original_order:
                        original_question = question

                question_above_in_new_list = False
                question_below_in_new_list = False
                for question in new_questions:
                    if question.original_order == question_above_index:
                        question_above_in_new_list = True
                    if question.original_order == question_below_index:
                        question_below_in_new_list = True

                if original_question.question_weight != 1:
                    offending_question = None
                    if (
                        original_question_above.question_weight != 1
                        and question_above_in_new_list
                    ):
                        offending_question = original_question_above
                    if (
                        original_question_below.question_weight != 1
                        and question_below_in_new_list
                    ):
                        offending_question = original_question_below
                    if offending_question:
                        warnings.append(
                            LaunchWarning(
                                relevant_question=new_question,
                                warning=f"Weighted pair {original_question.title} and {offending_question.title} is in list",
                            )
                        )
            return warnings

        # Questions are evenly distributed per day
        def _questions_evenly_distributed_per_day() -> list[LaunchWarning]:
            duration_of_period = end_date - start_date
            target_questions_per_day = (
                len(new_questions) / duration_of_period.days
            )
            list_of_dates = [
                (start_date + timedelta(days=i)).date()
                for i in range(duration_of_period.days)
            ]
            question_count_per_day = {date: 0 for date in list_of_dates}
            for question in new_questions:
                if question.scheduled_close_time:
                    if (
                        question.scheduled_close_time.date()
                        not in question_count_per_day
                    ):
                        question_count_per_day[
                            question.scheduled_close_time.date()
                        ] = 0
                    question_count_per_day[
                        question.scheduled_close_time.date()
                    ] += 1
            warnings = []
            for date_key, count in question_count_per_day.items():
                if not (
                    math.floor(target_questions_per_day)
                    <= count
                    <= math.ceil(target_questions_per_day)
                ):
                    warnings.append(
                        LaunchWarning(
                            relevant_question=None,
                            warning=f"{count} questions scheduled for {date_key} != target of {target_questions_per_day}",
                        )
                    )
            return warnings

        def _check_resolve_date_is_in_future() -> list[LaunchWarning]:
            warnings = []
            for question in new_questions:
                if question.scheduled_resolve_time:
                    if question.scheduled_resolve_time < datetime.now():
                        warnings.append(
                            LaunchWarning(
                                relevant_question=question,
                                warning=f"Resolve date {question.scheduled_resolve_time} is in the past",
                            )
                        )
            return warnings

        final_warnings.extend(_check_existing_times_preserved(question_type))
        final_warnings.extend(_check_no_new_overlapping_windows())
        final_warnings.extend(_check_window_duration(question_type))
        final_warnings.extend(_check_unique_windows())
        final_warnings.extend(_check_required_fields())
        final_warnings.extend(_check_numeric_fields())
        final_warnings.extend(_check_mc_fields())
        final_warnings.extend(_check_no_field_changes(question_type))
        final_warnings.extend(_check_parent_url_fields())
        final_warnings.extend(_check_earliest_open_time(question_type))
        final_warnings.extend(
            _check_adherence_to_end_and_start_time(question_type)
        )
        final_warnings.extend(_check_bridgewater_tournament())
        final_warnings.extend(_check_numeric_range())
        final_warnings.extend(_check_duplicate_titles(use_stored_titles=True))
        final_warnings.extend(_check_question_type_distribution())
        final_warnings.extend(_check_average_weight())
        final_warnings.extend(_check_order_changed())
        final_warnings.extend(_check_ordered_by_open_time())
        final_warnings.extend(_check_resolve_date_is_in_future())
        if question_type == "bots":
            final_warnings.extend(_check_same_number_of_questions())
            final_warnings.extend(
                await _no_inconsistencies_causing_annulment_gpt()
            )
        if question_type == "pros":
            final_warnings.extend(_weighted_pairs_are_not_in_list())
            final_warnings.extend(_questions_evenly_distributed_per_day())
        return final_warnings

    @classmethod
    def _find_overlapping_windows(
        cls, questions: list[LaunchQuestion]
    ) -> list[tuple[LaunchQuestion, LaunchQuestion]]:
        time_periods = []
        overlapping_pairs = []

        # Collect all valid time periods
        for question in questions:
            if (
                question.open_time is not None
                and question.scheduled_close_time is not None
            ):
                time_periods.append(
                    (
                        question,
                        question.open_time,
                        question.scheduled_close_time,
                    )
                )

        # Check each pair of time periods
        for i, (q1, start1, end1) in enumerate(time_periods):
            for j, (q2, start2, end2) in enumerate(time_periods):
                if (
                    i >= j
                ):  # Skip comparing the same pair or pairs we've already checked
                    continue

                # Check if periods are exactly the same
                if start1 == start2 and end1 == end2:
                    continue

                # Check for overlap
                if cls._open_window_overlaps_for_questions(q1, q2):
                    overlapping_pairs.append((q1, q2))

        return overlapping_pairs

    @staticmethod
    def _open_window_overlaps_for_questions(
        question_1: LaunchQuestion, question_2: LaunchQuestion
    ) -> bool:
        if (
            question_1.open_time is None
            or question_1.scheduled_close_time is None
            or question_2.open_time is None
            or question_2.scheduled_close_time is None
        ):
            raise ValueError("Question has no open or close time")
        return (
            question_1.open_time < question_2.scheduled_close_time
            and question_2.open_time < question_1.scheduled_close_time
        )


if __name__ == "__main__":
    CustomLogger.setup_logging()

    start_date = SheetOrganizer.compute_upcoming_day("monday")
    # start_date = datetime(year=2025, month=6, day=2)
    end_date = SheetOrganizer.compute_upcoming_day("friday")
    logger.info(f"Start date: {start_date}, End date: {end_date}")
    asyncio.run(
        SheetOrganizer.schedule_questions_from_file(
            input_file_path="temp/input_launch_questions.csv",
            bot_output_file_path="temp/bot_questions.csv",
            pro_output_file_path="temp/pro_questions.csv",
            num_pro_questions=13,
            chance_to_skip_slot=0.8,
            start_date=start_date,
            end_date=end_date,
        )
    )
