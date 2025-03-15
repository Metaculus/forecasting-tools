from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

from forecasting_tools.util.file_manipulation import (
    load_csv_file,
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
    range_min: int | None = None
    range_max: int | None = None
    zero_point: int | float | None = None
    open_lower_bound: bool | None = None
    open_upper_bound: bool | None = None
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
            "group_variable": self.group_variable,
            "options": "|".join(self.options) if self.options else "",
        }

    @classmethod
    def from_csv_row(cls, row: dict, original_order: int) -> LaunchQuestion:
        row["original_order"] = original_order
        return cls(**row)


class LaunchWarning(BaseModel, Jsonable):
    warning: str
    relevant_question: LaunchQuestion | None = None


class SheetOrganizer:

    @classmethod
    def load_questions_from_csv(cls, file_path: str) -> list[LaunchQuestion]:
        questions = load_csv_file(file_path)
        return [
            LaunchQuestion.from_csv_row(row, i)
            for i, row in enumerate(questions)
        ]

    @classmethod
    def save_questions_to_csv(
        cls, questions: list[LaunchQuestion], file_path: str
    ) -> None:
        write_csv_file(
            file_path, [question.to_csv_row() for question in questions]
        )

    @classmethod
    def find_overlapping_windows(
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

    @classmethod
    def find_processing_errors(  # NOSONAR
        cls,
        original_questions: list[LaunchQuestion],
        new_questions: list[LaunchQuestion],
        start_date: datetime,
        end_date: datetime,
        question_type: Literal["bots", "pros"],
    ) -> list[LaunchWarning]:
        final_warnings = []

        # Some questions will already have a open and close time. These must be respected and stay the same
        def _check_existing_times_preserved() -> list[LaunchWarning]:
            warnings = []
            for orig_q in original_questions:
                if (
                    orig_q.open_time is not None
                    and orig_q.scheduled_close_time is not None
                ):
                    for new_q in new_questions:
                        if new_q.title == orig_q.title and (
                            new_q.open_time != orig_q.open_time
                            or new_q.scheduled_close_time
                            != orig_q.scheduled_close_time
                        ):
                            warnings.append(
                                LaunchWarning(
                                    relevant_question=new_q,
                                    warning=f"Existing open/close times must be preserved. Original: {orig_q.open_time} to {orig_q.scheduled_close_time}",
                                )
                            )
            return warnings

        # No overlapping windows except for questions originally with a open/close time
        def _check_no_new_overlapping_windows() -> list[LaunchWarning]:
            warnings = []
            overlapping_pairs = cls.find_overlapping_windows(new_questions)

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
        def _check_window_duration() -> list[LaunchWarning]:
            warnings = []
            required_duration = (
                timedelta(hours=2)
                if question_type == "bots"
                else timedelta(days=2)
            )

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
                        question.question_weight
                    ), "question_weight is required"
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
                    except AssertionError as e:
                        warnings.append(
                            LaunchWarning(
                                relevant_question=question,
                                warning=f"Missing at least one numeric field. Error: {e}",
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
        def _check_no_field_changes() -> list[LaunchWarning]:
            warnings = []

            for orig_q in original_questions:
                for new_q in new_questions:
                    if orig_q.title == new_q.title:
                        for field in LaunchQuestion.model_fields.keys():
                            if field not in [
                                "open_time",
                                "scheduled_close_time",
                            ]:
                                if getattr(orig_q, field) != getattr(
                                    new_q, field
                                ):
                                    warnings.append(
                                        LaunchWarning(
                                            relevant_question=new_q,
                                            warning=f"Field {field} was changed",
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
        def _check_earliest_open_time() -> list[LaunchWarning]:
            warnings = []

            earliest_open = min(
                (
                    q.open_time
                    for q in new_questions
                    if q.open_time is not None
                ),
                default=None,
            )
            if earliest_open and earliest_open.date() != start_date.date():
                # Find the question with the earliest open time
                earliest_question = next(
                    q for q in new_questions if q.open_time == earliest_open
                )
                warnings.append(
                    LaunchWarning(
                        relevant_question=earliest_question,
                        warning=f"Earliest open time should be on {start_date.date()}, not {earliest_open.date()}",
                    )
                )
            return warnings

        # open times are between start_date and the questions' resolve date
        def _check_open_time_bounds() -> list[LaunchWarning]:
            warnings = []

            for question in new_questions:
                if question.open_time and (
                    question.open_time < start_date
                    or (
                        question.scheduled_resolve_time
                        and question.open_time
                        > question.scheduled_resolve_time
                    )
                ):
                    warnings.append(
                        LaunchWarning(
                            relevant_question=question,
                            warning=f"Open time {question.open_time} is before start date {start_date} or after resolve time {question.scheduled_resolve_time}",
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
        def _check_duplicate_titles() -> list[LaunchWarning]:
            warnings = []
            title_count = {}

            for question in new_questions:
                title_count[question.title] = (
                    title_count.get(question.title, 0) + 1
                )

            for question in new_questions:
                if title_count[question.title] > 1:
                    warnings.append(
                        LaunchWarning(
                            relevant_question=question,
                            warning=f"Duplicate title found: {question.title}",
                        )
                    )
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

        final_warnings.extend(_check_existing_times_preserved())
        final_warnings.extend(_check_no_new_overlapping_windows())
        final_warnings.extend(_check_window_duration())
        final_warnings.extend(_check_unique_windows())
        final_warnings.extend(_check_required_fields())
        final_warnings.extend(_check_numeric_fields())
        final_warnings.extend(_check_mc_fields())
        final_warnings.extend(_check_no_field_changes())
        final_warnings.extend(_check_parent_url_fields())
        final_warnings.extend(_check_earliest_open_time())
        final_warnings.extend(_check_open_time_bounds())
        final_warnings.extend(_check_bridgewater_tournament())
        final_warnings.extend(_check_numeric_range())
        final_warnings.extend(_check_duplicate_titles())
        final_warnings.extend(_check_question_type_distribution())
        final_warnings.extend(_check_average_weight())
        final_warnings.extend(_check_order_changed())
        final_warnings.extend(_check_ordered_by_open_time())
        final_warnings.extend(_check_same_number_of_questions())
        return final_warnings

    @classmethod
    def schedule_questions(
        cls, questions: list[LaunchQuestion], start_date: datetime
    ) -> list[LaunchQuestion]:
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
        questions_to_schedule.sort(
            key=lambda q: (
                q.scheduled_resolve_time
                if q.scheduled_resolve_time
                else datetime.max
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
                key=lambda q: q.open_time if q.open_time else datetime.max
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
            key=lambda q: q.open_time if q.open_time else datetime.max
        )

        return all_questions

    @classmethod
    def schedule_questions_from_file(
        cls,
        input_file_path: str,
        output_file_path: str,
        start_date: datetime,
        end_date: datetime,
        question_type: Literal["bots", "pros"],
    ) -> None:
        questions = cls.load_questions_from_csv(input_file_path)
        scheduled_questions = cls.schedule_questions(questions, start_date)
        cls.save_questions_to_csv(scheduled_questions, output_file_path)
        warnings = cls.find_processing_errors(
            questions,
            scheduled_questions,
            start_date,
            end_date,
            question_type,
        )
        for warning in warnings:
            logger.warning(warning)

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


if __name__ == "__main__":
    from forecasting_tools.util.custom_logger import CustomLogger

    CustomLogger.setup_logging()

    start_date = SheetOrganizer.compute_upcoming_day("monday")
    end_date = SheetOrganizer.compute_upcoming_day("friday")
    logger.info(f"Start date: {start_date}, End date: {end_date}")
    SheetOrganizer.schedule_questions_from_file(
        "temp/input_launch_questions.csv",
        "temp/ordered_questions.csv",
        start_date,
        end_date,
        "bots",
    )
