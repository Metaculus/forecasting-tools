from __future__ import annotations

import csv
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

from forecasting_tools.util.jsonable import Jsonable

full_datetime_format = "%m/%d/%Y %H:%M:%S"
sheet_date_format1 = "%m/%d/%Y"
sheet_date_format2 = "%m/%d/%y"


class LaunchQuestion(BaseModel, Jsonable):
    parent_url: str
    author: str
    title: str
    type: Literal["binary", "numeric", "multiple_choice"]
    resolution_criteria: str
    fine_print: str
    description: str
    question_weight: int
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
    def parse_boolean_fields(cls, value: Any) -> bool | None:
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
    question: LaunchQuestion
    warning: str


class SheetOrganizer:

    @classmethod
    def load_questions_from_csv(cls, file_path: str) -> list[LaunchQuestion]:
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            questions = [
                LaunchQuestion.from_csv_row(row, i)
                for i, row in enumerate(reader)
            ]
        return questions

    @classmethod
    def save_questions_to_csv(
        cls, questions: list[LaunchQuestion], file_path: str
    ) -> None:
        with open(file_path, "w") as f:
            writer = csv.DictWriter(
                f, fieldnames=LaunchQuestion.model_fields.keys()
            )
            writer.writeheader()
            writer.writerows([question.to_csv_row() for question in questions])

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
                if start1 < end2 and start2 < end1:
                    overlapping_pairs.append((q1, q2))

        return overlapping_pairs

    @classmethod
    def find_processing_errors(
        cls,
        original_questions: list[LaunchQuestion],
        new_questions: list[LaunchQuestion],
        start_date: datetime,
        end_date: datetime,
        question_type: Literal["bots", "pros"] = "bots",
    ) -> list[LaunchWarning]:
        failed_tests = []
        # Some questions will already have a open and close time. These must be respected and stay the same
        # Questions that resolve earlier should open earlier
        # No overlapping windows except for questions originally with a open/close time
        # If bots, The open time must be 2hr before schedueld close time unless. If pros it must be 2 days
        # No open/close windows are exactly the same
        # For each question All fields exist
        # All date questions
        # All text fields other than fine print and parent url
        # If numeric should include range_min and range_max and upper and lower bounds
        # If MC should include options and group_variable
        # No fields changed between original and new question other than open/close time
        # If there is a parent url, then the description, resolution criteria, and fine print should be ".p"
        # The earliest open time is on start_date
        # open times are between start_date and the questions' resolve date
        # None of the questions have "Bridgewater" in tournament name if pros
        # All numeric questions have their range_min and range_max larger than 100 difference
        # None of the questions have duplicate titles
        # If bot questions
        # 16-24% of questions are numeric
        # 16-24% of questions are MC
        # 56-64% of questions are binary
        # The average question_weight is greater than 0.8
        # The original order is different than the new ordering
        # Questions are ordered by open time
        return failed_tests

    @classmethod
    def schedule_questions(
        cls, questions: list[LaunchQuestion], start_date: datetime
    ) -> list[LaunchQuestion]:
        # I want to make a script to automatically choose valid open times for a set of questions and order these questions in terms of open time. Here are the constraints:
        # - I need to launch 50 questions a week
        # - The process should error if a question can't find valid open time that is before its resolution date
        # - The function in SheetOrganizer should take in a file path to the csv and an output file path.
        # - For questions without a preexisting open time, you should pick the earliest 2hr slot in the week that does not already have a question. There should only be 1 question per 2hr slot
        return questions

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
