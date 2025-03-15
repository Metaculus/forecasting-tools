from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from typing import Any

from pydantic import BaseModel, model_validator

from forecasting_tools.util.jsonable import Jsonable

sheet_datetime_format = "%m/%d/%Y %H:%M:%S"
sheet_date_format = "%m/%d/%Y"


class LaunchQuestion(BaseModel, Jsonable):
    parent_url: str
    author: str
    title: str
    type: str
    resolution_criteria: str
    fine_print: str
    description: str
    question_weight: str
    open_time: datetime | None = None
    scheduled_close_time: datetime | None = None
    scheduled_resolve_time: date
    range_min: int | None = None
    range_max: int | None = None
    zero_point: int | None = None
    open_lower_bound: int | None = None
    open_upper_bound: int | None = None
    group_variable: str | None = None
    options: list[str] | None = None
    original_order: int = 0

    model_config = {
        "json_encoders": {
            datetime: lambda dt: (
                dt.strftime(sheet_datetime_format) if dt else None
            ),
            date: lambda d: d.strftime(sheet_date_format) if d else None,
        }
    }

    @model_validator(mode="before")
    @classmethod
    def parse_dates(cls, data: dict[str, Any]) -> dict[str, Any]:
        # Parse open_time
        if isinstance(data.get("open_time"), str):
            value = data["open_time"].strip()
            if value and value != "None":
                data["open_time"] = datetime.strptime(
                    value, sheet_datetime_format
                )
            else:
                data["open_time"] = None

        # Parse scheduled_close_time
        if isinstance(data.get("scheduled_close_time"), str):
            value = data["scheduled_close_time"].strip()
            if value and value != "None":
                data["scheduled_close_time"] = datetime.strptime(
                    value, sheet_datetime_format
                )
            else:
                data["scheduled_close_time"] = None

        # Parse scheduled_resolve_time
        if isinstance(data.get("scheduled_resolve_time"), str):
            value = str(data["scheduled_resolve_time"]).strip()
            if value and value != "None":
                data["scheduled_resolve_time"] = datetime.strptime(
                    value, sheet_date_format
                ).date()
            else:
                data["scheduled_resolve_time"] = None

        return data

    @model_validator(mode="after")
    def validate_times(self: LaunchQuestion) -> LaunchQuestion:
        open_time = self.open_time
        close_time = self.scheduled_close_time
        resolve_date = self.scheduled_resolve_time

        if (open_time is None) != (close_time is None):
            raise ValueError(
                "Both open_time and scheduled_close_time must be provided together."
            )

        if open_time is not None and close_time is not None:
            if close_time != open_time + timedelta(hours=2):
                raise ValueError(
                    "open_time and scheduled_close_time are not exactly two hours apart."
                )
            if close_time.date() >= resolve_date:
                raise ValueError(
                    "scheduled_close_time must be before scheduled_resolve_time."
                )
            if open_time.date() > close_time.date():
                raise ValueError(
                    "open_time must be before scheduled_close_time."
                )

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
                self.open_time.strftime(sheet_datetime_format)
                if self.open_time
                else ""
            ),
            "scheduled_close_time": (
                self.scheduled_close_time.strftime(sheet_datetime_format)
                if self.scheduled_close_time
                else ""
            ),
            "scheduled_resolve_time": self.scheduled_resolve_time.strftime(
                "%m/%d/%Y"
            ),
            "range_min": self.range_min,
            "range_max": self.range_max,
            "zero_point": self.zero_point,
            "open_lower_bound": self.open_lower_bound,
            "open_upper_bound": self.open_upper_bound,
            "group_variable": self.group_variable,
            "options": self.options,
        }

    @classmethod
    def from_csv_row(cls, row: dict, original_order: int) -> LaunchQuestion:
        row["original_order"] = original_order
        return cls(**row)


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
