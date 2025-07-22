import csv
import logging
import os
import tempfile
from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest

from forecasting_tools.util.launch_utils import LaunchQuestion, SheetOrganizer

logger = logging.getLogger(__name__)


def get_template_question() -> dict[str, Any]:
    return {
        "parent_url": "https://example.com",
        "author": "Test Author",
        "title": "Test Question",
        "type": "binary",
        "resolution_criteria": "Test criteria",
        "fine_print": "Test fine print",
        "description": "Test description",
        "question_weight": "1",
        "open_time": "05/01/2023 10:00:00",
        "scheduled_close_time": "05/01/2023 12:00:00",
        "scheduled_resolve_time": "05/02/2023",
        "range_min": None,
        "range_max": None,
        "zero_point": None,
        "open_lower_bound": None,
        "open_upper_bound": None,
        "group_variable": None,
        "options": None,
    }


class TestLaunchQuestion:

    def test_valid_datetime_parsing(self) -> None:
        """Test valid datetime parsing in LaunchQuestion."""
        question = get_template_question()
        question["open_time"] = "05/01/2023 10:00:00"
        question["scheduled_close_time"] = "05/01/2023 12:00:00"
        question["scheduled_resolve_time"] = "05/02/2023"
        question = LaunchQuestion(**question)

        assert isinstance(question.open_time, datetime)
        assert isinstance(question.scheduled_close_time, datetime)
        assert question.open_time.day == 1
        assert question.open_time.month == 5
        assert question.open_time.year == 2023
        assert question.open_time.hour == 10
        assert question.scheduled_close_time.hour == 12

    def test_empty_dates(self) -> None:
        """Test handling of empty datetime fields."""
        question = get_template_question()
        question["open_time"] = ""
        question["scheduled_close_time"] = ""
        question["scheduled_resolve_time"] = ""
        question = LaunchQuestion(**question)

        assert question.open_time is None
        assert question.scheduled_close_time is None
        assert question.scheduled_resolve_time is None

    def test_different_resolve_date_formats(self) -> None:
        """Test different formats for resolve date."""
        # Two-digit year format
        question1 = get_template_question()
        question1["scheduled_resolve_time"] = "05/02/2023"
        question1 = LaunchQuestion(**question1)

        # Four-digit year format
        question2 = get_template_question()
        question2["scheduled_resolve_time"] = "05/02/23"
        question2 = LaunchQuestion(**question2)

        assert isinstance(question1.scheduled_resolve_time, datetime)
        assert question1.scheduled_resolve_time.day == 2
        assert question1.scheduled_resolve_time.month == 5
        assert question1.scheduled_resolve_time.year == 2023
        assert isinstance(question2.scheduled_resolve_time, datetime)
        assert question2.scheduled_resolve_time.day == 2
        assert question2.scheduled_resolve_time.month == 5
        assert question2.scheduled_resolve_time.year == 2023

    def test_close_open_resolve_times_in_order(self) -> None:
        """Test validation when close time is not before resolve time."""
        with pytest.raises(ValueError):
            question1 = get_template_question()
            question1["open_time"] = "05/02/2023 10:00:00"
            question1["scheduled_close_time"] = "05/02/2023 8:00:00"
            LaunchQuestion(**question1)

        with pytest.raises(ValueError):
            question2 = get_template_question()
            question2["open_time"] = "05/02/2023 10:00:00"
            question2["scheduled_resolve_time"] = "05/02/2023 8:00:00"
            LaunchQuestion(**question2)

        with pytest.raises(ValueError):
            question3 = get_template_question()
            question3["scheduled_close_time"] = "05/02/2023 10:00:00"
            question3["scheduled_resolve_time"] = "05/02/2023 8:00:00"
            LaunchQuestion(**question3)

    def test_from_csv_row(self) -> None:
        """Test from_csv_row creates valid LaunchQuestion."""
        question = LaunchQuestion.from_csv_row(get_template_question(), 5)
        assert question.original_order == 5
        assert isinstance(question.open_time, datetime)

    def test_numeric_question(self) -> None:
        question1 = get_template_question()
        question1["type"] = "numeric"
        question1["range_min"] = "1"
        question1["range_max"] = 10
        question1["zero_point"] = ""
        question1["open_lower_bound"] = "TRUE"
        question1["open_upper_bound"] = "FALSE"

        question1 = LaunchQuestion(**question1)
        assert question1.type == "numeric"
        assert question1.range_min == 1
        assert question1.range_max == 10
        assert question1.zero_point is None
        assert question1.open_lower_bound is True
        assert question1.open_upper_bound is False

        question2 = get_template_question()
        question2["type"] = "numeric"
        question2["range_min"] = 1
        question2["range_max"] = ""
        question2["zero_point"] = 0.3
        question2["open_lower_bound"] = ""
        question2["open_upper_bound"] = "TRUE"

        question2 = LaunchQuestion(**question2)
        assert question2.type == "numeric"
        assert question2.range_min == 1
        assert question2.range_max == None
        assert question2.zero_point == pytest.approx(0.3)
        assert question2.open_lower_bound is None
        assert question2.open_upper_bound is True

    def test_multiple_choice_question(self) -> None:
        question = get_template_question()
        question["type"] = "multiple_choice"
        question["options"] = "Option 1|Option 2|Option 3"
        question = LaunchQuestion(**question)
        assert question.type == "multiple_choice"
        assert question.options == ["Option 1", "Option 2", "Option 3"]


class TestSheetOrganizer:
    def create_temp_csv(self, rows: list[dict[str, Any]]) -> str:
        """Helper to create a temporary CSV file with test data."""
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", newline="") as f:
            if not rows:
                return path

            fieldnames = rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path

    def test_load_questions_from_csv(self) -> None:
        """Test loading questions from a CSV file."""
        template_question = get_template_question()
        question1 = template_question.copy()
        question1["title"] = "Question 1"
        question2 = template_question.copy()
        question2["title"] = "Question 2"
        test_data = [question1, question2]

        temp_path = self.create_temp_csv(test_data)
        try:
            questions = SheetOrganizer.load_questions_from_csv(temp_path)
            assert len(questions) == 2
            assert questions[0].title == "Question 1"
            assert questions[1].title == "Question 2"
            assert questions[0].original_order == 0
            assert questions[1].original_order == 1
        finally:
            os.unlink(temp_path)

    def test_load_empty_csv(self) -> None:
        """Test loading from an empty CSV."""
        temp_path = self.create_temp_csv([])
        try:
            questions = SheetOrganizer.load_questions_from_csv(temp_path)
            assert len(questions) == 0
        finally:
            os.unlink(temp_path)

    def test_find_no_overlapping_windows(self) -> None:
        """Test finding overlapping windows when none exist."""
        question1 = get_template_question()
        question1["open_time"] = "05/01/2023 10:00:00"
        question1["scheduled_close_time"] = "05/01/2023 12:00:00"

        question2 = get_template_question()
        question2["open_time"] = "05/01/2023 13:00:00"
        question2["scheduled_close_time"] = "05/01/2023 15:00:00"

        questions = [
            LaunchQuestion(**question1),
            LaunchQuestion(**question2),
        ]

        overlapping = SheetOrganizer._find_overlapping_windows(questions)
        assert len(overlapping) == 0

    def test_find_overlapping_windows(self) -> None:
        """Test finding overlapping time windows."""
        question1 = get_template_question()
        question1["title"] = "Question 1"
        question1["open_time"] = "05/01/2023 10:00:00"
        question1["scheduled_close_time"] = "05/01/2023 12:00:00"

        question2 = get_template_question()
        question2["title"] = "Question 2"
        question2["open_time"] = "05/01/2023 11:00:00"
        question2["scheduled_close_time"] = "05/01/2023 13:00:00"

        question3 = get_template_question()
        question3["title"] = "Question 3"
        question3["open_time"] = "05/01/2023 14:00:00"
        question3["scheduled_close_time"] = "05/01/2023 16:00:00"

        questions = [
            LaunchQuestion(**question1),
            LaunchQuestion(**question2),
            LaunchQuestion(**question3),
        ]

        overlapping = SheetOrganizer._find_overlapping_windows(questions)
        assert len(overlapping) == 1
        assert overlapping[0][0].title == "Question 1"
        assert overlapping[0][1].title == "Question 2"

    def test_identical_time_windows_not_overlapping(self) -> None:
        """Test that identical time windows are not considered overlapping."""
        same_open = "05/01/2023 10:00:00"
        same_close = "05/01/2023 12:00:00"

        question1 = get_template_question()
        question1["open_time"] = same_open
        question1["scheduled_close_time"] = same_close

        question2 = get_template_question()
        question2["open_time"] = same_open
        question2["scheduled_close_time"] = same_close

        questions = [
            LaunchQuestion(**question1),
            LaunchQuestion(**question2),
        ]

        overlapping = SheetOrganizer._find_overlapping_windows(questions)
        assert len(overlapping) == 0

    def test_questions_without_time_windows(self) -> None:
        """Test handling questions without time windows."""
        question1 = get_template_question()
        question1["open_time"] = ""
        question1["scheduled_close_time"] = ""

        question2 = get_template_question()
        question2["open_time"] = "05/01/2023 11:00:00"
        question2["scheduled_close_time"] = "05/01/2023 13:00:00"

        questions = [
            LaunchQuestion(**question1),
            LaunchQuestion(**question2),
        ]

        overlapping = SheetOrganizer._find_overlapping_windows(questions)
        assert len(overlapping) == 0

    def test_day_of_week_computation(self) -> None:
        """Test computing upcoming days with mocked current dates."""
        from datetime import datetime

        friday = datetime(2025, 3, 14, 10, 30, 0)
        with patch("forecasting_tools.util.launch_utils.datetime") as mock_datetime:
            mock_datetime.now.return_value = friday
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            monday = SheetOrganizer.compute_upcoming_day("monday")
            assert monday.day == 17
            assert monday.month == 3
            assert monday.year == 2025

            friday = SheetOrganizer.compute_upcoming_day("friday")
            assert friday.day == 21
            assert friday.month == 3
            assert friday.year == 2025

            saturday = SheetOrganizer.compute_upcoming_day("saturday")
            assert saturday.day == 15
            assert saturday.month == 3
            assert saturday.year == 2025

        sunday = datetime(2025, 3, 16, 10, 30, 0)
        with patch("forecasting_tools.util.launch_utils.datetime") as mock_datetime:
            mock_datetime.now.return_value = sunday
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            monday = SheetOrganizer.compute_upcoming_day("monday")
            assert monday.day == 17
            assert monday.month == 3
            assert monday.year == 2025

            friday = SheetOrganizer.compute_upcoming_day("friday")
            assert friday.day == 21
            assert friday.month == 3
            assert friday.year == 2025

            saturday = SheetOrganizer.compute_upcoming_day("saturday")
            assert saturday.day == 22
            assert saturday.month == 3
            assert saturday.year == 2025
