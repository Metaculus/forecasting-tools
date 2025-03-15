import csv
import os
import tempfile
from datetime import datetime
from typing import Any

import pytest

from forecasting_tools.util.launch_utils import LaunchQuestion, SheetOrganizer


class TestLaunchQuestion:
    def test_valid_datetime_parsing(self) -> None:
        """Test valid datetime parsing in LaunchQuestion."""
        question = LaunchQuestion(
            parent_url="https://example.com",
            author="Test Author",
            title="Test Question",
            type="binary",
            resolution_criteria="Test criteria",
            fine_print="Test fine print",
            description="Test description",
            question_weight="1",
            open_time="05/01/2023 10:00:00",
            scheduled_close_time="05/01/2023 12:00:00",
            scheduled_resolve_time="05/02/2023",
            range_min=None,
            range_max=None,
            zero_point=None,
            open_lower_bound=None,
            open_upper_bound=None,
            group_variable=None,
            options=None,
        )

        assert isinstance(question.open_time, datetime)
        assert isinstance(question.scheduled_close_time, datetime)
        assert question.open_time.day == 1
        assert question.open_time.month == 5
        assert question.open_time.year == 2023
        assert question.open_time.hour == 10
        assert question.scheduled_close_time.hour == 12

    def test_empty_optional_datetime(self) -> None:
        """Test handling of empty datetime fields."""
        question = LaunchQuestion(
            parent_url="https://example.com",
            author="Test Author",
            title="Test Question",
            type="binary",
            resolution_criteria="Test criteria",
            fine_print="Test fine print",
            description="Test description",
            question_weight="1",
            open_time=None,
            scheduled_close_time=None,
            scheduled_resolve_time="05/02/2023",
            range_min=None,
            range_max=None,
            zero_point=None,
            open_lower_bound=None,
            open_upper_bound=None,
            group_variable=None,
            options=None,
        )

        assert question.open_time is None
        assert question.scheduled_close_time is None

    def test_different_resolve_date_formats(self) -> None:
        """Test different formats for resolve date."""
        # Two-digit year format
        question1 = LaunchQuestion(
            parent_url="https://example.com",
            author="Test Author",
            title="Test Question",
            type="binary",
            resolution_criteria="Test criteria",
            fine_print="Test fine print",
            description="Test description",
            question_weight="1",
            open_time=None,
            scheduled_close_time=None,
            scheduled_resolve_time="05/02/23",
            range_min=None,
            range_max=None,
            zero_point=None,
            open_lower_bound=None,
            open_upper_bound=None,
            group_variable=None,
            options=None,
        )

        # Four-digit year format
        question2 = LaunchQuestion(
            parent_url="https://example.com",
            author="Test Author",
            title="Test Question",
            type="binary",
            resolution_criteria="Test criteria",
            fine_print="Test fine print",
            description="Test description",
            question_weight="1",
            open_time=None,
            scheduled_close_time=None,
            scheduled_resolve_time="05/02/2023",
            range_min=None,
            range_max=None,
            zero_point=None,
            open_lower_bound=None,
            open_upper_bound=None,
            group_variable=None,
            options=None,
        )

        assert isinstance(question1.scheduled_resolve_time, datetime)
        assert question1.scheduled_resolve_time.day == 2
        assert question1.scheduled_resolve_time.month == 5
        assert question1.scheduled_resolve_time.year == 2023
        assert isinstance(question2.scheduled_resolve_time, datetime)
        assert question2.scheduled_resolve_time.day == 2
        assert question2.scheduled_resolve_time.month == 5
        assert question2.scheduled_resolve_time.year == 2023

    def test_invalid_resolve_date_format(self) -> None:
        """Test invalid resolve date format."""
        with pytest.raises(ValueError):
            LaunchQuestion(
                parent_url="https://example.com",
                author="Test Author",
                title="Test Question",
                type="binary",
                resolution_criteria="Test criteria",
                fine_print="Test fine print",
                description="Test description",
                question_weight="1",
                open_time=None,
                scheduled_close_time=None,
                scheduled_resolve_time="2023-05-02",  # Invalid format
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            )

    def test_time_validation_only_one_provided(self) -> None:
        """Test validation when only one time is provided."""
        with pytest.raises(ValueError):
            LaunchQuestion(
                parent_url="https://example.com",
                author="Test Author",
                title="Test Question",
                type="binary",
                resolution_criteria="Test criteria",
                fine_print="Test fine print",
                description="Test description",
                question_weight="1",
                open_time="05/01/2023 10:00:00",
                scheduled_close_time=None,  # Only one provided
                scheduled_resolve_time="05/02/2023",
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            )

    def test_time_validation_not_two_hours_apart(self) -> None:
        """Test validation when times are not exactly 2 hours apart."""
        with pytest.raises(ValueError):
            LaunchQuestion(
                parent_url="https://example.com",
                author="Test Author",
                title="Test Question",
                type="binary",
                resolution_criteria="Test criteria",
                fine_print="Test fine print",
                description="Test description",
                question_weight="1",
                open_time="05/01/2023 10:00:00",
                scheduled_close_time="05/01/2023 13:00:00",  # 3 hours apart
                scheduled_resolve_time="05/02/2023",
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            )

    def test_close_time_not_before_resolve_time(self) -> None:
        """Test validation when close time is not before resolve time."""
        with pytest.raises(ValueError):
            LaunchQuestion(
                parent_url="https://example.com",
                author="Test Author",
                title="Test Question",
                type="binary",
                resolution_criteria="Test criteria",
                fine_print="Test fine print",
                description="Test description",
                question_weight="1",
                open_time="05/02/2023 10:00:00",
                scheduled_close_time="05/02/2023 12:00:00",  # Same day as resolve
                scheduled_resolve_time="05/02/2023",
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            )

    def test_open_time_after_close_time(self) -> None:
        """Test validation when open time is after close time."""
        # This should be caught by the 2-hour check, but testing specific error case
        with pytest.raises(ValueError):
            LaunchQuestion(
                parent_url="https://example.com",
                author="Test Author",
                title="Test Question",
                type="binary",
                resolution_criteria="Test criteria",
                fine_print="Test fine print",
                description="Test description",
                question_weight="1",
                open_time="05/02/2023 13:00:00",  # After close time
                scheduled_close_time="05/01/2023 12:00:00",
                scheduled_resolve_time="05/03/2023",
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            )

    def test_to_csv_row(self) -> None:
        """Test to_csv_row method returns correctly formatted dict."""
        question = LaunchQuestion(
            parent_url="https://example.com",
            author="Test Author",
            title="Test Question",
            type="binary",
            resolution_criteria="Test criteria",
            fine_print="Test fine print",
            description="Test description",
            question_weight="1",
            open_time="05/01/2023 10:00:00",
            scheduled_close_time="05/01/2023 12:00:00",
            scheduled_resolve_time="05/02/2023",
            range_min=None,
            range_max=None,
            zero_point=None,
            open_lower_bound=None,
            open_upper_bound=None,
            group_variable=None,
            options=None,
        )

        csv_row = question.to_csv_row()
        assert csv_row["parent_url"] == "https://example.com"
        assert csv_row["title"] == "Test Question"
        assert csv_row["open_time"] == "05/01/2023 10:00:00"
        assert csv_row["scheduled_resolve_time"] == "05/02/2023"

    def test_from_csv_row(self) -> None:
        """Test from_csv_row creates valid LaunchQuestion."""
        row = {
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

        question = LaunchQuestion.from_csv_row(row, 5)
        assert question.title == "Test Question"
        assert question.original_order == 5
        assert isinstance(question.open_time, datetime)


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
        test_data = [
            {
                "parent_url": "https://example.com/1",
                "author": "Author 1",
                "title": "Question 1",
                "type": "binary",
                "resolution_criteria": "Criteria 1",
                "fine_print": "Fine print 1",
                "description": "Description 1",
                "question_weight": "1",
                "open_time": "05/01/2023 10:00:00",
                "scheduled_close_time": "05/01/2023 12:00:00",
                "scheduled_resolve_time": "05/02/2023",
                "range_min": "",
                "range_max": "",
                "zero_point": "",
                "open_lower_bound": "",
                "open_upper_bound": "",
                "group_variable": "",
                "options": "",
            },
            {
                "parent_url": "https://example.com/2",
                "author": "Author 2",
                "title": "Question 2",
                "type": "binary",
                "resolution_criteria": "Criteria 2",
                "fine_print": "Fine print 2",
                "description": "Description 2",
                "question_weight": "1",
                "open_time": "05/03/2023 10:00:00",
                "scheduled_close_time": "05/03/2023 12:00:00",
                "scheduled_resolve_time": "05/04/2023",
                "range_min": "",
                "range_max": "",
                "zero_point": "",
                "open_lower_bound": "",
                "open_upper_bound": "",
                "group_variable": "",
                "options": "",
            },
        ]

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
        questions = [
            LaunchQuestion(
                parent_url="https://example.com/1",
                author="Author 1",
                title="Question 1",
                type="binary",
                resolution_criteria="Criteria 1",
                fine_print="Fine print 1",
                description="Description 1",
                question_weight="1",
                open_time="05/01/2023 10:00:00",
                scheduled_close_time="05/01/2023 12:00:00",
                scheduled_resolve_time="05/02/2023",
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            ),
            LaunchQuestion(
                parent_url="https://example.com/2",
                author="Author 2",
                title="Question 2",
                type="binary",
                resolution_criteria="Criteria 2",
                fine_print="Fine print 2",
                description="Description 2",
                question_weight="1",
                open_time="05/01/2023 13:00:00",  # After Q1 closes
                scheduled_close_time="05/01/2023 15:00:00",
                scheduled_resolve_time="05/02/2023",
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            ),
        ]

        overlapping = SheetOrganizer.find_overlapping_windows(questions)
        assert len(overlapping) == 0

    def test_find_overlapping_windows(self) -> None:
        """Test finding overlapping time windows."""
        questions = [
            LaunchQuestion(
                parent_url="https://example.com/1",
                author="Author 1",
                title="Question 1",
                type="binary",
                resolution_criteria="Criteria 1",
                fine_print="Fine print 1",
                description="Description 1",
                question_weight="1",
                open_time="05/01/2023 10:00:00",
                scheduled_close_time="05/01/2023 12:00:00",
                scheduled_resolve_time="05/02/2023",
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            ),
            LaunchQuestion(
                parent_url="https://example.com/2",
                author="Author 2",
                title="Question 2",
                type="binary",
                resolution_criteria="Criteria 2",
                fine_print="Fine print 2",
                description="Description 2",
                question_weight="1",
                open_time="05/01/2023 11:00:00",  # Overlaps with Q1
                scheduled_close_time="05/01/2023 13:00:00",
                scheduled_resolve_time="05/02/2023",
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            ),
            LaunchQuestion(
                parent_url="https://example.com/3",
                author="Author 3",
                title="Question 3",
                type="binary",
                resolution_criteria="Criteria 3",
                fine_print="Fine print 3",
                description="Description 3",
                question_weight="1",
                open_time="05/01/2023 14:00:00",  # No overlap
                scheduled_close_time="05/01/2023 16:00:00",
                scheduled_resolve_time="05/02/2023",
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            ),
        ]

        overlapping = SheetOrganizer.find_overlapping_windows(questions)
        assert len(overlapping) == 1
        assert overlapping[0][0].title == "Question 1"
        assert overlapping[0][1].title == "Question 2"

    def test_identical_time_windows_not_overlapping(self) -> None:
        """Test that identical time windows are not considered overlapping."""
        same_open = "05/01/2023 10:00:00"
        same_close = "05/01/2023 12:00:00"

        questions = [
            LaunchQuestion(
                parent_url="https://example.com/1",
                author="Author 1",
                title="Question 1",
                type="binary",
                resolution_criteria="Criteria 1",
                fine_print="Fine print 1",
                description="Description 1",
                question_weight="1",
                open_time=same_open,
                scheduled_close_time=same_close,
                scheduled_resolve_time="05/02/2023",
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            ),
            LaunchQuestion(
                parent_url="https://example.com/2",
                author="Author 2",
                title="Question 2",
                type="binary",
                resolution_criteria="Criteria 2",
                fine_print="Fine print 2",
                description="Description 2",
                question_weight="1",
                open_time=same_open,  # Same times as Q1
                scheduled_close_time=same_close,
                scheduled_resolve_time="05/02/2023",
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            ),
        ]

        overlapping = SheetOrganizer.find_overlapping_windows(questions)
        assert len(overlapping) == 0

    def test_questions_without_time_windows(self) -> None:
        """Test handling questions without time windows."""
        questions = [
            LaunchQuestion(
                parent_url="https://example.com/1",
                author="Author 1",
                title="Question 1",
                type="binary",
                resolution_criteria="Criteria 1",
                fine_print="Fine print 1",
                description="Description 1",
                question_weight="1",
                open_time=None,  # No time window
                scheduled_close_time=None,
                scheduled_resolve_time="05/02/2023",
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            ),
            LaunchQuestion(
                parent_url="https://example.com/2",
                author="Author 2",
                title="Question 2",
                type="binary",
                resolution_criteria="Criteria 2",
                fine_print="Fine print 2",
                description="Description 2",
                question_weight="1",
                open_time="05/01/2023 11:00:00",  # Has time window
                scheduled_close_time="05/01/2023 13:00:00",
                scheduled_resolve_time="05/02/2023",
                range_min=None,
                range_max=None,
                zero_point=None,
                open_lower_bound=None,
                open_upper_bound=None,
                group_variable=None,
                options=None,
            ),
        ]

        overlapping = SheetOrganizer.find_overlapping_windows(questions)
        assert len(overlapping) == 0  # No overlap since one has no time window
