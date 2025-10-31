import logging
import os

from code_tests.utilities_for_tests.misc_utils import replace_tzinfo_in_string
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.questions import (
    DateQuestion,
    DiscreteQuestion,
    MetaculusQuestion,
    NumericQuestion,
)

logger = logging.getLogger(__name__)


def test_metaculus_question_is_jsonable() -> None:
    temp_writing_path = "temp/temp_metaculus_question.json"
    read_report_path = "code_tests/unit_tests/test_data_models/forecasting_test_data/metaculus_questions.json"
    questions = DataOrganizer.load_questions_from_file_path(read_report_path)

    _assert_correct_number_of_questions(questions)

    DataOrganizer.save_questions_to_file_path(questions, temp_writing_path)
    questions_2 = DataOrganizer.load_questions_from_file_path(temp_writing_path)
    assert len(questions) == len(questions_2)
    for question, question_2 in zip(questions, questions_2):
        assert question.question_text == question_2.question_text
        assert question.id_of_post == question_2.id_of_post
        assert question.state == question_2.state

        _assert_tzinfo_is_not_none(question)
        _assert_tzinfo_is_not_none(question_2)

        logger.info(
            f"\nQuestion 1 string: {str(question)}\nQuestion 2 string: {str(question_2)}"
        )

        assert replace_tzinfo_in_string(str(question)) == replace_tzinfo_in_string(
            str(question_2)
        ), "The questions are not identical in all the necessary ways"

    _assert_correct_number_of_questions(questions_2)
    os.remove(temp_writing_path)


def _assert_tzinfo_is_not_none(question: MetaculusQuestion) -> None:
    assert question.date_accessed.tzinfo is not None
    if question.open_time is not None:
        assert question.open_time.tzinfo is not None
    if question.close_time is not None:
        assert question.close_time.tzinfo is not None
    if question.scheduled_resolution_time is not None:
        assert question.scheduled_resolution_time.tzinfo is not None


def _assert_correct_number_of_questions(questions: list[MetaculusQuestion]) -> None:
    for question_type in DataOrganizer.get_all_question_types():
        questions_of_type = [
            question for question in questions if type(question) == question_type
        ]
        if question_type == DateQuestion:
            assert (
                len(questions_of_type) == 2
            ), f"Expected 2 {question_type.__name__} questions, got {len(questions_of_type)}"
        else:
            assert (
                len(questions_of_type) > 0
            ), f"Expected > 0 {question_type.__name__} questions, got {len(questions_of_type)}"

        for question in questions_of_type:
            api_type_name = question_type.get_api_type_name()  # type: ignore
            assert question.get_question_type() == api_type_name
            assert question.question_type == api_type_name  # type: ignore

            if type(question) is NumericQuestion:
                assert question.cdf_size == 201
            elif type(question) is DiscreteQuestion:
                assert question.cdf_size is not None
                assert question.cdf_size < 201
