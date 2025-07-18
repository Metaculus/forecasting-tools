import os

from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)


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
        assert str(question) == str(question_2)

    _assert_correct_number_of_questions(questions_2)
    os.remove(temp_writing_path)


def _assert_correct_number_of_questions(questions: list[MetaculusQuestion]) -> None:
    numeric_questions = [
        question for question in questions if isinstance(question, NumericQuestion)
    ]
    binary_questions = [
        question for question in questions if isinstance(question, BinaryQuestion)
    ]
    multiple_choice_questions = [
        question
        for question in questions
        if isinstance(question, MultipleChoiceQuestion)
    ]
    date_questions = [
        question for question in questions if isinstance(question, DateQuestion)
    ]

    assert len(numeric_questions) > 0
    assert len(binary_questions) > 0
    assert len(multiple_choice_questions) > 0
    assert len(date_questions) == 2  # Consider adding explicit numbers above

    for question in numeric_questions:
        assert question.question_type == "numeric"
    for question in binary_questions:
        assert question.question_type == "binary"
    for question in multiple_choice_questions:
        assert question.question_type == "multiple_choice"
    for question in date_questions:
        assert question.question_type == "date"
    for question in questions:
        assert question.get_question_type() is not None
