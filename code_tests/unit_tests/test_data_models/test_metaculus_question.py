import os

from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)


def test_metaculus_question_is_jsonable() -> None:
    temp_writing_path = "temp/temp_metaculus_question.json"
    read_report_path = "code_tests/unit_tests/test_data_models/forecasting_test_data/metaculus_questions.json"
    questions = DataOrganizer.load_questions_from_file_path(read_report_path)
    assert any(isinstance(question, NumericQuestion) for question in questions)
    assert any(isinstance(question, BinaryQuestion) for question in questions)
    assert any(
        isinstance(question, MultipleChoiceQuestion) for question in questions
    )
    assert any(isinstance(question, DateQuestion) for question in questions)

    DataOrganizer.save_questions_to_file_path(questions, temp_writing_path)
    questions_2 = DataOrganizer.load_questions_from_file_path(
        temp_writing_path
    )
    assert len(questions) == len(questions_2)
    for question, question_2 in zip(questions, questions_2):
        assert question.question_text == question_2.question_text
        assert question.id_of_post == question_2.id_of_post
        assert question.state == question_2.state
        assert str(question) == str(question_2)

    os.remove(temp_writing_path)
