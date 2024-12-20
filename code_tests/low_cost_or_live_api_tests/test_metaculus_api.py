import logging
from datetime import datetime, timedelta

import pytest

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    QuestionState,
)
from forecasting_tools.forecasting.questions_and_reports.report_organizer import (
    ReportOrganizer,
)

logger = logging.getLogger(__name__)

# TODO:
# Can post numeric/date/multiple choice prediction
# Post binary prediction errors if given a non binary question id (and all other combinations of questions)
# Post numeric/date/multiple choice choice prediction errors if given a binary question id


def test_get_binary_question_type_from_id() -> None:
    question_id = ReportOrganizer.get_example_post_id_for_question_type(
        BinaryQuestion
    )
    question = MetaculusApi.get_question_by_post_id(question_id)
    assert isinstance(question, BinaryQuestion)
    assert question_id == question.id_of_post
    assert question.community_prediction_at_access_time is not None
    assert question.community_prediction_at_access_time == pytest.approx(0.01)
    assert question.state == QuestionState.OPEN
    assert_basic_question_attributes_not_none(question, question_id)


def test_get_numeric_question_type_from_id() -> None:
    question_id = ReportOrganizer.get_example_post_id_for_question_type(
        NumericQuestion
    )
    question = MetaculusApi.get_question_by_post_id(question_id)
    assert isinstance(question, NumericQuestion)
    assert question_id == question.id_of_post
    assert question.lower_bound == 0
    assert question.upper_bound == 200
    assert not question.open_lower_bound
    assert question.open_upper_bound
    assert_basic_question_attributes_not_none(question, question_id)


@pytest.mark.skip(reason="Date questions are not fully supported yet")
def test_get_date_question_type_from_id() -> None:
    question_id = ReportOrganizer.get_example_post_id_for_question_type(
        DateQuestion
    )
    question = MetaculusApi.get_question_by_post_id(question_id)
    assert isinstance(question, DateQuestion)
    assert question_id == question.id_of_post
    assert question.lower_bound == datetime(2020, 8, 25)
    assert question.upper_bound == datetime(2199, 12, 25)
    assert question.lower_bound_is_hard_limit
    assert not question.upper_bound_is_hard_limit
    assert_basic_question_attributes_not_none(question, question_id)


def test_get_multiple_choice_question_type_from_id() -> None:
    post_id = ReportOrganizer.get_example_post_id_for_question_type(
        MultipleChoiceQuestion
    )
    question = MetaculusApi.get_question_by_post_id(post_id)
    assert isinstance(question, MultipleChoiceQuestion)
    assert post_id == question.id_of_post
    assert len(question.options) == 6
    assert "0 or 1" in question.options
    assert "2 or 3" in question.options
    assert "4 or 5" in question.options
    assert "6 or 7" in question.options
    assert "8 or 9" in question.options
    assert "10 or more" in question.options
    assert_basic_question_attributes_not_none(question, post_id)


def test_post_comment_on_question() -> None:
    post_id = ReportOrganizer.get_example_post_id_for_question_type(
        BinaryQuestion
    )
    question = MetaculusApi.get_question_by_post_id(post_id)
    MetaculusApi.post_question_comment(
        question.id_of_post, "This is a test comment"
    )
    # No assertion needed, just check that the request did not raise an exception


@pytest.mark.skip(reason="There are no safe questions to post predictions on")
def test_post_binary_prediction_on_question() -> None:
    question = ForecastingTestManager.get_fake_binary_questions()
    assert isinstance(question, BinaryQuestion)
    question_id = question.id_of_post
    MetaculusApi.post_binary_question_prediction(question_id, 0.01)
    MetaculusApi.post_binary_question_prediction(question_id, 0.99)


def test_post_binary_prediction_error_when_out_of_range() -> None:
    question = ForecastingTestManager.get_fake_binary_questions()
    question_id = question.id_of_post
    with pytest.raises(ValueError):
        MetaculusApi.post_binary_question_prediction(question_id, 0)
    with pytest.raises(ValueError):
        MetaculusApi.post_binary_question_prediction(question_id, 1)
    with pytest.raises(ValueError):
        MetaculusApi.post_binary_question_prediction(question_id, -0.01)
    with pytest.raises(ValueError):
        MetaculusApi.post_binary_question_prediction(question_id, 1.1)


def test_questions_returned_from_list_questions() -> None:
    ai_tournament_id = (
        ForecastingTestManager.TOURNAMENT_WITH_MIXTURE_OF_OPEN_AND_NOT_OPEN
    )
    questions = MetaculusApi.get_all_open_questions_from_tournament(
        ai_tournament_id
    )
    assert len(questions) > 0
    # TODO: Add a tournament ID field and assert that the tournament is the same


def test_get_questions_from_tournament() -> None:
    questions = MetaculusApi.get_all_open_questions_from_tournament(
        ForecastingTestManager.TOURN_WITH_OPENNESS_AND_TYPE_VARIATIONS
    )
    score = 0
    if any(isinstance(question, BinaryQuestion) for question in questions):
        score += 1
    if any(isinstance(question, NumericQuestion) for question in questions):
        score += 1
    if any(isinstance(question, DateQuestion) for question in questions):
        score += 1
    if any(
        isinstance(question, MultipleChoiceQuestion) for question in questions
    ):
        score += 1
    assert (
        score > 1
    ), "There needs to be multiple question types in the tournament"

    for question in questions:
        assert question.state == QuestionState.OPEN
        assert_basic_question_attributes_not_none(
            question, question.id_of_post
        )


@pytest.mark.parametrize("num_questions_to_get", [30, 100])
def test_get_benchmark_questions(num_questions_to_get: int) -> None:
    if ForecastingTestManager.quarterly_cup_is_not_active():
        pytest.skip("Quarterly cup is not active")

    random_seed = 42
    questions = MetaculusApi.get_benchmark_questions(
        num_questions_to_get, random_seed
    )

    assert (
        len(questions) == num_questions_to_get
    ), f"Expected {num_questions_to_get} questions to be returned"
    for question in questions:
        assert isinstance(question, BinaryQuestion)
        assert question.date_accessed.date() == datetime.now().date()
        assert isinstance(question.num_forecasters, int)
        assert isinstance(question.num_predictions, int)
        assert isinstance(question.close_time, datetime)
        assert isinstance(question.scheduled_resolution_time, datetime)
        assert (
            question.num_predictions >= 40
        ), "Need to have critical mass of predictions to be confident in the results"
        assert (
            question.num_forecasters >= 40
        ), "Need to have critical mass of forecasters to be confident in the results"
        assert isinstance(question, BinaryQuestion)
        three_months_from_now = datetime.now() + timedelta(days=90)
        assert question.close_time < three_months_from_now
        assert question.scheduled_resolution_time < three_months_from_now
        assert question.state == QuestionState.OPEN
        assert question.community_prediction_at_access_time is not None
        logger.info(f"Found question: {question.question_text}")
    question_ids = [question.id_of_post for question in questions]
    assert len(question_ids) == len(
        set(question_ids)
    ), "Not all questions are unique"

    questions2 = MetaculusApi.get_benchmark_questions(
        num_questions_to_get, random_seed
    )
    question_ids1 = [q.id_of_post for q in questions]
    question_ids2 = [q.id_of_post for q in questions2]
    assert (
        question_ids1 == question_ids2
    ), "Questions retrieved with same random seed should return same IDs"


def assert_basic_question_attributes_not_none(
    question: MetaculusQuestion, question_id: int
) -> None:
    assert question.resolution_criteria is not None
    assert question.fine_print is not None
    assert question.background_info is not None
    assert question.question_text is not None
    assert question.close_time is not None
    # assert question.scheduled_resolution_time is not None
    assert isinstance(question.state, QuestionState)
    assert isinstance(question.page_url, str)
    assert (
        question.page_url
        == f"https://www.metaculus.com/questions/{question_id}"
    )
    assert isinstance(question.num_forecasters, int)
    assert isinstance(question.num_predictions, int)
    assert question.actual_resolution_time is None or isinstance(
        question.actual_resolution_time, datetime
    )
    assert isinstance(question.api_json, dict)
    assert question.close_time > datetime.now()
    if question.scheduled_resolution_time:
        assert question.scheduled_resolution_time >= question.close_time
    if (
        isinstance(question, BinaryQuestion)
        and question.state == QuestionState.OPEN
    ):
        assert question.community_prediction_at_access_time is not None
        assert 0 <= question.community_prediction_at_access_time <= 1
    assert question.id_of_question is not None
