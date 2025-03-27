import logging
from datetime import datetime, timedelta

import pytest
import typeguard

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    QuestionState,
)
from forecasting_tools.forecast_helpers.metaculus_api import (
    ApiFilter,
    MetaculusApi,
)

logger = logging.getLogger(__name__)

# TODO:
# Can post numeric/date/multiple choice prediction
# Post binary prediction errors if given a non binary question id (and all other combinations of questions)
# Post numeric/date/multiple choice choice prediction errors if given a binary question id


def test_get_binary_question_type_from_id() -> None:
    # Test question w/ <1% probability: https://www.metaculus.com/questions/578/human-extinction-by-2100/
    post_id = DataOrganizer.get_example_post_id_for_question_type(
        BinaryQuestion
    )
    question = MetaculusApi.get_question_by_post_id(post_id)
    assert isinstance(question, BinaryQuestion)
    assert post_id == question.id_of_post
    assert question.community_prediction_at_access_time is not None
    assert question.community_prediction_at_access_time <= 0.01
    assert question.state == QuestionState.OPEN
    assert_basic_question_attributes_not_none(question, post_id)


def test_get_numeric_question_type_from_id() -> None:
    question_id = DataOrganizer.get_example_post_id_for_question_type(
        NumericQuestion
    )
    question = MetaculusApi.get_question_by_post_id(question_id)
    assert isinstance(question, NumericQuestion)
    assert question_id == question.id_of_post
    assert question.lower_bound == 0
    assert question.upper_bound == 200
    assert not question.open_lower_bound
    assert question.open_upper_bound
    assert question.unit_of_measure == "years old"
    assert_basic_question_attributes_not_none(question, question_id)


@pytest.mark.skip(reason="Date questions are not fully supported yet")
def test_get_date_question_type_from_id() -> None:
    question_id = DataOrganizer.get_example_post_id_for_question_type(
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
    post_id = DataOrganizer.get_example_post_id_for_question_type(
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


def test_get_question_with_tournament_slug() -> None:
    question = MetaculusApi.get_question_by_url(
        "https://www.metaculus.com/questions/19741"
    )
    assert question.tournament_slugs == ["quarterly-cup-2024q1"]


def test_post_comment_on_question() -> None:
    post_id = DataOrganizer.get_example_post_id_for_question_type(
        BinaryQuestion
    )
    question = MetaculusApi.get_question_by_post_id(post_id)
    assert question.id_of_post is not None
    MetaculusApi.post_question_comment(
        question.id_of_post, "This is a test comment"
    )
    # No assertion needed, just check that the request did not raise an exception


def test_post_binary_prediction_on_question() -> None:
    question = MetaculusApi.get_question_by_url(
        "https://www.metaculus.com/questions/578/human-extinction-by-2100/"
    )
    assert isinstance(question, BinaryQuestion)
    question_id = question.id_of_question
    assert question_id is not None
    MetaculusApi.post_binary_question_prediction(question_id, 0.01)
    MetaculusApi.post_binary_question_prediction(question_id, 0.99)


def test_post_binary_prediction_error_when_out_of_range() -> None:
    question = MetaculusApi.get_question_by_url(
        "https://www.metaculus.com/questions/578/human-extinction-by-2100/"
    )
    question_id = question.id_of_post
    assert question_id is not None
    with pytest.raises(ValueError):
        MetaculusApi.post_binary_question_prediction(question_id, 0)
    with pytest.raises(ValueError):
        MetaculusApi.post_binary_question_prediction(question_id, 1)
    with pytest.raises(ValueError):
        MetaculusApi.post_binary_question_prediction(question_id, -0.01)
    with pytest.raises(ValueError):
        MetaculusApi.post_binary_question_prediction(question_id, 1.1)


def test_questions_returned_from_list_questions() -> None:
    if ForecastingTestManager.quarterly_cup_is_not_active():
        pytest.skip("Quarterly cup is not active")

    tournament_id = (
        ForecastingTestManager.TOURNAMENT_WITH_MIXTURE_OF_OPEN_AND_NOT_OPEN
    )
    questions = MetaculusApi.get_all_open_questions_from_tournament(
        tournament_id
    )
    assert len(questions) > 0
    assert all(question.state == QuestionState.OPEN for question in questions)

    quarterly_cup_slug = "quarterly-cup"
    questions = MetaculusApi.get_all_open_questions_from_tournament(
        quarterly_cup_slug
    )
    assert len(questions) > 0
    assert all(
        quarterly_cup_slug in question.tournament_slugs
        for question in questions
    )
    assert all(question.state == QuestionState.OPEN for question in questions)


def test_get_questions_from_tournament() -> None:
    if ForecastingTestManager.quarterly_cup_is_not_active():
        pytest.skip("Quarterly cup is not active")

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
    assert_basic_attributes_at_percentage(questions, 0.8)


@pytest.mark.parametrize("num_questions_to_get", [30])
def test_get_benchmark_questions(num_questions_to_get: int) -> None:
    questions = MetaculusApi.get_benchmark_questions(num_questions_to_get)

    assert (
        len(questions) == num_questions_to_get
    ), f"Expected {num_questions_to_get} questions to be returned"
    for question in questions:
        assert isinstance(
            question, BinaryQuestion
        ), f"Question {question.id_of_post} is not a BinaryQuestion"
        assert (
            question.date_accessed.date() == datetime.now().date()
        ), f"Question {question.id_of_post} was accessed at {question.date_accessed}, expected today"
        assert isinstance(
            question.num_forecasters, int
        ), f"Question {question.id_of_post} has {question.num_forecasters} forecasters, expected an int"
        assert isinstance(
            question.num_predictions, int
        ), f"Question {question.id_of_post} has {question.num_predictions} predictions, expected an int"
        assert isinstance(
            question.close_time, datetime
        ), f"Question {question.id_of_post} closes at {question.close_time}, expected a datetime"
        assert isinstance(
            question.scheduled_resolution_time, datetime
        ), f"Question {question.id_of_post} resolves at {question.scheduled_resolution_time}, expected a datetime"
        assert (
            question.num_predictions >= 40
        ), "Need to have critical mass of predictions to be confident in the results"
        assert (
            question.num_forecasters >= 40
        ), "Need to have critical mass of forecasters to be confident in the results"
        assert isinstance(
            question, BinaryQuestion
        ), f"Question {question.id_of_post} is not a BinaryQuestion"
        one_year_from_now = datetime.now() + timedelta(days=365)
        assert (
            question.close_time < one_year_from_now
        ), f"Question {question.id_of_post} closes at {question.close_time}, expected before {one_year_from_now}"
        assert (
            question.scheduled_resolution_time < one_year_from_now
        ), f"Question {question.id_of_post} resolves at {question.scheduled_resolution_time}, expected before {one_year_from_now}"
        assert (
            question.state == QuestionState.OPEN
        ), f"Question {question.id_of_post} is not open"
        assert (
            question.community_prediction_at_access_time is not None
        ), f"Community prediction at access time is None for question {question.id_of_post}"
        logger.info(f"Found question: {question.question_text}")
    question_ids = [question.id_of_post for question in questions]
    assert len(question_ids) == len(
        set(question_ids)
    ), "Not all questions are unique"

    questions2 = MetaculusApi.get_benchmark_questions(num_questions_to_get)
    question_ids1 = [q.id_of_post for q in questions]
    question_ids2 = [q.id_of_post for q in questions2]
    assert set(question_ids1) != set(
        question_ids2
    ), "Questions should not be the same (randomly sampled)"


@pytest.mark.parametrize(
    "api_filter, num_questions, randomly_sample",
    [
        (
            ApiFilter(
                num_forecasters_gte=100, allowed_statuses=["open", "resolved"]
            ),
            10,
            False,
        ),
        (
            ApiFilter(
                allowed_types=["binary"],
                allowed_statuses=["closed", "resolved"],
                scheduled_resolve_time_lt=datetime(2024, 1, 20),
                open_time_gt=datetime(2022, 12, 22),
            ),
            250,
            True,
        ),
        (
            ApiFilter(
                close_time_gt=datetime(2024, 1, 15),
                close_time_lt=datetime(2024, 1, 20),
                allowed_tournaments=["quarterly-cup-2024q1"],
            ),
            1,
            False,
        ),
        (
            ApiFilter(
                allowed_tournaments=[32506],  # Q4 AIB Metaculus Tournament
            ),
            None,
            False,
        ),
        (
            ApiFilter(
                num_forecasters_gte=50,
                allowed_types=["binary", "numeric"],
                allowed_statuses=["resolved"],
                publish_time_gt=datetime(2023, 12, 22),
                close_time_lt=datetime(2025, 12, 22),
            ),
            120,
            True,
        ),
        (
            ApiFilter(
                allowed_statuses=["resolved"],
                cp_reveal_time_gt=datetime(2023, 1, 1),
                cp_reveal_time_lt=datetime(2024, 1, 1),
            ),
            30,
            False,
        ),
    ],
)
async def test_get_questions_from_tournament_with_filter(
    api_filter: ApiFilter, num_questions: int | None, randomly_sample: bool
) -> None:
    questions = await MetaculusApi.get_questions_matching_filter(
        api_filter,
        num_questions=num_questions,
        randomly_sample=randomly_sample,
    )
    assert_questions_match_filter(questions, api_filter)
    if num_questions is not None:
        assert len(questions) == num_questions
    else:
        assert len(questions) > 0
    assert_basic_attributes_at_percentage(questions, 0.8)


@pytest.mark.skip(reason="This test takes a while to run")
@pytest.mark.parametrize(
    "status_filter",
    [
        [QuestionState.OPEN],
        [QuestionState.CLOSED],
        [QuestionState.RESOLVED],
        [QuestionState.OPEN, QuestionState.CLOSED],
        [QuestionState.CLOSED, QuestionState.RESOLVED],
    ],
)
async def test_question_status_filters(
    status_filter: list[QuestionState],
) -> None:
    api_filter = ApiFilter(
        allowed_statuses=[state.value for state in status_filter]
    )
    questions = await MetaculusApi.get_questions_matching_filter(
        api_filter, num_questions=250, randomly_sample=True
    )
    for question in questions:
        assert question.state in status_filter
    for expected_state in status_filter:
        assert any(question.state == expected_state for question in questions)


@pytest.mark.parametrize(
    "api_filter, num_questions_in_tournament, randomly_sample",
    [
        (
            ApiFilter(allowed_tournaments=["quarterly-cup-2024q1"]),
            46,
            False,
        ),
        (
            ApiFilter(allowed_tournaments=["quarterly-cup-2024q1"]),
            46,
            True,
        ),
        (
            ApiFilter(
                includes_bots_in_aggregates=False,
                allowed_tournaments=["aibq4"],
            ),
            1,
            False,
        ),
    ],
)
async def test_fails_to_get_questions_if_filter_is_too_restrictive(
    api_filter: ApiFilter,
    num_questions_in_tournament: int,
    randomly_sample: bool,
) -> None:
    requested_questions = num_questions_in_tournament + 50

    with pytest.raises(Exception):
        await MetaculusApi.get_questions_matching_filter(
            api_filter,
            num_questions=requested_questions,
            randomly_sample=randomly_sample,
        )


def assert_basic_attributes_at_percentage(
    questions: list[MetaculusQuestion], percentage: float
) -> None:
    passing = []
    failing_errors: list[Exception] = []
    failing_questions: list[MetaculusQuestion] = []
    for question in questions:
        assert question.id_of_post is not None
        try:
            assert_basic_question_attributes_not_none(
                question, question.id_of_post
            )
            passing.append(question)
        except Exception as e:
            failing_errors.append(e)
            failing_questions.append(question)
    all_errors = "\n".join(str(e) for e in failing_errors)
    assert (
        len(passing) / len(questions) >= percentage
    ), f"Failed {len(failing_questions)} questions. Most recent question: {failing_questions[-1].page_url}. All errors:\n{all_errors}"


def assert_basic_question_attributes_not_none(
    question: MetaculusQuestion, post_id: int
) -> None:
    assert (
        question.resolution_criteria is not None
    ), f"Resolution criteria is None for post ID {post_id}"
    assert (
        question.fine_print is not None
    ), f"Fine print is None for post ID {post_id}"
    assert (
        question.background_info is not None
    ), f"Background info is None for post ID {post_id}"
    assert (
        question.question_text is not None
    ), f"Question text is None for post ID {post_id}"
    assert (
        question.close_time is not None
    ), f"Close time is None for post ID {post_id}"
    assert (
        question.open_time is not None
    ), f"Open time is None for post ID {post_id}"
    assert (
        question.published_time is not None
    ), f"Published time is None for post ID {post_id}"
    assert (
        question.scheduled_resolution_time is not None
    ), f"Scheduled resolution time is None for post ID {post_id}"
    assert (
        question.includes_bots_in_aggregates is not None
    ), f"Includes bots in aggregates is None for post ID {post_id}"
    assert (
        question.cp_reveal_time is not None
    ), f"CP reveal time is None for post ID {post_id}"
    assert isinstance(
        question.state, QuestionState
    ), f"State is not a QuestionState for post ID {post_id}"
    assert isinstance(
        question.page_url, str
    ), f"Page URL is not a string for post ID {post_id}"
    assert (
        question.page_url == f"https://www.metaculus.com/questions/{post_id}"
    ), f"Page URL does not match expected URL for post ID {post_id}"
    assert isinstance(
        question.num_forecasters, int
    ), f"Num forecasters is not an int for post ID {post_id}"
    assert isinstance(
        question.num_predictions, int
    ), f"Num predictions is not an int for post ID {post_id}"
    assert question.actual_resolution_time is None or isinstance(
        question.actual_resolution_time, datetime
    ), f"Actual resolution time is not a datetime for post ID {post_id}"
    assert isinstance(
        question.api_json, dict
    ), f"API JSON is not a dict for post ID {post_id}"
    assert (
        question.close_time is not None
    ), f"Close time is None for post ID {post_id}"
    if question.scheduled_resolution_time:
        assert (
            question.scheduled_resolution_time >= question.close_time
        ), f"Scheduled resolution time is not after close time for post ID {post_id}"
    if (
        isinstance(question, BinaryQuestion)
        and question.community_prediction_at_access_time is not None
    ):
        assert (
            0 <= question.community_prediction_at_access_time <= 1
        ), f"Community prediction at access time is not between 0 and 1 for post ID {post_id}"
    assert (
        question.id_of_question is not None
    ), f"ID of question is None for post ID {post_id}"
    assert (
        question.id_of_post is not None
    ), f"ID of post is None for post ID {post_id}"
    assert question.date_accessed > datetime.now() - timedelta(
        days=1
    ), f"Date accessed is not in the past for post ID {post_id}"
    assert isinstance(
        question.already_forecasted, bool
    ), f"Already forecasted is not a boolean for post ID {post_id}"
    if isinstance(question, NumericQuestion):
        assert (
            question.unit_of_measure is not None
        ), f"Unit of measure is None for post ID {post_id}"


def assert_questions_match_filter(  # NOSONAR
    questions: list[MetaculusQuestion], filter: ApiFilter
) -> None:
    for question in questions:
        if filter.num_forecasters_gte is not None:
            assert (
                question.num_forecasters is not None
                and question.num_forecasters >= filter.num_forecasters_gte
            ), f"Question {question.id_of_post} has {question.num_forecasters} forecasters, expected > {filter.num_forecasters_gte}"

        if filter.allowed_types:
            question_type = type(question)
            type_name = question_type.get_api_type_name()
            assert (
                type_name in filter.allowed_types
            ), f"Question {question.id_of_post} has type {type_name}, expected one of {filter.allowed_types}"

        if filter.allowed_statuses:
            assert (
                question.state
                and question.state.value in filter.allowed_statuses
            ), f"Question {question.id_of_post} has state {question.state}, expected one of {filter.allowed_statuses}"

        if filter.scheduled_resolve_time_gt:
            assert (
                question.scheduled_resolution_time
                and question.scheduled_resolution_time
                > filter.scheduled_resolve_time_gt
            ), f"Question {question.id_of_post} resolves at {question.scheduled_resolution_time}, expected after {filter.scheduled_resolve_time_gt}"

        if filter.scheduled_resolve_time_lt:
            assert (
                question.scheduled_resolution_time
                and question.scheduled_resolution_time
                < filter.scheduled_resolve_time_lt
            ), f"Question {question.id_of_post} resolves at {question.scheduled_resolution_time}, expected before {filter.scheduled_resolve_time_lt}"

        if filter.publish_time_gt:
            assert (
                question.published_time
                and question.published_time > filter.publish_time_gt
            ), f"Question {question.id_of_post} published at {question.published_time}, expected after {filter.publish_time_gt}"

        if filter.publish_time_lt:
            assert (
                question.published_time
                and question.published_time < filter.publish_time_lt
            ), f"Question {question.id_of_post} published at {question.published_time}, expected before {filter.publish_time_lt}"

        if filter.close_time_gt:
            assert (
                question.close_time
                and question.close_time > filter.close_time_gt
            ), f"Question {question.id_of_post} closes at {question.close_time}, expected after {filter.close_time_gt}"

        if filter.close_time_lt:
            assert (
                question.close_time
                and question.close_time < filter.close_time_lt
            ), f"Question {question.id_of_post} closes at {question.close_time}, expected before {filter.close_time_lt}"

        if filter.open_time_gt:
            assert (
                question.open_time and question.open_time > filter.open_time_gt
            ), f"Question {question.id_of_post} opened at {question.open_time}, expected after {filter.open_time_gt}"

        if filter.open_time_lt:
            assert (
                question.open_time and question.open_time < filter.open_time_lt
            ), f"Question {question.id_of_post} opened at {question.open_time}, expected before {filter.open_time_lt}"

        if filter.cp_reveal_time_gt:
            assert (
                question.cp_reveal_time
                and question.cp_reveal_time > filter.cp_reveal_time_gt
            ), f"Question {question.id_of_post} CP reveal time is {question.cp_reveal_time}, expected after {filter.cp_reveal_time_gt}"

        if filter.cp_reveal_time_lt:
            assert (
                question.cp_reveal_time
                and question.cp_reveal_time < filter.cp_reveal_time_lt
            ), f"Question {question.id_of_post} CP reveal time is {question.cp_reveal_time}, expected before {filter.cp_reveal_time_lt}"

        if filter.allowed_tournaments and all(
            isinstance(tournament, str)
            for tournament in filter.allowed_tournaments
        ):
            # TODO: Handle when an allowed tournament is an int ID rather than a slug
            # TODO: As of Jan 25, 2025 you can pass in a question series slug and get back questions.
            #       this should be collected in the question
            assert any(
                slug in filter.allowed_tournaments
                for slug in question.tournament_slugs
            ), f"Question {question.id_of_post} tournaments {question.tournament_slugs} not in allowed tournaments {filter.allowed_tournaments}"

        if filter.community_prediction_exists is not None:
            assert filter.allowed_types == [
                "binary"
            ], "Community prediction filter only works for binary questions at the moment"
            question = typeguard.check_type(question, BinaryQuestion)
            filter_passes = (
                question.community_prediction_at_access_time is not None
                if filter.community_prediction_exists
                else question.community_prediction_at_access_time is None
            )
            assert (
                filter_passes
            ), f"Question {question.id_of_post} has no community prediction at access time"
