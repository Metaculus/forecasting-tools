from datetime import datetime
from unittest.mock import Mock

import pendulum
import pytest
import time_machine

from code_tests.unit_tests.forecasting_test_manager import ForecastingTestManager
from forecasting_tools import MetaculusClient
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.official_bots.uniform_probability_bot import (
    UniformProbabilityBot,
)
from run_bots import (
    AllowedTourn,
    RunBotConfig,
    ScheduleConfig,
    TournConfig,
    get_questions_for_config,
)

NUM_QUESTIONS_FOR_SINGLE_MOCK_CALL = 20
PERCENT_ALREADY_FORECASTED = 0.5
PERCENT_NOT_FORECASTED = 1 - PERCENT_ALREADY_FORECASTED
PERCENT_FORECAST_LONG_TIME_AGO = 0.1
PERCENT_FORECAST_SHORT_TIME_AGO = 0.4
assert (
    PERCENT_FORECAST_LONG_TIME_AGO + PERCENT_FORECAST_SHORT_TIME_AGO
    == PERCENT_ALREADY_FORECASTED
)


def create_mock_questions() -> list[MetaculusQuestion]:
    questions: list[MetaculusQuestion] = []
    num_to_make_already_forecasted = int(
        NUM_QUESTIONS_FOR_SINGLE_MOCK_CALL * PERCENT_ALREADY_FORECASTED
    )
    for i in range(NUM_QUESTIONS_FOR_SINGLE_MOCK_CALL):
        already_forecasted = i < num_to_make_already_forecasted
        question = ForecastingTestManager.get_fake_binary_question(
            already_forecasted=already_forecasted
        )
        question.close_time = pendulum.now().add(days=60)
        question.open_time = pendulum.now().subtract(days=30)
        questions.append(question)

    already_forecasted_questions = [q for q in questions if q.already_forecasted]
    normalized_percent_long = (
        PERCENT_FORECAST_LONG_TIME_AGO / PERCENT_ALREADY_FORECASTED
    )
    normalized_percent_short = (
        PERCENT_FORECAST_SHORT_TIME_AGO / PERCENT_ALREADY_FORECASTED
    )
    num_long_time_ago = int(len(already_forecasted_questions) * normalized_percent_long)
    num_short_time_ago = int(
        len(already_forecasted_questions) * normalized_percent_short
    )
    for question in already_forecasted_questions:
        if num_long_time_ago > 0:
            question.api_json = {
                "question": {
                    "my_forecasts": {
                        "latest": {
                            "start_time": pendulum.now().subtract(days=20).timestamp()
                        }
                    }
                }
            }
            num_long_time_ago -= 1
        elif num_short_time_ago > 0:
            question.api_json = {
                "question": {
                    "my_forecasts": {
                        "latest": {
                            "start_time": pendulum.now().subtract(days=1).timestamp()
                        }
                    }
                }
            }
            num_short_time_ago -= 1
    return questions


def mock_metaculus_api_call(mocker: Mock) -> Mock:
    mock_function = mocker.patch(
        f"{MetaculusClient.get_questions_matching_filter.__module__}.{MetaculusClient.get_questions_matching_filter.__qualname__}",
        return_value=create_mock_questions(),
    )
    return mock_function


def create_test_cases() -> list[tuple[list[AllowedTourn], datetime, int]]:
    out_of_hours = min(ScheduleConfig.UTC_morning_hour - 2, 0)
    morning_hour = ScheduleConfig.UTC_morning_hour
    afternoon_hour = ScheduleConfig.UTC_afternoon_hour
    dates = [
        pendulum.datetime(2025, 5, 11, morning_hour + 1, 0, 0, tz="UTC"),
        pendulum.datetime(2025, 5, 11, afternoon_hour + 1, 0, 0, tz="UTC"),
        pendulum.datetime(2025, 5, 11, out_of_hours + 1, 0, 0, tz="UTC"),
        pendulum.datetime(2025, 5, 12, morning_hour + 1, 0, 0, tz="UTC"),
        pendulum.datetime(2025, 5, 12, afternoon_hour + 1, 0, 0, tz="UTC"),
        pendulum.datetime(2025, 5, 12, out_of_hours + 1, 0, 0, tz="UTC"),
        pendulum.datetime(2025, 5, 13, morning_hour + 1, 0, 0, tz="UTC"),
        pendulum.datetime(2025, 5, 13, afternoon_hour + 1, 0, 0, tz="UTC"),
        pendulum.datetime(2025, 5, 13, out_of_hours + 1, 0, 0, tz="UTC"),
    ]
    configs = [
        TournConfig.aib_only,
        [AllowedTourn.METACULUS_CUP],
        TournConfig.main_site_tourns,
        # TournConfig.aib_and_site,
        TournConfig.everything,
    ]

    test_cases = []
    for date in dates:
        for config in configs:
            num_aib_tourns = len(set([t for t in config if t in TournConfig.aib_only]))
            num_regularly_forecasted_tourns = len(
                set([t for t in config if t in TournConfig.every_x_days_tourns])
            )
            num_main_site_tourns = len(
                set([t for t in config if t in TournConfig.main_site_tourns])
            )
            assert (
                num_aib_tourns + num_regularly_forecasted_tourns + num_main_site_tourns
                == len(config)
            ), "Did not account for all tourns in config"

            is_morning_window = ScheduleConfig.is_morning_window(time=date)
            is_afternoon_window = ScheduleConfig.is_afternoon_window(time=date)
            is_interval_day = ScheduleConfig.is_interval_day(time=date)

            expected_aib_questions = (
                NUM_QUESTIONS_FOR_SINGLE_MOCK_CALL
                * PERCENT_ALREADY_FORECASTED
                * num_aib_tourns
            )

            no_or_stale_forecasted_questions = (
                NUM_QUESTIONS_FOR_SINGLE_MOCK_CALL * PERCENT_NOT_FORECASTED
                + NUM_QUESTIONS_FOR_SINGLE_MOCK_CALL * PERCENT_FORECAST_LONG_TIME_AGO
            )
            expected_regularly_forecasted_questions = (
                no_or_stale_forecasted_questions * num_regularly_forecasted_tourns
                if is_morning_window and is_interval_day
                else 0
            )
            expected_main_site_questions = (
                no_or_stale_forecasted_questions * num_main_site_tourns
                if is_afternoon_window and is_interval_day
                else 0
            )

            expected_questions = (
                expected_aib_questions
                + expected_regularly_forecasted_questions
                + expected_main_site_questions
            )
            test_cases.append(
                (
                    f"{len(config)} tourns on {date} (is_morning_window: {is_morning_window}, is_afternoon_window: {is_afternoon_window}, is_interval_day: {is_interval_day})",
                    config,
                    date,
                    expected_questions,
                )
            )
    return test_cases


@pytest.mark.parametrize(
    "test_name, allowed_tourns, current_time, expected_num_questions",
    create_test_cases(),
)
async def test_basic_get_questions(
    mocker: Mock,
    test_name: str,  # Purely for visualizing in test runner
    allowed_tourns: list[AllowedTourn],
    current_time: datetime,
    expected_num_questions: int,
) -> None:
    with time_machine.travel(current_time):
        assert pendulum.now().date() == current_time.date()
        mock_metaculus_api_call(mocker)
        questions = await get_questions_for_config(
            RunBotConfig(
                mode="mock",
                bot=UniformProbabilityBot(),
                estimated_cost_per_question=0.00,
                tournaments=allowed_tourns,
            ),
            max_questions=1000,
        )
        assert len(questions) == expected_num_questions
