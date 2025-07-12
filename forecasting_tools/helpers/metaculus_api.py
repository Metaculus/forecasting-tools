from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import time
from datetime import datetime, timedelta
from typing import Any, Literal, TypeVar

import requests
import typeguard
from pydantic import BaseModel

from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.util.misc import raise_for_status_with_additional_info

logger = logging.getLogger(__name__)

Q = TypeVar("Q", bound=MetaculusQuestion)


class MetaculusApi:
    """
    Documentation for the API can be found at https://www.metaculus.com/api/
    """

    # NOTE: The tourament slug can be used for ID as well (e.g. "aibq2" or "quarterly-cup")
    AI_WARMUP_TOURNAMENT_ID = (
        3294  # https://www.metaculus.com/tournament/ai-benchmarking-warmup/
    )
    AI_COMPETITION_ID_Q3 = 3349  # https://www.metaculus.com/tournament/aibq3/
    AI_COMPETITION_ID_Q4 = 32506  # https://www.metaculus.com/tournament/aibq4/
    AI_COMPETITION_ID_Q1 = 32627  # https://www.metaculus.com/tournament/aibq1/
    AI_COMPETITION_ID_Q2 = 32721  # https://www.metaculus.com/tournament/aibq2/
    PRO_COMPARISON_TOURNAMENT_Q1 = 32631
    PRO_COMPARISON_TOURNAMENT_Q2 = (
        32761  # https://www.metaculus.com/tournament/pro-benchmark-q22025
    )
    ACX_2025_TOURNAMENT = 32564
    Q3_2024_QUARTERLY_CUP = 3366
    Q4_2024_QUARTERLY_CUP = 3672
    Q1_2025_QUARTERLY_CUP = 32630
    CURRENT_QUARTERLY_CUP_ID = "metaculus-cup"  # Consider this parameter deprecated
    METACULUS_CUP_2025_1_ID = 32726
    CURRENT_METACULUS_CUP_ID = "metaculus-cup"
    CURRENT_AI_COMPETITION_ID = AI_COMPETITION_ID_Q2
    TEST_QUESTION_URLS = [
        "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
        "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
        "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
    ]

    API_BASE_URL = "https://www.metaculus.com/api"
    MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST = 100

    @classmethod
    def post_question_comment(cls, post_id: int, comment_text: str) -> None:
        response = requests.post(
            f"{cls.API_BASE_URL}/comments/create/",
            json={
                "on_post": post_id,
                "text": comment_text,
                "is_private": True,
                "included_forecast": True,
            },
            **cls._get_auth_headers(),  # type: ignore
        )
        logger.info(f"Posted comment on post {post_id}")
        raise_for_status_with_additional_info(response)

    @classmethod
    def post_binary_question_prediction(
        cls, question_id: int, prediction_in_decimal: float
    ) -> None:
        logger.info(f"Posting prediction on question {question_id}")
        if prediction_in_decimal < 0.01 or prediction_in_decimal > 0.99:
            raise ValueError("Prediction value must be between 0.001 and 0.99")
        payload = {
            "probability_yes": prediction_in_decimal,
        }
        cls._post_question_prediction(question_id, payload)

    @classmethod
    def post_numeric_question_prediction(
        cls, question_id: int, cdf_values: list[float]
    ) -> None:
        """
        If the question is numeric, forecast must be a dictionary that maps
        quartiles or percentiles to datetimes, or a 201 value cdf.
        In this case we use the cdf.
        """
        logger.info(f"Posting prediction on question {question_id}")
        if len(cdf_values) != 201:
            raise ValueError("CDF must contain exactly 201 values")
        if not all(0 <= x <= 1 for x in cdf_values):
            raise ValueError("All CDF values must be between 0 and 1")
        if not all(a <= b for a, b in zip(cdf_values, cdf_values[1:])):
            raise ValueError("CDF values must be monotonically increasing")
        payload = {
            "continuous_cdf": cdf_values,
        }
        cls._post_question_prediction(question_id, payload)

    @classmethod
    def post_multiple_choice_question_prediction(
        cls, question_id: int, options_with_probabilities: dict[str, float]
    ) -> None:
        """
        If the question is multiple choice, forecast must be a dictionary that
        maps question.options labels to floats.
        """
        payload = {
            "probability_yes_per_category": options_with_probabilities,
        }
        cls._post_question_prediction(question_id, payload)

    @classmethod
    def get_question_by_url(cls, question_url: str) -> MetaculusQuestion:
        """
        URL looks like https://www.metaculus.com/questions/28841/will-eric-adams-be-the-nyc-mayor-on-january-1-2025/
        """
        match = re.search(r"/questions/(\d+)", question_url)
        if not match:
            raise ValueError(f"Could not find question ID in URL: {question_url}")
        question_id = int(match.group(1))
        return cls.get_question_by_post_id(question_id)

    @classmethod
    def get_question_by_post_id(cls, post_id: int) -> MetaculusQuestion:
        logger.info(f"Retrieving question details for question {post_id}")
        url = f"{cls.API_BASE_URL}/posts/{post_id}/"
        response = requests.get(
            url,
            **cls._get_auth_headers(),  # type: ignore
        )
        raise_for_status_with_additional_info(response)
        json_question = json.loads(response.content)
        metaculus_question = MetaculusApi._metaculus_api_json_to_question(json_question)
        logger.info(f"Retrieved question details for question {post_id}")
        return metaculus_question

    @classmethod
    async def get_questions_matching_filter(
        cls,
        api_filter: ApiFilter,
        num_questions: int | None = None,
        randomly_sample: bool = False,
        error_if_question_target_missed: bool = True,
    ) -> list[MetaculusQuestion]:
        """
        Will return a list of questions that match the filter.
        If num questions is not set, it will only grab the first page of questions from API.
        If you use filter criteria that are not directly built into the API,
        then there maybe questions that match the filter even if the first page does not contain any.

        Requiring a number will go through pages until it finds the number of questions or runs out of pages.
        """
        if num_questions is not None:
            assert num_questions > 0, "Must request at least one question"
        if randomly_sample:
            assert (
                num_questions is not None
            ), "Must request at least one question if randomly sampling"
            questions = await cls._filter_using_randomized_strategy(
                api_filter, num_questions, error_if_question_target_missed
            )
        else:
            questions = await cls._filter_sequential_strategy(api_filter, num_questions)
        if (
            num_questions is not None
            and len(questions) != num_questions
            and error_if_question_target_missed
        ):
            raise ValueError(
                f"Requested number of questions ({num_questions}) does not match number of questions found ({len(questions)})"
            )
        if len(set(q.id_of_post for q in questions)) != len(questions):
            raise ValueError("Not all questions found are unique")
        logger.info(
            f"Returning {len(questions)} questions matching the Metaculus API filter"
        )
        return questions

    @classmethod
    def get_all_open_questions_from_tournament(
        cls,
        tournament_id: int | str,
    ) -> list[MetaculusQuestion]:
        logger.info(f"Retrieving questions from tournament {tournament_id}")
        api_filter = ApiFilter(
            allowed_tournaments=[tournament_id],
            allowed_statuses=["open"],
        )
        questions = asyncio.run(cls.get_questions_matching_filter(api_filter))
        logger.info(
            f"Retrieved {len(questions)} questions from tournament {tournament_id}"
        )
        return questions

    @classmethod
    def get_benchmark_questions(
        cls,
        num_of_questions_to_return: int,
        days_to_resolve_in: int | None = None,
        max_days_since_opening: int | None = 365,
        num_forecasters_gte: int = 30,
        error_if_question_target_missed: bool = True,
    ) -> list[BinaryQuestion]:
        logger.info(f"Retrieving {num_of_questions_to_return} benchmark questions")
        date_into_future = (
            datetime.now() + timedelta(days=days_to_resolve_in)
            if days_to_resolve_in
            else None
        )
        date_into_past = (
            datetime.now() - timedelta(days=max_days_since_opening)
            if max_days_since_opening
            else None
        )
        api_filter = ApiFilter(
            allowed_statuses=["open"],
            allowed_types=["binary"],
            num_forecasters_gte=num_forecasters_gte,
            scheduled_resolve_time_lt=date_into_future,
            includes_bots_in_aggregates=False,
            community_prediction_exists=True,
            open_time_gt=date_into_past,
        )
        questions = asyncio.run(
            cls.get_questions_matching_filter(
                api_filter,
                num_questions=num_of_questions_to_return,
                randomly_sample=True,
                error_if_question_target_missed=error_if_question_target_missed,
            )
        )
        questions = typeguard.check_type(questions, list[BinaryQuestion])
        return questions

    @classmethod
    def _get_auth_headers(cls) -> dict[str, dict[str, str]]:
        METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
        if METACULUS_TOKEN is None:
            raise ValueError("METACULUS_TOKEN environment variable not set")
        return {
            "headers": {
                "Authorization": f"Token {METACULUS_TOKEN}",
                "Accept-Language": "en",
            }
        }

    @classmethod
    def _post_question_prediction(
        cls, question_id: int, forecast_payload: dict
    ) -> None:
        url = f"{cls.API_BASE_URL}/questions/forecast/"
        response = requests.post(
            url,
            json=[
                {
                    "question": question_id,
                    **forecast_payload,
                },
            ],
            **cls._get_auth_headers(),  # type: ignore
        )
        logger.info(f"Posted prediction on question {question_id}")
        raise_for_status_with_additional_info(response)

    @classmethod
    def _get_questions_from_api(cls, params: dict[str, Any]) -> list[MetaculusQuestion]:
        random_sleep_time = random.uniform(2, 3)
        logger.debug(
            f"Sleeping for {random_sleep_time:.1f} seconds before next request"
        )
        time.sleep(random_sleep_time)
        num_requested = params.get("limit")
        assert (
            num_requested is None
            or num_requested <= cls.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
        ), "You cannot get more than 100 questions at a time"
        url = f"{cls.API_BASE_URL}/posts/"
        response = requests.get(url, params=params, **cls._get_auth_headers())  # type: ignore
        raise_for_status_with_additional_info(response)
        data = json.loads(response.content)
        results = data["results"]
        supported_posts = [
            q
            for q in results
            if "notebook" not in q
            and "group_of_questions" not in q
            and "conditional" not in q
        ]
        removed_posts = [post for post in results if post not in supported_posts]
        if len(removed_posts) > 0:
            logger.warning(
                f"Removed {len(removed_posts)} posts that "
                "are not supported (e.g. notebook or group question)"
            )

        questions = []
        for q in supported_posts:
            try:
                questions.append(cls._metaculus_api_json_to_question(q))
            except Exception as e:
                logger.warning(
                    f"Error processing post ID {q['id']}: {e.__class__.__name__} {e}"
                )

        return questions

    @classmethod
    def _metaculus_api_json_to_question(cls, api_json: dict) -> MetaculusQuestion:
        assert "question" in api_json, f"Question not found in API JSON: {api_json}"
        question_type_string = api_json["question"]["type"]  # type: ignore
        if question_type_string == BinaryQuestion.get_api_type_name():
            question_type = BinaryQuestion
        elif question_type_string == NumericQuestion.get_api_type_name():
            question_type = NumericQuestion
        elif question_type_string == MultipleChoiceQuestion.get_api_type_name():
            question_type = MultipleChoiceQuestion
        elif question_type_string == DateQuestion.get_api_type_name():
            question_type = DateQuestion
        else:
            raise ValueError(f"Unknown question type: {question_type_string}")
        question = question_type.from_metaculus_api_json(api_json)
        return question

    @classmethod
    async def _filter_using_randomized_strategy(
        cls,
        api_filter: ApiFilter,
        num_questions: int,
        error_if_not_enough_questions: bool,
    ) -> list[MetaculusQuestion]:
        number_of_questions_matching_filter = (
            cls._determine_how_many_questions_match_filter(api_filter)
        )
        if (
            number_of_questions_matching_filter < num_questions
            and error_if_not_enough_questions
        ):
            raise ValueError(
                f"Not enough questions matching filter ({number_of_questions_matching_filter} before local filtering) to sample {num_questions} questions. Set error_if_not_enough_questions to False to return as many as possible"
            )

        questions_per_page = cls.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
        total_pages = math.ceil(
            number_of_questions_matching_filter / questions_per_page
        )
        target_qs_to_sample_from = num_questions * 2

        # Create randomized list of all possible page indices
        available_page_indices = list(range(total_pages))
        random.shuffle(available_page_indices)

        questions: list[MetaculusQuestion] = []
        for page_index in available_page_indices:
            if len(questions) >= target_qs_to_sample_from:
                break

            offset = page_index * questions_per_page
            page_questions, _ = cls._grab_filtered_questions_with_offset(
                api_filter, offset
            )
            questions.extend(page_questions)

        if len(questions) < num_questions and error_if_not_enough_questions:
            raise ValueError(
                f"Exhausted all {total_pages} pages but only found {len(questions)} questions, needed {num_questions}. Set error_if_not_enough_questions to False to return as many as possible"
            )
        assert len(set(q.id_of_post for q in questions)) == len(
            questions
        ), "Not all questions found are unique"

        if len(questions) > num_questions:
            random_sample = random.sample(questions, num_questions)
        else:
            random_sample = questions
        logger.info(
            f"Sampled {len(random_sample)} questions from {len(questions)} questions that matched the filterwhich were taken from {total_pages} randomly selected pages which each had at max {cls.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST} questions matching the filter"
        )

        return random_sample

    @classmethod
    async def _filter_sequential_strategy(
        cls, api_filter: ApiFilter, num_questions: int | None
    ) -> list[MetaculusQuestion]:
        if num_questions is None:
            questions, _ = cls._grab_filtered_questions_with_offset(api_filter, 0)
            return questions

        questions: list[MetaculusQuestion] = []
        more_questions_available = True
        page_num = 0
        while len(questions) < num_questions and more_questions_available:
            offset = page_num * cls.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
            new_questions, continue_searching = (
                cls._grab_filtered_questions_with_offset(api_filter, offset)
            )
            questions.extend(new_questions)
            if not continue_searching:
                more_questions_available = False
            page_num += 1
        return questions[:num_questions]

    @classmethod
    def _determine_how_many_questions_match_filter(cls, filter: ApiFilter) -> int:
        """
        Search Metaculus API with binary search to find the number of questions
        matching the filter.
        """
        estimated_max_questions = 20000
        left, right = 0, estimated_max_questions
        last_successful_offset = 0

        while left <= right:
            mid = (left + right) // 2
            offset = mid * cls.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST

            _, found_questions_before_running_local_filter = (
                cls._grab_filtered_questions_with_offset(filter, offset)
            )

            if found_questions_before_running_local_filter:
                left = mid + 1
                last_successful_offset = offset
            else:
                right = mid - 1

        final_page_questions, _ = cls._grab_filtered_questions_with_offset(
            filter, last_successful_offset
        )
        total_questions = last_successful_offset + len(final_page_questions)

        if total_questions >= estimated_max_questions:
            raise ValueError(
                f"Total questions ({total_questions}) exceeded estimated max ({estimated_max_questions})"
            )
        logger.info(
            f"Estimating that there are {total_questions} questions matching the filter -> {str(filter)[:200]}"
        )
        return total_questions

    @classmethod
    def _grab_filtered_questions_with_offset(
        cls,
        api_filter: ApiFilter,
        offset: int = 0,
    ) -> tuple[list[MetaculusQuestion], bool]:
        url_params: dict[str, Any] = {
            "limit": cls.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST,
            "offset": offset,
            "order_by": "-published_at",
            "with_cp": "true",
        }

        if api_filter.allowed_types:
            url_params["forecast_type"] = api_filter.allowed_types

        if api_filter.allowed_statuses:
            url_params["statuses"] = api_filter.allowed_statuses

        if api_filter.scheduled_resolve_time_gt:
            url_params["scheduled_resolve_time__gt"] = (
                api_filter.scheduled_resolve_time_gt.strftime("%Y-%m-%d")
            )
        if api_filter.scheduled_resolve_time_lt:
            url_params["scheduled_resolve_time__lt"] = (
                api_filter.scheduled_resolve_time_lt.strftime("%Y-%m-%d")
            )

        if api_filter.publish_time_gt:
            url_params["published_at__gt"] = api_filter.publish_time_gt.strftime(
                "%Y-%m-%d"
            )
        if api_filter.publish_time_lt:
            url_params["published_at__lt"] = api_filter.publish_time_lt.strftime(
                "%Y-%m-%d"
            )

        if api_filter.open_time_gt:
            url_params["open_time__gt"] = api_filter.open_time_gt.strftime("%Y-%m-%d")
        if api_filter.open_time_lt:
            url_params["open_time__lt"] = api_filter.open_time_lt.strftime("%Y-%m-%d")

        if api_filter.allowed_tournaments:
            url_params["tournaments"] = api_filter.allowed_tournaments

        questions = cls._get_questions_from_api(url_params)
        questions_were_found_before_local_filter = len(questions) > 0

        if api_filter.num_forecasters_gte is not None:
            questions = cls._filter_questions_by_forecasters(
                questions, api_filter.num_forecasters_gte
            )

        if api_filter.close_time_gt or api_filter.close_time_lt:
            questions = cls._filter_questions_by_close_time(
                questions, api_filter.close_time_gt, api_filter.close_time_lt
            )

        if api_filter.includes_bots_in_aggregates is not None:
            questions = cls._filter_questions_by_includes_bots_in_aggregates(
                questions, api_filter.includes_bots_in_aggregates
            )

        if api_filter.community_prediction_exists is not None:
            if not any(t in api_filter.allowed_types for t in ["binary"]):
                raise ValueError(
                    "Community prediction filter only works for binary questions at the moment"
                )
            questions = typeguard.check_type(questions, list[BinaryQuestion])
            questions = cls._filter_questions_by_community_prediction_exists(
                questions, api_filter.community_prediction_exists
            )
            questions = typeguard.check_type(questions, list[MetaculusQuestion])

        if api_filter.cp_reveal_time_gt or api_filter.cp_reveal_time_lt:
            questions = cls._filter_questions_by_cp_reveal_time(
                questions,
                api_filter.cp_reveal_time_gt,
                api_filter.cp_reveal_time_lt,
            )

        return questions, questions_were_found_before_local_filter

    @classmethod
    def _filter_questions_by_forecasters(
        cls, questions: list[Q], min_forecasters: int
    ) -> list[Q]:
        questions_with_enough_forecasters: list[Q] = []
        for question in questions:
            assert question.num_forecasters is not None
            if question.num_forecasters >= min_forecasters:
                questions_with_enough_forecasters.append(question)
        return questions_with_enough_forecasters

    @classmethod
    def _filter_questions_by_includes_bots_in_aggregates(
        cls, questions: list[Q], includes_bots_in_aggregates: bool
    ) -> list[Q]:
        return [
            question
            for question in questions
            if question.includes_bots_in_aggregates == includes_bots_in_aggregates
        ]

    @classmethod
    def _filter_questions_by_close_time(
        cls,
        questions: list[Q],
        close_time_gt: datetime | None,
        close_time_lt: datetime | None,
    ) -> list[Q]:
        questions_with_close_time: list[Q] = []
        for question in questions:
            if question.close_time is not None:
                if close_time_gt and question.close_time <= close_time_gt:
                    continue
                if close_time_lt and question.close_time >= close_time_lt:
                    continue
                questions_with_close_time.append(question)
        return questions_with_close_time

    @classmethod
    def _filter_questions_by_community_prediction_exists(
        cls, questions: list[BinaryQuestion], community_prediction_exists: bool
    ) -> list[BinaryQuestion]:
        return [
            question
            for question in questions
            if (question.community_prediction_at_access_time is not None)
            == community_prediction_exists
        ]

    @classmethod
    def _filter_questions_by_cp_reveal_time(
        cls,
        questions: list[Q],
        cp_reveal_time_gt: datetime | None,
        cp_reveal_time_lt: datetime | None,
    ) -> list[Q]:
        questions_with_cp_reveal_time: list[Q] = []
        for question in questions:
            if question.cp_reveal_time is not None:
                if cp_reveal_time_gt and question.cp_reveal_time <= cp_reveal_time_gt:
                    continue
                if cp_reveal_time_lt and question.cp_reveal_time >= cp_reveal_time_lt:
                    continue
                questions_with_cp_reveal_time.append(question)
        return questions_with_cp_reveal_time


class ApiFilter(BaseModel):
    num_forecasters_gte: int | None = None
    allowed_types: list[Literal["binary", "numeric", "multiple_choice", "date"]] = [
        "binary",
        "numeric",
        "multiple_choice",
        "date",
    ]
    allowed_statuses: list[Literal["open", "upcoming", "resolved", "closed"]] | None = (
        None
    )
    scheduled_resolve_time_gt: datetime | None = None
    scheduled_resolve_time_lt: datetime | None = None
    publish_time_gt: datetime | None = None
    publish_time_lt: datetime | None = None
    close_time_gt: datetime | None = None
    close_time_lt: datetime | None = None
    open_time_gt: datetime | None = None
    open_time_lt: datetime | None = None
    allowed_tournaments: list[str | int] | None = None
    includes_bots_in_aggregates: bool | None = None
    community_prediction_exists: bool | None = None
    cp_reveal_time_gt: datetime | None = None
    cp_reveal_time_lt: datetime | None = None
