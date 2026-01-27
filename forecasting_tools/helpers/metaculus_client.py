from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, List, Literal, TypeVar, overload

import pendulum
import requests
import typeguard
from pydantic import BaseModel, Field, model_validator

from forecasting_tools.data_models.coherence_link import (
    CoherenceLink,
    DetailedCoherenceLink,
    NeedsUpdateResponse,
)
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    ConditionalQuestion,
    MetaculusQuestion,
    QuestionBasicType,
)
from forecasting_tools.data_models.user_response import (
    TokenResponse,
    TokenUserResponse,
    UserResponse,
)
from forecasting_tools.util.misc import (
    add_timezone_to_dates_in_base_model,
    raise_for_status_with_additional_info,
    retry_with_exponential_backoff,
)

logger = logging.getLogger(__name__)


Q = TypeVar("Q", bound=MetaculusQuestion)
T = TypeVar("T", bound=BaseModel)

GroupQuestionMode = Literal["exclude", "unpack_subquestions"]
"""
If group_question_mode is "exclude", then group questions will be removed from the list of questions.
If group_question_mode is "unpack_subquestions", then each subquestion in the group question will be added as a separate question.
"""
QuestionFullType = Literal[
    "binary", "numeric", "multiple_choice", "date", "group_of_questions", "conditional"
]
QuestionStateAsString = Literal[
    "open", "upcoming", "resolved", "closed"
]  # Also see QuestionState enum


class ApiFilter(BaseModel):
    num_forecasters_gte: int | None = None
    allowed_types: list[QuestionBasicType] = [
        "binary",
        "numeric",
        "multiple_choice",
        "date",
        "discrete",
        "conditional",
    ]
    allowed_subquestion_types: list[QuestionBasicType] | None = None
    group_question_mode: GroupQuestionMode = "exclude"
    allowed_statuses: list[QuestionStateAsString] | None = None
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
    is_previously_forecasted_by_user: bool | None = None
    order_by: str = (
        "-published_time"  # Alternatives include things like "-weekly_movement" + is asc, - is desc
    )
    is_in_main_feed: bool | None = (
        None
        # TODO (Sep 5, 2025): Instead of checking if default project visibility is normal, add the url parameter "for_main_feed=true"
    )
    other_url_parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def add_timezone_to_dates(self) -> ApiFilter:
        return add_timezone_to_dates_in_base_model(self)


class MetaculusClient:
    """
    Documentation for the API can be found at https://www.metaculus.com/api/
    """

    # NOTE: The tourament slug can be used for ID as well (e.g. "aibq2" or "quarterly-cup" instead of 32721 or 32630)
    AI_COMPETITION_ID_Q3 = 3349  # https://www.metaculus.com/tournament/aibq3/
    AI_COMPETITION_ID_Q4 = 32506  # https://www.metaculus.com/tournament/aibq4/
    AI_COMPETITION_ID_Q1 = 32627  # https://www.metaculus.com/tournament/aibq1/
    AI_COMPETITION_ID_Q2 = 32721  # https://www.metaculus.com/tournament/aibq2/
    AIB_FALL_2025_ID = 32813  # https://www.metaculus.com/tournament/fall-aib-2025/
    AIB_SPRING_2026_ID = 32916  # https://www.metaculus.com/tournament/spring-aib-2026/

    PRO_COMPARISON_TOURNAMENT_Q1 = 32631
    PRO_COMPARISON_TOURNAMENT_Q2 = (
        32761  # https://www.metaculus.com/tournament/pro-benchmark-q22025
    )

    ACX_2025_TOURNAMENT = 32564
    AI_2027_TOURNAMENT_ID = "ai-2027"
    MAIN_FEED = 144  # site_main

    Q3_2024_QUARTERLY_CUP = 3366
    Q4_2024_QUARTERLY_CUP = 3672
    Q1_2025_QUARTERLY_CUP = 32630
    METACULUS_CUP_2025_1_ID = 32726  # Summer cup 2025
    METACULUS_CUP_FALL_2025_ID = 32828
    METACULUS_CUP_SPRING_2026_ID = (
        32921  # https://www.metaculus.com/tournament/metaculus-cup-spring-2026/
    )

    Q3_2025_MARKET_PULSE_ID = "market-pulse-25q3"
    Q4_2025_MARKET_PULSE_ID = "market-pulse-25q4"
    Q1_2026_MARKET_PULSE_ID = "market-pulse-26q1"

    CURRENT_METACULUS_CUP_ID = METACULUS_CUP_SPRING_2026_ID
    CURRENT_QUARTERLY_CUP_ID = CURRENT_METACULUS_CUP_ID  # Consider this parameter deprecated since quarterly cup is no longer active
    CURRENT_AI_COMPETITION_ID = AIB_SPRING_2026_ID
    CURRENT_MINIBENCH_ID = "minibench"
    CURRENT_MARKET_PULSE_ID = Q1_2026_MARKET_PULSE_ID

    TEST_QUESTION_URLS = [
        "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
        "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
        "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
    ]

    MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST = 100

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 30,
        sleep_seconds_between_requests: float = 3.5,
        sleep_jitter_seconds: float = 1,
        token: str | None = None,
    ):
        self.base_url = base_url or os.getenv(
            "METACULUS_API_BASE_URL", "https://www.metaculus.com/api"
        )
        self.timeout = timeout
        self.sleep_time_between_requests_min = sleep_seconds_between_requests
        self.sleep_jitter_seconds = sleep_jitter_seconds

        self.token = token or os.getenv("METACULUS_TOKEN")
        if self.token is None:
            logger.warning(
                "METACULUS_TOKEN environment variable and/or token field not set"
            )

    @retry_with_exponential_backoff()
    def get_user_bots(self) -> list[UserResponse]:
        self._sleep_between_requests()
        response = requests.get(
            f"{self.base_url}/users/me/bots/",
            **self._get_auth_headers(),  # type: ignore
            timeout=self.timeout,
        )
        raise_for_status_with_additional_info(response)
        content = json.loads(response.content)
        bots = [
            UserResponse(id=bot_data["id"], username=bot_data["username"])
            for bot_data in content
        ]
        logger.info(f"Retrieved {len(bots)} bots for current user")
        return bots

    @retry_with_exponential_backoff()
    def get_bot_token(self, bot_id: int) -> TokenResponse:
        self._sleep_between_requests()
        response = requests.get(
            f"{self.base_url}/users/me/bots/{bot_id}/token/",
            **self._get_auth_headers(),  # type: ignore
            timeout=self.timeout,
        )
        raise_for_status_with_additional_info(response)
        content = json.loads(response.content)
        token_response = TokenResponse(token=content["token"])
        logger.info(f"Retrieved token for bot {bot_id}")
        return token_response

    @retry_with_exponential_backoff()
    def create_bot(self, username: str) -> TokenUserResponse:
        self._sleep_between_requests()
        response = requests.post(
            f"{self.base_url}/users/me/bots/create/",
            json={"username": username},
            **self._get_auth_headers(),  # type: ignore
            timeout=self.timeout,
        )
        raise_for_status_with_additional_info(response)
        content = json.loads(response.content)

        # Combine token and user data for TokenUserResponse
        combined_data = {
            "token": content["token"],
            "id": content["user"]["id"],
            "username": content["user"]["username"],
        }
        bot_response = TokenUserResponse(**combined_data)
        logger.info(f"Created bot with username '{username}' and id {bot_response.id}")
        return bot_response

    @retry_with_exponential_backoff()
    def post_question_comment(
        self,
        post_id: int,
        comment_text: str,
        is_private: bool = True,
        included_forecast: bool = True,
    ) -> None:
        self._sleep_between_requests()
        response = requests.post(
            f"{self.base_url}/comments/create/",
            json={
                "on_post": post_id,
                "text": comment_text,
                "is_private": is_private,
                "included_forecast": included_forecast,
            },
            **self._get_auth_headers(),  # type: ignore
            timeout=self.timeout,
        )
        logger.info(f"Posted comment on post {post_id}")
        raise_for_status_with_additional_info(response)

    def post_binary_question_prediction(
        self, question_id: int, prediction_in_decimal: float
    ) -> None:
        logger.info(f"Posting prediction on question {question_id}")
        if prediction_in_decimal < 0.001 or prediction_in_decimal > 0.999:
            raise ValueError("Prediction value must be between 0.001 and 0.999")
        payload = {
            "probability_yes": prediction_in_decimal,
        }
        self._post_question_prediction(question_id, payload)

    def post_numeric_question_prediction(
        self, question_id: int, cdf_values: list[float]
    ) -> None:
        """
        If the question is numeric, forecast must be a dictionary that maps
        quartiles or percentiles to datetimes, or a 201 value cdf.
        In this case we use the cdf.
        """
        logger.info(f"Posting prediction on question {question_id}")
        if not all(0 <= x <= 1 for x in cdf_values):
            raise ValueError("All CDF values must be between 0 and 1")
        if not all(a <= b for a, b in zip(cdf_values, cdf_values[1:])):
            raise ValueError("CDF values must be monotonically increasing")
        payload = {
            "continuous_cdf": cdf_values,
        }
        self._post_question_prediction(question_id, payload)

    def post_multiple_choice_question_prediction(
        self, question_id: int, options_with_probabilities: dict[str, float]
    ) -> None:
        """
        If the question is multiple choice, forecast must be a dictionary that
        maps question.options labels to floats.
        """
        payload = {
            "probability_yes_per_category": options_with_probabilities,
        }
        self._post_question_prediction(question_id, payload)

    @overload
    def get_question_by_url(
        self,
        question_url: str,
        group_question_mode: Literal["exclude"] = "exclude",
    ) -> MetaculusQuestion: ...

    @overload
    def get_question_by_url(
        self,
        question_url: str,
        group_question_mode: GroupQuestionMode = "exclude",
    ) -> MetaculusQuestion | list[MetaculusQuestion]: ...

    def get_question_by_url(
        self,
        question_url: str,
        group_question_mode: GroupQuestionMode = "exclude",
    ) -> MetaculusQuestion | list[MetaculusQuestion]:
        """
        Handles both question and community URLs:
        - Question: https://www.metaculus.com/questions/28841/...
        - Community: https://www.metaculus.com/c/risk/38787/
        """
        question_match = re.search(r"/questions/(\d+)", question_url)
        if question_match:
            post_id = int(question_match.group(1))
            return self.get_question_by_post_id(post_id, group_question_mode)

        community_match = re.search(r"/c/[^/]+/(\d+)", question_url)
        if community_match:
            post_id = int(community_match.group(1))
            return self.get_question_by_post_id(post_id, group_question_mode)

        raise ValueError(
            f"Could not find question or collection ID in URL: {question_url}"
        )

    @overload
    def get_question_by_post_id(
        self,
        post_id: int,
        group_question_mode: Literal["exclude"] = "exclude",
    ) -> MetaculusQuestion: ...

    @overload
    def get_question_by_post_id(
        self,
        post_id: int,
        group_question_mode: GroupQuestionMode = "exclude",
    ) -> MetaculusQuestion | list[MetaculusQuestion]: ...

    @retry_with_exponential_backoff()
    def get_question_by_post_id(
        self,
        post_id: int,
        group_question_mode: GroupQuestionMode = "exclude",
    ) -> MetaculusQuestion | list[MetaculusQuestion]:
        logger.info(f"Retrieving question details for question {post_id}")
        self._sleep_between_requests()

        url = f"{self.base_url}/posts/{post_id}/"
        headers = self._get_auth_headers()
        response = requests.get(
            url,
            **headers,  # type: ignore
            timeout=self.timeout,
        )
        raise_for_status_with_additional_info(response)
        json_question = json.loads(response.content)
        metaculus_questions = self._post_json_to_questions_while_handling_groups(
            json_question, group_question_mode
        )
        logger.info(f"Retrieved question details for question {post_id}")
        if len(metaculus_questions) == 1:
            return metaculus_questions[0]
        else:
            if group_question_mode == "exclude":
                raise ValueError(
                    f"Expected 1 question but got {len(metaculus_questions)}. You probably accessed a group question. Group question mode is set to {group_question_mode}."
                )
            return metaculus_questions

    async def get_questions_matching_filter(
        self,
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
            questions = await self._filter_using_randomized_strategy(
                api_filter, num_questions, error_if_question_target_missed
            )
        else:
            questions = await self._filter_sequential_strategy(
                api_filter, num_questions
            )
        if (
            num_questions is not None
            and len(questions) != num_questions
            and error_if_question_target_missed
        ):
            raise ValueError(
                f"Requested number of questions ({num_questions}) does not match number of questions found ({len(questions)})"
            )
        if len(set(q.id_of_question for q in questions)) != len(questions):
            raise ValueError("Not all questions found are unique")
        logger.info(
            f"Returning {len(questions)} questions matching the Metaculus API filter"
        )
        return questions

    def get_all_open_questions_from_tournament(
        self,
        tournament_id: int | str,
        group_question_mode: GroupQuestionMode = "unpack_subquestions",
    ) -> list[MetaculusQuestion]:
        logger.info(f"Retrieving questions from tournament {tournament_id}")
        api_filter = ApiFilter(
            allowed_tournaments=[tournament_id],
            allowed_statuses=["open"],
            group_question_mode=group_question_mode,
        )
        questions = asyncio.run(self.get_questions_matching_filter(api_filter))
        logger.info(
            f"Retrieved {len(questions)} questions from tournament {tournament_id}"
        )
        return questions

    def get_benchmark_questions(
        self,
        num_of_questions_to_return: int,
        days_to_resolve_in: int | None = None,
        max_days_since_opening: int | None = 365,
        num_forecasters_gte: int = 30,
        error_if_question_target_missed: bool = True,
        group_question_mode: GroupQuestionMode = "exclude",
    ) -> list[BinaryQuestion]:
        logger.info(f"Retrieving {num_of_questions_to_return} benchmark questions")
        date_into_future = (
            pendulum.now(tz="UTC") + timedelta(days=days_to_resolve_in)
            if days_to_resolve_in
            else None
        )
        date_into_past = (
            pendulum.now(tz="UTC") - timedelta(days=max_days_since_opening)
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
            group_question_mode=group_question_mode,
        )
        questions = asyncio.run(
            self.get_questions_matching_filter(
                api_filter,
                num_questions=num_of_questions_to_return,
                randomly_sample=True,
                error_if_question_target_missed=error_if_question_target_missed,
            )
        )
        questions = typeguard.check_type(questions, list[BinaryQuestion])
        return questions

    @retry_with_exponential_backoff()
    def get_current_user_id(self) -> int:
        self._sleep_between_requests()
        response = requests.get(
            f"{self.base_url}/users/me",
            **self._get_auth_headers(),  # type: ignore
        )
        raise_for_status_with_additional_info(response)
        content = json.loads(response.content)
        return int(content["id"])

    @retry_with_exponential_backoff()
    def post_question_link(
        self,
        question1_id: int,
        question2_id: int,
        direction: int,
        strength: int,
        link_type: str,
    ) -> int:
        """
        Posts a link between questions
        :param question1_id
        :param question2_id
        :param direction: +1 for positive, -1 for negative
        :param strength: 1 for low, 2 for medium, 5 for high
        :param link_type: only supports "causal" for now
        :return: id of the created link
        """
        self._sleep_between_requests()
        response = requests.post(
            f"{self.base_url}/coherence/links/create/",
            json={
                "question1_id": question1_id,
                "question2_id": question2_id,
                "direction": direction,
                "strength": strength,
                "type": link_type,
            },
            **self._get_auth_headers(),  # type: ignore
            timeout=self.timeout,
        )
        logger.info(f"Posted question link between {question1_id} and {question2_id}")
        raise_for_status_with_additional_info(response)
        content = json.loads(response.content)
        return content["id"]

    @retry_with_exponential_backoff()
    def get_links_for_question(self, question_id: int) -> List[DetailedCoherenceLink]:
        """
        Returns all links associated with a specific question
        direction is +1 for positive and -1 for negative
        strength is 1 for low, 2 for medium and 5 for high
        """
        self._sleep_between_requests()
        response = requests.get(
            f"{self.base_url}/coherence/question/{question_id}/links/",
            **self._get_auth_headers(),  # type: ignore
            timeout=self.timeout,
        )
        raise_for_status_with_additional_info(response)
        content = json.loads(response.content)["data"]
        links = [
            DetailedCoherenceLink.from_metaculus_api_json(link) for link in content
        ]
        return links

    @retry_with_exponential_backoff()
    def delete_question_link(self, link_id: int):
        self._sleep_between_requests()
        response = requests.delete(
            f"{self.base_url}/coherence/links/{link_id}/delete/",
            **self._get_auth_headers(),  # type: ignore
            timeout=self.timeout,
        )
        logger.info(f"Deleted question link with id {link_id}")
        raise_for_status_with_additional_info(response)

    @retry_with_exponential_backoff()
    def get_needs_update_questions(
        self,
        question_id: int,
        last_datetime: datetime,
        user_id_for_links: int | None = None,
    ) -> NeedsUpdateResponse:
        self._sleep_between_requests()
        json_data: dict[str, Any] = {"datetime": last_datetime.isoformat()}
        if user_id_for_links:
            json_data["user_id_for_links"] = user_id_for_links
        response = requests.get(
            f"{self.base_url}/coherence/question/{question_id}/links/needs-update/",
            **self._get_auth_headers(),  # type: ignore
            timeout=self.timeout,
            json=json_data,
        )
        raise_for_status_with_additional_info(response)
        content = json.loads(response.content)
        questions = [
            DataOrganizer.get_question_from_question_json(json_q)
            for json_q in content["questions"]
        ]
        links = [
            CoherenceLink.model_validate(json_link) for json_link in content["links"]
        ]
        result = NeedsUpdateResponse(questions=questions, links=links)
        return result

    def _get_auth_headers(self) -> dict[str, dict[str, str]]:
        if self.token is None:
            raise ValueError(
                "METACULUS_TOKEN environment variable and/or token field not set"
            )
        headers = {
            "headers": {
                "Authorization": f"Token {self.token}",
                "Accept-Language": "en",
            }
        }
        return headers

    @retry_with_exponential_backoff()
    def _post_question_prediction(
        self, question_id: int, forecast_payload: dict
    ) -> None:
        url = f"{self.base_url}/questions/forecast/"
        self._sleep_between_requests()
        response = requests.post(
            url,
            json=[
                {
                    "question": question_id,
                    "source": "api",
                    **forecast_payload,
                },
            ],
            **self._get_auth_headers(),  # type: ignore
            timeout=self.timeout,
        )
        logger.info(f"Posted prediction on question {question_id}")
        raise_for_status_with_additional_info(response)

    @retry_with_exponential_backoff()
    def _get_questions_from_api(
        self, params: dict[str, Any], group_question_mode: GroupQuestionMode
    ) -> list[MetaculusQuestion]:
        self._sleep_between_requests()
        num_requested = params.get("limit")
        assert (
            num_requested is None
            or num_requested <= self.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
        ), "You cannot get more than 100 questions at a time"
        url = f"{self.base_url}/posts/"
        response = requests.get(url, params=params, **self._get_auth_headers(), timeout=self.timeout)  # type: ignore
        raise_for_status_with_additional_info(response)
        data = json.loads(response.content)
        results = data["results"]
        supported_posts = [q for q in results if "notebook" not in q]
        removed_posts = [post for post in results if post not in supported_posts]
        if len(removed_posts) > 0:
            logger.warning(
                f"Removed {len(removed_posts)} posts that "
                "are not supported (e.g. notebook or conditional question)"
            )

        questions: list[MetaculusQuestion] = []
        for q in supported_posts:
            try:
                new_questions = self._post_json_to_questions_while_handling_groups(
                    q, group_question_mode
                )
                questions.extend(new_questions)
            except Exception as e:
                logger.warning(
                    f"Error processing post ID {q['id']}: {e.__class__.__name__} {e}"
                )

        return questions

    def _post_json_to_questions_while_handling_groups(
        self, post_json_from_api: dict, group_question_mode: GroupQuestionMode
    ) -> list[MetaculusQuestion]:
        if "group_of_questions" in post_json_from_api:
            if group_question_mode == "exclude":
                logger.debug(
                    f"Excluding group question post {post_json_from_api['id']}"
                )
                return []
            elif group_question_mode == "unpack_subquestions":
                if "group_of_questions" in post_json_from_api:
                    logger.debug(
                        f"Unpacking subquestions for group question post {post_json_from_api['id']}"
                    )
                    questions = self._unpack_group_question(post_json_from_api)
                    return questions
            else:
                raise ValueError("group_question_mode option not supported")
        elif "conditional" in post_json_from_api:
            post_json_from_api["question"] = post_json_from_api["conditional"]
            post_json_from_api["question"]["type"] = "conditional"
        return [DataOrganizer.get_question_from_post_json(post_json_from_api)]

    @staticmethod
    def _unpack_group_question(post_json_from_api: dict) -> list[MetaculusQuestion]:
        group_json: dict = post_json_from_api["group_of_questions"]
        questions: list[MetaculusQuestion] = []
        question_jsons: list[dict] = group_json["questions"]
        question_ids: list[int] = [q["id"] for q in question_jsons]
        for question_json in question_jsons:
            # Reformat the json to make it look like a normal post
            new_question_json = copy.deepcopy(question_json)
            new_question_json["fine_print"] = group_json["fine_print"]
            new_question_json["description"] = group_json["description"]
            new_question_json["resolution_criteria"] = group_json["resolution_criteria"]

            new_post_json = copy.deepcopy(post_json_from_api)
            new_post_json["question"] = new_question_json

            question_obj = DataOrganizer.get_question_from_post_json(new_post_json)
            question_obj.question_ids_of_group = question_ids.copy()
            questions.append(question_obj)
        return questions

    async def _filter_using_randomized_strategy(
        self,
        api_filter: ApiFilter,
        num_questions: int,
        error_if_not_enough_questions: bool,
    ) -> list[MetaculusQuestion]:
        number_of_questions_matching_filter = (
            self._determine_how_many_questions_match_filter(api_filter)
        )
        if (
            number_of_questions_matching_filter < num_questions
            and error_if_not_enough_questions
        ):
            raise ValueError(
                f"Not enough questions matching filter ({number_of_questions_matching_filter} before local filtering) to sample {num_questions} questions. Set error_if_not_enough_questions to False to return as many as possible"
            )

        questions_per_page = self.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
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
            page_questions, _ = self._grab_filtered_questions_with_offset(
                api_filter, offset
            )
            questions.extend(page_questions)

        if len(questions) < num_questions and error_if_not_enough_questions:
            raise ValueError(
                f"Exhausted all {total_pages} pages but only found {len(questions)} questions, needed {num_questions}. Set error_if_not_enough_questions to False to return as many as possible"
            )
        assert len(set(q.id_of_question for q in questions)) == len(
            questions
        ), "Not all questions found are unique"

        if len(questions) > num_questions:
            random_sample = random.sample(questions, num_questions)
        else:
            random_sample = questions
        logger.info(
            f"Sampled {len(random_sample)} questions from {len(questions)} questions that matched the filter which were taken from {total_pages} randomly selected pages which each had at max {self.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST} questions matching the filter"
        )

        return random_sample

    async def _filter_sequential_strategy(
        self, api_filter: ApiFilter, num_questions: int | None
    ) -> list[MetaculusQuestion]:
        if num_questions is None:
            questions, _ = self._grab_filtered_questions_with_offset(api_filter, 0)
            return questions

        questions: list[MetaculusQuestion] = []
        more_questions_available = True
        page_num = 0
        while len(questions) < num_questions and more_questions_available:
            offset = page_num * self.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST
            new_questions, continue_searching = (
                self._grab_filtered_questions_with_offset(api_filter, offset)
            )
            questions.extend(new_questions)
            if not continue_searching:
                more_questions_available = False
            page_num += 1
        return questions[:num_questions]

    def _determine_how_many_questions_match_filter(self, filter: ApiFilter) -> int:
        """
        Search Metaculus API with binary search to find the number of questions
        matching the filter.
        """
        estimated_max_questions = 20000
        left, right = 0, estimated_max_questions
        last_successful_offset = 0

        while left <= right:
            mid = (left + right) // 2
            offset = mid * self.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST

            _, found_questions_before_running_local_filter = (
                self._grab_filtered_questions_with_offset(filter, offset)
            )

            if found_questions_before_running_local_filter:
                left = mid + 1
                last_successful_offset = offset
            else:
                right = mid - 1

        final_page_questions, _ = self._grab_filtered_questions_with_offset(
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

    def _grab_filtered_questions_with_offset(
        self,
        api_filter: ApiFilter,
        offset: int = 0,
    ) -> tuple[list[MetaculusQuestion], bool]:
        url_params = self._create_url_params_for_search(api_filter, offset)
        questions = self._get_questions_from_api(
            url_params, api_filter.group_question_mode
        )
        questions_were_found_before_local_filter = len(questions) > 0
        filtered_questions = self._apply_local_filters(questions, api_filter)
        return filtered_questions, questions_were_found_before_local_filter

    def _create_url_params_for_search(
        self, api_filter: ApiFilter, offset: int = 0
    ) -> dict[str, Any]:
        url_params: dict[str, Any] = {
            "limit": self.MAX_QUESTIONS_FROM_QUESTION_API_PER_REQUEST,
            "offset": offset,
            "order_by": api_filter.order_by,
            "with_cp": "true",
            "include_conditional_cps": "true",
        }

        if api_filter.allowed_types:
            type_filter: list[QuestionFullType] = api_filter.allowed_types  # type: ignore
            if api_filter.group_question_mode == "unpack_subquestions":
                type_filter.extend(["group_of_questions"])
            url_params["forecast_type"] = type_filter

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

        if api_filter.is_previously_forecasted_by_user:
            user_id = self.get_current_user_id()
            if user_id:
                url_params["forecaster_id"] = user_id

        url_params.update(api_filter.other_url_parameters)
        return url_params

    def _apply_local_filters(
        self, input_questions: list[MetaculusQuestion], api_filter: ApiFilter
    ) -> list[MetaculusQuestion]:
        questions = copy.deepcopy(input_questions)

        if api_filter.is_in_main_feed is not None:
            questions = self._filter_questions_by_is_in_main_feed(
                questions, api_filter.is_in_main_feed
            )

        if api_filter.allowed_types:
            questions = self._filter_questions_by_type(
                questions, api_filter.allowed_types
            )

        if api_filter.allowed_subquestion_types is not None:
            questions = self._filter_questions_by_subquestions_type(
                questions, api_filter.allowed_subquestion_types
            )

        if api_filter.allowed_statuses:
            questions = self._filter_by_status(questions, api_filter.allowed_statuses)

        if api_filter.num_forecasters_gte is not None:
            questions = self._filter_questions_by_forecasters(
                questions, api_filter.num_forecasters_gte
            )

        if api_filter.community_prediction_exists is not None:
            if not any(t in api_filter.allowed_types for t in ["binary"]):
                raise ValueError(
                    "Community prediction filter only works for binary questions at the moment"
                )
            questions = typeguard.check_type(questions, list[BinaryQuestion])
            questions = self._filter_questions_by_community_prediction_exists(
                questions, api_filter.community_prediction_exists
            )
            questions = typeguard.check_type(questions, list[MetaculusQuestion])

        if api_filter.close_time_gt or api_filter.close_time_lt:
            questions = self._filter_questions_by_close_time(
                questions, api_filter.close_time_gt, api_filter.close_time_lt
            )

        if api_filter.open_time_gt or api_filter.open_time_lt:
            questions = self._filter_questions_by_open_time(
                questions, api_filter.open_time_gt, api_filter.open_time_lt
            )

        if api_filter.scheduled_resolve_time_gt or api_filter.scheduled_resolve_time_lt:
            questions = self._filter_by_scheduled_resolve_time(
                questions,
                api_filter.scheduled_resolve_time_gt,
                api_filter.scheduled_resolve_time_lt,
            )

        if api_filter.includes_bots_in_aggregates is not None:
            questions = self._filter_questions_by_includes_bots_in_aggregates(
                questions, api_filter.includes_bots_in_aggregates
            )

        if api_filter.cp_reveal_time_gt or api_filter.cp_reveal_time_lt:
            questions = self._filter_questions_by_cp_reveal_time(
                questions,
                api_filter.cp_reveal_time_gt,
                api_filter.cp_reveal_time_lt,
            )

        return questions

    @staticmethod
    def _filter_questions_by_is_in_main_feed(
        questions: list[Q], is_in_main_feed: bool
    ) -> list[Q]:
        return [
            question
            for question in questions
            if question.is_in_main_feed == is_in_main_feed
        ]

    @staticmethod
    def _filter_questions_by_type(
        questions: list[Q], allowed_types: list[QuestionBasicType]
    ) -> list[Q]:
        return [
            question
            for question in questions
            if question.get_api_type_name() in allowed_types
        ]

    @staticmethod
    def _filter_group_questions_by_subquestions_type(
        questions: list[MetaculusQuestion], allowed_types: list[QuestionBasicType]
    ) -> set[int]:
        questions_by_post_id: dict[int, list[MetaculusQuestion]] = defaultdict(list)
        for question in questions:
            if question.question_ids_of_group is not None:
                questions_by_post_id[question.id_of_post].append(question)

        disallowed_question_ids: set[int] = set()
        for group_questions in questions_by_post_id.values():
            has_disallowed_type = any(
                q.get_api_type_name() not in allowed_types for q in group_questions
            )
            if has_disallowed_type:
                disallowed_question_ids.update(
                    q.id_of_question for q in group_questions
                )

        return disallowed_question_ids

    @staticmethod
    def _filter_conditional_questions_by_subquestions_type(
        questions: list[MetaculusQuestion], allowed_types: list[QuestionBasicType]
    ) -> set[int]:
        disallowed_ids: set[int] = set()

        for question in questions:
            if question.get_question_type() != "conditional":
                continue
            conditional_question: ConditionalQuestion = question  # type: ignore
            subquestions = conditional_question.get_all_subquestions().values()
            has_disallowed_type = any(
                subquestion.get_api_type_name() not in allowed_types
                for subquestion in subquestions
            )

            if has_disallowed_type:
                disallowed_ids.add(question.id_of_question)

        return disallowed_ids

    @staticmethod
    def _filter_questions_by_subquestions_type(
        questions: list[MetaculusQuestion], allowed_types: list[QuestionBasicType]
    ) -> list[MetaculusQuestion]:
        disallowed_group_questions = (
            MetaculusClient._filter_group_questions_by_subquestions_type(
                questions, allowed_types
            )
        )
        disallowed_conditional_questions = (
            MetaculusClient._filter_conditional_questions_by_subquestions_type(
                questions, allowed_types
            )
        )
        return [
            question
            for question in questions
            if question.id_of_question not in disallowed_group_questions
            and question.id_of_question not in disallowed_conditional_questions
        ]

    @staticmethod
    def _filter_by_status(
        questions: list[Q], statuses: list[QuestionStateAsString]
    ) -> list[Q]:
        return [
            question
            for question in questions
            if question.state is not None and question.state.value in statuses
        ]

    @staticmethod
    def _filter_questions_by_forecasters(
        questions: list[Q], min_forecasters: int
    ) -> list[Q]:
        questions_with_enough_forecasters: list[Q] = []
        for question in questions:
            assert question.num_forecasters is not None
            if question.num_forecasters >= min_forecasters:
                questions_with_enough_forecasters.append(question)
        return questions_with_enough_forecasters

    @staticmethod
    def _filter_questions_by_includes_bots_in_aggregates(
        questions: list[Q], includes_bots_in_aggregates: bool
    ) -> list[Q]:
        return [
            question
            for question in questions
            if question.includes_bots_in_aggregates == includes_bots_in_aggregates
        ]

    @staticmethod
    def _filter_questions_by_community_prediction_exists(
        questions: list[BinaryQuestion], community_prediction_exists: bool
    ) -> list[BinaryQuestion]:
        return [
            question
            for question in questions
            if (question.community_prediction_at_access_time is not None)
            == community_prediction_exists
        ]

    @staticmethod
    def _filter_by_date(
        questions: list[Q],
        date_field_getter: Callable[[Q], datetime | None],
        date_gt: datetime | None,
        date_lt: datetime | None,
    ) -> list[Q]:
        if date_gt and date_gt.tzinfo is None:
            date_gt = date_gt.replace(tzinfo=pendulum.timezone("UTC"))
        if date_lt and date_lt.tzinfo is None:
            date_lt = date_lt.replace(tzinfo=pendulum.timezone("UTC"))

        filtered_questions: list[Q] = []
        for question in questions:
            question_date = date_field_getter(question)
            if question_date is not None:
                if date_gt and question_date <= date_gt:
                    continue
                if date_lt and question_date >= date_lt:
                    continue
                filtered_questions.append(question)
        return filtered_questions

    def _filter_questions_by_close_time(
        self,
        questions: list[Q],
        close_time_gt: datetime | None,
        close_time_lt: datetime | None,
    ) -> list[Q]:
        return self._filter_by_date(
            questions,
            lambda q: q.close_time,
            close_time_gt,
            close_time_lt,
        )

    def _filter_by_scheduled_resolve_time(
        self,
        questions: list[Q],
        scheduled_resolve_time_gt: datetime | None,
        scheduled_resolve_time_lt: datetime | None,
    ) -> list[Q]:
        return self._filter_by_date(
            questions,
            lambda q: q.scheduled_resolution_time,
            scheduled_resolve_time_gt,
            scheduled_resolve_time_lt,
        )

    def _filter_questions_by_open_time(
        self,
        questions: list[Q],
        open_time_gt: datetime | None,
        open_time_lt: datetime | None,
    ) -> list[Q]:
        return self._filter_by_date(
            questions,
            lambda q: q.open_time,
            open_time_gt,
            open_time_lt,
        )

    def _filter_questions_by_cp_reveal_time(
        self,
        questions: list[Q],
        cp_reveal_time_gt: datetime | None,
        cp_reveal_time_lt: datetime | None,
    ) -> list[Q]:
        return self._filter_by_date(
            questions,
            lambda q: q.cp_reveal_time,
            cp_reveal_time_gt,
            cp_reveal_time_lt,
        )

    @classmethod
    def dev(cls) -> MetaculusClient:
        return cls(base_url="https://dev.metaculus.com/api")

    @classmethod
    def local(cls) -> MetaculusClient:
        return cls(base_url="http://127.0.0.1:8000/api")

    def _sleep_between_requests(self) -> None:
        higher_bound = self.sleep_time_between_requests_min + self.sleep_jitter_seconds
        lower_bound = self.sleep_time_between_requests_min
        random_sleep_time = random.uniform(lower_bound, higher_bound)
        logger.debug(
            f"Sleeping for {random_sleep_time:.1f} seconds before next request"
        )
        time.sleep(random_sleep_time)

    #################### ADMIN FUNCTIONS ####################

    @retry_with_exponential_backoff()
    def resolve_question(
        self, question_id: int, resolution: str | None, resolve_time: datetime
    ) -> None:
        self._sleep_between_requests()
        if resolution is None:
            logger.warning(
                f"Resolution is None, Question ID {question_id} not resolved"
            )
            return
        response = requests.post(
            f"{self.base_url}/questions/{question_id}/resolve/",
            json={
                "resolution": resolution,
                "actual_resolve_time": resolve_time.strftime("%Y-%m-%dT%H:%M:%S"),
            },  # TODO: Find all instances of strftime that doesn't use timezone, and add timezone
            **self._get_auth_headers(),  # type: ignore
            timeout=self.timeout,
        )
        raise_for_status_with_additional_info(response)
        logger.info(f"Resolved question ID {question_id} with resolution {resolution}")

    @retry_with_exponential_backoff()
    def unresolve_question(self, question_id: int) -> None:
        self._sleep_between_requests()
        response = requests.post(
            f"{self.base_url}/questions/{question_id}/unresolve/",
            **self._get_auth_headers(),  # type: ignore
            timeout=self.timeout,
        )
        raise_for_status_with_additional_info(response)
        logger.info(f"Unresolved question ID {question_id}")

    @retry_with_exponential_backoff()
    def create_question(self, question: MetaculusQuestion) -> MetaculusQuestion:
        question_data = self._get_post_create_data(question)
        self._sleep_between_requests()
        response = requests.post(
            f"{self.base_url}/posts/create/",
            json=question_data,
            headers={
                "accept": "application/json",
                "Authorization": f"Token {self.token}",
            },
            timeout=self.timeout,
        )

        raise_for_status_with_additional_info(response)
        created_question_list = self._post_json_to_questions_while_handling_groups(
            json.loads(response.content), "exclude"
        )
        if len(created_question_list) != 1:
            raise ValueError(
                f"Expected 1 question to be created, got {len(created_question_list)}. Questions created: {[question.page_url for question in created_question_list]}"
            )
        single_created_question = created_question_list[0]
        logger.info(f"Created new questions: {single_created_question.page_url}")

        assert single_created_question.id_of_post is not None
        full_created_question = self.get_question_by_post_id(
            single_created_question.id_of_post
        )

        return full_created_question

    @retry_with_exponential_backoff()
    def approve_question(self, question: MetaculusQuestion) -> None:
        self._sleep_between_requests()
        response = requests.post(
            f"{self.base_url}/posts/{question.id_of_post}/approve/",
            data={
                "published_at": (
                    question.published_time.strftime("%Y-%m-%dT%H:%M:%S")
                    if question.published_time
                    else None
                ),
                "open_time": (
                    question.open_time.strftime("%Y-%m-%dT%H:%M:%S")
                    if question.open_time
                    else None
                ),
                "cp_reveal_time": (
                    question.cp_reveal_time.strftime("%Y-%m-%dT%H:%M:%S")
                    if question.cp_reveal_time
                    else None
                ),
                "scheduled_close_time": (
                    question.close_time.strftime("%Y-%m-%dT%H:%M:%S")
                    if question.close_time
                    else None
                ),
                "scheduled_resolve_time": (
                    question.scheduled_resolution_time.strftime("%Y-%m-%dT%H:%M:%S")
                    if question.scheduled_resolution_time
                    else None
                ),
            },
            timeout=self.timeout,
            headers={
                "accept": "application/json",
                "Authorization": f"Token {self.token}",
            },
        )

        raise_for_status_with_additional_info(response)
        logger.info(f"Approved question {question.page_url}")

    @staticmethod
    def _get_post_create_data(question: MetaculusQuestion) -> dict:
        if question.published_time is None:
            publish_time = question.open_time
        else:
            publish_time = question.published_time

        return {
            "title": question.question_text,
            "short_title": question.custom_metadata.get("short_name", None)
            or question.question_text,
            "default_project": question.default_project_id,
            "categories": [category.id for category in question.categories],
            "published_at": (
                publish_time.strftime("%Y-%m-%dT%H:%M:%S") if publish_time else None
            ),
            "is_automatically_translated": False,
            "question": {
                "title": question.question_text,
                "description": question.background_info or "",
                "type": getattr(question, "question_type", "binary"),
                "possibilities": None,  # deprecated
                "resolution": None,
                "include_bots_in_aggregates": (
                    question.includes_bots_in_aggregates
                    if isinstance(question.includes_bots_in_aggregates, bool)
                    else True  # defaults to True since mini bench is a bot thing
                ),
                "default_aggregation_method": "unweighted",
                "default_score_type": "spot_peer",
                "question_weight": (
                    question.question_weight if question.question_weight else 1.0
                ),
                "scaling": {
                    "range_max": getattr(question, "upper_bound", None),
                    "range_min": getattr(question, "lower_bound", None),
                    "zero_point": getattr(question, "zero_point", None),
                },
                "open_upper_bound": getattr(question, "open_upper_bound", None),
                "open_lower_bound": getattr(question, "open_lower_bound", None),
                "inbound_outcome_count": getattr(question, "cdf_size", 201) - 1,
                "options": getattr(question, "options", None),
                "group_variable": getattr(question, "option_is_instance_of", None)
                or "",
                "label": "",  # only group questions
                "resolution_criteria": question.resolution_criteria or "",
                "open_time": (
                    question.open_time.strftime("%Y-%m-%dT%H:%M:%S")
                    if question.open_time
                    else None
                ),
                "cp_reveal_time": (
                    question.cp_reveal_time.strftime("%Y-%m-%dT%H:%M:%S")
                    if question.cp_reveal_time
                    else None
                ),
                "scheduled_close_time": (
                    question.close_time.strftime("%Y-%m-%dT%H:%M:%S")
                    if question.close_time
                    else None
                ),
                "scheduled_resolve_time": (
                    question.scheduled_resolution_time.strftime("%Y-%m-%dT%H:%M:%S")
                    if question.scheduled_resolution_time
                    else None
                ),
                "unit": getattr(question, "unit_of_measure", None) or "",
                "fine_print": question.fine_print or "",
                "group_rank": None,  # only group questions
            },
        }
