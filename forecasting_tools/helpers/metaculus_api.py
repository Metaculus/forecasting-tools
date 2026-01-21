from __future__ import annotations

from datetime import datetime
from typing import List, Literal, overload

import typing_extensions

from forecasting_tools.data_models.coherence_link import DetailedCoherenceLink
from forecasting_tools.data_models.questions import BinaryQuestion, MetaculusQuestion
from forecasting_tools.helpers.metaculus_client import (
    ApiFilter,
    GroupQuestionMode,
    MetaculusClient,
)


@typing_extensions.deprecated(
    "MetaculusApi is deprecated, please replace instances of MetaculusApi.method(...) with MetaculusClient().method(...)"
)
class MetaculusApi:
    """
    Documentation for the API can be found at https://www.metaculus.com/api/
    """

    # NOTE: The tourament slug can be used for ID as well (e.g. "aibq2" or "quarterly-cup" instead of 32721 or 32630)
    AI_WARMUP_TOURNAMENT_ID = (
        3294  # https://www.metaculus.com/tournament/ai-benchmarking-warmup/
    )
    AI_COMPETITION_ID_Q3 = 3349  # https://www.metaculus.com/tournament/aibq3/
    AI_COMPETITION_ID_Q4 = 32506  # https://www.metaculus.com/tournament/aibq4/
    AI_COMPETITION_ID_Q1 = 32627  # https://www.metaculus.com/tournament/aibq1/
    AI_COMPETITION_ID_Q2 = 32721  # https://www.metaculus.com/tournament/aibq2/
    AIB_FALL_2025_ID = 32813  # https://www.metaculus.com/tournament/fall-aib-2025/
    PRO_COMPARISON_TOURNAMENT_Q1 = 32631
    PRO_COMPARISON_TOURNAMENT_Q2 = (
        32761  # https://www.metaculus.com/tournament/pro-benchmark-q22025
    )
    ACX_2025_TOURNAMENT = 32564
    Q3_2024_QUARTERLY_CUP = 3366
    Q4_2024_QUARTERLY_CUP = 3672
    Q1_2025_QUARTERLY_CUP = 32630
    METACULUS_CUP_2025_1_ID = 32726  # Summer cup 2025
    METACULUS_CUP_FALL_2025_ID = 32828
    AI_2027_TOURNAMENT_ID = "ai-2027"
    # MAIN_FEED = 144 # site_main

    Q3_2025_MARKET_PULSE_ID = "market-pulse-25q3"
    Q4_2025_MARKET_PULSE_ID = "market-pulse-25q4"

    CURRENT_METACULUS_CUP_ID = METACULUS_CUP_FALL_2025_ID
    CURRENT_QUARTERLY_CUP_ID = CURRENT_METACULUS_CUP_ID  # Consider this parameter deprecated since quarterly cup is no longer active
    CURRENT_AI_COMPETITION_ID = AIB_FALL_2025_ID
    CURRENT_MINIBENCH_ID = "minibench"
    CURRENT_MARKET_PULSE_ID = Q4_2025_MARKET_PULSE_ID

    TEST_QUESTION_URLS = [
        "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
        "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
        "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
    ]

    METACULUS_CLIENT = MetaculusClient()

    @classmethod
    def post_question_comment(
        cls,
        post_id: int,
        comment_text: str,
        is_private: bool = True,
        included_forecast: bool = True,
    ) -> None:
        return cls.METACULUS_CLIENT.post_question_comment(
            post_id, comment_text, is_private, included_forecast
        )

    @classmethod
    def post_question_link(
        cls,
        question1_id: int,
        question2_id: int,
        direction: int,
        strength: int,
        link_type: str,
    ) -> int:
        return cls.METACULUS_CLIENT.post_question_link(
            question1_id, question2_id, direction, strength, link_type
        )

    @classmethod
    def get_links_for_question(cls, question_id: int) -> List[DetailedCoherenceLink]:
        return cls.METACULUS_CLIENT.get_links_for_question(question_id)

    @classmethod
    def delete_question_link(cls, link_id: int):
        return cls.METACULUS_CLIENT.delete_question_link(link_id)

    @classmethod
    def post_binary_question_prediction(
        cls, question_id: int, prediction_in_decimal: float
    ) -> None:
        return cls.METACULUS_CLIENT.post_binary_question_prediction(
            question_id, prediction_in_decimal
        )

    @classmethod
    def post_numeric_question_prediction(
        cls, question_id: int, cdf_values: list[float]
    ) -> None:
        """
        If the question is numeric, forecast must be a dictionary that maps
        quartiles or percentiles to datetimes, or a 201 value cdf.
        In this case we use the cdf.
        """
        return cls.METACULUS_CLIENT.post_numeric_question_prediction(
            question_id, cdf_values
        )

    @classmethod
    def post_multiple_choice_question_prediction(
        cls, question_id: int, options_with_probabilities: dict[str, float]
    ) -> None:
        """
        If the question is multiple choice, forecast must be a dictionary that
        maps question.options labels to floats.
        """
        return cls.METACULUS_CLIENT.post_multiple_choice_question_prediction(
            question_id, options_with_probabilities
        )

    @overload
    @classmethod
    def get_question_by_url(
        cls,
        question_url: str,
        group_question_mode: Literal["exclude"] = "exclude",
    ) -> MetaculusQuestion: ...

    @overload
    @classmethod
    def get_question_by_url(
        cls,
        question_url: str,
        group_question_mode: GroupQuestionMode = "exclude",
    ) -> MetaculusQuestion | list[MetaculusQuestion]: ...

    @classmethod
    def get_question_by_url(
        cls,
        question_url: str,
        group_question_mode: GroupQuestionMode = "exclude",
    ) -> MetaculusQuestion | list[MetaculusQuestion]:
        """
        Handles both question and community URLs:
        - Question: https://www.metaculus.com/questions/28841/...
        - Community: https://www.metaculus.com/c/risk/38787/
        """
        return cls.METACULUS_CLIENT.get_question_by_url(
            question_url, group_question_mode
        )

    @overload
    @classmethod
    def get_question_by_post_id(
        cls,
        post_id: int,
        group_question_mode: Literal["exclude"] = "exclude",
    ) -> MetaculusQuestion: ...

    @overload
    @classmethod
    def get_question_by_post_id(
        cls,
        post_id: int,
        group_question_mode: GroupQuestionMode = "exclude",
    ) -> MetaculusQuestion | list[MetaculusQuestion]: ...

    @classmethod
    def get_question_by_post_id(
        cls,
        post_id: int,
        group_question_mode: GroupQuestionMode = "exclude",
    ) -> MetaculusQuestion | list[MetaculusQuestion]:
        return cls.METACULUS_CLIENT.get_question_by_post_id(
            post_id, group_question_mode
        )

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
        return await cls.METACULUS_CLIENT.get_questions_matching_filter(
            api_filter, num_questions, randomly_sample, error_if_question_target_missed
        )

    @classmethod
    def get_all_open_questions_from_tournament(
        cls,
        tournament_id: int | str,
        group_question_mode: GroupQuestionMode = "unpack_subquestions",
    ) -> list[MetaculusQuestion]:
        return cls.METACULUS_CLIENT.get_all_open_questions_from_tournament(
            tournament_id, group_question_mode
        )

    @classmethod
    def get_benchmark_questions(
        cls,
        num_of_questions_to_return: int,
        days_to_resolve_in: int | None = None,
        max_days_since_opening: int | None = 365,
        num_forecasters_gte: int = 30,
        error_if_question_target_missed: bool = True,
        group_question_mode: GroupQuestionMode = "exclude",
    ) -> list[BinaryQuestion]:
        return cls.METACULUS_CLIENT.get_benchmark_questions(
            num_of_questions_to_return,
            days_to_resolve_in,
            max_days_since_opening,
            num_forecasters_gte,
            error_if_question_target_missed,
            group_question_mode,
        )

    ### ADMIN FUNCTIONS ###
    @classmethod
    def resolve_question(
        cls, question_id: int, resolution: str | None, resolve_time: datetime
    ) -> None:
        return cls.METACULUS_CLIENT.resolve_question(
            question_id, resolution, resolve_time
        )

    @classmethod
    def unresolve_question(cls, question_id: int) -> None:
        return cls.METACULUS_CLIENT.unresolve_question(question_id)

    @classmethod
    def create_question(cls, question: MetaculusQuestion) -> MetaculusQuestion:
        return cls.METACULUS_CLIENT.create_question(question)

    @classmethod
    def approve_question(cls, question: MetaculusQuestion) -> None:
        return cls.METACULUS_CLIENT.approve_question(question)
