import asyncio
import json
import logging
from datetime import datetime

from pydantic import BaseModel

from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.helpers.metaculus_client import ApiFilter, MetaculusClient
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


class QuestionPair(BaseModel):
    ai_question: BinaryQuestion
    real_site_question: BinaryQuestion

    @property
    def comparison_time(self) -> datetime:
        assert self.ai_question.close_time is not None
        return self.ai_question.close_time

    @property
    def real_site_cp(self) -> float:
        real_site_cp = self.real_site_question.get_cp_at_time(self.comparison_time)
        assert real_site_cp is not None
        return real_site_cp

    @property
    def ai_cp(self) -> float:
        assert self.ai_question.community_prediction_at_access_time is not None
        return self.ai_question.community_prediction_at_access_time

    @property
    def difference_in_community_prediction(self) -> float:
        assert self.ai_cp is not None
        assert self.real_site_cp is not None
        return abs(self.ai_cp - self.real_site_cp)


async def main() -> None:
    api_filter = ApiFilter(
        allowed_statuses=["resolved"],
        allowed_types=["binary"],
        group_question_mode="unpack_subquestions",
        allowed_tournaments=[MetaculusClient.CURRENT_AI_COMPETITION_ID],
        other_url_parameters={"include_cp_history": "true"},
    )
    # questions_for_interestingness = (
    #     await MetaculusClient().get_questions_matching_filter(
    #         api_filter,
    #         num_questions=1000,
    #         error_if_question_target_missed=False,
    #     )
    # )
    # question_pairs = get_matching_real_site_questions(questions_for_interestingness)  # type: ignore
    # find_interesting_question_pairs(question_pairs)

    new_api_filter = api_filter.model_copy(
        update={"allowed_statuses": ["open", "resolved", "closed"]}
    )
    questions_for_low_probability = (
        await MetaculusClient().get_questions_matching_filter(
            new_api_filter,
            num_questions=1000,
            error_if_question_target_missed=False,
        )
    )
    new_question_pairs = get_matching_real_site_questions(questions_for_low_probability)  # type: ignore
    display_extreme_probability_site_questions(new_question_pairs)


def get_matching_real_site_questions(
    fall_aib_questions: list[BinaryQuestion],
) -> list[QuestionPair]:
    question_pairs = []
    for ai_question in fall_aib_questions:
        assert ai_question.background_info is not None
        assert isinstance(ai_question, BinaryQuestion)

        try:
            end_json = ai_question.background_info.split("\n")[-1].strip("`")
            end_json = json.loads(end_json)
        except Exception as e:
            logger.error(
                f"Error parsing background info for {ai_question.page_url}: {e}"
            )
            continue

        post_id = int(end_json["info"]["post_id"])
        real_site_questions = MetaculusClient().get_question_by_post_id(
            post_id, group_question_mode="unpack_subquestions"
        )
        if ai_question.resolution_string == "annulled":
            continue
        if isinstance(real_site_questions, list):
            continue
        real_site_question = real_site_questions
        assert isinstance(real_site_question, BinaryQuestion)
        assert (
            ai_question.community_prediction_at_access_time is not None
            and real_site_question.community_prediction_at_access_time is not None
        ), f"Community prediction is None for {ai_question.page_url} or {real_site_question.page_url}"
        question_pairs.append(
            QuestionPair(ai_question=ai_question, real_site_question=real_site_question)
        )
    return question_pairs


def find_interesting_question_pairs(
    question_pairs: list[QuestionPair],
) -> None:
    interesting_question_pairs: list[QuestionPair] = []
    for pair in question_pairs:
        if pair.difference_in_community_prediction > 0.15:
            interesting_question_pairs.append(pair)
        logger.info(
            f"Difference in community prediction: {pair.difference_in_community_prediction}. Real site CP: {pair.real_site_question.page_url}, AI CP: {pair.ai_question.page_url}"
        )
    logger.info(f"Found {len(interesting_question_pairs)} interesting question pairs")
    for i, pair in enumerate(interesting_question_pairs):
        logger.info("-" * 75)
        logger.info(pair.real_site_question.question_text)
        logger.info("-" * 75)
        logger.info(f"AI question CP: {pair.ai_cp}")
        logger.info(f"Real site question CP: {pair.real_site_cp}")
        logger.info(f"Comparison time: {pair.comparison_time}")
        logger.info(f"AI question text: {pair.ai_question.question_text}")
        logger.info(f"AI URL: {pair.ai_question.page_url}")
        logger.info(f"Real site URL: {pair.real_site_question.page_url}")
        logger.info(f"Resolution string: {pair.real_site_question.resolution_string}")
    logger.info("=" * 75)


def display_extreme_probability_site_questions(
    question_pairs: list[QuestionPair],
) -> None:
    extreme_probability_pairs: list[QuestionPair] = []

    high_probability = 0.5
    low_probability = 0.5

    for pair in question_pairs:
        if pair.real_site_cp > high_probability or pair.real_site_cp < low_probability:
            extreme_probability_pairs.append(pair)
            logger.info("-" * 75)
            logger.info(
                f"Low probability site question: {pair.real_site_question.page_url}"
            )
            logger.info(f"AI question: {pair.ai_question.question_text}")
            logger.info(f"AI URL: {pair.ai_question.page_url}")
            logger.info(f"Comparison time: {pair.comparison_time}")
            logger.info(f"Real site question CP: {pair.real_site_cp}")
            logger.info(f"AI question CP: {pair.ai_cp}")
    logger.info("-" * 75)
    percentage_low_probability = (
        len(extreme_probability_pairs) / len(question_pairs) * 100
    )
    high_probability_pairs = [
        pair for pair in question_pairs if pair.real_site_cp > high_probability
    ]
    low_probability_pairs = [
        pair for pair in question_pairs if pair.real_site_cp < low_probability
    ]
    logger.info(
        f"Percentage of extreme probability site questions: {percentage_low_probability}% of {len(question_pairs)} questions"
    )
    logger.info(
        f"Percentage of high probability site questions: {len(high_probability_pairs) / len(question_pairs) * 100}% of {len(question_pairs)} questions"
    )
    logger.info(
        f"Percentage of low probability site questions: {len(low_probability_pairs) / len(question_pairs) * 100}% of {len(question_pairs)} questions"
    )
    logger.info(
        f"Average extreme probability site question CP: {sum(pair.real_site_cp for pair in extreme_probability_pairs) / len(extreme_probability_pairs)}"
    )
    logger.info(
        f"Average CP of high probability site questions: {sum(pair.real_site_cp for pair in high_probability_pairs) / len(high_probability_pairs)}"
    )
    logger.info(
        f"Average CP of low probability site questions: {sum(pair.real_site_cp for pair in low_probability_pairs) / len(low_probability_pairs)}"
    )


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(main())
