import asyncio
import json

from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.helpers.metaculus_client import ApiFilter, MetaculusClient


async def main() -> None:
    fall_aib_questions = await MetaculusClient().get_questions_matching_filter(
        api_filter=ApiFilter(
            allowed_statuses=["resolved"],
            allowed_types=["binary"],
            group_question_mode="unpack_subquestions",
            allowed_tournaments=[MetaculusClient.CURRENT_AI_COMPETITION_ID],
        ),
        num_questions=1000,
        error_if_question_target_missed=False,
    )
    matching_real_site_questions = []
    interesting_question_pairs: list[tuple[BinaryQuestion, BinaryQuestion]] = []
    for ai_question in fall_aib_questions:
        assert ai_question.background_info is not None
        assert isinstance(ai_question, BinaryQuestion)
        end_json = ai_question.background_info.split("\n")[-1].strip("`")
        end_json = json.loads(end_json)
        post_id = int(end_json["info"]["post_id"])
        real_site_question = MetaculusClient().get_question_by_post_id(post_id)
        assert isinstance(real_site_question, BinaryQuestion)
        matching_real_site_questions.append(real_site_question)
        assert (
            ai_question.community_prediction_at_access_time is not None
            and real_site_question.community_prediction_at_access_time is not None
        )
        difference_in_community_prediction = (
            ai_question.community_prediction_at_access_time
            - real_site_question.community_prediction_at_access_time
        )
        print(
            f"Difference in community prediction: {difference_in_community_prediction} - {ai_question.question_text}"
        )
        if difference_in_community_prediction > 0.3:
            interesting_question_pairs.append((ai_question, real_site_question))
    print(f"Found {len(interesting_question_pairs)} interesting question pairs")
    for ai_question, real_site_question in interesting_question_pairs:
        print(f"AI question: {ai_question.community_prediction_at_access_time}")
        print(
            f"Real site question: {real_site_question.community_prediction_at_access_time}"
        )
        print(f"AI question: {ai_question.question_text}")
        print(f"AI URL: {ai_question.page_url}")
        print(f"Real site URL: {real_site_question.page_url}")
        print("-" * 100)
        print(real_site_question.question_text)
        print("-" * 100)


if __name__ == "__main__":
    asyncio.run(main())
