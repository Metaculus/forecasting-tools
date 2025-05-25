import asyncio
import logging
import math

from forecasting_tools.benchmarking.question_research_snapshot import (
    QuestionResearchSnapshot,
)
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def snapshot_questions() -> None:
    target_questions_to_use = 500
    chosen_questions = MetaculusApi.get_benchmark_questions(
        target_questions_to_use,
        error_if_question_target_missed=False,
    )
    for question in chosen_questions:
        assert question.community_prediction_at_access_time is not None
    snapshots = []
    file_name = (
        f"logs/forecasts/question_snapshots_v2_{len(chosen_questions)}qs.json"
    )
    batch_size = 20
    num_batches = math.ceil(len(chosen_questions) / batch_size)
    for batch_index in range(num_batches):
        batch_questions = chosen_questions[
            batch_index * batch_size : (batch_index + 1) * batch_size
        ]
        batch_snapshots = await asyncio.gather(
            *[
                QuestionResearchSnapshot.create_snapshot_of_question(question)
                for question in batch_questions
            ]
        )
        snapshots.extend(batch_snapshots)
        QuestionResearchSnapshot.save_object_list_to_file_path(
            snapshots, file_name
        )
        logger.info(f"Saved {len(snapshots)} snapshots to {file_name}")
    QuestionResearchSnapshot.save_object_list_to_file_path(
        snapshots, file_name
    )


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(snapshot_questions())
