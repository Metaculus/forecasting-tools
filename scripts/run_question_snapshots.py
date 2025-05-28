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
    # --- Parameters ---
    target_questions_to_use = 500
    chosen_questions = MetaculusApi.get_benchmark_questions(
        target_questions_to_use,
        max_days_since_opening=365 + 180,
        days_to_resolve_in=None,
        num_forecasters_gte=20,
        error_if_question_target_missed=False,
    )
    file_name = f"logs/forecasts/question_snapshots_v1.6_{len(chosen_questions)}qs__>20f__<1.5yr_open.json"
    batch_size = 20
    raise NotImplementedError(
        f"Found {len(chosen_questions)} questions, but this script is not implemented yet"
    )

    # --- Execute the snapshotting ---
    logger.info(f"Retrieved {len(chosen_questions)} questions")
    for question in chosen_questions:
        assert question.community_prediction_at_access_time is not None
    snapshots = []

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
