import asyncio
import logging
import math
import random
from datetime import datetime

import typeguard

from forecasting_tools.auto_optimizers.question_plus_research import (
    QuestionPlusResearch,
)
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.helpers.metaculus_api import MetaculusApi
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def grab_questions(include_research: bool) -> None:
    # --- Parameters ---
    target_questions_to_use = 500
    target_training_size = 50
    chosen_questions = MetaculusApi.get_benchmark_questions(
        target_questions_to_use,
        max_days_since_opening=365 + 180,
        days_to_resolve_in=None,
        num_forecasters_gte=15,
        error_if_question_target_missed=False,
    )
    train_test_base_file_name = "logs/forecasts/benchmarks/questions_v4.0"
    file_name = f"{train_test_base_file_name}_{len(chosen_questions)}qs__>15f__<1.5yr_open__{datetime.now().strftime('%Y-%m-%d')}.json"
    batch_size = 20

    # --- Validate the questions ---
    logger.info(f"Retrieved {len(chosen_questions)} questions")
    for question in chosen_questions:
        assert question.community_prediction_at_access_time is not None

    # --- Execute the research snapshotting if needed ---
    if not include_research:
        chosen_questions = typeguard.check_type(
            chosen_questions, list[MetaculusQuestion]
        )
        DataOrganizer.save_questions_to_file_path(chosen_questions, file_name)
    else:
        snapshots: list[QuestionPlusResearch] = []

        num_batches = math.ceil(len(chosen_questions) / batch_size)
        for batch_index in range(num_batches):
            batch_questions = chosen_questions[
                batch_index * batch_size : (batch_index + 1) * batch_size
            ]
            batch_snapshots = await asyncio.gather(
                *[
                    QuestionPlusResearch.create_snapshot_of_question(question)
                    for question in batch_questions
                ]
            )
            snapshots.extend(batch_snapshots)
            random.shuffle(snapshots)
            QuestionPlusResearch.save_object_list_to_file_path(snapshots, file_name)
            logger.info(f"Saved {len(snapshots)} snapshots to {file_name}")
        QuestionPlusResearch.save_object_list_to_file_path(snapshots, file_name)

    # --- Split into train and test ---
    if not include_research:
        train_test_questions = chosen_questions
    else:
        train_test_questions = [snapshot.question for snapshot in snapshots]
    train_test_questions = typeguard.check_type(
        train_test_questions, list[MetaculusQuestion]
    )

    test_size = len(train_test_questions) - target_training_size
    random.shuffle(train_test_questions)
    train_questions = train_test_questions[:target_training_size]
    test_questions = train_test_questions[target_training_size:]
    assert len(train_test_questions) == target_training_size + test_size
    DataOrganizer.save_questions_to_file_path(
        train_questions,
        f"{train_test_base_file_name}.train__{target_training_size}qs.json",
    )
    DataOrganizer.save_questions_to_file_path(
        test_questions, f"{train_test_base_file_name}.test__{test_size}qs.json"
    )

    # --- Visualize and randomly sample questions ---
    random.shuffle(train_test_questions)
    for question in train_test_questions:
        logger.info(f"URL: {question.page_url} - Question: {question.question_text}")


if __name__ == "__main__":
    CustomLogger.setup_logging()

    chosen_mode = input("Include research? (y/n): ")
    if chosen_mode == "y":
        include_research = True
    elif chosen_mode == "n":
        include_research = False
    else:
        raise ValueError("Invalid mode")
    asyncio.run(grab_questions(include_research))
