import asyncio

from forecasting_tools.benchmarking.question_research_snapshot import (
    QuestionResearchSnapshot,
)
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.util.custom_logger import CustomLogger


async def snapshot_questions() -> None:
    num_questions_to_use = 500
    chosen_questions = MetaculusApi.get_benchmark_questions(
        num_questions_to_use,
    )
    snapshots = []
    file_name = (
        f"logs/forecasts/question_snapshots_v1_{num_questions_to_use}qs.json"
    )
    for question in chosen_questions:
        snapshot = await QuestionResearchSnapshot.create_snapshot_of_question(
            question
        )
        snapshots.append(snapshot)
        if len(snapshots) % 20 == 0:
            QuestionResearchSnapshot.save_object_list_to_file_path(
                snapshots, file_name
            )
    QuestionResearchSnapshot.save_object_list_to_file_path(
        snapshots, file_name
    )


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(snapshot_questions())
