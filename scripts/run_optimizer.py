import asyncio
import logging

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.benchmarking.prompt_optimizer import PromptOptimizer
from forecasting_tools.benchmarking.question_research_snapshot import (
    QuestionResearchSnapshot,
)
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def run_optimizer() -> None:
    evaluation_questions = QuestionResearchSnapshot.load_json_from_file_path(
        "logs/forecasts/question_snapshots_v2_157qs.json"
    )
    forecast_llm = GeneralLlm(
        model="openrouter/openai/gpt-4.1-nano",
        temperature=0.3,
    )
    ideation_llm = "openrouter/google/gemini-2.5-pro-preview"
    for snapshot in evaluation_questions:
        snapshot.question.background_info = None

    logger.info(f"Loaded {len(evaluation_questions)} evaluation questions")
    optimizer = PromptOptimizer(
        evaluation_questions=evaluation_questions,
        num_prompts_to_try=10,
        forecast_llm=forecast_llm,
        ideation_llm_name=ideation_llm,
        file_or_folder_to_save_benchmarks="logs/forecasts/benchmarks/",
    )
    evaluation_result = await optimizer.create_optimized_prompt()
    evaluated_prompts = evaluation_result.evaluated_prompts
    for evaluated_prompt in evaluated_prompts:
        logger.info(
            f"Name: {evaluated_prompt.prompt_config.original_idea.short_name}"
        )
        logger.info(f"Config: {evaluated_prompt.prompt_config}")
        logger.info(f"Code: {evaluated_prompt.benchmark.code}")
        logger.info(
            f"Forecast Bot Class Name: {evaluated_prompt.benchmark.forecast_bot_class_name}"
        )
        logger.info(f"Cost: {evaluated_prompt.benchmark.total_cost}")
        logger.info(f"Score: {evaluated_prompt.score}")

    logger.info(f"Best prompt: {evaluation_result.best_prompt}")


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(run_optimizer())
