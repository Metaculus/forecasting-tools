import asyncio
import logging

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.benchmarking.prompt_evaluator import PromptEvaluator
from forecasting_tools.benchmarking.prompt_optimizer import PromptOptimizer
from forecasting_tools.benchmarking.question_research_snapshot import (
    QuestionResearchSnapshot,
    ResearchType,
)
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def run_optimizer() -> None:
    # -------- Configure the optimizer -----
    evaluation_questions = QuestionResearchSnapshot.load_json_from_file_path(
        "logs/forecasts/question_snapshots_v1.6.train__112qs.json"
    )
    forecast_llm = GeneralLlm(
        model="openrouter/openai/gpt-4.1",
        temperature=0.3,
    )
    ideation_llm = "openrouter/google/gemini-2.5-pro-preview"
    remove_background_info = True
    start_fresh_optimization_runs = 1
    num_iterations_per_run = 4
    questions_batch_size = 112

    # ----- Run the optimizer -----
    if remove_background_info:
        for snapshot in evaluation_questions:
            snapshot.question.background_info = None
    for run in range(start_fresh_optimization_runs):
        logger.info(f"Run {run + 1} of {start_fresh_optimization_runs}")
        logger.info(f"Loaded {len(evaluation_questions)} evaluation questions")
        evaluator = PromptEvaluator(
            evaluation_questions=evaluation_questions,
            research_type=ResearchType.ASK_NEWS_SUMMARIES,
            concurrent_evaluation_batch_size=questions_batch_size,
            file_or_folder_to_save_benchmarks="logs/forecasts/benchmarks/",
        )
        optimizer = PromptOptimizer(
            iterations=num_iterations_per_run,
            forecast_llm=forecast_llm,
            ideation_llm_name=ideation_llm,
            evaluator=evaluator,
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
