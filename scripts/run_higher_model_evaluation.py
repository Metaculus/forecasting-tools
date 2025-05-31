import asyncio
import logging

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.benchmarking.prompt_evaluator import PromptEvaluator
from forecasting_tools.benchmarking.question_research_snapshot import (
    QuestionResearchSnapshot,
    ResearchType,
)
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def run_higher_model_evaluation() -> None:
    # --- Evaluation Parameters ---
    evaluation_questions = QuestionResearchSnapshot.load_json_from_file_path(
        "logs/forecasts/question_snapshots_v1.6.test__230qs.json"
    )

    questions_batch_size = 115
    forecast_llm = GeneralLlm(
        model="openrouter/openai/gpt-4.1-nano",
        temperature=0.3,
    )
    benchmark_files = [
        "logs/forecasts/benchmarks/benchmarks_prompt_optimization_v2.1__nano_112qs.json",
        "logs/forecasts/benchmarks/benchmarks_prompt_optimization_v2.2__nano_112qs.json",
        "logs/forecasts/benchmarks/benchmarks_prompt_optimization_v2.3__nano_112qs.json",
        "logs/forecasts/benchmarks/benchmarks_prompt_optimization_v2.4__nano_112qs.json",
        "logs/forecasts/benchmarks/benchmarks_prompt_optimization_v2.5__nano_112qs.json",
    ]
    top_n_prompts = 3
    include_worse_benchmark = True
    research_reports_per_question = 1
    num_predictions_per_research_report = 1

    # --- Run the evaluation ---
    evaluator = PromptEvaluator(
        evaluation_questions=evaluation_questions,
        research_type=ResearchType.ASK_NEWS_SUMMARIES,
        concurrent_evaluation_batch_size=questions_batch_size,
        file_or_folder_to_save_benchmarks="logs/forecasts/benchmarks/",
    )
    evaluation_result = await evaluator.evaluate_best_benchmarked_prompts(
        forecast_llm=forecast_llm,
        benchmark_files=benchmark_files,
        top_n_prompts=top_n_prompts,
        include_control_group_prompt=True,
        include_worst_prompt=include_worse_benchmark,
        research_reports_per_question=research_reports_per_question,
        num_predictions_per_research_report=num_predictions_per_research_report,
    )
    for evaluated_prompt in evaluation_result.evaluated_prompts:
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
    asyncio.run(run_higher_model_evaluation())
