import asyncio
import logging

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.auto_optimizers.bot_evaluator import BotEvaluator
from forecasting_tools.auto_optimizers.prompt_data_models import ResearchTool, ToolName
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def run_higher_model_evaluation() -> None:
    # --- Evaluation Parameters ---
    evaluation_questions = DataOrganizer.load_questions_from_file_path(
        "logs/forecasts/benchmarks/questions_v4.0.test__282qs.json"
    )

    questions_batch_size = 25
    both_reason_and_research_llm: list[tuple[str, GeneralLlm]] = [
        (
            "openrouter/openai/gpt-4.1-nano",
            GeneralLlm(
                model="openrouter/openai/gpt-4.1-nano",
                temperature=0.3,
            ),
        ),
    ]
    benchmark_files = [
        "logs/forecasts/benchmarks/benchmarks_research_optimization_v4.1__gpt4.1-nano__Qv4.0.train_50.jsonl",
        "logs/forecasts/benchmarks/benchmarks_research_optimization_v4.2__gpt4.1-nano__Qv4.0.train_50.jsonl",
        "logs/forecasts/benchmarks/benchmarks_research_optimization_v4.3__gpt4.1-nano__Qv4.0.train_50.jsonl",
    ]
    top_n_prompts = 4
    include_worse_benchmark = False
    research_reports_per_question = 1
    num_predictions_per_research_report = 1
    remove_background_info = True
    research_tools = [
        ResearchTool(
            tool_name=ToolName.PERPLEXITY_LOW_COST,
            max_calls=7,
        ),
        ResearchTool(
            tool_name=ToolName.ASKNEWS,
            max_calls=2,
        ),
        ResearchTool(
            tool_name=ToolName.DATA_ANALYZER,
            max_calls=1,
        ),
        ResearchTool(
            tool_name=ToolName.PERPLEXITY_REASONING_PRO_SEARCH,
            max_calls=1,
        ),
    ]

    # --- Run the evaluation ---
    for question in evaluation_questions:
        if remove_background_info:
            question.background_info = None

    for research_and_reason_llm in both_reason_and_research_llm:
        research_llm, reason_llm = research_and_reason_llm
        evaluator = BotEvaluator(
            input_questions=evaluation_questions,
            research_type=None,
            concurrent_evaluation_batch_size=questions_batch_size,
            file_or_folder_to_save_benchmarks="logs/forecasts/benchmarks/",
        )
        evaluation_result = await evaluator.evaluate_best_benchmarked_prompts(
            forecast_llm=reason_llm,
            research_llm_name=research_llm,
            research_tools=research_tools,
            benchmark_files=benchmark_files,
            top_n_prompts=top_n_prompts,
            include_control_group_prompt=True,
            include_worst_prompt=include_worse_benchmark,
            research_reports_per_question=research_reports_per_question,
            num_predictions_per_research_report=num_predictions_per_research_report,
        )
        for evaluated_prompt in evaluation_result.evaluated_bots:
            logger.info(
                f"Name: {evaluated_prompt.bot_config.originating_idea.short_name}"
            )
            logger.info(f"Config: {evaluated_prompt.bot_config}")
            logger.info(f"Code: {evaluated_prompt.benchmark.code}")
            logger.info(
                f"Forecast Bot Class Name: {evaluated_prompt.benchmark.forecast_bot_class_name}"
            )
            logger.info(f"Cost: {evaluated_prompt.benchmark.total_cost}")
            logger.info(f"Score: {evaluated_prompt.score}")

        logger.info(f"Best prompt: {evaluation_result.best_bot}")


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(run_higher_model_evaluation())
