from __future__ import annotations

import asyncio
import logging

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.auto_optimizers.question_plus_research import (
    QuestionPlusResearch,
)
from forecasting_tools.cp_benchmarking.benchmarker import Benchmarker
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.util.custom_logger import CustomLogger
from run_bots import get_default_bot_dict

logger = logging.getLogger(__name__)


def get_chosen_q2_bots() -> list[ForecastBot]:
    # Expected cost: $126
    chosen_bot_names = [
        "METAC_DEEPSEEK_R1_TOKEN",  # Regular Deepseek
        "METAC_GPT_4O_TOKEN",  # Regular gpt-4o
        "METAC_O1_TOKEN",  # Regular o1
        "METAC_GEMINI_2_5_PRO_PREVIEW_TOKEN",  # Regular Gemini
        # "METAC_O3_HIGH_TOKEN",  # o3-high
    ]

    bots = [get_default_bot_dict()[bot_name].bot for bot_name in chosen_bot_names]
    return bots


async def benchmark_forecast_bots() -> None:
    # ----- Configure the benchmarker -----
    concurrent_batch_size = 15
    bots = get_chosen_q2_bots()
    additional_code_to_snapshot = []
    # num_questions_to_use = 150
    # chosen_questions = MetaculusApi.get_benchmark_questions(
    #     num_questions_to_use,
    # )
    snapshots = QuestionPlusResearch.load_json_from_file_path(
        "logs/forecasts/question_snapshots_v1.6.train__112qs.json"
    )
    chosen_questions = [snapshot.question for snapshot in snapshots]
    remove_background_info = (
        True  # AIB (and real life questions) often don't have detailed background info
    )
    num_predictions_per_report: int | None = 1

    # ----- Run the benchmarker -----

    if remove_background_info:
        for question in chosen_questions:
            question.background_info = None

    if num_predictions_per_report is not None:
        for bot in bots:
            bot.predictions_per_research_report = num_predictions_per_report

    with MonetaryCostManager() as cost_manager:
        for bot in bots:
            bot.publish_reports_to_metaculus = False
        benchmarks = await Benchmarker(
            questions_to_use=chosen_questions,
            forecast_bots=bots,
            file_path_to_save_reports="logs/forecasts/benchmarks/",
            concurrent_question_batch_size=concurrent_batch_size,
            additional_code_to_snapshot=additional_code_to_snapshot,
        ).run_benchmark()
        for i, benchmark in enumerate(benchmarks):
            logger.info(f"Benchmark {i+1} of {len(benchmarks)}: {benchmark.name}")
            try:
                logger.info(
                    f"- Final Score: {benchmark.average_expected_baseline_score}"
                )
            except Exception:
                logger.info(
                    "- Final Score: Couldn't calculate score (potentially no forecasts?)"
                )
            logger.info(f"- Total Cost: {benchmark.total_cost}")
            logger.info(f"- Time taken: {benchmark.time_taken_in_minutes}")
        logger.info(f"Total Cost: {cost_manager.current_usage}")


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(benchmark_forecast_bots())


# def get_decomposition_bots() -> list[ForecastBot]:
#     google_gemini_2_5_pro_preview = GeneralLlm(
#         # model="gemini/gemini-2.5-pro-preview-03-25",
#         model="openrouter/google/gemini-2.5-pro-preview",
#         temperature=0.3,
#         timeout=120,
#     )
#     perplexity_reasoning_pro = GeneralLlm.search_context_model(
#         # model="perplexity/sonar-reasoning-pro",
#         model="openrouter/perplexity/sonar-reasoning-pro",
#         temperature=0.3,
#         search_context_size="high",
#     )
#     gpt_4o = GeneralLlm(
#         model="openai/gpt-4o",
#         temperature=0.3,
#     )
#     bots = [
#         Q2TemplateBot2025(
#             llms={
#                 "default": google_gemini_2_5_pro_preview,
#                 "researcher": "asknews/news-summaries",
#                 "summarizer": gpt_4o,
#             },
#             research_reports_per_question=1,
#             predictions_per_research_report=5,
#         ),
#         Q2TemplateBotWithDecompositionV1(
#             llms={
#                 "default": google_gemini_2_5_pro_preview,
#                 "decomposer": perplexity_reasoning_pro,
#                 "researcher": perplexity_reasoning_pro,
#                 "summarizer": gpt_4o,
#             },
#             research_reports_per_question=1,
#             predictions_per_research_report=5,
#         ),
#         Q2TemplateBotWithDecompositionV2(
#             llms={
#                 "default": google_gemini_2_5_pro_preview,
#                 "decomposer": perplexity_reasoning_pro,
#                 "researcher": "asknews/news-summaries",
#                 "summarizer": gpt_4o,
#             },
#             research_reports_per_question=1,
#             predictions_per_research_report=5,
#         ),
#     ]
#     bots = typeguard.check_type(bots, list[ForecastBot])
#     return bots
