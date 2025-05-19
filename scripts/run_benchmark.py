from __future__ import annotations

import asyncio
import logging

import typeguard

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecast_bots.experiments.q2t_w_decomposition import (
    Q2TemplateBotWithDecompositionV1,
    Q2TemplateBotWithDecompositionV2,
    QuestionDecomposer,
    QuestionOperationalizer,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.official_bots.q2_template_bot import (
    Q2TemplateBot2025,
)
from forecasting_tools.forecast_helpers.benchmarker import Benchmarker
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.util.custom_logger import CustomLogger
from run_bot import get_all_bots

logger = logging.getLogger(__name__)


def get_all_tournament_bots() -> list[ForecastBot]:
    return asyncio.run(get_all_bots())


def get_decomposition_bots() -> list[ForecastBot]:
    google_gemini_2_5_pro_preview = GeneralLlm(
        # model="gemini/gemini-2.5-pro-preview-03-25",
        model="openrouter/google/gemini-2.5-pro-preview",
        temperature=0.3,
        timeout=120,
    )
    perplexity_reasoning_pro = GeneralLlm.search_context_model(
        # model="perplexity/sonar-reasoning-pro",
        model="openrouter/perplexity/sonar-reasoning-pro",
        temperature=0.3,
        search_context_size="high",
    )
    gpt_4o = GeneralLlm(
        model="openai/gpt-4o",
        temperature=0.3,
    )
    bots = [
        Q2TemplateBot2025(
            llms={
                "default": google_gemini_2_5_pro_preview,
                "researcher": "asknews/news-summaries",
                "summarizer": gpt_4o,
            },
            research_reports_per_question=1,
            predictions_per_research_report=5,
        ),
        Q2TemplateBotWithDecompositionV1(
            llms={
                "default": google_gemini_2_5_pro_preview,
                "decomposer": perplexity_reasoning_pro,
                "researcher": perplexity_reasoning_pro,
                "summarizer": gpt_4o,
            },
            research_reports_per_question=1,
            predictions_per_research_report=5,
        ),
        Q2TemplateBotWithDecompositionV2(
            llms={
                "default": google_gemini_2_5_pro_preview,
                "decomposer": perplexity_reasoning_pro,
                "researcher": "asknews/news-summaries",
                "summarizer": gpt_4o,
            },
            research_reports_per_question=1,
            predictions_per_research_report=5,
        ),
    ]
    bots = typeguard.check_type(bots, list[ForecastBot])
    return bots


async def benchmark_forecast_bots() -> None:
    num_questions_to_use = 500
    concurrent_batch_size = 2
    bots = get_decomposition_bots()
    additional_code_to_snapshot = [
        QuestionDecomposer,
        QuestionOperationalizer,
    ]
    chosen_questions = MetaculusApi.get_benchmark_questions(
        num_questions_to_use,
        days_to_resolve_in=365,  # 6 * 30,  # 6 months
        max_days_since_opening=365,
    )

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
            logger.info(
                f"Benchmark {i+1} of {len(benchmarks)}: {benchmark.name}"
            )
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
