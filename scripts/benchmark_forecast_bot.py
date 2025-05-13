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
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.official_bots.q2_template_bot import (
    Q2TemplateBot2025,
)
from forecasting_tools.forecast_helpers.benchmarker import Benchmarker
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def benchmark_forecast_bot() -> None:
    questions_to_use = 3
    perplexity_reasoning_pro = GeneralLlm(
        model="openrouter/perplexity/perplexity-reasoning-pro",
        temperature=0.3,
        web_search_options={"search_context_size": "high"},
        reasoning_effort="high",
    )
    google_gemini_2_5_pro_preview = GeneralLlm(
        model="openrouter/google/gemini-2.5-pro-preview",
        temperature=0.3,
    )
    gpt_4o = GeneralLlm(
        model="openrouter/openai/gpt-4o",
        temperature=0.3,
    )
    with MonetaryCostManager() as cost_manager:
        bots = [
            Q2TemplateBot2025(
                llms={
                    "default": perplexity_reasoning_pro,
                    "researcher": perplexity_reasoning_pro,
                    "summarizer": gpt_4o,
                },
            ),
            Q2TemplateBotWithDecompositionV1(
                llms={
                    "default": google_gemini_2_5_pro_preview,
                    "decomposer": google_gemini_2_5_pro_preview,
                    "researcher": perplexity_reasoning_pro,
                    "summarizer": gpt_4o,
                },
            ),
            Q2TemplateBotWithDecompositionV2(
                llms={
                    "default": google_gemini_2_5_pro_preview,
                    "decomposer": google_gemini_2_5_pro_preview,
                    "researcher": perplexity_reasoning_pro,
                    "summarizer": gpt_4o,
                },
            ),
        ]
        bots = typeguard.check_type(bots, list[ForecastBot])
        benchmarks = await Benchmarker(
            number_of_questions_to_use=questions_to_use,
            forecast_bots=bots,
            file_path_to_save_reports="logs/forecasts/benchmarks/",
            concurrent_question_batch_size=20,
        ).run_benchmark()
        for i, benchmark in enumerate(benchmarks):
            logger.info(
                f"Benchmark {i+1} of {len(benchmarks)}: {benchmark.name}"
            )
            logger.info(
                f"- Final Score: {benchmark.average_expected_baseline_score}"
            )
            logger.info(f"- Total Cost: {benchmark.total_cost}")
            logger.info(f"- Time taken: {benchmark.time_taken_in_minutes}")
        logger.info(f"Total Cost: {cost_manager.current_usage}")


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(benchmark_forecast_bot())
