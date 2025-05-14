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

logger = logging.getLogger(__name__)


async def benchmark_forecast_bot() -> None:
    num_questions_to_use = 1
    google_gemini_2_5_pro_preview = GeneralLlm(
        model="openrouter/google/gemini-2.5-pro-preview",
        temperature=0.3,
    )
    gemini_grounded_model = GeneralLlm.grounded_model(
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
                    "default": google_gemini_2_5_pro_preview,
                    "researcher": gemini_grounded_model,
                    "summarizer": gpt_4o,
                },
                research_reports_per_question=1,
                predictions_per_research_report=5,
            ),
            Q2TemplateBotWithDecompositionV1(
                llms={
                    "default": google_gemini_2_5_pro_preview,
                    "decomposer": gemini_grounded_model,
                    "researcher": gemini_grounded_model,
                    "summarizer": gpt_4o,
                },
                research_reports_per_question=1,
                predictions_per_research_report=5,
            ),
            Q2TemplateBotWithDecompositionV2(
                llms={
                    "default": google_gemini_2_5_pro_preview,
                    "decomposer": gemini_grounded_model,
                    "researcher": gemini_grounded_model,
                    "summarizer": gpt_4o,
                },
                research_reports_per_question=1,
                predictions_per_research_report=5,
            ),
        ]
        bots = typeguard.check_type(bots, list[ForecastBot])
        chosen_questions = MetaculusApi.get_benchmark_questions(
            num_questions_to_use,
            days_to_resolve_in=6 * 30,  # 6 months
            max_days_since_opening=365,
        )
        benchmarks = await Benchmarker(
            questions_to_use=chosen_questions,
            forecast_bots=bots,
            file_path_to_save_reports="logs/forecasts/benchmarks/",
            concurrent_question_batch_size=20,
            additional_code_to_snapshot=[
                QuestionDecomposer,
                QuestionOperationalizer,
            ],
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
