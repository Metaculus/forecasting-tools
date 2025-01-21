from __future__ import annotations

import asyncio
import logging

import typeguard

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecasting.forecast_bots.experiments.q3t_w_exa import (
    Q3TemplateWithExa,
)
from forecasting_tools.forecasting.forecast_bots.experiments.q4_main_binary_bot import (
    Q3TemplatePlusQ4VeritasBinaryPrompt,
)
from forecasting_tools.forecasting.forecast_bots.experiments.q4v_w_exa import (
    Q4VeritasWithExa,
)
from forecasting_tools.forecasting.forecast_bots.experiments.q4v_w_exa_and_o1_preview import (
    Q4VeritasWithExaAndO1Preview,
)
from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastBot,
)
from forecasting_tools.forecasting.helpers.benchmarker import Benchmarker
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def benchmark_forecast_bot() -> None:
    questions_to_use = 120
    with MonetaryCostManager() as cost_manager:
        bots = [
            Q3TemplateWithExa(),
            Q3TemplatePlusQ4VeritasBinaryPrompt(),
            Q4VeritasWithExa(),
            Q4VeritasWithExaAndO1Preview(),
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
                f"- Final Score: {benchmark.average_inverse_expected_log_score}"
            )
            logger.info(f"- Total Cost: {benchmark.total_cost}")
            logger.info(f"- Time taken: {benchmark.time_taken_in_minutes}")
        logger.info(f"Total Cost: {cost_manager.current_usage}")


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(benchmark_forecast_bot())
