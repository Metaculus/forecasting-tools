from __future__ import annotations

import asyncio
import logging

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecasting.forecast_bots.main_bot import MainBot
from forecasting_tools.forecasting.helpers.benchmarker import Benchmarker
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def benchmark_forecast_bot() -> None:
    with MonetaryCostManager() as cost_manager:
        bot = MainBot()
        benchmark = await Benchmarker().run_benchmark(
            number_of_questions_to_use=100,
            forecast_bots=[bot],
            file_path_to_save_reports="logs/forecasts/benchmarks/",
        )
        logger.info(f"Total Cost: {cost_manager.current_usage}")
        logger.info(f"Final Score: {benchmark.average_expected_log_score}")


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(benchmark_forecast_bot())
