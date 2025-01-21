from __future__ import annotations

import argparse
import asyncio
import os
import sys

import dotenv

# Dynamically determine the absolute path to the top-level directory
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)
dotenv.load_dotenv()

import logging

from forecasting_tools.forecasting.forecast_bots.main_bot import MainBot
from forecasting_tools.forecasting.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ForecastReport,
)
from forecasting_tools.util.custom_logger import CustomLogger

CustomLogger.setup_logging()

logger = logging.getLogger(__name__)


async def run_forecasts(skip_previous: bool, tournament_id: int) -> None:
    """
    Make a copy of this file called run_bot.py (i.e. remove template) and fill in your bot details.
    This will be run in the workflows
    """
    forecaster = MainBot(
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=skip_previous,
    )
    reports = await forecaster.forecast_on_tournament(
        tournament_id, return_exceptions=True
    )
    valid_reports = [
        report for report in reports if isinstance(report, ForecastReport)
    ]
    exceptions = [
        report for report in reports if isinstance(report, BaseException)
    ]
    minor_exceptions = [
        report.errors for report in valid_reports if report.errors
    ]

    for report in valid_reports:
        await asyncio.sleep(5)
        try:
            ForecastDatabaseManager.add_forecast_report_to_database(
                report, ForecastRunType.REGULAR_FORECAST
            )
        except Exception as e:
            logger.error(f"Error adding forecast report to database: {e}")

    if exceptions:
        raise RuntimeError(
            f"{len(exceptions)} errors occurred while forecasting: {exceptions}"
        )
    if minor_exceptions:
        logger.error(
            f"{len(minor_exceptions)} minor exceptions occurred while forecasting: {minor_exceptions}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run forecasts with specified bot type"
    )
    parser.add_argument(
        "--skip-previous",
        type=bool,
        required=True,
        help="Skip previously forecasted questions",
    )
    parser.add_argument(
        "--tournament",
        type=int,
        required=True,
        help="Tournament to forecast on",
    )
    args = parser.parse_args()

    skip_previous = args.skip_previous
    tournament_id = args.tournament

    asyncio.run(run_forecasts(skip_previous, tournament_id))
