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

from forecasting_tools.forecasting.forecast_bots.template_bot import (
    TemplateBot,
)

logger = logging.getLogger(__name__)


async def run_morning_forecasts(
    skip_previous: bool, tournament_id: int
) -> None:
    """
    Make a copy of this file called run_bot.py (i.e. remove template) and fill in your bot details.
    This will be run in the workflows
    """
    forecaster = TemplateBot(
        research_reports_per_question=3,
        predictions_per_research_report=3,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=skip_previous,
        use_research_summary_to_forecast=False,
    )
    reports = await forecaster.forecast_on_tournament(
        tournament_id, return_exceptions=True
    )
    exceptions = [
        report for report in reports if isinstance(report, Exception)
    ]
    if exceptions:
        raise Exception(
            f"{len(exceptions)} errors occurred while forecasting: {exceptions}"
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
        "--tournament-id",
        type=int,
        required=True,
        help="Tournament ID to forecast on",
    )
    args = parser.parse_args()

    skip_previous = args.skip_previous
    tournament_id = args.tournament_id

    asyncio.run(run_morning_forecasts(skip_previous, tournament_id))
