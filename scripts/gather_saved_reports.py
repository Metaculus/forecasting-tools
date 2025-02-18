from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime

import dotenv

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.forecast_helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.util import file_manipulation
from forecasting_tools.util.custom_logger import CustomLogger

print(__name__)
logger = logging.getLogger(__name__)


async def gather_reports_for_date_range(
    start_date: datetime, end_date: datetime | None
) -> list[BinaryReport]:
    if end_date is None:
        end_date = datetime.now()
    forecast_path = "logs/forecasts/forecast_team"
    absolute_forecast_path = file_manipulation.get_absolute_path(forecast_path)

    files = os.listdir(absolute_forecast_path)
    logger.info(f"Found {len(files)} forecast files")

    target_files = []
    for file_name in files:
        datetime_of_file = datetime.strptime(
            file_name[:19], "%Y-%m-%d-%H-%M-%S"
        )
        if start_date <= datetime_of_file <= end_date:
            target_files.append(file_name)

    if not target_files:
        logger.warning(
            f"No forecast files found for range {start_date} to {end_date}"
        )
        return []

    logger.info(
        f"Found {len(target_files)} forecast files for range {start_date} to {end_date}"
    )

    all_reports: list[BinaryReport] = []
    for file_name in target_files:
        file_path = os.path.join(absolute_forecast_path, file_name)
        reports = BinaryReport.load_json_from_file_path(file_path)
        all_reports.extend(reports)
    return all_reports


async def republish_reports(all_reports: list[BinaryReport]) -> None:
    for report in all_reports:
        assert isinstance(report, BinaryReport)
        try:
            await report.publish_report_to_metaculus()
            logger.info(
                f"Successfully published forecast for question {report.question.id_of_post}"
            )
            ForecastDatabaseManager.add_forecast_report_to_database(
                report, ForecastRunType.REGULAR_FORECAST
            )
            time.sleep(10)  # Wait between publications to avoid rate limiting
        except Exception as e:
            logger.error(f"Failed to publish report: {e}")


if __name__ == "__main__":
    dotenv.load_dotenv()
    CustomLogger.setup_logging()
    start_date = datetime.today().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_date = None
    reports = asyncio.run(gather_reports_for_date_range(start_date, end_date))
    logger.info(f"Found {len(reports)} reports")
    log_path = f"logs/forecasts/republish/republish_{start_date.strftime('%Y-%m-%d')}.json"
    BinaryReport.save_object_list_to_file_path(reports, log_path)
    # asyncio.run(republish_reports(reports))
