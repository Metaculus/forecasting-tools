import asyncio

from forecasting_tools.agents_and_tools.base_rates.base_rate_researcher import (
    BaseRateReport,
)
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.util.coda_utils import CodaRow


def test_forecast_report_turns_into_coda_row() -> None:
    example_reports = get_forecast_example_reports()
    for example_report in example_reports:
        coda_row = ForecastDatabaseManager._turn_report_into_coda_row(
            example_report, ForecastRunType.UNIT_TEST_FORECAST
        )
        assert isinstance(coda_row, CodaRow)
        assert ForecastDatabaseManager.REPORTS_TABLE.check_that_row_matches_columns(
            coda_row
        )


def test_base_rate_report_turns_into_coda_row() -> None:
    example_reports = get_base_rate_example_reports()
    for example_report in example_reports:
        coda_row = ForecastDatabaseManager._turn_report_into_coda_row(
            example_report, ForecastRunType.UNIT_TEST_FORECAST
        )
        assert isinstance(coda_row, CodaRow)
        assert ForecastDatabaseManager.REPORTS_TABLE.check_that_row_matches_columns(
            coda_row
        )


async def test_forecast_report_can_be_added_to_coda() -> None:
    example_reports = get_forecast_example_reports()[:2]
    for example_report in example_reports:
        ForecastDatabaseManager.add_forecast_report_to_database(
            example_report, ForecastRunType.UNIT_TEST_FORECAST
        )
        await asyncio.sleep(5)
    # No assert, make sure it doesn't error


async def test_base_rate_report_can_be_added_to_coda() -> None:
    example_reports = get_base_rate_example_reports()[:2]
    for example_report in example_reports:
        ForecastDatabaseManager.add_base_rate_report_to_database(
            example_report, ForecastRunType.UNIT_TEST_BASE_RATE
        )
        await asyncio.sleep(5)
    # No assert, make sure it doesn't error


def get_forecast_example_reports() -> list[ForecastReport]:
    metaculus_data_path = "code_tests/unit_tests/test_data_models/forecasting_test_data/metaculus_forecast_report_examples.json"
    metaculus_reports = DataOrganizer.load_reports_from_file_path(metaculus_data_path)
    return metaculus_reports


def get_base_rate_example_reports() -> list[BaseRateReport]:
    base_rate_data_path = "code_tests/unit_tests/test_data_models/forecasting_test_data/base_rate_reports.json"
    base_rate_reports = BaseRateReport.load_json_from_file_path(base_rate_data_path)
    return base_rate_reports
