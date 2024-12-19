import asyncio

from forecasting_tools.forecasting.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ForecastReport,
)
from forecasting_tools.forecasting.questions_and_reports.report_organizer import (
    ReportOrganizer,
)
from forecasting_tools.forecasting.sub_question_researchers.base_rate_researcher import (
    BaseRateReport,
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
    metaculus_data_path = "code_tests/unit_tests/test_forecasting/forecasting_test_data/metaculus_forecast_report_examples.json"
    metaculus_reports = ReportOrganizer.load_reports_from_file_path(
        metaculus_data_path
    )
    return metaculus_reports


def get_base_rate_example_reports() -> list[BaseRateReport]:
    base_rate_data_path = "code_tests/unit_tests/test_forecasting/forecasting_test_data/base_rate_reports.json"
    base_rate_reports = (
        BaseRateReport.convert_project_file_path_to_object_list(
            base_rate_data_path
        )
    )
    return base_rate_reports
