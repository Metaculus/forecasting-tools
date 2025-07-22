import logging
from enum import Enum

from forecasting_tools.agents_and_tools.base_rates.base_rate_researcher import (
    BaseRateReport,
)
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.util.coda_utils import CodaCell, CodaColumn, CodaRow, CodaTable

logger = logging.getLogger(__name__)


class ForecastRunType(Enum):
    UNIT_TEST_FORECAST = "unit_test_run"
    WEB_APP_FORECAST = "web_app_run"
    REGULAR_FORECAST = "regular_run"
    UNIT_TEST_BASE_RATE = "unit_test_base_rate"
    WEB_APP_BASE_RATE = "web_app_base_rate"
    REGULAR_BASE_RATE = "regular_base_rate"
    WEB_APP_NICHE_LIST = "web_app_niche_list"
    WEB_APP_KEY_FACTORS = "web_app_key_factors"
    WEB_APP_ESTIMATOR = "web_app_estimator"
    WEB_APP_QUESTION_GENERATOR = "web_app_question_generator"
    WEB_APP_CHAT = "web_app_chat"


class ForecastDatabaseManager:
    # NOTE: The column ids here are not sensitive information
    # (you need the API key to publish anything), but they are hardcoded
    # TODO: Change database to something more robust
    # and based on environment variables (i.e. not Coda.io)
    # NOTE: Coda was originally used in the mvp due to familiarity
    # and quickness to make visuals and a custom dashboard online
    # and just hasn't been updated yet

    QUESTION_COLUMN = CodaColumn("Question", "c-rDW4kf2dUl")
    BACKGROUND_INFO_COLUMN = CodaColumn("Background Info", "c-eOYV3GdLJL")
    RESOLUTION_CRITERIA_COLUMN = CodaColumn("Resolution Criteria", "c-sb19UcdKNF")
    FINE_PRINT_COLUMN = CodaColumn("Fine Print", "c-cxeoRPkdzX")
    PREDICTION_COLUMN = CodaColumn("Prediction", "c-IEjaL5UKx3")
    EXPLANATION_COLUMN = CodaColumn("Explanation", "c-VV-ZSafEGk")
    PAGE_URL = CodaColumn("Page URL", "c-0YQYHx0Tia")
    RUN_TYPE_COLUMN = CodaColumn("Run Type", "c-lco-HpDPJs")
    PRICE_ESTIMATE_COLUMN = CodaColumn("Price Estimate", "c-ybaqRY7YPr")

    REPORTS_TABLE_COLUMNS = [
        QUESTION_COLUMN,
        BACKGROUND_INFO_COLUMN,
        RESOLUTION_CRITERIA_COLUMN,
        FINE_PRINT_COLUMN,
        PREDICTION_COLUMN,
        EXPLANATION_COLUMN,
        PAGE_URL,
        RUN_TYPE_COLUMN,
        PRICE_ESTIMATE_COLUMN,
    ]
    REPORTS_TABLE_KEY_COLUMNS = []

    REPORTS_TABLE = CodaTable(
        "ygtubEdAK8",
        "grid-pGvVYv9Hfu",
        REPORTS_TABLE_COLUMNS,
        REPORTS_TABLE_KEY_COLUMNS,
    )

    @staticmethod
    def add_forecast_report_to_database(
        metaculus_report: ForecastReport, run_type: ForecastRunType
    ) -> None:
        metaculus_report_copy = metaculus_report.model_copy()
        coda_row = ForecastDatabaseManager._turn_report_into_coda_row(
            metaculus_report_copy, run_type
        )
        try:
            ForecastDatabaseManager.REPORTS_TABLE.add_row_to_table(coda_row)
        except Exception as e:
            logger.error(f"Error while uploading metaculus report: {e}")
            metaculus_report_copy.explanation = "ERROR while uploading to Coda"
            coda_row = ForecastDatabaseManager._turn_report_into_coda_row(
                metaculus_report_copy, run_type
            )
            ForecastDatabaseManager.REPORTS_TABLE.add_row_to_table(coda_row)

    @classmethod
    def add_general_report_to_database(
        cls,
        question_text: str | None,
        background_info: str | None,
        resolution_criteria: str | None,
        fine_print: str | None,
        prediction: float | None,
        explanation: str | None,
        page_url: str | None,
        price_estimate: float | None,
        run_type: ForecastRunType,
    ) -> None:
        if explanation is None:
            explanation = ""
        try:
            coda_row = cls.__create_coda_row_from_column_values(
                question_text,
                background_info,
                resolution_criteria,
                fine_print,
                prediction,
                explanation,
                page_url,
                price_estimate,
                run_type,
            )
            cls.REPORTS_TABLE.add_row_to_table(coda_row)
        except Exception as e:
            logger.warning(f"Error while uploading to coda: {e}")
            explanation = f"ERROR while uploading to Coda. Here is shortened version: {explanation[:10000]}..."
            coda_row = cls.__create_coda_row_from_column_values(
                question_text,
                background_info,
                resolution_criteria,
                fine_print,
                prediction,
                explanation,
                page_url,
                price_estimate,
                run_type,
            )
            cls.REPORTS_TABLE.add_row_to_table(coda_row)

    @classmethod
    def add_base_rate_report_to_database(
        cls, report: BaseRateReport, run_type: ForecastRunType
    ) -> None:
        report_copy = report.model_copy()
        coda_row = cls._turn_report_into_coda_row(report_copy, run_type)
        try:
            cls.REPORTS_TABLE.add_row_to_table(coda_row)
        except Exception as e:
            logger.error(f"Error while uploading metaculus report: {e}")
            report_copy.markdown_report = "ERROR while uploading to Coda"
            coda_row = cls._turn_report_into_coda_row(report_copy, run_type)
            cls.REPORTS_TABLE.add_row_to_table(coda_row)

    @classmethod
    def _turn_report_into_coda_row(
        cls, report: ForecastReport | BaseRateReport, run_type: ForecastRunType
    ) -> CodaRow:
        """
        This function turns a metaculus report into a coda row
        """
        if isinstance(report, ForecastReport):
            question_text = report.question.question_text
            background_info = report.question.background_info
            resolution_criteria = report.question.resolution_criteria
            fine_print = report.question.fine_print
            page_url = report.question.page_url
            prediction = str(report.prediction)
            try:
                explanation = (
                    report.summary
                )  # The Full Explanation is too long for coda (api data limit per request is 85kb). Thus summary used instead
            except Exception as e:
                logger.warning(f"Error while getting summary: {e}")
                explanation = report.explanation[:10000]
            price_estimate = report.price_estimate
        elif isinstance(report, BaseRateReport):
            question_text = report.question
            background_info = ""
            resolution_criteria = ""
            fine_print = ""
            page_url = ""
            prediction = report.historical_rate
            explanation = report.markdown_report
            price_estimate = report.price_estimate
        row = cls.__create_coda_row_from_column_values(
            question_text,
            background_info,
            resolution_criteria,
            fine_print,
            prediction,
            explanation,
            page_url,
            price_estimate,
            run_type,
        )
        return row

    @staticmethod
    def __create_coda_row_from_column_values(
        question_text: str | None,
        background_info: str | None,
        resolution_criteria: str | None,
        fine_print: str | None,
        prediction: float | str | None,
        explanation: str | None,
        page_url: str | None,
        price_estimate: float | None,
        run_type: ForecastRunType,
    ) -> CodaRow:
        cells = [
            CodaCell(ForecastDatabaseManager.QUESTION_COLUMN, question_text),
            CodaCell(ForecastDatabaseManager.BACKGROUND_INFO_COLUMN, background_info),
            CodaCell(
                ForecastDatabaseManager.RESOLUTION_CRITERIA_COLUMN,
                resolution_criteria,
            ),
            CodaCell(ForecastDatabaseManager.FINE_PRINT_COLUMN, fine_print),
            CodaCell(
                ForecastDatabaseManager.PREDICTION_COLUMN,
                prediction,
            ),
            CodaCell(
                ForecastDatabaseManager.EXPLANATION_COLUMN,
                explanation,
            ),
            CodaCell(ForecastDatabaseManager.PAGE_URL, page_url),
            CodaCell(ForecastDatabaseManager.PRICE_ESTIMATE_COLUMN, price_estimate),
            CodaCell(ForecastDatabaseManager.RUN_TYPE_COLUMN, run_type.value),
        ]
        return CodaRow(cells)
