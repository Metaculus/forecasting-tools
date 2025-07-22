from __future__ import annotations

import logging
import os
import sys

import dotenv
import streamlit as st
from pydantic import BaseModel

dotenv.load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(top_level_dir)

from forecasting_tools.agents_and_tools.base_rates.base_rate_researcher import (
    BaseRateReport,
    BaseRateResearcher,
)
from forecasting_tools.front_end.helpers.report_displayer import ReportDisplayer
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class BaseRateInput(Jsonable, BaseModel):
    question_text: str


class BaseRatePage(ToolPage):
    PAGE_DISPLAY_NAME: str = "🦕 Find a Historical Base Rate"
    URL_PATH: str = "/base-rate-generator"
    INPUT_TYPE = BaseRateInput
    OUTPUT_TYPE = BaseRateReport
    EXAMPLES_FILE_PATH = (
        "forecasting_tools/front_end/example_outputs/base_rate_page_examples.json"
    )
    QUESTION_TEXT_BOX = "base_rate_question_text"

    @classmethod
    async def _display_intro_text(cls) -> None:
        # No intro text to display
        pass

    @classmethod
    async def _get_input(cls) -> BaseRateInput | None:
        with st.form("base_rate_form"):
            question_text = st.text_input(
                "Enter your question here", key=cls.QUESTION_TEXT_BOX
            )
            submitted = st.form_submit_button("Submit")
            if submitted and question_text:
                input_to_tool = BaseRateInput(question_text=question_text)
                return input_to_tool
        return None

    @classmethod
    async def _run_tool(cls, input: BaseRateInput) -> BaseRateReport:
        with st.spinner("Analyzing... This may take a minute or two..."):
            return await BaseRateResearcher(input.question_text).make_base_rate_report()

    @classmethod
    async def _save_run_to_coda(
        cls,
        input_to_tool: BaseRateInput,
        output: BaseRateReport,
        is_premade: bool,
    ) -> None:
        if is_premade:
            output.price_estimate = 0
        ForecastDatabaseManager.add_base_rate_report_to_database(
            output, ForecastRunType.WEB_APP_BASE_RATE
        )

    @classmethod
    async def _display_outputs(cls, outputs: list[BaseRateReport]) -> None:
        for report in outputs:
            with st.expander(report.question, expanded=True):
                st.markdown(ReportDisplayer.clean_markdown(report.markdown_report))


if __name__ == "__main__":
    BaseRatePage.main()
