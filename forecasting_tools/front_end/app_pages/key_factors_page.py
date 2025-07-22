from __future__ import annotations

import logging
import os
import re
import sys

import dotenv
import streamlit as st
from pydantic import BaseModel

dotenv.load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(top_level_dir)

from forecasting_tools.agents_and_tools.research.key_factors_researcher import (
    KeyFactorsResearcher,
    ScoredKeyFactor,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.front_end.helpers.report_displayer import ReportDisplayer
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.helpers.metaculus_api import MetaculusApi
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class KeyFactorsInput(Jsonable, BaseModel):
    metaculus_url: str


class KeyFactorsOutput(Jsonable, BaseModel):
    question_text: str
    markdown: str
    cost: float
    scored_key_factors: list[ScoredKeyFactor] | None = None


class KeyFactorsPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "🔑 Key Factors Researcher"
    URL_PATH: str = "/key-factors"
    INPUT_TYPE = KeyFactorsInput
    OUTPUT_TYPE = KeyFactorsOutput
    EXAMPLES_FILE_PATH = (
        "forecasting_tools/front_end/example_outputs/key_factors_page_examples.json"
    )

    @classmethod
    async def _display_intro_text(cls) -> None:
        # No intro text needed
        pass

    @classmethod
    async def _get_input(cls) -> KeyFactorsInput | None:
        with st.form("key_factors_form"):
            metaculus_url = st.text_input("Metaculus Question URL")
            submitted = st.form_submit_button("Find Key Factors")
            if submitted and metaculus_url:
                return KeyFactorsInput(metaculus_url=metaculus_url)
        return None

    @classmethod
    async def _run_tool(cls, input: KeyFactorsInput) -> KeyFactorsOutput:
        with st.spinner("Finding key factors... This may take a minute or two..."):
            question_id = cls.__extract_question_id(input.metaculus_url)
            metaculus_question = MetaculusApi.get_question_by_post_id(question_id)

            with MonetaryCostManager() as cost_manager:
                num_questions_to_research = 16
                num_key_factors_to_return = 7
                key_factors = await KeyFactorsResearcher.find_and_sort_key_factors(
                    metaculus_question,
                    num_questions_to_research_with=num_questions_to_research,
                    num_key_factors_to_return=num_key_factors_to_return,
                )
                cost = cost_manager.current_usage
                markdown = ScoredKeyFactor.turn_key_factors_into_markdown_list(
                    key_factors
                )
                return KeyFactorsOutput(
                    question_text=metaculus_question.question_text,
                    markdown=markdown,
                    cost=cost,
                    scored_key_factors=key_factors,
                )

    @classmethod
    async def _save_run_to_coda(
        cls,
        input_to_tool: KeyFactorsInput,
        output: KeyFactorsOutput,
        is_premade: bool,
    ) -> None:
        if is_premade:
            output.cost = 0
        ForecastDatabaseManager.add_general_report_to_database(
            question_text=output.question_text,
            background_info=None,
            resolution_criteria=None,
            fine_print=None,
            prediction=None,
            explanation=output.markdown,
            page_url=None,
            price_estimate=output.cost,
            run_type=ForecastRunType.WEB_APP_KEY_FACTORS,
        )

    @classmethod
    async def _display_outputs(cls, outputs: list[KeyFactorsOutput]) -> None:
        for output in outputs:
            with st.expander(f"Key Factors for: {output.question_text}", expanded=True):
                st.markdown(f"Cost: ${output.cost:.2f}")
                st.markdown(ReportDisplayer.clean_markdown(output.markdown))

    @classmethod
    def __extract_question_id(cls, url: str) -> int:
        match = re.search(r"/questions/(\d+)/", url)
        if match:
            return int(match.group(1))
        raise ValueError(
            "Invalid Metaculus question URL. Please ensure it's in the format: https://metaculus.com/questions/[ID]/[question-title]/"
        )


if __name__ == "__main__":
    KeyFactorsPage.main()
