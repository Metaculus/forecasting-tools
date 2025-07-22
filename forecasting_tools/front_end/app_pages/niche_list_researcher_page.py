import logging

import streamlit as st
from pydantic import BaseModel

from forecasting_tools.agents_and_tools.base_rates.niche_list_researcher import (
    FactCheckedItem,
    NicheListResearcher,
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
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class NicheListOutput(Jsonable, BaseModel):
    question_text: str
    niche_list_items: list[FactCheckedItem]
    cost: float

    @property
    def markdown_output(self) -> str:
        return FactCheckedItem.make_markdown_with_valid_and_invalid_lists(
            self.niche_list_items
        )


class NicheListInput(Jsonable, BaseModel):
    question_text: str


class NicheListResearchPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "📋 Niche List Researcher"
    URL_PATH: str = "/niche-list-researcher"
    INPUT_TYPE = NicheListInput
    OUTPUT_TYPE = NicheListOutput
    EXAMPLES_FILE_PATH = (
        "forecasting_tools/front_end/example_outputs/niche_list_page_examples.json"
    )

    @classmethod
    async def _display_intro_text(cls) -> None:
        # markdown = textwrap.dedent(
        #     """
        #     Enter a description of the niche topic you want to research and create a comprehensive list for. The tool will have problems with lists that include more than 15-30 items.
        #     The AI will attempt to find all relevant instances and fact-check them. Examples:
        #     - Times there has been a declaration of a public health emergency of international concern by the World Health Organization
        #     - Times that Apple was successfully sued for patent violations
        #     """
        # )
        # st.markdown(markdown)
        pass

    @classmethod
    async def _get_input(cls) -> NicheListInput | None:
        with st.form("niche_list_form"):
            question_text = st.text_input("Enter your niche list research query here")
            submitted = st.form_submit_button("Research and Generate List")
            if submitted and question_text:
                return NicheListInput(question_text=question_text)
        return None

    @classmethod
    async def _run_tool(cls, input: NicheListInput) -> NicheListOutput:
        with st.spinner(
            "Researching and fact-checking... This may take several minutes..."
        ):
            with MonetaryCostManager() as cost_manager:
                generator = NicheListResearcher(input.question_text)
                fact_checked_items = await generator.research_niche_reference_class(
                    return_invalid_items=True
                )

                cost = cost_manager.current_usage

                return NicheListOutput(
                    question_text=input.question_text,
                    cost=cost,
                    niche_list_items=fact_checked_items,
                )

    @classmethod
    async def _save_run_to_coda(
        cls,
        input_to_tool: NicheListInput,
        output: NicheListOutput,
        is_premade: bool,
    ) -> None:
        if is_premade:
            output.cost = 0
        ForecastDatabaseManager.add_general_report_to_database(
            question_text=input_to_tool.question_text,
            background_info=None,
            resolution_criteria=None,
            fine_print=None,
            prediction=len(output.niche_list_items),
            explanation=output.markdown_output,
            page_url=None,
            price_estimate=output.cost,
            run_type=ForecastRunType.WEB_APP_NICHE_LIST,
        )

    @classmethod
    async def _display_outputs(cls, outputs: list[NicheListOutput]) -> None:
        for output in outputs:
            with st.expander(f"{output.question_text}", expanded=True):
                st.markdown(f"**Cost:** ${output.cost:.2f}")
                st.markdown(ReportDisplayer.clean_markdown(output.markdown_output))


if __name__ == "__main__":
    NicheListResearchPage.main()
