from __future__ import annotations

import logging
from datetime import datetime, timedelta

import streamlit as st
from pydantic import BaseModel

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecast_helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.research_agents.question_generator import (
    QuestionGenerator,
    SimpleQuestion,
)
from forecasting_tools.util.jsonable import Jsonable
from front_end.helpers.report_displayer import ReportDisplayer
from front_end.helpers.tool_page import ToolPage

logger = logging.getLogger(__name__)


class QuestionGeneratorInput(Jsonable, BaseModel):
    topic: str
    number_of_questions: int
    resolve_before_date: datetime
    resolve_after_date: datetime
    model: str


class QuestionGeneratorOutput(Jsonable, BaseModel):
    questions: list[SimpleQuestion]
    original_input: QuestionGeneratorInput
    cost: float


class QuestionGeneratorPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "❓ Question Generator"
    URL_PATH: str = "/question-generator"
    INPUT_TYPE = QuestionGeneratorInput
    OUTPUT_TYPE = QuestionGeneratorOutput
    EXAMPLES_FILE_PATH = (
        "front_end/example_outputs/question_generator_page_examples.json"
    )

    @classmethod
    async def _display_intro_text(cls) -> None:
        pass

    @classmethod
    async def _get_input(cls) -> QuestionGeneratorInput | None:
        with st.form("question_generator_form"):
            topic = st.text_input(
                "Topic (optional)",
                value="Lithuanian politics and technology",
            )
            number_of_questions = st.number_input(
                "Number of questions to generate",
                min_value=1,
                max_value=10,
                value=5,
            )
            model = st.text_input(
                "Litellm Model (e.g.: o1, claude-3-7-sonnet-latest, openrouter/deepseek/deepseek-r1)",
                value="o1",
            )
            col1, col2 = st.columns(2)
            with col1:
                resolve_after_date = st.date_input(
                    "Resolve after date",
                    value=datetime.now().date(),
                )
            with col2:
                resolve_before_date = st.date_input(
                    "Resolve before date",
                    value=(datetime.now() + timedelta(days=90)).date(),
                )

            submitted = st.form_submit_button("Generate Questions")
            if submitted:
                return QuestionGeneratorInput(
                    topic=topic,
                    number_of_questions=number_of_questions,
                    resolve_before_date=datetime.combine(
                        resolve_before_date, datetime.min.time()
                    ),
                    resolve_after_date=datetime.combine(
                        resolve_after_date, datetime.min.time()
                    ),
                    model=model,
                )
        return None

    @classmethod
    async def _run_tool(
        cls, input: QuestionGeneratorInput
    ) -> QuestionGeneratorOutput:
        with st.spinner(
            "Generating questions... This may take a few minutes..."
        ):
            with MonetaryCostManager() as cost_manager:
                generator = QuestionGenerator(model=input.model)
                questions = await generator.generate_questions(
                    number_of_questions=input.number_of_questions,
                    topic=input.topic,
                    resolve_before_date=input.resolve_before_date,
                    resolve_after_date=input.resolve_after_date,
                )
                cost = cost_manager.current_usage

                return QuestionGeneratorOutput(
                    questions=questions,
                    original_input=input,
                    cost=cost,
                )

    @classmethod
    async def _save_run_to_coda(
        cls,
        input_to_tool: QuestionGeneratorInput,
        output: QuestionGeneratorOutput,
        is_premade: bool,
    ) -> None:
        if is_premade:
            output.cost = 0
        ForecastDatabaseManager.add_general_report_to_database(
            question_text=f"Topic: {input_to_tool.topic}",
            background_info=str(input_to_tool),
            resolution_criteria=None,
            fine_print=None,
            prediction=None,
            explanation=str(output.questions),
            page_url=None,
            price_estimate=output.cost,
            run_type=ForecastRunType.WEB_APP_QUESTION_GENERATOR,
        )

    @classmethod
    async def _display_outputs(
        cls, outputs: list[QuestionGeneratorOutput]
    ) -> None:
        for output in outputs:
            st.markdown(
                f"**Cost of below questions:** ${output.cost:.2f} | **Topic:** {output.original_input.topic if output.original_input.topic else 'N/A'}"
            )
            for question in output.questions:
                with st.expander(question.question_text):
                    st.markdown("### Question")
                    st.markdown(
                        ReportDisplayer.clean_markdown(question.question_text)
                    )
                    st.markdown("### Resolution Criteria")
                    st.markdown(
                        ReportDisplayer.clean_markdown(
                            question.resolution_criteria
                        )
                    )
                    st.markdown("### Fine Print")
                    st.markdown(
                        ReportDisplayer.clean_markdown(question.fine_print)
                    )
                    st.markdown("### Background Information")
                    st.markdown(
                        ReportDisplayer.clean_markdown(
                            question.background_information
                        )
                    )
                    st.markdown("### Expected Resolution Date")
                    st.markdown(
                        question.expected_resolution_date.strftime("%Y-%m-%d")
                    )


if __name__ == "__main__":
    QuestionGeneratorPage.main()
