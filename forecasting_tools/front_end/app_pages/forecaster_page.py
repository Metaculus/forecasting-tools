import json
import logging
import re

import dotenv
import streamlit as st
from pydantic import BaseModel

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.forecast_bots.bot_lists import get_all_important_bot_classes
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.main_bot import MainBot
from forecasting_tools.front_end.helpers.report_displayer import ReportDisplayer
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.helpers.metaculus_api import MetaculusApi
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class ForecastInput(Jsonable, BaseModel):
    question: BinaryQuestion


class ForecasterPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "🔍 Forecast a Question"
    URL_PATH: str = "/forecast"
    INPUT_TYPE = ForecastInput
    OUTPUT_TYPE = BinaryReport
    EXAMPLES_FILE_PATH = (
        "forecasting_tools/front_end/example_outputs/forecast_page_examples.json"
    )

    QUESTION_TEXT_BOX = "question_text_box"
    RESOLUTION_CRITERIA_BOX = "resolution_criteria_box"
    FINE_PRINT_BOX = "fine_print_box"
    BACKGROUND_INFO_BOX = "background_info_box"
    NUM_BACKGROUND_QUESTIONS_BOX = "num_background_questions_box"
    NUM_BASE_RATE_QUESTIONS_BOX = "num_base_rate_questions_box"
    METACULUS_URL_INPUT = "metaculus_url_input"
    FETCH_BUTTON = "fetch_button"
    BOT_CHOICE_KEY = "forecaster_bot_choice"
    DEFAULT_BOT_NAME = MainBot.__name__

    @classmethod
    async def _display_intro_text(cls) -> None:
        cls._display_bot_selector_and_config()

    @classmethod
    def _get_available_bot_classes(cls) -> list[type[ForecastBot]]:
        bot_classes = get_all_important_bot_classes()
        ordered: list[type[ForecastBot]] = []
        for bot_class in bot_classes:
            if bot_class is MainBot:
                ordered.insert(0, bot_class)
            elif bot_class not in ordered:
                ordered.append(bot_class)
        if MainBot not in ordered:
            ordered.insert(0, MainBot)
        return ordered

    @classmethod
    def _get_selected_bot_class(cls) -> type[ForecastBot]:
        bot_classes = cls._get_available_bot_classes()
        bot_class_by_name = {bot.__name__: bot for bot in bot_classes}
        chosen_name = st.session_state.get(cls.BOT_CHOICE_KEY, cls.DEFAULT_BOT_NAME)
        return bot_class_by_name.get(chosen_name, MainBot)

    @classmethod
    def _display_bot_selector_and_config(cls) -> None:
        bot_classes = cls._get_available_bot_classes()
        bot_names = [bot.__name__ for bot in bot_classes]
        if cls.BOT_CHOICE_KEY not in st.session_state:
            st.session_state[cls.BOT_CHOICE_KEY] = cls.DEFAULT_BOT_NAME

        default_index = (
            bot_names.index(st.session_state[cls.BOT_CHOICE_KEY])
            if st.session_state[cls.BOT_CHOICE_KEY] in bot_names
            else 0
        )
        st.selectbox(
            "Forecasting Bot",
            options=bot_names,
            index=default_index,
            key=cls.BOT_CHOICE_KEY,
            help=(
                "The bot used to forecast. "
                f"`{cls.DEFAULT_BOT_NAME}` is the default and the verified "
                "highest-accuracy bot."
            ),
        )

        bot_class = cls._get_selected_bot_class()
        with st.expander("Bot Configuration", expanded=False):
            cls._render_bot_config(bot_class)

    @classmethod
    def _render_bot_config(cls, bot_class: type[ForecastBot]) -> None:
        try:
            bot_instance = bot_class(
                research_reports_per_question=1,
                predictions_per_research_report=5,
                publish_reports_to_metaculus=False,
                folder_to_save_reports_to=None,
            )
        except Exception as exception:
            st.error(f"Could not instantiate {bot_class.__name__}: {exception}")
            return

        docstring = (bot_class.__doc__ or "").strip()
        if docstring:
            st.markdown(f"**Description:** {docstring}")
        st.markdown(f"**Bot Class:** `{bot_class.__name__}`")
        st.markdown(
            f"**Research Reports per Question:** "
            f"{bot_instance.research_reports_per_question}"
        )
        st.markdown(
            f"**Predictions per Research Report:** "
            f"{bot_instance.predictions_per_research_report}"
        )
        st.markdown("**LLM Configuration:**")
        st.code(
            json.dumps(bot_instance.make_llm_dict(), indent=2, default=str),
            language="json",
        )

    @classmethod
    async def _get_input(cls) -> ForecastInput | None:
        cls.__display_metaculus_url_input()
        with st.form("forecast_form"):
            question_text = st.text_input(
                "Yes/No Binary Question", key=cls.QUESTION_TEXT_BOX
            )
            resolution_criteria = st.text_area(
                "Resolution Criteria (optional)",
                key=cls.RESOLUTION_CRITERIA_BOX,
            )
            fine_print = st.text_area("Fine Print (optional)", key=cls.FINE_PRINT_BOX)
            background_info = st.text_area(
                "Background Info (optional)", key=cls.BACKGROUND_INFO_BOX
            )

            submitted = st.form_submit_button("Submit")

            if submitted:
                if not question_text:
                    st.error("Question Text is required.")
                    return None
                question = BinaryQuestion(
                    question_text=question_text,
                    background_info=background_info,
                    resolution_criteria=resolution_criteria,
                    fine_print=fine_print,
                    page_url="",
                    api_json={},
                )
                return ForecastInput(
                    question=question,
                )
        return None

    @classmethod
    async def _run_tool(cls, input: ForecastInput) -> BinaryReport:
        bot_class = cls._get_selected_bot_class()
        with st.spinner(
            f"Forecasting with `{bot_class.__name__}`... "
            "This may take a minute or two..."
        ):
            report = await bot_class(
                research_reports_per_question=1,
                predictions_per_research_report=5,
                publish_reports_to_metaculus=False,
                folder_to_save_reports_to=None,
            ).forecast_question(input.question)
            assert isinstance(report, BinaryReport)
            return report

    @classmethod
    async def _save_run_to_coda(
        cls,
        input_to_tool: ForecastInput,
        output: BinaryReport,
        is_premade: bool,
    ) -> None:
        if is_premade:
            output.price_estimate = 0
        ForecastDatabaseManager.add_forecast_report_to_database(
            output, run_type=ForecastRunType.WEB_APP_FORECAST
        )

    @classmethod
    async def _display_outputs(cls, outputs: list[BinaryReport]) -> None:
        ReportDisplayer.display_report_list(outputs)

    @classmethod
    def __display_metaculus_url_input(cls) -> None:
        with st.expander("Use an existing Metaculus Binary question"):
            st.write("Enter a Metaculus question URL to autofill the form below.")

            metaculus_url = st.text_input(
                "Metaculus Question URL", key=cls.METACULUS_URL_INPUT
            )
            fetch_button = st.button("Fetch Question", key=cls.FETCH_BUTTON)

            if fetch_button and metaculus_url:
                with st.spinner("Fetching question details..."):
                    try:
                        question_id = cls.__extract_question_id(metaculus_url)
                        metaculus_question = MetaculusApi.get_question_by_post_id(
                            question_id
                        )
                        if isinstance(metaculus_question, BinaryQuestion):
                            cls.__autofill_form(metaculus_question)
                        else:
                            st.error(
                                "Only binary questions are supported at this time."
                            )
                    except Exception as e:
                        st.error(
                            f"An error occurred while fetching the question: {e.__class__.__name__}: {e}"
                        )

    @classmethod
    def __extract_question_id(cls, url: str) -> int:
        match = re.search(r"/questions/(\d+)/", url)
        if match:
            return int(match.group(1))
        raise ValueError(
            "Invalid Metaculus question URL. Please ensure it's in the format: https://metaculus.com/questions/[ID]/[question-title]/"
        )

    @classmethod
    def __autofill_form(cls, question: BinaryQuestion) -> None:
        st.session_state[cls.QUESTION_TEXT_BOX] = question.question_text
        st.session_state[cls.BACKGROUND_INFO_BOX] = question.background_info or ""
        st.session_state[cls.RESOLUTION_CRITERIA_BOX] = (
            question.resolution_criteria or ""
        )
        st.session_state[cls.FINE_PRINT_BOX] = question.fine_print or ""


if __name__ == "__main__":
    dotenv.load_dotenv()
    ForecasterPage.main()
