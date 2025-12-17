import asyncio
import logging
from datetime import datetime

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.official_bots.template_bot_2025_fall import (
    FallTemplateBot2025,
)
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)


class FallResearchOnlyBot2025(FallTemplateBot2025):

    _max_concurrent_questions = (
        3  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _instructions = clean_indents(
        """
        To run the forecast:
        1. Consider what principles associated with good forecasting you plan to use in this situation, if any (e.g. base rates, bias identification, premortems, simulations, scope sensitivity, aggregation, etc)
        2. Make a research plan
        3. Conduct the research (iterate as needed)
        4. Write down the main facts from the research you conducted that you will consider in your forecast
        5. Do any analysis you need to do, and then write down your rationale for the forecast
        6. Write down your forecast in accordance with the format requested of you
        """
    )

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        config_dict = super()._llm_config_defaults()
        if "researcher" in config_dict:
            config_dict.pop("researcher")
        if "summarizer" in config_dict:
            config_dict["summarizer"] = None
        return config_dict

    async def run_research(self, question: MetaculusQuestion) -> str:
        return "No separate research phase used. Research phase is combined with the forecast"

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {self._instructions}

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        async with self._concurrency_limiter:
            return await self._binary_prompt_to_forecast(question, prompt)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {self._instructions}

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        async with self._concurrency_limiter:
            return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            {self._instructions}

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        async with self._concurrency_limiter:
            return await self._numeric_prompt_to_forecast(question, prompt)

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            {self._instructions}

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        async with self._concurrency_limiter:
            return await self._date_prompt_to_forecast(question, prompt)
