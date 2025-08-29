import asyncio
import logging
from datetime import datetime

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.official_bots.fall_template_bot import (
    FallTemplateBot2025,
)

logger = logging.getLogger(__name__)


class FallResearchOnlyBot2025(FallTemplateBot2025):

    _max_concurrent_questions = (
        5  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _research_prompt = clean_indents(
        """
        To run the forecast:
        1. Remember and consider the principles that are often helpful when trying to forecast well
        2. Make a research plan
        3. Conduct the research (iterate as needed)
        4. Write down the main facts from the research you conducted that you will consider in your forecast
        5. Write down your rationale for the forecast
        6. Write down your forecast in accordance with the format requested of you
        """
    )

    async def run_research(self, question: MetaculusQuestion) -> str:
        researcher = self.get_llm("researcher")
        if not (
            researcher == "no_research" or researcher == "None" or researcher == None
        ):
            raise ValueError("Research only bot does not use a researcher")
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

            Before answering you write:
            (a) {self._research_prompt}
            (b) The time left until the outcome to the question is known.
            (c) The status quo outcome if nothing changed.
            (d) A brief description of a scenario that results in a No outcome.
            (e) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

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

            Before answering you write:
            (a) {self._research_prompt}
            (b) The time left until the outcome to the question is known.
            (c) The status quo outcome if nothing changed.
            (d) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
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

            Before answering you write:
            (a) {self._research_prompt}
            (b) The time left until the outcome to the question is known.
            (c) The outcome if nothing changed.
            (d) The outcome if the current trend continued.
            (e) The expectations of experts and markets.
            (f) A brief description of an unexpected scenario that results in a low outcome.
            (g) A brief description of an unexpected scenario that results in a high outcome.

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
        return await self._numeric_prompt_to_forecast(question, prompt)
