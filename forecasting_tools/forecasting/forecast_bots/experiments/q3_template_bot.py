from datetime import datetime

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.gpt4o import Gpt4o
from forecasting_tools.ai_models.perplexity import Perplexity
from forecasting_tools.forecasting.forecast_bots.template_bot import (
    TemplateBot,
)
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ReasonedPrediction,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
    MetaculusQuestion,
)


class Q3TemplateBot(TemplateBot):
    """
    Find the q3 bot here: https://github.com/Metaculus/metac-bot/commit/e459f2958f66658783057da46e257896b49607be
    """

    FINAL_DECISION_LLM = Gpt4o(
        temperature=0.1
    )  # Q3 Bot used the default llama index temperature which as of Dec 21 2024 is 0.1

    async def run_research(self, question: MetaculusQuestion) -> str:
        system_prompt = clean_indents(
            """
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.
            """
        )

        # Note: The original q3 bot did not set temperature, and I could not find the default temperature of perplexity
        response = await Perplexity(
            temperature=0.1, system_prompt=system_prompt
        ).invoke(question.question_text)
        return response

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) What the outcome would be if nothing changed.
            (c) What you would forecast if there was only a quarter of the time left.
            (d) What you would forecast if there was 4x the time left.

            You write your rationale and then the last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.FINAL_DECISION_LLM.invoke(prompt)
        prediction = self._extract_forecast_from_binary_rationale(
            reasoning, max_prediction=0.99, min_prediction=0.01
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )
