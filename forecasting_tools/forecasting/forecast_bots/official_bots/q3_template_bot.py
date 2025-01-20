import re
from datetime import datetime

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.gpt4o import Gpt4o
from forecasting_tools.ai_models.perplexity import Perplexity
from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastBot,
)
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ReasonedPrediction,
)
from forecasting_tools.forecasting.questions_and_reports.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    NumericDistribution,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)


class Q3TemplateBot(ForecastBot):
    """
    This is the template bot for the Q3 2024 Metaculus AI Tournament.
    It should be exactly the same except for Perplexity running on a new model (the original model was deprecated)
    Find the q3 bot here: https://github.com/Metaculus/metac-bot/commit/e459f2958f66658783057da46e257896b49607be
    This comment was last updated on Jan 20 2025
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

    def _extract_forecast_from_binary_rationale(
        self, rationale: str, max_prediction: float, min_prediction: float
    ) -> float:
        assert 0 <= max_prediction <= 1
        assert 0 <= min_prediction <= 1
        assert max_prediction >= min_prediction
        matches = re.findall(r"(\d+)%", rationale)
        if matches:
            # Return the last number found before a '%'
            original_number = int(matches[-1]) / 100
            clamped_number = min(
                max_prediction, max(min_prediction, original_number)
            )
            assert (
                min_prediction <= clamped_number <= max_prediction
            ), f"Clamped number {clamped_number} is not between {min_prediction} and {max_prediction}"
            return clamped_number
        else:
            raise ValueError(
                f"Could not extract prediction from response: {rationale}"
            )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        raise NotImplementedError("Multiple choice was not supported in Q3")

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        raise NotImplementedError("Numeric was not supported in Q3")
