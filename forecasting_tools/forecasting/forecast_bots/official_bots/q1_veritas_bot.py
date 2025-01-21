import asyncio
from datetime import datetime

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.gpt4o import Gpt4o
from forecasting_tools.forecasting.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot,
)
from forecasting_tools.forecasting.helpers.smart_searcher import SmartSearcher
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ReasonedPrediction,
)
from forecasting_tools.forecasting.questions_and_reports.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
)


class Q1VeritasBot(Q1TemplateBot):
    FINAL_DECISION_LLM = Gpt4o(temperature=0.1)

    async def run_research(self, question: MetaculusQuestion) -> str:
        searcher = SmartSearcher(
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster. The superforecaster will give
            you a question they intend to forecast on. To be a great assistant, you generate
            a concise but detailed rundown of the most relevant news, including if the question
            would resolve Yes or No based on current information. You do not produce forecasts yourself.

            The question is:
            {question.question_text}

            Background information:
            {question.background_info if question.background_info else "No background information provided."}

            Resolution criteria:
            {question.resolution_criteria if question.resolution_criteria else "No resolution criteria provided."}

            Fine print:
            {question.fine_print if question.fine_print else "No fine print provided."}
            """
        )
        response = await searcher.invoke(prompt)
        return response

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        assert isinstance(
            question, BinaryQuestion
        ), "Question must be a BinaryQuestion"
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.
            Your interview question is:
            {question.question_text}

            Background information:
            {question.background_info if question.background_info else "No background information provided."}

            Resolution criteria:
            {question.resolution_criteria if question.resolution_criteria else "No resolution criteria provided."}

            Fine print:
            {question.fine_print if question.fine_print else "No fine print provided."}


            Your research assistant says:
            ```
            {research}
            ```

            Today is {datetime.now().strftime("%Y-%m-%d")}.


            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) What the outcome would be if nothing changed.
            (c) The most important factors that will influence a successful/unsuccessful resolution.
            (d) What do you not know that should give you pause and lower confidence? Remember people are statistically overconfident.
            (e) What you would forecast if you were to only use historical precedent (i.e. how often this happens in the past) without any current information.
            (f) What you would forecast if there was only a quarter of the time left.
            (g) What you would forecast if there was 4x the time left.

            You write your rationale and then the last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        gpt_forecast = await self.FINAL_DECISION_LLM.invoke(prompt)
        prediction = self._extract_forecast_from_binary_rationale(
            gpt_forecast, max_prediction=0.95, min_prediction=0.05
        )
        reasoning = (
            gpt_forecast
            + "\nThe original forecast may have been clamped between 5% and 95%."
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        new_questions: list[BinaryQuestion] = []
        for option in question.options:
            new_question = BinaryQuestion(
                question_text=f'Will the outcome be option "{option}" for the question "{question.question_text}"?',
                background_info=question.background_info,
                resolution_criteria=f'The question resolves yes if the below criteria resolves for the option "{option}". Here is the overall question criteria:\n{question.resolution_criteria}',
                fine_print=question.fine_print,
                id_of_post=0,
            )
            new_questions.append(new_question)
        binary_forecasts = await asyncio.gather(
            *[
                self._run_forecast_on_binary(new_question, research)
                for new_question in new_questions
            ]
        )

        options_message = "\n".join(
            [
                f"{opt}: {pred.prediction_value*100:.1f}%"
                for opt, pred in zip(question.options, binary_forecasts)
            ]
        )

        consistency_prompt = clean_indents(
            f"""
            You are a professional forecaster. You've made individual probability assessments for each option in a multiple choice question.
            Your task is to make these probabilities consistent (sum to 100%) while preserving their relative relationships.
            The probabilities were made by domain experts smarter in this field than you, but they may conflict. Your goal is to synthesize their assessments.

            The question being asked is:
            {question.question_text}

            The resolution criteria are:
            {question.resolution_criteria}

            The fine print is:
            {question.fine_print}

            The options and their initial probabilities are:
            {options_message}

            Write a brief explanation of your adjustments, then output the final probabilities in this exact format:
            Option_A: XX
            Option_B: XX
            ...
            """
        )

        final_distribution = await self.FINAL_DECISION_LLM.invoke(
            consistency_prompt
        )
        prediction = self._extract_forecast_from_multiple_choice_rationale(
            final_distribution, question.options
        )
        reasoning = (
            "Individual option assessments and reasoning:\n"
            + "\n\n".join(
                [
                    f"#### Option: {opt}\n"
                    f"Probability: {pred.prediction_value*100:.1f}%\n"
                    f"Reasoning:\n{pred.reasoning}"
                    for opt, pred in zip(question.options, binary_forecasts)
                ]
            )
            + "\n\nFinal consistent distribution reasoning:\n"
            + final_distribution
        )

        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=reasoning,
            sub_predictions=binary_forecasts,
        )
