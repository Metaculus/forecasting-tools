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
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    NumericDistribution,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
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
            The probabilities were made by domain experts smarter in this field than you, but they may conflict. Your goal is to synthesize their assessments. Change them as little as possible.

            The question being asked is:
            {question.question_text}

            The resolution criteria are:
            {question.resolution_criteria}

            The fine print is:
            {question.fine_print}

            Your research assistant says:
            ```
            {research}
            ```

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

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        initial_prediction: ReasonedPrediction[NumericDistribution] = (
            await super()._run_forecast_on_numeric(question, research)
        )

        new_questions: list[BinaryQuestion] = []
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        for (
            percentile
        ) in initial_prediction.prediction_value.declared_percentiles:
            new_question = BinaryQuestion(
                question_text=f'Will the value be less than or equal to {percentile.value} for the question "{question.question_text}"?',
                background_info=f"{question.background_info}\n{upper_bound_message}\n{lower_bound_message}",
                resolution_criteria=f"The question resolves yes if the value is less than or equal to {percentile.value} (assume the units inferred below). Here is the overall question criteria:\n{question.resolution_criteria}",
                fine_print=question.fine_print,
                id_of_post=0,
            )
            new_questions.append(new_question)

        # Get binary forecasts for each percentile
        binary_forecasts = await asyncio.gather(
            *[
                self._run_forecast_on_binary(new_question, research)
                for new_question in new_questions
            ]
        )

        # Create message showing initial percentile assessments
        percentile_message = "\n".join(
            [
                f"Percentile {int(pred.prediction_value * 100)}: {percentile.value}"
                for percentile, pred in zip(
                    initial_prediction.prediction_value.declared_percentiles,
                    binary_forecasts,
                )
            ]
        )

        consistency_prompt = clean_indents(
            f"""
            You are a professional forecaster. You've made individual probability assessments for different percentiles in a numeric question.
            Your task is to make these percentiles consistent while preserving their relative relationships as much as possible.
            The probabilities were made by domain experts smarter in this field than you, but they may conflict.
            Your goal is to make your own assessment and then synthesize it with their assessment.
            Every consecutive percentile should have a higher value
            Please fill in the gaps and give the percentiles as listed in the example below

            The question being asked is:
            {question.question_text}

            The resolution criteria are:
            {question.resolution_criteria}

            The fine print is:
            {question.fine_print}

            Your research assistant says:
            ```
            {research}
            ```

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            The initial assessment is below:
            {percentile_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.
            (g) Given the initial assessment was made by experts smarter than you, how would you adjust your own assessment to synthesize your views?

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

        final_distribution = await self.FINAL_DECISION_LLM.invoke(
            consistency_prompt
        )
        prediction = self._extract_forecast_from_numeric_rationale(
            final_distribution, question
        )

        reasoning = (
            "Individual percentile assessments and reasoning:\n"
            + "\n\n".join(
                [
                    f"#### Percentile {int(percentile.percentile * 100)}\n"
                    f"Value: {percentile.value}\n"
                    f"Binary Assessment: {pred.prediction_value*100:.1f}%\n"
                    f"Reasoning:\n{pred.reasoning}"
                    for percentile, pred in zip(
                        initial_prediction.prediction_value.declared_percentiles,
                        binary_forecasts,
                    )
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
