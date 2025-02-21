import asyncio
from datetime import datetime

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot2025,
)
from forecasting_tools.forecast_helpers.prediction_extractor import (
    PredictionExtractor,
)


class Q1TemplateBotWithConvertToBinary(Q1TemplateBot2025):

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

            Previously you have found the following probabilities for the options:
            {options_message}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )

        final_distribution = await self._get_final_decision_llm().invoke(
            consistency_prompt
        )
        prediction = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                final_distribution, question.options
            )
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

            Previously you have found the following percentiles and probabilities:
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

        final_distribution = await self._get_final_decision_llm().invoke(
            consistency_prompt
        )
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
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
            + "\n\n#### Final consistent distribution reasoning:\n"
            + final_distribution
        )

        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=reasoning,
        )
