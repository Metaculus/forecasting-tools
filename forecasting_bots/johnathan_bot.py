import logging
import re
from datetime import datetime

from forecasting_tools import (
    BinaryQuestion,
    MetaculusQuestion,
    ReasonedPrediction,
    TemplateBot,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
    NumericDistribution,
    PredictedOption,
    Percentile,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.gemini2exp import Gemini2Exp
from forecasting_tools.ai_models.gemini2flash import Gemini2Flash
from forecasting_tools.ai_models.gemini2flashthinking import (
    Gemini2FlashThinking,
)
from forecasting_tools.forecasting.helpers.smart_searcher import SmartSearcher

logger = logging.getLogger(__name__)



class JohnathanBot(TemplateBot):
    """
    A composite bot that utilizes different Gemini models for research and forecasting.
    """

    # Use Gemini2Exp for final forecasting decisions
    FINAL_DECISION_LLM = Gemini2Exp(temperature=0.7)

    def __init__(
        self,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        skip_questions_that_error: bool = True,
        use_flash_thinking_for_research: bool = True,  # Toggle between Flash and FlashThinking
        **kwargs,
    ):
        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_questions_that_error=skip_questions_that_error,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            **kwargs,
        )
        # Choose between Gemini2FlashThinking and Gemini2Flash for research
        if use_flash_thinking_for_research:
            self.RESEARCH_LLM = Gemini2FlashThinking(temperature=0.7)
        else:
            self.RESEARCH_LLM = Gemini2Flash(temperature=0.7)


    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Conducts research using either Gemini2FlashThinking or Gemini2Flash (with grounding via SmartSearcher).
        """
        try:
            if isinstance(self.RESEARCH_LLM, Gemini2FlashThinking):
                prompt = clean_indents(
                    f"""
                    You are a research assistant helping with a forecasting question.
                    Generate a concise but detailed analysis of relevant information, including if the question would resolve Yes or No based on current information.
                    Focus on speed and key points rather than exhaustive detail.

                    Question: {question.question_text}
                    Resolution Criteria: {question.resolution_criteria}
                    Fine Print: {question.fine_print}
                    Background: {question.background_info}
                    """
                )
            else:  # Gemini2Flash
                prompt = clean_indents(
                    f"""
                    You are a research assistant helping with a forecasting question.
                    Generate a concise but detailed analysis of relevant information, including if the question would resolve Yes or No based on current information.
                    Focus on speed and key points rather than exhaustive detail.

                    Question: {question.question_text}
                    Resolution Criteria: {question.resolution_criteria}
                    Fine Print: {question.fine_print}
                    Background: {question.background_info}
                    """
                )
            # Ground the research using SmartSearcher
            research_text = await SmartSearcher().invoke(prompt)
            return research_text
        except Exception as e:
            logger.error(f"Research failed: {str(e)}")
            return "Research unavailable due to model limitations."

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """
        Generates a forecast on a binary question using Gemini2Exp.
        """
        try:
            prompt = clean_indents(
                f"""
                You are a professional forecaster making a detailed prediction.

                Question: {question.question_text}
                Background: {question.background_info}
                Resolution Criteria: {question.resolution_criteria}
                Fine Print: {question.fine_print}

                Research findings:
                {research}

                Today's date: {datetime.now().strftime("%Y-%m-%d")}

                Please provide a detailed analysis of:
                1. Time remaining until resolution and key milestones
                2. Current status quo outcome and historical trends
                3. Comprehensive scenario leading to No, with key factors
                4. Comprehensive scenario leading to Yes, with key factors
                5. Expert opinions and market signals if relevant

                Remember to weigh the status quo heavily as change happens slowly.

                End your response with: "Probability: ZZ%" (a number between 0-100)
                """
            )
            reasoning = await self.FINAL_DECISION_LLM.invoke(prompt)
            prediction = self._extract_forecast_from_binary_rationale(
                reasoning, max_prediction=1, min_prediction=0
            )
            return ReasonedPrediction(
                prediction_value=prediction, reasoning=reasoning
            )
        except Exception as e:
            logger.error(f"Forecast failed: {str(e)}")
            return ReasonedPrediction(
                prediction_value=0.5,
                reasoning="Forecast failed due to model limitations.",
            )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        """
        Generates a forecast on a multiple-choice question using Gemini2Exp.
        """
        try:
            prompt = clean_indents(
                f"""
                You are a professional forecaster making a detailed prediction.

                Question: {question.question_text}
                Options: {question.options}
                Background: {question.background_info}
                Resolution Criteria: {question.resolution_criteria}
                Fine Print: {question.fine_print}

                Research findings:
                {research}

                Today's date: {datetime.now().strftime("%Y-%m-%d")}

                Please provide a detailed analysis of:
                1. Time remaining until resolution and key milestones
                2. Current status quo outcome and historical trends
                3. Comprehensive analysis for each option, with key factors supporting and opposing each outcome
                4. Expert opinions and market signals if relevant

                Remember to weigh the status quo heavily as change happens slowly.

                End your response with a probability distribution over the options, summing to 100%.
                The last thing you write is your final probabilities for the N options in this order {question.options} as:
                Option_A: Probability_A
                Option_B: Probability_B
                ...
                Option_N: Probability_N
                ...

                """
            )
            reasoning = await self.FINAL_DECISION_LLM.invoke(prompt)
            prediction = self._extract_forecast_from_multiple_choice_rationale(
                reasoning, question.options
            )
            return ReasonedPrediction(
                prediction_value=prediction, reasoning=reasoning
            )


        except Exception as e:
            logger.error(f"Forecast failed: {str(e)}")
            return ReasonedPrediction(
                prediction_value=PredictedOptionList(
                    predicted_options=[
                        PredictedOption(option_name=option, probability=1 / len(question.options))
                        for option in question.options
                    ]
                ),
                reasoning="Forecast failed due to model limitations.",
            )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        Generates a forecast on a numeric question using Gemini2Exp.
        """
        try:
            if question.open_upper_bound:
                upper_bound_message = ""
            else:
                upper_bound_message = f"The outcome can not be higher than {question.upper_bound}."
            if question.open_lower_bound:
                lower_bound_message = ""
            else:
                lower_bound_message = f"The outcome can not be lower than {question.lower_bound}."

            prompt = clean_indents(
                f"""
                You are a professional forecaster making a detailed prediction.

                Question: {question.question_text}
                Background: {question.background_info}
                Resolution Criteria: {question.resolution_criteria}
                Fine Print: {question.fine_print}

                Research findings:
                {research}

                Today's date: {datetime.now().strftime("%Y-%m-%d")}

                {lower_bound_message}
                {upper_bound_message}

                Please provide a detailed analysis of:
                1. Time remaining until resolution and key milestones
                2. Current status quo value and historical trends
                3. Comprehensive scenario leading to a low outcome, with key factors
                4. Comprehensive scenario leading to a high outcome, with key factors
                5. Expert opinions and market signals if relevant

                Remember to consider the full range of possible outcomes.

                End your response with your probability distribution in the following format:
                10th percentile: X
                25th percentile: Y
                50th percentile: Z
                75th percentile: A
                90th percentile: B
                """
            )
            reasoning = await self.FINAL_DECISION_LLM.invoke(prompt)
            prediction = self._extract_forecast_from_numeric_rationale(
                reasoning, question
            )
            return ReasonedPrediction(
                prediction_value=prediction, reasoning=reasoning
            )
        except Exception as e:
            logger.error(f"Forecast failed: {str(e)}")
            return ReasonedPrediction(
                prediction_value=NumericDistribution(
                    declared_percentiles=[
                        Percentile(value=question.lower_bound, percentile=0.01),
                        Percentile(value=question.upper_bound, percentile=0.99),
                    ],
                    open_upper_bound=question.open_upper_bound,
                    open_lower_bound=question.open_lower_bound,
                    upper_bound=question.upper_bound,
                    lower_bound=question.lower_bound,
                    zero_point=question.zero_point,
                ),
                reasoning="Forecast failed due to model limitations.",
            )