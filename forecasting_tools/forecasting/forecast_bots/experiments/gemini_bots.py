import logging
from datetime import datetime

from forecasting_tools import (
    BinaryQuestion,
    MetaculusQuestion,
    ReasonedPrediction,
    TemplateBot,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.gemini2exp import Gemini2Exp
from forecasting_tools.ai_models.gemini2flash import Gemini2Flash
from forecasting_tools.ai_models.gemini2flashthinking import (
    Gemini2FlashThinking,
)

logger = logging.getLogger(__name__)


class GeminiFlashThinkingExpBot(TemplateBot):
    FINAL_DECISION_LLM = Gemini2FlashThinking(temperature=0.7)

    async def run_research(self, question: MetaculusQuestion) -> str:
        try:
            prompt = clean_indents(
                f"""
                You are a research assistant helping with a forecasting question.
                Generate a very brief analysis focusing only on the most important points.
                Keep your response under 500 words.

                Question: {question.question_text}
                Resolution Criteria: {question.resolution_criteria}
                Fine Print: {question.fine_print}
                Background: {question.background_info}
                """
            )
            return await self.FINAL_DECISION_LLM.invoke(prompt)
        except Exception as e:
            logger.error(f"Research failed: {str(e)}")
            return "Research unavailable due to model limitations."

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        try:
            prompt = clean_indents(
                f"""
                You are a quick-thinking forecaster. Keep your response under 300 words.

                Question: {question.question_text}
                Research: {research}

                Give a very brief analysis:
                1. Time to resolution
                2. Status quo
                3. Key scenario for No
                4. Key scenario for Yes

                End with: "Probability: ZZ%" (0-100)
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


class GeminiExpBot(TemplateBot):
    FINAL_DECISION_LLM = Gemini2Exp(temperature=0.7)

    async def run_research(self, question: MetaculusQuestion) -> str:
        prompt = clean_indents(
            f"""
            You are a thorough research assistant helping with a forecasting question.
            Generate a comprehensive analysis of relevant information, including if the question would resolve Yes or No based on current information.
            Include historical analogies and expert opinions where relevant.

            Question: {question.question_text}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            Background: {question.background_info}
            """
        )
        return await self.FINAL_DECISION_LLM.invoke(prompt)

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
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


class GeminiFlash2Bot(TemplateBot):
    FINAL_DECISION_LLM = Gemini2Flash(temperature=0.7)

    async def run_research(self, question: MetaculusQuestion) -> str:
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
        return await self.FINAL_DECISION_LLM.invoke(prompt)

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a quick-thinking forecaster making a rapid prediction.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}

            Research findings:
            {research}

            Today's date: {datetime.now().strftime("%Y-%m-%d")}

            Give a rapid analysis of:
            1. Time to resolution
            2. Status quo outcome
            3. Quick No scenario
            4. Quick Yes scenario

            End with: "Probability: ZZ%" (0-100)
            """
        )
        reasoning = await self.FINAL_DECISION_LLM.invoke(prompt)
        prediction = self._extract_forecast_from_binary_rationale(
            reasoning, max_prediction=1, min_prediction=0
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )
