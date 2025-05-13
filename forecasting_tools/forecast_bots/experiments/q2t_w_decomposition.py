import asyncio
import logging

from forecasting_tools.agents_and_tools.question_generators.question_decomposer import (
    QuestionDecomposer,
)
from forecasting_tools.agents_and_tools.question_generators.question_operationalizer import (
    QuestionOperationalizer,
)
from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.official_bots.q2_template_bot import (
    Q2TemplateBot2025,
)
from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher

logger = logging.getLogger(__name__)


class Q2TemplateBotWithDecompositionV1(Q2TemplateBot2025):
    """
    Runs forecasts on sub questions separately
    """

    async def run_research(self, question: MetaculusQuestion) -> str:
        ask_news_research = await AskNewsSearcher().get_formatted_news_async(
            question.question_text
        )

        model = self.get_llm("decomposer", "string_name")
        decomposition_result = await QuestionDecomposer(
            model=model
        ).decompose_into_questions(
            fuzzy_topic_or_question=question.question_text,
            number_of_questions=5,
        )
        logger.info(f"Decomposition result: {decomposition_result}")
        sub_questions = decomposition_result.questions
        operationalized_questions = await asyncio.gather(
            *[
                QuestionOperationalizer(
                    model=model
                ).question_title_to_simple_question(question)
                for question in sub_questions
            ]
        )
        metaculus_questions = (
            SimpleQuestion.simple_questions_to_metaculus_question(
                operationalized_questions
            )
        )

        sub_predictor = Q2TemplateBot2025(
            llms=self._llms,
            predictions_per_research_report=5,
            research_reports_per_question=1,
        )
        forecasts = await sub_predictor.forecast_questions(
            metaculus_questions, return_exceptions=True
        )

        formatted_forecasts = ""
        for forecast in forecasts:
            if isinstance(forecast, BaseException):
                logger.error(f"Error forecasting on question: {forecast}")
                continue
            formatted_forecasts += (
                f"Question: {forecast.question.question_text}\n"
            )
            formatted_forecasts += f"Prediction: {forecast.make_readable_prediction(forecast.prediction)}\n"

        research = clean_indents(
            f"""
            {ask_news_research}

            FORECASTS:
            Below are some previous forecasts you have made on related questions
            {formatted_forecasts}
            """
        )
        logger.info(research)
        return research
