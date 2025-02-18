import logging

from forecasting_tools.research_agents.question_responder import (
    QuestionResponder,
)

logger = logging.getLogger(__name__)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher


class GeneralResearcher(QuestionResponder):
    NAME = "General Search"
    DESCRIPTION_OF_WHEN_TO_USE = "Use this responder when the question doesn't match well with any of the other responders or you need simple online information"

    async def respond_with_markdown(self) -> str:
        prompt = clean_indents(
            f"""
            You are a discerning super genius expert helping a superforecaster forecasting for Metaculus. You are doing research on a question to help them make a better prediction.
            Your goal is to answer the question, targeting information most likely to change the forecaster's prediction and save the forecaster research time.

            You will be paid $10 if you give a concise, accurate answer to the question.
            You will be given a $50 tip if you can find key information that changes their prediction by 5% or more.
            You will be given a $200 tip if you can find and clearly highlight key information that changes their prediction by 10% or more.
            You will be fired if you make up information or present information in a misleading way.

            When figuring out what information to highlight, consider the following:
            - Philip Tetlock's book "Superforecasting" and his "Forecasting ten commandments"
            - Historical rates, numbers, and stats are almost always important
            - Being specific is always better than being general.
            - Consider the type of context that could be given to evaluate the trustworthiness of the source (e.g. don't say "The source says that the scientific consensus is..." say "According to a large scale survey of [X] participants, it was found that 80% of scientists believe [belief] though this was a telephone survey which may be subject to sampling bias. Margin of error was not mentioned."

            You have been asked the following question:
            "{self.question}"

            Please answer the question. Cite your sources inline. Only give 1-2 paragraphs and/or some bullet points. Your goal is to highlight the most important information.
            """
        )
        model = SmartSearcher(temperature=0)
        answer = await model.invoke(prompt)
        logger.info(f"Answered question: {self.question}")
        return answer
