import logging

from forecasting_tools.agents_and_tools.base_rates.base_rate_researcher import (
    BaseRateResearcher,
)
from forecasting_tools.agents_and_tools.deprecated.configured_llms import (
    BasicLlm,
)
from forecasting_tools.agents_and_tools.deprecated.general_researcher import (
    GeneralResearcher,
)
from forecasting_tools.agents_and_tools.deprecated.question_responder import (
    QuestionResponder,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents

logger = logging.getLogger(__name__)


class QuestionRouter:
    AVAILABLE_REPONDERS: list[type[QuestionResponder]] = [
        GeneralResearcher,
        BaseRateResearcher,
    ]

    async def answer_question_with_markdown_using_routing(
        self, question: str
    ) -> str:
        available_responder_descriptions = ""

        for responder in self.AVAILABLE_REPONDERS:
            available_responder_descriptions += (
                f"{responder.NAME}: {responder.DESCRIPTION_OF_WHEN_TO_USE}\n"
            )

        q1_routing_prompt = clean_indents(
            f"""
            You are a research manager. You have to choose one of {len(self.AVAILABLE_REPONDERS)} research strategies to answer a question.

            Your job is to suggest the best strategy to answer the following question:
            {question}

            The possible strategies for answering the question are as follows:
            {available_responder_descriptions}

            Lets take this step by step:
            1. List out the responders whose description matches the type of question you have
            2. Of the ones who's description matches, pick the one that you think is most likely to give a good answer
            3. Write down the name of the strategy in all caps

            Remember to give the research strategy name exactly as written, and put it in all caps
            """
        )
        model = BasicLlm(temperature=0)
        response = await model.invoke(q1_routing_prompt)
        response_to_be_logged = response.replace("\n", "|")
        logger.info(f"Response to routing prompt: {response_to_be_logged}")

        chosen_responder = GeneralResearcher
        default_strategy_chosen = True
        for responder in self.AVAILABLE_REPONDERS:
            if responder.NAME.upper() in response:
                chosen_responder = responder
                default_strategy_chosen = False

        logger.info(f"Chose responder strategy: {chosen_responder.NAME}")
        answer = await chosen_responder(question).respond_with_markdown()
        logger.info(
            f"Answered question with strategy: {chosen_responder.NAME}"
        )

        if default_strategy_chosen:
            return f"Defaulting to strategy {chosen_responder.NAME}:\n"
        else:
            return f"Using strategy {chosen_responder.NAME}:\n{answer}"
