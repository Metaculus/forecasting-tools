# Steps to take:
# - Get Topic (randomly generate topics if not given a topic or search if given a direction)
# - Topic -> Find questions that shed light on the topic (Question title operationalizer)
# 	- General news
# 	- Useful questions
# - Question title -> Turn into full question
# - Refine Resolution/fine print maker
# - Research Background information
# - Return question
# NOTE: Start at any part of the process that the person asks you to

from __future__ import annotations

import asyncio

from agents import function_tool
from pydantic import BaseModel

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_helpers.structure_output import (
    structure_output,
)


class DecompositionResult(BaseModel):
    reasoning: str
    questions: list[str]


class QuestionDecomposer:
    def __init__(
        self,
        model: str | GeneralLlm = "openrouter/perplexity/sonar-reasoning-pro",
    ) -> None:
        self.model: GeneralLlm = GeneralLlm.to_llm(model)

    async def decompose_into_questions(
        self,
        fuzzy_topic_or_question: str | None,
        related_research: str | None,
        additional_context: str | None,
        number_of_questions: int = 5,
    ) -> DecompositionResult:
        prompt = clean_indents(
            f"""
            # Instructions
            You are a research assistant to a superforecaster

            You want to take an overarching topic or question they have given you and decompose
            it into a list of sub questions that that will lead to better understanding and forecasting
            the topic or question.

            Your research process should look like this:
            1. First get general news on the topic
            2. Then pick 3 things to follow up with. Search perplexity with these in parallel
            3. Then brainstorm 2x the number of question requested number of key questions requested
            4. Pick your top questions
            5. Give your final answer as:
                - Reasoning
                - Research Summary
                - List of Questions

            Don't forget to INCLUDE Links (including to each question if possible)!
            Copy the links IN FULL to all your answers so others can know where you got your information.

            # Question requireemnts
            - The question can be forecast and will be resolvable with public information
                - Good: "Will SpaceX launch a rocket in 2023?"
                - Bad: "Will Elon mention his intention to launch in a private meeting by the end of 2023?"
            - The question should be specific and not vague
            - The question should have an inferred date
            - The question should shed light on the topic and have high VOI (Value of Information)

            # Good candidates for follow up question to get context
            - Anything that shed light on a good base rate (especially ones that already have data)
            - If there might be a index, site, etc that would allow for a clear resolution
            - Consider if it would be best to ask a binary ("Will X happen"), numeric ("How many?"), or multiple choice question ("Which of these will occur?")


            # Your Task
            ## Topic/Question to Decompose
            Please decompose the following topic or question into a list of {number_of_questions} sub questions.

            Question/Topic: {fuzzy_topic_or_question}

            ## Additional Context/Criteria
            {additional_context}

            ## Related Research
            {related_research}
            """
        )
        final_output = await self.model.invoke(prompt)
        structured_output = await structure_output(
            str(final_output), DecompositionResult
        )
        return structured_output

    @function_tool
    @staticmethod
    def decompose_into_questions_tool(
        fuzzy_topic_or_question: str,
        number_of_questions: int,
        related_research: str,
        additional_criteria_or_context_from_user: str | None,
    ) -> DecompositionResult:
        """
        Decompose a topic or question into a list of sub questions that helps to understand and forecast the topic or question.

        Args:
            fuzzy_topic_or_question: The topic or question to decompose.
            number_of_questions: The number of questions to decompose the topic or question into. Default to 5.
            related_research: Include as much research as possible to help make a good question (especially include important drivers/influencers of the topic)
            additional_criteria_or_context_from_user: Additional criteria or context from the user (default to None)

        Returns:
            A DecompositionResult object with the following fields:
            - reasoning: The reasoning for the decomposition.
            - questions: A list of sub questions.
        """
        return asyncio.run(
            QuestionDecomposer().decompose_into_questions(
                fuzzy_topic_or_question=fuzzy_topic_or_question,
                number_of_questions=number_of_questions,
                additional_context=additional_criteria_or_context_from_user,
                related_research=related_research,
            )
        )


# agent_instructions_v4 = """
# # Instructions
# You are a research assistant to a superforecaster

# You want to take an overarching topic or question they have given you and decompose
# it into a list of sub questions that that will lead to better understanding and forecasting
# the topic or question.

# Your research process should look like this:
# 1. First get general news on the topic
# 2. Then pick 3 things to follow up with. Search perplexity with these in parallel
# 3. Then brainstorm 2x the number of question requested number of key questions requested
# 4. Pick your top questions
# 5. Give your final answer as:
#     - Reasoning
#     - Research Summary
#     - List of Questions

# Don't forget to INCLUDE Links (including to each question if possible)!
# Copy the links IN FULL to all your answers so others can know where you got your information.

# # Question requireemnts
# - The question can be forecast and will be resolvable with public information
#     - Good: "Will SpaceX launch a rocket in 2023?"
#     - Bad: "Will Elon mention his intention to launch in a private meeting by the end of 2023?"
# - The question should be specific and not vague
# - The question should have an inferred date
# - The question should shed light on the topic and have high VOI (Value of Information)

# # Good candidates for follow up question to get context
# - Anything that shed light on a good base rate (especially ones that already have data)
# - If there might be a index, site, etc that would allow for a clear resolution
# - Consider if it would be best to ask a binary ("Will X happen"), numeric ("How many?"), or multiple choice question ("Which of these will occur?")

# DO NOT ask follow up questions. Just execute the plan the best you can.
# """


# agent_instructions_v1 = """
# # Instructions
# You are a research assistant to a superforecaster

# You want to take an overarching topic or question they have given you and decompose
# it into a list of sub questions that that will lead to better understanding and forecasting
# the topic or question.

# Your research process should look like this:
# 1. First get general news on the topic and run a perplexity search
# 2. Pick 3 ideas things to follow up with and search perplexity with these
# 3. Then brainstorm 2x the number of question requested number of key questions requested
# 4. Pick  your top questions
# 5. Give your final answer as:
#     - Reasoning
#     - Research Summary
#     - List of Questions

# # Question requireemnts
# - The question can be forecast and will be resolvable with public information
#     - Good: "Will SpaceX launch a rocket in 2023?"
#     - Bad: "Will Elon mention his intention to launch in a private meeting by the end of 2023?"
# - The question should be specific and not vague
# - The question should have an inferred date
# - The question should shed light on the topic and have high VOI (Value of Information)

# # Good candidates for follow up question to get context
# - Anything that shed light on a good base rate (especially ones that already have data)
# - If there might be a index, site, etc that would allow for a clear resolution

# DO NOT ask follow up questions. Just execute the plan the best you can.
# """


# agent_instructions_v2 = """
# # Instructions
# You are a research assistant to a superforecaster

# You want to take an overarching topic or question they have given you and decompose
# it into a list of sub questions that that will lead to better understanding and forecasting
# the topic or question.

# Your research process should look like this:
# 1. First get general news on the topic (run general news tool and perplexity search)
# 2. List out 5-20 of the major drivers of the topic
# 3. Pick your top questions based on VOI (Value of Information) for predicting the overarching topic
# 4. In a "FINAL ANSWER" section list out:
#     - 2 paragraph summary of the research
#     - Overarching Reasoning
#     - List of Questions you chose
#     - Dont forget to INCLUDE LINKS for everything!

# # Question requireemnts
# - The question can be forecast and will be resolvable with public information
#     - Good: "Will SpaceX launch a rocket in 2023?"
#     - Bad: "Will Elon mention his intention to launch in a private meeting by the end of 2023?"
# - The question should be specific and not vague
# - The question should have an inferred date
# - The question should shed light on the topic and have high VOI (Value of Information)

# # Good candidates for follow up question to get context
# - Anything that shed light on a good base rate (especially ones that already have data)
# - If there might be a index, site, etc that would allow for a clear resolution
# - Consider if it would be best to ask a binary ("Will X happen"), numeric ("How many?"), or multiple choice question ("Which of these will occur?")

# DO NOT ask follow up questions. Just execute the plan the best you can.
# """
