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
from dataclasses import dataclass

from agents import Agent, AgentOutputSchema, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher


@function_tool
async def get_general_news(topic: str) -> str:
    """
    Get general news context for a topic using AskNews.
    """
    return await AskNewsSearcher().get_formatted_news_async(topic)


@function_tool
async def perplexity_search(query: str) -> str:
    """
    Use Perplexity (sonar-reasoning-pro) to search for information on a topic.
    """
    llm = GeneralLlm(
        model="perplexity/sonar-reasoning-pro",
        reasoning_effort="high",
        web_search_options={"search_context_size": "high"},
    )
    return await llm.invoke(query)


agent_instructions = """
# Instructions
You are a research assistant to a superforecaster

You want to take an overarching topic or question they have given you and decompose
it into a list of sub questions that that will lead to better understanding and forecasting
the topic or question.

Your research process should look like this:
1. First get general news on the topic and run a perplexity search
2. Pick 3 ideas things to follow up with and search perplexity with these
3. Then brainstorm 2x the number of question requested number of key questions requested
4. Pick and submit your top questions

You only give your final result in json after you have gone through all the steps

# Question requireemnts
- The question can be forecast and will be resolvable with public information
    - Good: "Will SpaceX launch a rocket in 2023?"
    - Bad: "Will Elon mention his intention to launch in a private meeting by the end of 2023?"
- The question should be specific and not vague
- The question should have an inferred date
- The question should shed light on the topic

# Good candidates for follow up question to get context
- Anything that shed light on a good base rate (especially ones that already have data)
- If there might be a index, site, etc that would allow for a clear resolution

Whatever you do, DO NOT give ANY Json until you have gotten the results of the perplexity tool call.
DO NOT ask follow up questions. Just execute the plan the best you can.
"""
# # Example
# Lets say the question is "Will AI be a net negative for society?" and you are asked for 5 questions
# 1. Run a general news search and perplexity search
# 2. Pick 3 ideas things to follow up with and search perplexity with these
# 3. Brainstorm 10 questions
# 4. Pick your top 5 questions
# [
#     "..."
# ]


@dataclass
class DecompositionResult:
    reasoning: str
    research_summary: str
    questions: list[str]


class QuestionDecomposer:
    def __init__(
        self,
        model: str = "anthropic/claude-3-7-sonnet-latest",
    ) -> None:
        self.model = model

    async def decompose_into_questions(
        self,
        fuzzy_topic_or_question: str | None = None,
        number_of_questions: int = 5,
        additional_context: str | None = None,
    ) -> DecompositionResult:
        input_prompt = (
            f"Please write me {number_of_questions} forecasting questions."
            + (
                f" Topic: {fuzzy_topic_or_question}."
                if fuzzy_topic_or_question
                else ""
            )
            + (
                f" Additional context: {additional_context}."
                if additional_context
                else ""
            )
        )
        agent = Agent(
            name="Question Decomposer Agent",
            instructions=agent_instructions,
            tools=[get_general_news, perplexity_search],
            model=LitellmModel(model=self.model),
            output_type=AgentOutputSchema(DecompositionResult),
        )
        result = await Runner.run(agent, input_prompt)
        output = result.final_output_as(
            DecompositionResult, raise_if_incorrect_type=True
        )
        return output

    @function_tool
    @staticmethod
    def decompose_into_questions_tool(
        fuzzy_topic_or_question: str,
        number_of_questions: int,
        summary_of_chat: str,
    ) -> DecompositionResult:
        """
        Decompose a topic or question into a list of sub questions.

        Args:
            fuzzy_topic_or_question: The topic or question to decompose.
            number_of_questions: The number of questions to decompose the topic or question into. Default to 5.
            summary_of_chat: A summary of the relevant chat history.

        Returns:
            A list of sub questions.
        """
        return asyncio.run(
            QuestionDecomposer().decompose_into_questions(
                fuzzy_topic_or_question, number_of_questions, summary_of_chat
            )
        )
