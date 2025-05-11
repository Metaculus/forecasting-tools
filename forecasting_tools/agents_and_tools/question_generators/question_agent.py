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
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from agents import Agent, AgentOutputSchema, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from pydantic import Field

from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.agents_and_tools.question_generators.topic_generator import (
    TopicGenerator,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher


@dataclass
class BasicQuestion:
    question_text: str
    resolution_criteria: str
    fine_print: str
    expected_resolution_date: datetime
    question_type: Literal["binary", "numeric", "multiple_choice"] = "binary"
    multiple_choice_options: list[str] = Field(default_factory=list)


@function_tool
def generate_topics(
    topic_hint: str | None = None, number_of_topics: int = 5
) -> list[str]:
    """
    Generate a list of forecasting topics. If topic_hint is provided, use it as a direction; otherwise, generate random topics.
    """
    if topic_hint:
        return [topic_hint]
    return asyncio.run(
        TopicGenerator.generate_random_topic(number_of_topics=number_of_topics)
    )


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


agent_instructions = f"""
You are a Metaculus Question Agent. Your job is to write clear, unambiguous, and interesting forecasting questions for Metaculus, following this process:

1. If the user provides a topic or direction, use it. Otherwise, use the generate_topics tool to create topics.
2. For each topic, decide if you need more context:
    - If the topic is news-related or recent, use get_general_news to gather background.
    - If the topic is technical, ambiguous, or requires deeper research, use perplexity_search.
    - Otherwise, use your own knowledge.
3. For each topic, write a question that sheds light on the future, following these guidelines:
    - The question should be about a future event or outcome.
    - The resolution criteria must be specific and unambiguous.
    - Fine print should cover all edge cases and ensure the question passes the clairvoyance test (no ambiguity after the event resolves).
    - Provide relevant background information.
    - Set an expected resolution date (use a reasonable future date if not specified).
    - Choose the question type: binary (yes/no), numeric, or multiple_choice. If numeric, specify bounds and if they are open/closed. If multiple_choice, provide options.
4. Return a list of SimpleQuestion objects, one for each topic.

{SimpleQuestion.get_field_descriptions()}

If the user asks you to do only a specific step, please do that.
"""


class BaseAgent(ABC):
    def __init__(self) -> None:
        self._agent = None

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            raise NotImplementedError(
                "Subclasses must implement a _agent field"
            )
        return self._agent

    @property
    def readable_agent(self) -> Agent:
        readable_agent = Agent(
            name=f"{self.__class__.__name__} Markdownifier",
            instructions=clean_indents(
                f"""
                You are the secretary of a {self.agent.name} and take json outputthey give and turn it into readable markdown.
                You include ALL the information they give, with no additional information.
                However you make it readable and easy-to-understand Markdown
                """
            ),
            tools=[
                self.agent.as_tool(
                    tool_name=f"run_{self.agent.name.lower()}",
                    tool_description="You must call this tool in order to make it markdown",
                )
            ],
            model=LitellmModel(model="gpt-4o-mini"),
            output_type=AgentOutputSchema(str),
        )
        return readable_agent


class QuestionCreationAgent(BaseAgent):
    def __init__(
        self,
        model: str = "anthropic/claude-3-7-sonnet-latest",
    ) -> None:
        self._agent = Agent(
            name="Metaculus Question Agent",
            instructions=agent_instructions,
            tools=[generate_topics, get_general_news, perplexity_search],
            model=LitellmModel(model=model),
            output_type=AgentOutputSchema(list[BasicQuestion]),
        )

    async def generate_questions(
        self, topic_hint: str | None = None, number_of_questions: int = 5
    ) -> list[BasicQuestion]:
        input_prompt = (
            f"Write {number_of_questions} forecasting questions."
            + (f" Topic: {topic_hint}." if topic_hint else "")
            + " Return a list of Questions objects."
        )
        result = await Runner.run(self.agent, input_prompt)
        return result.final_output
