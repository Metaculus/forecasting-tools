from agents import Agent, Handoff, Tool, function_tool

from forecasting_tools.agents_and_tools.question_generators.question_agent import (
    QuestionCreationAgent,
)
from forecasting_tools.agents_and_tools.question_generators.topic_generator import (
    TopicGenerator,
)


@function_tool
async def generate_random_topics() -> str:
    """
    Generate a list of random topics to help come up with ideas for questions to forecast.
    Output: List of topics that include links to the source.
    """
    topics = await TopicGenerator.generate_random_news_items(
        number_of_items=10
    )
    topic_list = ""
    for topic in topics:
        topic_list += f"- {topic}\n"
    return topic_list


def get_agents_for_chat_app() -> list[Agent | Handoff]:
    return [QuestionCreationAgent().readable_agent]


def get_tools_for_chat_app() -> list[Tool]:
    return []  # generate_random_topics]
