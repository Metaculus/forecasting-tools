import random

from agents import Tool, function_tool

from forecasting_tools.agents_and_tools.question_generators.topic_generator import (
    TopicGenerator,
)


@function_tool
async def generate_random_number(min_num: int, max_num: int) -> int:
    """
    Generate a random number between min_num and max_num (default to 1 and 100).
    """
    return random.randint(min_num, max_num)


@function_tool
async def generate_random_topics(num_topics: int = 10) -> str:
    """
    Generate a list of random topics to help come up with ideas for questions to forecast.
    """
    topics = await TopicGenerator.generate_random_news_items(
        number_of_items=num_topics
    )
    topic_list = ""
    for topic in topics:
        topic_list += f"- {topic}\n"
    return topic_list


def get_tools_for_chat_app() -> list[Tool]:
    return [generate_random_number, generate_random_topics]
