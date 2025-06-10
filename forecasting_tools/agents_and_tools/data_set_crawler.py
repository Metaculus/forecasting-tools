import asyncio
import logging

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.computer_use import ComputerUse
from forecasting_tools.agents_and_tools.data_analyzer import DataAnalyzer
from forecasting_tools.agents_and_tools.misc_tools import perplexity_pro_search
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AiAgent,
    agent_tool,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents

logger = logging.getLogger(__name__)


class DataCrawlerResult(BaseModel):
    answer: str


class DataCrawler:

    def __init__(
        self, llm: str = "openrouter/gemini/gemini-2.5-pro-preview"
    ) -> None:
        self.llm = llm

    async def search_for_data_set(self, query: str) -> DataCrawlerResult:
        instructions = clean_indents(
            f"""
            You are a data researcher helping to find and analyze datasets to answer questions.

            Your task is to:
            1. Use general Perplexity research to find what datasets might be available
                a. Search multiple times in parallel if needed.
                b. Run up to 2 iterations of searches (using one iteration to inform the next)
            2. Use the computer use tool to find and download relevant datasets or analyze graphs/visuals that could help answer the question
            3. Use the data analyzer tool to analyze the downloaded datasets (if any), do math, and provide insights
            4. Give a 3 paragraph report of your findings
                a. What the dataset is, where you found it
                b. Data analysis you did
                c. Any graphs/visuals you analyzed

            The question you need to help answer is:
            {query}

            Follow these guidelines:
            - Look for datasets on sites like Kaggle, data.gov, FRED, or other public data repositories
            - Make sure to download any relevant datasets you find
            - If you can't download the dataset, analyze any graphs available visibly
            - Analyze the data to provide insights that help answer the question
            - If you can't find relevant data, explain why and suggest alternative approaches
            - If you are trying to find base rates or historical trends consider:
                - Whether growth rate of a graph is more useful than actual values
                - Try to give quantiles whenever possible (10%, 25%, 50%, 75%, 90%) (e.g. The graph is above value X 10% of the time, value Y 25% of the time, etc.)
            """
        )

        agent = AiAgent(
            name="Data Set Crawler",
            instructions=instructions,
            model=self.llm,
            tools=[
                perplexity_pro_search,
                ComputerUse.computer_use_tool,
                DataAnalyzer.data_analysis_tool,
            ],
            handoffs=[],
        )

        result = AgentRunner.run_streamed(
            agent,
            "Please Follow your instructions.",
            max_turns=10,
        )

        final_answer = ""
        async for event in result.stream_events():
            if event.type == "run_item_stream_event":
                event_message = f"{event.item.type}: {event.item.raw_item}\n"
                final_answer += event_message
                logger.info(event_message)

        final_answer += f"\n\nFinal output: {result.final_output}"
        return DataCrawlerResult(answer=final_answer)

    @agent_tool
    @staticmethod
    def data_set_crawler_tool(query: str) -> str:
        """
        This tool helps find and analyze datasets relevant to a question.
        It will:
        1. Search for and download relevant datasets or analyze graphs/visuals that could help answer the question
        2. Analyze the datasets (if any) to help answer the question

        The tool is best used for questions that would benefit from data analysis and for which there is probably a dataset out there for,such as:
        - Questions about trends over time or historical trends
        - Questions requiring statistical analysis
        - Questions about relationships between variables
        - Questions that can be answered with numerical data or graphs/visuals
        """
        data_crawler = DataCrawler()
        result = asyncio.run(data_crawler.search_for_data_set(query))
        return result.answer
