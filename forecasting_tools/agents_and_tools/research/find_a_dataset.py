import asyncio
import logging

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.minor_tools import (
    perplexity_reasoning_pro_search,
)
from forecasting_tools.agents_and_tools.other.data_analyzer import DataAnalyzer
from forecasting_tools.agents_and_tools.research.computer_use import ComputerUse
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AiAgent,
    agent_tool,
    event_to_tool_message,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents

logger = logging.getLogger(__name__)


class DatasetFinderResult(BaseModel):
    steps: list[str]
    final_answer: str

    @property
    def as_string(self) -> str:
        steps_str = ""
        for step in self.steps:
            steps_str += f"{step}\n\n"
        return steps_str + "\n\n**Final Answer**\n" + self.final_answer + "\n\n"


class DatasetFinder:

    def __init__(self, llm: str = "openrouter/google/gemini-2.5-pro-preview") -> None:
        self.llm = llm

    async def find_and_analyze_data_set(self, query: str) -> DatasetFinderResult:
        instructions = clean_indents(
            f"""
            You are a data researcher helping to find and analyze datasets to answer questions.

            Your task is to:
            1. Use general Perplexity research to find what datasets might be available
                a. Run multiple times in parallel if searches are not dependent on each other
                b. Run up to 2 iterations of searches (using one iteration to inform the next)
            2. Use the computer use tool to find and download relevant datasets or analyze graphs/visuals that could help answer the question
                a. Run up to 2 in parallel to try to find multiple datasets (in case one doesn't work out)
                b. If it doesn't work in 2 tries, give up stop here, and explain why
            3. Use the data analyzer tool to analyze the downloaded datasets (if any), do math, and provide insights
                a. Come up with multiple analysis methods (ideally 3 and if its super simple just run the analysis 3 times)
                b. Run each approach in parallel
                c. Share the different results of the methods (and raise a concern if they deviate significantly)
            4. Give a 3 paragraph report of:
                a. What the dataset is, where you found it, its url
                b. Findings from any data analysis you did
                c. Findings from any graphs/visuals you analyzed

            Follow these guidelines:
            - Look for datasets on sites like Kaggle, data.gov, FRED, or other public data repositories
            - Make sure to download any relevant datasets you find
            - If you can't download the dataset, analyze any graphs available visibly
            - Analyze the data to provide insights that help answer the question
            - If you can't find relevant data, explain why and suggest alternative approaches
            - If you are trying to find base rates or historical trends consider:
                - Whether growth rate of a graph is more useful than actual values
                - Try to give quantiles whenever possible (10%, 25%, 50%, 75%, 90%) (e.g. The graph is above value X 10% of the time, value Y 25% of the time, etc.)
            - Consider exploring relationships between variables
            - Be generally proactive in trying different analyses that would be useful for the user.

            The question you need to help answer is below:
            {query}
            """
        )

        agent = AiAgent(
            name="Dataset Finder",
            instructions=instructions,
            model=AgentSdkLlm(model=self.llm),
            tools=[
                perplexity_reasoning_pro_search,
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

        steps = []
        async for event in result.stream_events():
            event_message = event_to_tool_message(event)
            if event_message:
                steps.append(event_message + "\n\n---\n\n")
                logger.info(event_message)
        final_answer = result.final_output
        return DatasetFinderResult(steps=steps, final_answer=final_answer)

    @agent_tool
    @staticmethod
    def find_a_dataset_tool(query: str) -> str:
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
        data_crawler = DatasetFinder()
        result = asyncio.run(data_crawler.find_and_analyze_data_set(query))
        return result.as_string
