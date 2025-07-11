import asyncio
import logging

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.minor_tools import (
    perplexity_quick_search_high_context,
)
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AiAgent,
    CodingTool,
    agent_tool,
    event_to_tool_message,
    general_trace_or_span,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents

logger = logging.getLogger(__name__)


class AvailableFile(BaseModel):
    file_name: str
    file_id: str


class DataAnalyzer:

    def __init__(self, model: str = "gpt-4.1") -> None:
        self.model = model

    async def run_data_analysis(
        self,
        instructions: str,
        additional_context: str | None = None,
        available_files: list[AvailableFile] | None = None,
    ) -> str:
        logger.warning("Cost tracking not supported for Data Analysis Agent")
        # NOTE: See example usage here: https://github.com/openai/openai-agents-python/blob/main/examples/tools/code_interpreter.py
        if not available_files:
            available_files = []
        available_files_context = "\n".join(
            [
                f"- File Name: {file.file_name} | File ID: {file.file_id}"
                for file in available_files
            ]
        )

        instructions = clean_indents(
            f"""
            You are a data analyst who uses code to solve problems.

            **You have been given the following instructions:**
            {instructions}

            **You have been given the following additional context:**
            {additional_context}

            **You have access to the following files:**
            {available_files_context}
            """
        )

        coding_tool = CodingTool(
            tool_config={
                "type": "code_interpreter",
                "container": {
                    "type": "auto",
                    "file_ids": [file.file_id for file in available_files],
                },
            }
        )

        agent = AiAgent(
            name="Data Analyzer",
            instructions=instructions,
            model=self.model,
            tools=[
                perplexity_quick_search_high_context,
                coding_tool,
            ],
            handoffs=[],
        )

        result = AgentRunner.run_streamed(
            agent, "Please follow your instructions.", max_turns=10
        )

        final_answer = ""
        async for event in result.stream_events():
            event_message = event_to_tool_message(event)
            if event_message:
                final_answer += event_message + "\n\n"
        final_answer += f"Final output: {result.final_output}"
        return final_answer

    @agent_tool
    @staticmethod
    def data_analysis_tool(
        instruction: str,
        additional_context: str | None = None,
        files: list | None = None,
    ) -> str:
        """
        This tool attempts to use code to achieve the user's instructions.
        Avoid giving it code when possible, just give step by step instructions (or a general goal).
        Can run analysis on files.
        Additional context should include any other constraints or requests from the user, and as much other information that is relevant to the task as possible.

        Format files as a list of dicts with the following format: {"file_name": "string", "file_id": "string"}
        """
        data_analysis = DataAnalyzer()
        available_files = [AvailableFile(**file) for file in files] if files else []
        return asyncio.run(
            data_analysis.run_data_analysis(
                instruction, additional_context, available_files
            )
        )


if __name__ == "__main__":
    with general_trace_or_span("Test Span 1"):
        answer = asyncio.run(
            DataAnalyzer().run_data_analysis(
                instructions="Please multiple 52.675 x 6.547 x 9867.5476 x 4356.5",
            )
        )
    print(answer)
