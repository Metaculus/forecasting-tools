import io
import logging

import pytest

from forecasting_tools.agents_and_tools.other.data_analyzer import DataAnalyzer
from forecasting_tools.agents_and_tools.other.hosted_file import (
    FileToUpload,
    HostedFile,
)
from forecasting_tools.agents_and_tools.research.find_a_dataset import DatasetFinder
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AgentTool,
    AiAgent,
)
from forecasting_tools.front_end.app_pages.chat_page import ChatPage
from forecasting_tools.helpers.metaculus_api import MetaculusApi
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)


_DATA_ANALYZER_TEST_CSV = (
    "question_text,question_type,resolution\n"
    "Will it rain tomorrow?,binary,YES\n"
    "Will the S&P 500 close above 5000?,binary,NO\n"
    "How many countries?,numeric,42\n"
    "Will AI pass the Turing test in 2030?,binary,NO\n"
    "Will renewables exceed 50%?,binary,YES\n"
)


def get_tool_tests() -> list[tuple[str, AgentTool]]:
    tools = []
    for tool in ChatPage.get_chat_tools():
        tools.append((tool.name, tool))
    return tools


def _upload_data_analyzer_test_file() -> HostedFile:
    file_buffer = io.BytesIO(_DATA_ANALYZER_TEST_CSV.encode("utf-8"))
    file_buffer.name = "bot_forecasts_q1.csv"
    [hosted] = HostedFile.upload_files_to_openai(
        [FileToUpload(file_data=file_buffer, file_name="bot_forecasts_q1.csv")]
    )
    return hosted


@pytest.mark.parametrize("name, function_tool", get_tool_tests())
async def test_chat_app_function_tools(name: str, function_tool: AgentTool) -> None:
    if function_tool == DatasetFinder.find_a_dataset_tool:
        pytest.skip("DatasetFinder is not supported in this test")

    data_analyzer_instructions = ""
    if function_tool == DataAnalyzer.data_analysis_tool:
        hosted = _upload_data_analyzer_test_file()
        data_analyzer_instructions = (
            f'- For data analyzer tool, use the file ID "{hosted.file_id}" '
            f'and file name "{hosted.file_name}" and ask for the number of '
            f"binary questions."
        )

    instructions = clean_indents(
        f"""
        You are a software engineer testing a piece of code.
        You are being given a tool you have access to.
        Please make up some inputs to the tool, and then run the tool with the inputs.
        Check whether the results of the tool match its description.

        If the results make sense generally, and there are no errors say: "<TOOL SUCCESSFULLY TESTED>"
        If there are errors, or the results indicate that the tool does something very different than expected say: "<TOOL FAILED TEST>" and then state the error/output verbatim then explain why the output is not right.

        Here is what to do for some specific tools:
        - For metaculus question tools, use the question ID 37328 and tournament slug '{MetaculusApi.CURRENT_METACULUS_CUP_ID}'
        {data_analyzer_instructions}
        - For computer use tool, ask it to download a csv from https://fred.stlouisfed.org/series/GDP and make sure it returns to you a download link and a OpenAI File ID.
        """
    )
    llm = AgentSdkLlm(model="openrouter/anthropic/claude-sonnet-4.6")
    agent = AiAgent(
        name="Tool Test Agent",
        instructions=instructions,
        model=llm,
        tools=[function_tool],
    )
    result = await AgentRunner.run(agent, "Please test the tool")
    final_answer = result.final_output
    logger.info(f"Full result: {result}")
    logger.info(f"Raw responses: {result.raw_responses}")
    if "<TOOL SUCCESSFULLY TESTED>" in final_answer:
        assert True  # NOSONAR
    elif "<TOOL FAILED TEST>" in final_answer:
        assert False, f"Tool failed to test. The LLM says: {final_answer}"
    else:
        assert (
            False
        ), f"Tool did not return a valid response. The LLM says: {final_answer}"


# TODO: Test the below:
# - File upload works in chat app
# - Data analyzer works with pdf
# - Ability to use and upload multiple files
# - Files are not uploaded again if they are already uploaded
