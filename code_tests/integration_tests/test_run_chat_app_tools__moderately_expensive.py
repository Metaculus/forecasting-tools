import logging

import pytest

from forecasting_tools.agents_and_tools.research.find_a_dataset import DatasetFinder
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AgentTool,
    AiAgent,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.front_end.app_pages.chat_page import ChatPage
from forecasting_tools.helpers.metaculus_api import MetaculusApi

logger = logging.getLogger(__name__)


def get_tool_tests() -> list[tuple[str, AgentTool]]:
    tools = []
    for tool in ChatPage.get_chat_tools():
        tools.append((tool.name, tool))
    return tools


@pytest.mark.parametrize("name, function_tool", get_tool_tests())
async def test_chat_app_function_tools(name: str, function_tool: AgentTool) -> None:
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
        - For data analyzer tool, use the file ID "file-KCPeaFiP8Szp7PnhXVPFH5" and file name "bot_forecasts_q1.csv" and ask for number of binary questions
        - For computer use tool, ask it to download a csv from https://fred.stlouisfed.org/series/GDP and make sure it returns to you a download link and a OpenAI File ID.
        """
    )
    if function_tool == DatasetFinder.find_a_dataset_tool:
        pytest.skip("DatasetFinder is not supported in this test")
    llm = AgentSdkLlm(model="openrouter/openai/gpt-4.1")
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
