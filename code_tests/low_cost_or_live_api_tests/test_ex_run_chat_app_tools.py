import pytest

from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AgentTool,
    AiAgent,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.front_end.app_pages.chat_page import ChatPage


def get_tool_tests() -> list[tuple[str, AgentTool]]:
    tools = []
    for tool in ChatPage.get_chat_tools():
        tools.append((tool.name, tool))
    return tools


@pytest.mark.parametrize("name, function_tool", get_tool_tests())
async def test_chat_app_function_tools(
    name: str, function_tool: AgentTool
) -> None:
    instructions = clean_indents(
        """
        You are a software engineer testing a piece of code.
        You are being given a tool you have access to.
        Please make up some inputs to the tool, and then run the tool with the inputs.
        Check whether the results of the tool match its description.

        If the results make sense generally, and there are no errors say: "<TOOL SUCCESSFULLY TESTED>"
        If there are errors, or the results indicate that the tool does something very different than expected say: "<TOOL FAILED TEST>" and then state the error/output verbatim then explain why the output is not right.

        For metaculus question tools, use the question ID 37328 and tournament slug "metaculus-cup"
        """
    )
    llm = AgentSdkLlm(model="openrouter/openai/gpt-4.1")
    agent = AiAgent(
        name="Test Agent",
        instructions=instructions,
        model=llm,
        tools=[function_tool],
    )
    result = await AgentRunner.run(agent, "Please test the tool")
    final_answer = result.final_output
    if "<TOOL SUCCESSFULLY TESTED>" in final_answer:
        assert True  # NOSONAR
    elif "<TOOL FAILED TEST>" in final_answer:
        assert False, f"Tool failed to test. The LLM says: {final_answer}"
    else:
        assert (
            False
        ), f"Tool did not return a valid response. The LLM says: {final_answer}"
