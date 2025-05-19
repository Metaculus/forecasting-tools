import asyncio
from typing import Callable, overload

from agents import Agent, FunctionTool, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from agents.tool import ToolFunction


class AgentSdkLlm(LitellmModel):
    """
    Wrapper around openai-agent-sdk's LiteLlm Model for later extension
    """

    async def get_response(self, *args, **kwargs):  # NOSONAR
        response = await super().get_response(*args, **kwargs)
        await asyncio.sleep(
            0.0001
        )  # For whatever reason, it seems you need to await a coroutine to get the litellm cost callback to work
        return response


class AgentRunner(Runner):
    """
    Wrapper around OpenAI AgentSDK Runner in case changes are needed in the future
    """

    pass


AgentTool = FunctionTool  # Alias for FunctionTool for later extension


class AiAgent(Agent):
    """
    Wrapper around OpenAI AgentSDK Agent for later extension
    """

    pass


# def agent_tool(*args, **kwargs) -> AgentTool | Callable[[ToolFunction[...]], AgentTool]:
#     return function_tool(*args, **kwargs)


@overload
def agent_tool(func: ToolFunction[...], **kwargs) -> FunctionTool:
    """Overload for usage as @function_tool (no parentheses)."""
    ...


@overload
def agent_tool(**kwargs) -> Callable[[ToolFunction[...]], FunctionTool]:
    """Overload for usage as @function_tool(...)."""
    ...


def agent_tool(
    func: ToolFunction[...] | None = None, **kwargs
) -> AgentTool | Callable[[ToolFunction[...]], AgentTool]:
    if func is None:

        def decorator(f):
            return function_tool(f, **kwargs)

        return decorator
    return function_tool(func, **kwargs)
