import asyncio

import nest_asyncio
from agents import (
    Agent,
    CodeInterpreterTool,
    FunctionTool,
    Runner,
    function_tool,
)
from agents.extensions.models.litellm_model import LitellmModel
from agents.stream_events import StreamEvent

from forecasting_tools.ai_models.model_tracker import ModelTracker

nest_asyncio.apply()


class AgentSdkLlm(LitellmModel):
    """
    Wrapper around openai-agent-sdk's LiteLlm Model for later extension
    """

    async def get_response(self, *args, **kwargs):  # NOSONAR
        ModelTracker.give_cost_tracking_warning_if_needed(self.model)
        response = await super().get_response(*args, **kwargs)
        await asyncio.sleep(
            0.0001
        )  # For whatever reason, it seems you need to await a coroutine to get the litellm cost callback to work
        return response


AgentRunner = Runner  # Alias for Runner for later extension
AgentTool = FunctionTool  # Alias for FunctionTool for later extension
AiAgent = Agent  # Alias for Agent for later extension
CodingTool = (
    CodeInterpreterTool  # Alias for CodeInterpreterTool for later extension
)
agent_tool = function_tool  # Alias for function_tool for later extension


def event_to_tool_message(event: StreamEvent) -> str | None:
    text = ""
    if event.type == "run_item_stream_event":
        item = event.item
        if item.type == "message_output_item":
            content = item.raw_item.content[0]
            if content.type == "output_text":
                # text = content.text
                text = ""  # the text is already streamed separate from this function
            elif content.type == "output_refusal":
                text = content.refusal
            else:
                text = "Error: unknown content type"
        elif item.type == "tool_call_item":
            if item.raw_item.type == "code_interpreter_call":
                text = f"\nCode interpreter code:\n```python\n{item.raw_item.code}\n```\n"
            else:
                tool_name = getattr(item.raw_item, "name", "unknown_tool")
                tool_args = getattr(item.raw_item, "arguments", {})
                text = f"Tool call: {tool_name}({tool_args})"
        elif item.type == "tool_call_output_item":
            output = getattr(item, "output", str(item.raw_item))
            text = f"Tool output:\n\n{output}"
        elif item.type == "handoff_call_item":
            handoff_info = getattr(item.raw_item, "name", "handoff")
            text = f"Handoff call: {handoff_info}"
        elif item.type == "handoff_output_item":
            text = f"Handoff output: {str(item.raw_item)}"
        elif item.type == "reasoning_item":
            text = f"Reasoning: {str(item.raw_item)}"
    # elif event.type == "agent_updated_stream_event":
    #     text += f"Agent updated: {event.new_agent.name}\n\n"
    if text == "":
        return None
    return text
