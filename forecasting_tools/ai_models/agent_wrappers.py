import asyncio
import logging
import time
from typing import Any

import nest_asyncio
from agents import (
    Agent,
    CodeInterpreterTool,
    FunctionTool,
    Runner,
    Span,
    Trace,
    custom_span,
)
from agents import function_tool as ft
from agents import generation_span, trace
from agents.extensions.models.litellm_model import LitellmModel
from agents.stream_events import StreamEvent
from agents.tracing.setup import GLOBAL_TRACE_PROVIDER
from agents.tracing.span_data import CustomSpanData, GenerationSpanData
from agents.tracing.traces import TraceImpl

from forecasting_tools.ai_models.model_tracker import ModelTracker

logger = logging.getLogger(__name__)


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


class NoOpContextManager:
    """A context manager that does nothing when used in with statements"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    @property
    def trace_id(self) -> str:
        return "no-op-trace-id"


def _current_trace_exists() -> bool:
    try:
        current_trace = GLOBAL_TRACE_PROVIDER.get_current_trace()
    except Exception as e:
        logger.warning(f"Error getting current trace: {e}")
        current_trace = None
    return current_trace is not None


def general_trace_or_span(
    name: str, data: dict[str, Any] | None = None, **kwargs
) -> Span[CustomSpanData] | Trace:
    if _current_trace_exists():
        return custom_span(name, data, **kwargs)
    else:
        return trace(workflow_name=name, metadata=data, **kwargs)


def track_generation(*args, **kwargs) -> Span[GenerationSpanData]:
    if _current_trace_exists():
        return generation_span(*args, **kwargs)
    else:
        with trace("One Off Generation"):
            return generation_span(*args, **kwargs)


AgentRunner = Runner  # Alias for Runner for later extension
AgentTool = FunctionTool  # Alias for FunctionTool for later extension
AiAgent = Agent  # Alias for Agent for later extension
CodingTool = CodeInterpreterTool  # Alias for CodeInterpreterTool for later extension
agent_tool = ft  # Alias for function_tool for later extension
ImplementedTrace = TraceImpl  # Alias for TraceImpl for later extension


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
                text = (
                    f"\nCode interpreter code:\n```python\n{item.raw_item.code}\n```\n"
                )
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


if __name__ == "__main__":
    # Test tracing/spans. See https://platform.openai.com/traces for visual confirmation
    with trace("Test Trace A"):
        with track_generation(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": "Test Input"}],
        ):
            time.sleep(1)
        with track_generation(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": "Test Input"}],
        ):
            time.sleep(1)
            with track_generation(
                model="gpt-4o-mini",
                input=[{"role": "user", "content": "Test Input"}],
            ):
                time.sleep(1)
        with custom_span("Test Span 2"):
            time.sleep(1)

    with custom_span("Test Span 3"):
        with track_generation(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": "Test Input"}],
        ):
            time.sleep(1)

    with general_trace_or_span("Test Trace B"):
        with track_generation(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": "Test Input"}],
        ):
            time.sleep(1)
        with track_generation(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": "Test Input"}],
        ):
            time.sleep(1)
            with track_generation(
                model="gpt-4o-mini",
                input=[{"role": "user", "content": "Test Input"}],
            ):
                time.sleep(1)
                with general_trace_or_span("Test Span 2"):
                    time.sleep(1)

    with trace("Test Trace C"):
        with general_trace_or_span("Test Span 1"):
            with track_generation(
                model="gpt-4o-mini",
                input=[{"role": "user", "content": "Test Input"}],
            ):
                time.sleep(1)
            with track_generation(
                model="gpt-4o-mini",
                input=[{"role": "user", "content": "Test Input"}],
            ):
                time.sleep(1)
                with track_generation(
                    model="gpt-4o-mini",
                    input=[{"role": "user", "content": "Test Input"}],
                ):
                    time.sleep(1)
