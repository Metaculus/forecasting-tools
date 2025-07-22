import logging

import pytest
from openai.types.responses import ResponseTextDeltaEvent

from forecasting_tools.ai_models.agent_wrappers import AgentRunner, AgentSdkLlm, AiAgent
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("model", ["openrouter/openai/gpt-4o", "gpt-4.1"])
async def test_agent_sdk_llm_works(model: str) -> None:
    agent = AiAgent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=AgentSdkLlm(model=model),
    )
    prompt = "Hello, world!"
    with MonetaryCostManager(1) as cost_manager:
        response = await AgentRunner.run(agent, prompt)
        assert response is not None, "Response is None"
        assert cost_manager.current_usage > 0, "No cost was incurred"


@pytest.mark.parametrize("model", ["openrouter/openai/gpt-4o", "gpt-4.1"])
async def test_streamed_agent_sdk_llm_works(model: str) -> None:
    agent = AiAgent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=AgentSdkLlm(model=model),
    )
    prompt = "Hello, world!"
    with MonetaryCostManager(1) as cost_manager:
        result = AgentRunner.run_streamed(agent, prompt)
        streamed_text = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                streamed_text += event.data.delta
        logger.info(f"Cost: {cost_manager.current_usage}")
        logger.info(f"Streamed text: {streamed_text}")
        assert cost_manager.current_usage > 0, "No cost was incurred"
        assert streamed_text, "Streamed text is empty"
