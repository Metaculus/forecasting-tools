import asyncio
import logging

import pytest
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent

from code_tests.unit_tests.test_ai_models.models_to_test import (
    GeneralLlmInstancesToTest,
    ModelTest,
)
from forecasting_tools.ai_models.general_llm import AgentSdkLlm, GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "test_name, test", GeneralLlmInstancesToTest().all_tests_with_names()
)
def test_general_llm_instances_run(
    test_name: str,
    test: ModelTest,
) -> None:
    model = test.llm
    model_input = test.model_input
    with MonetaryCostManager(1) as cost_manager:
        response = asyncio.run(model.invoke(model_input))
        assert response is not None, "Response is None"
        assert response != "", "Response is an empty string"
        logger.info(f"Cost for {test_name}: {cost_manager.current_usage}")
        if not model.model.startswith("exa/"):
            assert cost_manager.current_usage > 0, "No cost was incurred"


def test_timeout_works() -> None:
    model = GeneralLlm(model="gpt-4o", timeout=0.1)
    model_input = "Hello, world!"
    with pytest.raises(Exception):
        asyncio.run(model.invoke(model_input))

    model = GeneralLlm(model="gpt-4o-mini", timeout=50)
    response = asyncio.run(model.invoke(model_input))
    assert response is not None, "Response is None"


def test_litellm_params_work() -> None:
    # Make sure it doesn't raise an exception
    GeneralLlm(
        model="gpt-4o",
        temperature=0.1,
        max_tokens=100,
        pass_through_unknown_kwargs=False,
    )

    # Make sure it raises an exception if a non-litellm param is passed
    with pytest.raises(Exception):
        GeneralLlm(
            model="gpt-4o",
            temperature=0.1,
            non_litellm_param=100,
            pass_through_unknown_kwargs=False,
        )

    # Make sure it doesn't raise an exception if a non-litellm param is passed
    GeneralLlm(
        model="gpt-4o",
        temperature=0.1,
        non_litellm_param=100,
        pass_through_unknown_kwargs=True,
    )


def test_citations_are_populated() -> None:
    model = GeneralLlm(
        model="openrouter/perplexity/sonar", populate_citations=True
    )
    response = asyncio.run(model.invoke("When did Abraham Lincoln die?"))
    logger.info(f"Response: {response}")
    assert response, "Response is empty"
    assert (
        "http:" in response or "www." in response
    ), "Citations are not populated"

    model = GeneralLlm(
        model="openrouter/perplexity/sonar", populate_citations=False
    )
    response = asyncio.run(model.invoke("When did Abraham Lincoln die?"))
    logger.info(f"Response: {response}")
    assert response, "Response is empty"
    assert (
        "http:" not in response and "www." not in response
    ), "Citations are populated"


async def test_agent_sdk_llm_works() -> None:
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=AgentSdkLlm(model="openrouter/openai/gpt-3.5-turbo"),
    )
    prompt = "Hello, world!"
    with MonetaryCostManager(1) as cost_manager:
        response = await Runner.run(agent, prompt)
        assert response is not None, "Response is None"
        assert hasattr(
            cost_manager, "current_usage"
        ), "Cost manager missing current_usage"
        assert cost_manager.current_usage > 0, "No cost was incurred"


async def test_streamted_agent_sdk_llm_works() -> None:
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=AgentSdkLlm(model="openrouter/openai/gpt-3.5-turbo"),
    )
    prompt = "Hello, world!"
    with MonetaryCostManager(1) as cost_manager:
        result = Runner.run_streamed(agent, prompt)
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
