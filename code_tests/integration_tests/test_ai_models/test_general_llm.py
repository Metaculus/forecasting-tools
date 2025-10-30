import asyncio
import logging

import pytest

from code_tests.unit_tests.test_ai_models.models_to_test import LLMTestData, ModelTest
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

logger = logging.getLogger(__name__)


def _all_tests() -> list[ModelTest]:
    test_data = LLMTestData()
    return [
        ModelTest(GeneralLlm(model="gpt-4o"), test_data.get_cheap_user_message()),
        ModelTest(
            GeneralLlm(model="gpt-4o"),
            test_data.get_cheap_vision_message_data(),
        ),
        ModelTest(
            GeneralLlm(model="openai/gpt-4o"),
            [{"role": "user", "content": test_data.get_cheap_user_message()}],
        ),
        ModelTest(
            GeneralLlm(model="o3-mini", reasoning_effort="low"),
            test_data.get_cheap_user_message(),
        ),
        ModelTest(
            GeneralLlm(model="metaculus/gpt-4o"),
            test_data.get_cheap_user_message(),
        ),
        ModelTest(
            GeneralLlm(model="claude-3-7-sonnet-latest"),
            test_data.get_cheap_user_message(),
        ),
        ModelTest(
            GeneralLlm(model="claude-3-7-sonnet-latest"),
            test_data.get_cheap_vision_message_data(),
        ),
        ModelTest(
            GeneralLlm(model="perplexity/sonar-pro"),
            test_data.get_cheap_user_message(),
        ),
        ModelTest(
            GeneralLlm(model="openrouter/openai/gpt-4o"),
            test_data.get_cheap_user_message(),
        ),
        ModelTest(
            GeneralLlm(model="exa/exa-pro"),
            test_data.get_cheap_user_message(),
        ),
        ModelTest(
            GeneralLlm(model="exa/exa"),
            test_data.get_cheap_user_message(),
        ),
        ModelTest(
            GeneralLlm(model="asknews/news-summaries"),
            test_data.get_cheap_user_message(),
        ),
        ModelTest(
            GeneralLlm(
                model="openai/gpt-5",
                responses_api=True,
                tools=[{"type": "web_search"}],
                reasoning_effort="minimal",
            ),
            "What is the latest News on the Middle East? Do a single very quick search. Go as fast as you can. I just want headlines.",
        ),
    ]


def all_tests_with_names() -> list[tuple[str, ModelTest]]:
    tests = _all_tests()
    pairs = []
    for test in tests:
        input_type = type(test.model_input)
        pairs.append((f"{test.llm.model}-{input_type.__name__}", test))
    return pairs


@pytest.mark.parametrize("test_name, test", all_tests_with_names())
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
        if not model.model.startswith("exa/") and not model.model.startswith(
            "asknews/"
        ):
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
    model = GeneralLlm(model="openrouter/perplexity/sonar", populate_citations=True)
    response = asyncio.run(model.invoke("When did Abraham Lincoln die?"))
    logger.info(f"Response: {response}")
    assert response, "Response is empty"
    assert "http" in response or "www." in response, "Citations are not populated"

    model = GeneralLlm(model="openrouter/perplexity/sonar", populate_citations=False)
    response = asyncio.run(model.invoke("When did Abraham Lincoln die?"))
    logger.info(f"Response: {response}")
    assert response, "Response is empty"
    assert "http" not in response and "www." not in response, "Citations are populated"


async def test_exa_errors_with_prompt_too_long() -> None:
    model = GeneralLlm(model="exa/exa-pro")
    prompt = """
        What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability?
        What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability?
        What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability?
        What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability?
        What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability?
        What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability?
        What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability?
        What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability?
        What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability?
        What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability?
        What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability?
        What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability?
        What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability? What are the key factors that led to the rise and fall of the Roman Empire? Please analyze the political, economic, social, and military aspects thatcontributed to its expansion and eventual decline.Consider the role of leadership, technological innovations,cultural integration, and external pressures.How did the empire's vast size and diverse population impact its governance and stability?
        """
    assert len(prompt) > 10_000, "The prompt must be longer than 10k tokens to error"
    with pytest.raises(Exception):
        await model.invoke(prompt)
