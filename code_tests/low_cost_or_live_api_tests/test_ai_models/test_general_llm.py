import asyncio

import pytest

from code_tests.unit_tests.test_ai_models.models_to_test import (
    GeneralLlmInstancesToTest,
    ModelTest,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm


@pytest.mark.parametrize(
    "test_name, test", GeneralLlmInstancesToTest().all_tests_with_names()
)
def test_general_llm_instances_run(
    test_name: str,
    test: ModelTest,
) -> None:
    model = test.llm
    model_input = test.model_input
    response = asyncio.run(model.invoke(model_input))
    assert response is not None, "Response is None"


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
    GeneralLlm(model="gpt-4o", temperature=0.1, max_tokens=100)

    # Make sure it raises an exception if a non-litellm param is passed
    with pytest.raises(Exception):
        GeneralLlm(model="gpt-4o", temperature=0.1, non_litellm_param=100)
