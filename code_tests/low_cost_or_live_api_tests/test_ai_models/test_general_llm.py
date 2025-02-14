import asyncio

import pytest

from code_tests.unit_tests.test_ai_models.models_to_test import (
    GeneralLlmInstancesToTest,
    ModelTest,
)


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
