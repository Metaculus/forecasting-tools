import asyncio
import logging
from typing import Any, Coroutine
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from code_tests.unit_tests.test_ai_models.ai_mock_manager import AiModelMockManager
from code_tests.unit_tests.test_ai_models.models_to_test import ModelsToTest
from forecasting_tools.ai_models.ai_utils.ai_misc import validate_complex_type
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse
from forecasting_tools.ai_models.deprecated_model_classes.gpt4o import Gpt4o
from forecasting_tools.ai_models.model_interfaces.ai_model import AiModel
from forecasting_tools.ai_models.model_interfaces.outputs_text import OutputsText

logger = logging.getLogger(__name__)

OUTPUTS_TEXT_ERROR_MESSAGE = "Model must be OutputsText"


@pytest.mark.parametrize("subclass", ModelsToTest.OUTPUTS_TEXT)
def test_errors_if_does_not_return_expected_values(
    mocker: Mock, subclass: type[AiModel]
) -> None:
    if not issubclass(subclass, OutputsText):
        raise ValueError(OUTPUTS_TEXT_ERROR_MESSAGE)

    AiModelMockManager.mock_ai_model_direct_call_with_predefined_mock_value(
        mocker, subclass
    )  # We assume the default mock value does not ask for only a number as an answer
    ai_model = subclass()
    cheap_input = ai_model._get_cheap_input_for_invoke()
    with pytest.raises(Exception):
        asyncio.run(
            ai_model.invoke_and_unsafely_run_and_return_generated_code(
                cheap_input, int, 1
            )
        )


class PydanticModelExample(BaseModel):
    int_value: int
    float_value: float


class SubPydanticModel2(BaseModel):
    str_value: str
    list_value: list[int]


class PydanticModelExample2(BaseModel):
    str_value: str
    list_value: list[int]
    sub_model: SubPydanticModel2
    sub_model_list: list[SubPydanticModel2]


instance_of_test_model_1 = PydanticModelExample(int_value=1, float_value=1.0)
sub_model_instance_1 = SubPydanticModel2(str_value="hello", list_value=[1, 2, 3])
instance_of_test_model_2 = PydanticModelExample2(
    str_value="hello",
    list_value=[1, 2, 3],
    sub_model=sub_model_instance_1,
    sub_model_list=[sub_model_instance_1, sub_model_instance_1],
)


@pytest.mark.parametrize(
    ("type_to_return", "mock_value_for_invoke", "expected_output"),
    [
        (int, "1", 1),
        (float, "1.0", 1.0),
        (str, '"string"', "string"),
        (list, "[1,2,3]", [1, 2, 3]),
        (dict, '{"key":"value"}', {"key": "value"}),
        (bool, "True", True),
        (tuple[int, int], "(1,2)", (1, 2)),
        (list[int], "[1,2,3]", [1, 2, 3]),
        (list[bool], "[True,False]", [True, False]),
        (list[str], '["string1","string2"]', ["string1", "string2"]),
        (list[str], '["string1",\n"string2"]', ["string1", "string2"]),
        (list[str], '[\n"string1",\n"string2"\n]', ["string1", "string2"]),
        (list[str], '["string1","string2",]', ["string1", "string2"]),
        (list[str], '[\n"string1",\n"string2",\n]', ["string1", "string2"]),
        (list[str], "['string1','string2']", ["string1", "string2"]),
        (list[str], "['string1',\n'string2']", ["string1", "string2"]),
        (list[str], "[\n'string1',\n'string2'\n]", ["string1", "string2"]),
        (list[str], "['string1','string2',]", ["string1", "string2"]),
        (list[str], "[\n'string1',\n'string2',\n]", ["string1", "string2"]),
        (list[str], "[]", []),
        (list[str], '[""]', []),
        (list[str], "['']", []),
        (list[int], "[]", []),
        (list[dict], '[{"key":"value"}]', [{"key": "value"}]),
        (
            PydanticModelExample,
            '{"int_value":1,"float_value":1.0,"list_value":[1,2,3]}',
            instance_of_test_model_1,
        ),
        (
            PydanticModelExample2,
            '{"str_value":"hello","list_value":[1,2,3],"sub_model":{"str_value":"hello","list_value":[1,2,3]},"sub_model_list":[{"str_value":"hello","list_value":[1,2,3]},{"str_value":"hello","list_value":[1,2,3]}]}',
            instance_of_test_model_2,
        ),
        (
            list[PydanticModelExample],
            '[{"int_value":1,"float_value":1.0,"list_value":[1,2,3]},{"int_value":1,"float_value":1.0,"list_value":[1,2,3]}]',
            [instance_of_test_model_1, instance_of_test_model_1],
        ),
        (list[str], '```["string1","string2"]```', ["string1", "string2"]),
        (
            list[str],
            '```json\n["string1","string2"]```',
            ["string1", "string2"],
        ),
        (
            list[str],
            '```python["string1","string2"]```',
            ["string1", "string2"],
        ),
        (
            list[str],
            '```markdown\n["string1","string2"]\n```',
            ["string1", "string2"],
        ),
        (
            list[str],
            'Here is a list of strings \n["string1","string2"]\n',
            ["string1", "string2"],
        ),
        (
            list[str],
            'Here is a list of strings ["string1","string2"]\n',
            ["string1", "string2"],
        ),
        (
            list[dict],
            'Here is a list of dictionaries [{"key":"value"},{"key":"value"}]\n',
            [{"key": "value"}, {"key": "value"}],
        ),
        (dict, 'Here is a dictionary {"key":"value"}', {"key": "value"}),
        (
            PydanticModelExample,
            'Here is a pydantic model: {"int_value":1,"float_value":1.0,"list_value":[1,2,3]}',
            instance_of_test_model_1,
        ),
        (list[tuple[int, int]], "[(1,2),(3,4)]", [(1, 2), (3, 4)]),
        (
            list[list[list[str]]],
            '[[["string1","string2"]]]',
            [[["string1", "string2"]]],
        ),
        (
            list[dict[str, list[int]]],
            '[{"key":[1,2,3]},{"key":[1,2,3]}]',
            [{"key": [1, 2, 3]}, {"key": [1, 2, 3]}],
        ),
        (int | float, "1", 1),
        (int | float, "1.0", 1.0),
    ],
)
def test_type_verification_works_for_valid_types(
    mocker: Mock,
    type_to_return: type,
    mock_value_for_invoke: str,
    expected_output: Any,
) -> None:
    mock_the_value_output_of_invoke(mocker, Gpt4o, mock_value_for_invoke)
    ai_model = Gpt4o()
    cheap_input = ai_model._get_cheap_input_for_invoke()
    coroutine = ai_model.invoke_and_return_verified_type(
        cheap_input, type_to_return, allowed_invoke_tries_for_failed_output=1
    )
    assert_output_is_of_expected_type_and_value(
        coroutine, type_to_return, expected_output
    )


@pytest.mark.parametrize(
    ("type_to_return", "mock_value_for_invoke"),
    [
        (int, "1.0"),
        (float, "1"),
        (int | float, "Hello"),
        (dict, "[1,2,3]"),
        (list[int], '[1,2,3,"hello"]'),
        (str, "1"),
        (bool, "1"),
        (list[str], "[1,2,3]"),
        (
            PydanticModelExample,
            '{"int_value":"Hello","float_value":1.0,"list_value":[1,2,3]}',
        ),
        (
            dict,
            'Here is a dictionary {"key":"value"} Here is a dictionary {"key":"value"}',
        ),
        (dict, ' Here are two dictionaries {"key":"value"}{"key2":"value2"}'),
    ],
)
def test_type_verification_fails_for_invalid_types(
    mocker: Mock, type_to_return: type, mock_value_for_invoke: str
) -> None:
    mock_the_value_output_of_invoke(mocker, Gpt4o, mock_value_for_invoke)
    ai_model = Gpt4o()
    cheap_input = ai_model._get_cheap_input_for_invoke()
    coroutine = ai_model.invoke_and_return_verified_type(
        cheap_input, type_to_return, allowed_invoke_tries_for_failed_output=1
    )
    with pytest.raises(Exception):
        asyncio.run(coroutine)


@pytest.mark.parametrize(
    ("mock_value_for_invoke", "expected_output_type", "expected_value"),
    [
        (
            '```python\nfinal_result = {"key":"value"}\n```',
            dict,
            {"key": "value"},
        ),
        ('```\nfinal_result = {"key":"value"}```', dict, {"key": "value"}),
        ('final_result = {"key":"value"}', dict, {"key": "value"}),
        ("a=1\nb=2\nfinal_result = a+b", int, 3),
        ("import math\nfinal_result = math.sqrt(4)", float, 2),
        ("print('hello')", Exception, None),
        ("raise Exception('error')", Exception, None),
        ("final_result = (1,2,3)", tuple[int, int, int], (1, 2, 3)),
        ("final_result = [1,2,3]", list[int], [1, 2, 3]),
    ],
)
def test_invoke_for_running_code(
    mocker: Mock,
    mock_value_for_invoke: str,
    expected_output_type: type,
    expected_value: Any,
) -> None:
    mock_the_value_output_of_invoke(mocker, Gpt4o, mock_value_for_invoke)
    ai_model = Gpt4o()
    cheap_input = ai_model._get_cheap_input_for_invoke()
    coroutine = ai_model.invoke_and_unsafely_run_and_return_generated_code(
        cheap_input, expected_output_type, 1
    )
    if expected_value is None:
        assert issubclass(expected_output_type, Exception)
        with pytest.raises(expected_output_type):
            asyncio.run(coroutine)
    else:
        result, code = asyncio.run(coroutine)
        assert validate_complex_type(result, expected_output_type)
        assert result == expected_value
        assert isinstance(code, str)


def test_schema_generation_works() -> None:
    class TestPydanticModel(BaseModel):
        int_value: int
        float_value: float
        list_value: list[int]

    format_instructions = Gpt4o().get_schema_format_instructions_for_pydantic_type(
        TestPydanticModel
    )
    logger.debug(format_instructions)
    assert "int_value" in format_instructions
    assert "float_value" in format_instructions
    assert "list_value" in format_instructions


def mock_the_value_output_of_invoke(
    mocker: Mock, ai_model: type[AiModel], mock_value: str
) -> None:
    mock_response = TextTokenCostResponse(
        data=mock_value,
        total_tokens_used=1,
        prompt_tokens_used=1,
        completion_tokens_used=1,
        cost=1.0,
        model="mock model",
    )
    AiModelMockManager.mock_ai_model_direct_call_with_value(
        mocker, ai_model, mock_response
    )


def assert_output_is_of_expected_type_and_value(
    coroutine: Coroutine, expected_type: type, expected_value: Any
) -> None:
    model_response = asyncio.run(coroutine)
    assert validate_complex_type(model_response, expected_type)
    assert model_response == expected_value
