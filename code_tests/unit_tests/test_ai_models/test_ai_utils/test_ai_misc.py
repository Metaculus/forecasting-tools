import asyncio
import logging
from typing import Any, Union

import pytest

from forecasting_tools.ai_models.ai_utils.ai_misc import (
    clean_indents,
    retry_async_function,
    strip_code_block_markdown,
    validate_complex_type,
)

logger = logging.getLogger(__name__)
from pydantic import BaseModel


def test_retry_decorator() -> None:

    NUMBER_OF_RETRIES = 3
    SUCCESS_STRING = "Success"

    @retry_async_function(NUMBER_OF_RETRIES)
    async def sample_function_for_retry_that_errors(
        success_on_try: int, call_count: dict
    ) -> str:
        call_count["count"] += 1
        if call_count["count"] < success_on_try:
            raise RuntimeError("Simulated failure")
        return SUCCESS_STRING

    call_count = {"count": 0}

    try_where_function_will_succeed = 1
    result = asyncio.run(
        sample_function_for_retry_that_errors(
            try_where_function_will_succeed, call_count
        )
    )
    assert result == SUCCESS_STRING
    assert call_count["count"] == try_where_function_will_succeed

    call_count["count"] = 0
    try_where_function_will_succeed = NUMBER_OF_RETRIES
    result = asyncio.run(
        sample_function_for_retry_that_errors(
            try_where_function_will_succeed, call_count
        )
    )
    assert result == SUCCESS_STRING
    assert call_count["count"] == try_where_function_will_succeed

    call_count["count"] = 0
    try_where_function_will_succeed = NUMBER_OF_RETRIES + 1
    with pytest.raises(RuntimeError):
        asyncio.run(
            sample_function_for_retry_that_errors(
                try_where_function_will_succeed, call_count
            )
        )
    assert call_count["count"] == NUMBER_OF_RETRIES


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


instance_of_test_model_1 = PydanticModelExample(int_value=1, float_value=1.0)
instance_of_test_model_2 = PydanticModelExample2(
    str_value="hello",
    list_value=[1, 2, 3],
    sub_model=SubPydanticModel2(str_value="hello", list_value=[1, 2, 3]),
)


@pytest.mark.parametrize(
    ("type_to_check", "value_to_check", "expected_output"),
    [
        (int, 1, True),
        (float, 1.0, True),
        (str, "hello", True),
        (list, [1, 2, 3], True),
        (dict, {"key": "value"}, True),
        (tuple, (1, 2, 3), True),
        (int, "hello", False),
        (float, "hello", False),
        (str, 1, False),
        (list, "hello", False),
        (dict, "hello", False),
        (tuple, "hello", False),
        (list[str], ["hello", "world"], True),
        (dict[str, int], {"key": 1}, True),
        (tuple[int, str], (1, "hello"), True),
        (list[str], ["hello", 1], False),
        (dict[str, int], {"key": "value"}, False),
        (tuple[int, str], (1, 2), False),
        (Union[int, str], 1, True),
        (Union[int, str], "hello", True),
        (int | str, "Hello", True),
        (int | float, "Hello", False),
        (tuple[int | None, str], (None, "Hello"), True),
        (tuple[int | None, str], (1, "Hello"), True),
        (tuple[int | None, str], ("Hello", "Hello"), False),
        (PydanticModelExample, instance_of_test_model_1, True),
        (PydanticModelExample2, instance_of_test_model_2, True),
        (PydanticModelExample, instance_of_test_model_2, False),
        (PydanticModelExample2, instance_of_test_model_1, False),
        (
            list[PydanticModelExample],
            [instance_of_test_model_1, instance_of_test_model_1],
            True,
        ),
        (
            list[PydanticModelExample],
            [instance_of_test_model_1, instance_of_test_model_2],
            False,
        ),
        (BaseModel, instance_of_test_model_1, True),
    ],
)
def test_validate_complex_type(
    type_to_check: type, value_to_check: Any, expected_output: bool
) -> None:
    assert validate_complex_type(value_to_check, type_to_check) == expected_output


@pytest.mark.parametrize(
    ("test_case", "expected_output"),
    [
        (
            """
        This is a test
        and this is a test
        and more tests
        """,
            """
This is a test
and this is a test
and more tests
""",
        ),
        (
            """
        This is a test
and this is a test
        and more tests
        """,
            """
This is a test
and this is a test
and more tests
""",
        ),
        (
            """
        This is a test
            and this is a test
        and more tests
        """,
            """
This is a test
    and this is a test
and more tests
""",
        ),
        (
            """
            This is a test
        and this is a test
        and more tests
        """,
            """
This is a test
and this is a test
and more tests
""",
        ),
        (
            """This is a test
        and this is a test
        and more tests
        """,
            """This is a test
and this is a test
and more tests
""",
        ),
        ("This is a test", "This is a test"),
        ("", ""),
    ],
)
def test_stripping_indents_from_lines_works(
    test_case: str, expected_output: str
) -> None:
    logger.info(test_case)
    cleaned_text = clean_indents(test_case)
    logger.info(cleaned_text)
    assert cleaned_text == expected_output


@pytest.mark.parametrize(
    ("string_input", "expected_output"),
    [
        ('```json\n{"key": "value"}\n```', '{"key": "value"}'),
        ('```python\n{"key": "value"}\n```', '{"key": "value"}'),
        ('```{"key": "value"}```', '{"key": "value"}'),
        ('```{"key": "value"}\n```', '{"key": "value"}'),
        ('```\n{"key": "value"}```', '{"key": "value"}'),
        ('```\n{"key": "value"}\n```', '{"key": "value"}'),
        ('{"key": "value"}', '{"key": "value"}'),
    ],
)
def test_strip_code_block_markdown(string_input: str, expected_output: str) -> None:
    stripped_string = strip_code_block_markdown(string_input)
    assert stripped_string == expected_output
