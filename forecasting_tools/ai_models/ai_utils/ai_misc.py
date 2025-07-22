import asyncio
import logging
from typing import Any, Callable, TypeGuard, TypeVar, Union, cast, get_args, get_origin

T = TypeVar("T")

logger = logging.getLogger(__name__)


async def try_function_till_tries_run_out(
    tries: int, function: Callable, *args, **kwargs
) -> Any:
    tries_left = tries
    while tries_left > 0:
        try:
            response = await function(*args, **kwargs)
            return response
        except Exception as e:
            tries_left -= 1
            if tries_left == 0:
                raise e
            logger.warning(f"Retrying function {function.__name__} due to error: {e}")
            await asyncio.sleep(1)


def retry_async_function(tries: int) -> Callable:
    def decorator(function: Callable) -> Callable:
        async def wrapper(*args, **kwargs) -> Any:
            return await try_function_till_tries_run_out(
                tries, function, *args, **kwargs
            )

        return wrapper

    return decorator


def validate_complex_type(value: T, expected_type: type[T]) -> TypeGuard[T]:
    # NOTE: Consider using typeguard.check_type instead of this function
    value = cast(expected_type, value)
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is None:
        # Base case: expected_type is not a generic alias (like int, str, etc.)
        return isinstance(value, expected_type)

    if origin is Union:
        # Special handling for Union types (e.g., Union[int, str])
        return any(validate_complex_type(value, arg) for arg in args)

    if origin is tuple:
        # Special handling for tuple types
        if not isinstance(value, tuple) or len(value) != len(args):
            return False
        return all(validate_complex_type(v, t) for v, t in zip(value, args))

    if origin is list:
        # Special handling for list types
        if not isinstance(value, list):
            return False
        return all(validate_complex_type(v, args[0]) for v in value)

    if origin is dict:
        # Special handling for dict types
        if not isinstance(value, dict):
            return False
        key_type, value_type = args
        return all(
            validate_complex_type(k, key_type) and validate_complex_type(v, value_type)
            for k, v in value.items()
        )

    # Fallback for other types
    return isinstance(value, expected_type)


def clean_indents(text: str) -> str:
    """
    Cleans indents from the text, optimized for prompts
    Note, this is not the same as textwrap.dedent (see the test for this function for examples)
    """
    lines = text.split("\n")
    try:
        indent_level_of_first_line = find_indent_level_of_string(lines[0])
        indent_level_of_second_line = find_indent_level_of_string(lines[1])
        greatest_indent_level_of_first_two_lines = max(
            indent_level_of_first_line, indent_level_of_second_line
        )
    except IndexError:
        greatest_indent_level_of_first_two_lines = find_indent_level_of_string(lines[0])

    new_lines = []
    for line in lines:
        indent_level_of_line = find_indent_level_of_string(line)
        if indent_level_of_line >= greatest_indent_level_of_first_two_lines:
            new_line = line[greatest_indent_level_of_first_two_lines:]
        else:
            new_line = line.lstrip()
        new_lines.append(new_line)

    combined_new_lines = "\n".join(new_lines)
    return combined_new_lines


def find_indent_level_of_string(string: str) -> int:
    return len(string) - len(string.lstrip())


def strip_code_block_markdown(string: str) -> str:
    string = string.strip()
    if string.startswith("```json") and string.endswith("```"):
        string = string[7:-3].strip()
    elif string.startswith("```python") and string.endswith("```"):
        string = string[9:-3].strip()
    elif string.startswith("```markdown") and string.endswith("```"):
        string = string[11:-3].strip()
    elif string.startswith("```") and string.endswith("```"):
        string = string[3:-3].strip()
    return string
