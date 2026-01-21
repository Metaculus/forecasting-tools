import asyncio
import json
import logging
import random
import re
import time
import urllib.parse
from datetime import datetime
from functools import wraps
from typing import Any, Callable, TypeGuard, TypeVar, Union, cast, get_args, get_origin

import aiohttp
import pendulum
import requests
from pydantic import BaseModel

T = TypeVar("T")
B = TypeVar("B", bound=BaseModel)

logger = logging.getLogger(__name__)


def raise_for_status_with_additional_info(
    response: requests.Response | aiohttp.ClientResponse,
) -> None:
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        response_text = str(response.text)
        response_reason = response.reason
        try:
            response_json = response.json()
        except Exception:
            response_json = None
        if "!DOCTYPE html".lower() in response_text.lower():
            response_text = "Response text is a HTML page"

        try:
            status_code = response.status_code  # type: ignore
        except Exception:
            status_code = None

        error_message = (
            f"HTTPError. Url: {response.url}. Status code: {status_code}. "
            f"Response reason: {response_reason}. Response text: {response_text}. "
            f"Response JSON: {response_json}."
        )
        logger.error(error_message)
        raise requests.exceptions.HTTPError(error_message) from e


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 2.5,
    exponential_base: float = 3,
    jitter_factor: float = 8.0,
    max_delay: float = 75.0,
    retry_on_exceptions: tuple[type[Exception], ...] = (
        requests.exceptions.RequestException,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    ),
) -> Callable:
    if jitter_factor < 1:
        raise ValueError("jitter_factor must be greater than 1")
    if exponential_base < 1:
        raise ValueError("exponential_base must be greater than 1")
    if initial_delay < 0:
        raise ValueError("initial_delay must be greater than 0")
    if max_delay < 0:
        raise ValueError("max_delay must be greater than 0")
    if max_retries < 0:
        raise ValueError("max_retries must be greater than 0")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries = 0
            delay = initial_delay

            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except retry_on_exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}. Last error: {e}"
                        )
                        raise

                    jitter = random.uniform(1, jitter_factor)
                    sleep_time = min(delay * jitter, max_delay)
                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} after {sleep_time:.1f}s. Error: {e}"
                    )
                    time.sleep(sleep_time)
                    delay *= exponential_base

        return wrapper

    return decorator


def is_markdown_citation(v: str) -> bool:
    pattern = r"\[\d+\]\(https?://\S+\)"
    return bool(re.match(pattern, v))


def extract_url_from_markdown_link(markdown_link: str) -> str:
    match = re.search(r"\((\S+)\)", markdown_link)
    if match:
        return match.group(1)
    else:
        raise ValueError(
            "Citation must be in the markdown friendly format [number](url)"
        )


def cast_and_check_type(value: Any, expected_type: type[T]) -> T:
    if not validate_complex_type(value, expected_type):
        raise ValueError(f"Value {value} is not of type {expected_type}")
    return cast(expected_type, value)


def make_text_fragment_url(quote: str, url: str) -> str:
    less_than_10_words = len(quote.split()) < 10
    if less_than_10_words:
        text_fragment = quote
    else:
        first_five_words = " ".join(quote.split()[:5])
        last_five_words = " ".join(quote.split()[-5:])
        encoded_first_five_words = urllib.parse.quote(first_five_words, safe="")
        encoded_last_five_words = urllib.parse.quote(last_five_words, safe="")
        text_fragment = f"{encoded_first_five_words},{encoded_last_five_words}"  # Comma indicates that anything can be included in between
    text_fragment = text_fragment.replace("(", "%28").replace(")", "%29")
    text_fragment = text_fragment.replace("-", "%2D").strip(",")
    text_fragment = text_fragment.replace(" ", "%20")
    fragment_url = f"{url}#:~:text={text_fragment}"
    return fragment_url


def fill_in_citations(
    urls_for_citations: list[str], text: str, use_citation_brackets: bool
) -> str:
    final_text = text
    for i, url in enumerate(urls_for_citations):
        citation_num = i + 1
        if use_citation_brackets:
            markdown_url = f"\\[[{citation_num}]({url})\\]"
        else:
            markdown_url = f"[{citation_num}]({url})"

        # Combined regex pattern for all citation types
        pattern = re.compile(
            r"(?:\\\[)?(\[{}\](?:\(.*?\))?)(?:\\\])?".format(citation_num)
        )
        # Matches:
        # [1]
        # [1](some text)
        # \[[1]\]
        # \[[1](some text)\]
        final_text = pattern.sub(markdown_url, final_text)
    return final_text


def get_schema_of_base_model(model_class: type[BaseModel]) -> str:
    schema = {k: v for k, v in model_class.model_json_schema().items()}

    reduced_schema = schema
    if "title" in reduced_schema:
        del reduced_schema["title"]
    if "type" in reduced_schema:
        del reduced_schema["type"]
    schema_str = json.dumps(reduced_schema)
    return schema_str


def add_timezone_to_dates_in_base_model(base_model: B) -> B:
    for field_name in type(base_model).model_fields.keys():
        value = getattr(base_model, field_name)
        if isinstance(value, datetime):
            if value is None:
                date = None
            elif value.tzinfo is None:
                date = value.replace(tzinfo=pendulum.timezone("UTC"))
            else:
                date = value
            setattr(base_model, field_name, date)
    return base_model


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
        if not args:
            return True  # bare list type, no element types specified
        return all(validate_complex_type(v, args[0]) for v in value)

    if origin is dict:
        # Special handling for dict types
        if not isinstance(value, dict):
            return False
        if not args:
            return True  # bare dict type, no key or value types specified
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
