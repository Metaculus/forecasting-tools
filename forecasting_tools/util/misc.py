import logging
import re
from typing import Any, TypeVar, cast

import requests

from forecasting_tools.ai_models.ai_utils.ai_misc import validate_complex_type

T = TypeVar("T")

logger = logging.getLogger(__name__)


def raise_for_status_with_additional_info(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        response_text = response.text
        response_reason = response.reason
        try:
            response_json = response.json()
        except Exception:
            response_json = None
        error_message = f"HTTPError. Url: {response.url}. Response reason: {response_reason}. Response text: {response_text}. Response JSON: {response_json}"
        logger.error(error_message)
        raise requests.exceptions.HTTPError(error_message) from e


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
