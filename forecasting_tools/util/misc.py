import json
import logging
import re
import urllib.parse
from typing import Any, TypeVar, cast

import aiohttp
import requests
from pydantic import BaseModel

from forecasting_tools.ai_models.ai_utils.ai_misc import validate_complex_type

T = TypeVar("T")

logger = logging.getLogger(__name__)


def raise_for_status_with_additional_info(
    response: requests.Response | aiohttp.ClientResponse,
) -> None:
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


def make_text_fragment_url(quote: str, url: str) -> str:
    less_than_10_words = len(quote.split()) < 10
    if less_than_10_words:
        text_fragment = quote
    else:
        first_five_words = " ".join(quote.split()[:5])
        last_five_words = " ".join(quote.split()[-5:])
        encoded_first_five_words = urllib.parse.quote(
            first_five_words, safe=""
        )
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
