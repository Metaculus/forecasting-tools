import json
from unittest.mock import patch

from forecasting_tools.ai_models.keenable_searcher import (
    KEYED_SEARCH_URL,
    PUBLIC_SEARCH_URL,
    KeenableSearcher,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_formats_keyless_results_and_sends_title_header() -> None:
    searcher = KeenableSearcher(api_key=None, max_results=3)
    payload = {
        "results": [
            {
                "title": "A",
                "url": "https://a.example/page",
                "description": "short",
                "snippet": "long",
            },
            {"title": "No URL", "description": "drop"},
            {"title": "B", "url": "https://b.example/page", "snippet": "fallback"},
        ]
    }
    with patch(
        "forecasting_tools.ai_models.keenable_searcher.urlopen",
        return_value=_FakeResponse(payload),
    ) as urlopen:
        text = searcher.get_formatted_results("Will X happen?")

    request = urlopen.call_args[0][0]
    assert request.full_url == PUBLIC_SEARCH_URL
    header_map = {key.lower(): value for key, value in request.header_items()}
    assert header_map.get("x-keenable-title") == "forecasting-tools"
    assert "x-api-key" not in header_map
    body = json.loads(request.data.decode("utf-8"))
    assert body == {"query": "Will X happen?", "max_results": 3}
    assert "https://a.example/page" in text
    assert "short" in text
    assert "fallback" in text
    assert "No URL" not in text


def test_keyed_endpoint_used_when_api_key_is_set() -> None:
    searcher = KeenableSearcher(api_key="secret", max_results=2)
    with patch(
        "forecasting_tools.ai_models.keenable_searcher.urlopen",
        return_value=_FakeResponse({"results": []}),
    ) as urlopen:
        text = searcher.get_formatted_results("q")
    request = urlopen.call_args[0][0]
    assert request.full_url == KEYED_SEARCH_URL
    header_map = {key.lower(): value for key, value in request.header_items()}
    assert header_map.get("x-api-key") == "secret"
    assert text == "No Keenable search results found."
