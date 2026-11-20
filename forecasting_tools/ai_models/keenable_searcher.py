import json
import logging
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

PUBLIC_SEARCH_URL = "https://api.keenable.ai/v1/search/public"
KEYED_SEARCH_URL = "https://api.keenable.ai/v1/search"
APP_TITLE = "forecasting-tools"


class KeenableSearcher:
    """Keyless web search via Keenable. Optional KEENABLE_API_KEY raises limits."""

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 10,
        timeout_seconds: int = 30,
    ):
        self.api_key = (api_key or os.getenv("KEENABLE_API_KEY") or "").strip() or None
        self.max_results = max(1, min(max_results, 50))
        self.timeout_seconds = timeout_seconds

    async def invoke(self, query: str) -> str:
        import asyncio

        return await asyncio.to_thread(self.get_formatted_results, query)

    def get_formatted_results(self, query: str) -> str:
        results = self._search(query)
        if not results:
            return "No Keenable search results found."
        lines = []
        for i, item in enumerate(results, start=1):
            title = item.get("title") or item.get("url") or "Untitled"
            url = item.get("url") or ""
            snippet = item.get("description") or item.get("snippet") or ""
            lines.append(f"{i}. {title}\n   {url}\n   {snippet}".rstrip())
        return "\n\n".join(lines)

    def _search(self, query: str) -> list[dict[str, Any]]:
        url = KEYED_SEARCH_URL if self.api_key else PUBLIC_SEARCH_URL
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Keenable-Title": APP_TITLE,
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        payload = json.dumps(
            {"query": query, "max_results": self.max_results}
        ).encode("utf-8")
        request = Request(url, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            logger.error("Keenable search HTTP %s: %s", error.code, error.reason)
            return []
        except URLError as error:
            logger.error("Keenable search error: %s", error.reason)
            return []
        return [item for item in (body.get("results") or []) if item.get("url")]
