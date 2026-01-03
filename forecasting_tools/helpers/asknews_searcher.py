from __future__ import annotations

import asyncio
import logging
import os
from typing import Literal

from asknews_sdk import AsyncAskNewsSDK

try:
    from asknews_sdk.dto.news import SearchResponseDictItem
except ImportError:
    pass

try:
    from asknews_sdk.dto.deepnews import CreateDeepNewsResponse
except ImportError:
    pass

from forecasting_tools.helpers.asknews_cache import AskNewsCache

# NOTE: More information available here:
# https://docs.asknews.app/en/news
# https://docs.asknews.app/en/deepnews

logger = logging.getLogger(__name__)


class AskNewsSearcher:
    _default_search_depth = 1
    _default_max_depth = 1
    _default_model = "deepseek-basic"
    _default_sources = ["asknews"]
    _default_rate_limit = 12

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        api_key: str | None = None,
        cache_mode: (
            Literal["use_cache", "use_cache_with_fallback", "no_cache"] | None
        ) = None,
    ) -> None:
        if cache_mode is None:
            cache_mode = os.getenv("ASKNEWS_CACHE_MODE", "no_cache")  # type: ignore
        if cache_mode not in ["use_cache", "use_cache_with_fallback", "no_cache"]:
            raise ValueError(f"Invalid cache mode: {cache_mode}")

        self.client_id = client_id or os.getenv("ASKNEWS_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("ASKNEWS_SECRET")
        self.api_key = api_key or os.getenv("ASKNEWS_API_KEY")

        # Check if both authentication methods are defined
        has_oauth = bool(self.client_id and self.client_secret)
        has_api_key = bool(self.api_key)

        if has_oauth and has_api_key:
            raise ValueError(
                "Cannot use both OAuth (client_id/client_secret) and API key authentication. "
                "Please provide either ASKNEWS_CLIENT_ID + ASKNEWS_SECRET or ASKNEWS_API_KEY, not both."
            )

        # Validate OAuth credentials
        if has_oauth and (not self.client_id or not self.client_secret):
            raise ValueError(
                "Incomplete OAuth credentials. Both ASKNEWS_CLIENT_ID and ASKNEWS_SECRET must be set."
            )

        # Validate that at least one authentication method is provided
        if not has_oauth and not has_api_key:
            raise ValueError(
                "No authentication credentials provided. "
                "Please set either ASKNEWS_CLIENT_ID + ASKNEWS_SECRET or ASKNEWS_API_KEY"
            )

        self.cache = AskNewsCache(cache_mode=cache_mode)

    def get_formatted_news(self, query: str) -> str:
        return asyncio.run(self.get_formatted_news_async(query))

    async def get_formatted_news_async(self, query: str) -> str:
        """
        Use the AskNews `news` endpoint to get news context for your query.
        The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
        """
        cached_result = self.cache.get(query)
        if cached_result is not None:
            logger.info(f"Found cached result for query: {query}")
            return cached_result

        async with AsyncAskNewsSDK(
            client_id=self.client_id,
            client_secret=self.client_secret,
            api_key=self.api_key,
            scopes=set(["news"]),
        ) as ask:

            # get the latest news related to the query (within the past 48 hours)
            hot_response = await ask.news.search_news(
                query=query,  # your natural language query
                n_articles=6,  # control the number of articles to include in the context, originally 5
                return_type="both",
                strategy="latest news",  # enforces looking at the latest news only
            )

            await asyncio.sleep(
                self._default_rate_limit
            )  # AskNews free tier has a ratelimit of 1 call per 10 seconds

            # get context from the "historical" database that contains a news archive going back to 2023
            historical_response = await ask.news.search_news(
                query=query,
                n_articles=10,
                return_type="both",
                strategy="news knowledge",  # looks for relevant news within the past 160 days
            )

            hot_articles = hot_response.as_dicts
            historical_articles = historical_response.as_dicts
            formatted_articles = "Here are the relevant news articles:\n\n"

            if hot_articles:
                formatted_articles += self._format_articles(hot_articles)
            if historical_articles:
                formatted_articles += self._format_articles(historical_articles)
            if not hot_articles and not historical_articles:
                formatted_articles += "No articles were found.\n\n"
                self.cache.set(query, formatted_articles)
                return formatted_articles

            self.cache.set(query, formatted_articles)
            return formatted_articles

    def _format_articles(self, articles: list[SearchResponseDictItem]) -> str:
        formatted_articles = ""
        sorted_articles = sorted(articles, key=lambda x: x.pub_date, reverse=True)

        for article in sorted_articles:
            pub_date = article.pub_date.strftime("%B %d, %Y %I:%M %p")
            formatted_articles += f"**{article.eng_title}**\n{article.summary}\nOriginal language: {article.language}\nPublish date: {pub_date}\nSource:[{article.source_id}]({article.article_url})\n\n"

        return formatted_articles

    async def call_preconfigured_version(self, preset: str, prompt: str) -> str:
        if "asknews/news-summaries" in preset:
            return await self.get_formatted_news_async(prompt)

        if "deep-research" not in preset:
            raise ValueError(f"Preset {preset} not found")
        parts_of_preset = preset.split("/")
        try:
            model_name = parts_of_preset[3]
        except IndexError:
            model_name = "deepseek-basic"

        if "asknews/deep-research/low-depth" in preset:
            research = await self.get_formatted_deep_research(
                prompt,
                sources=self._default_sources,
                search_depth=1,
                max_depth=1,
                model=model_name,
            )
        elif "asknews/deep-research/medium-depth" in preset:
            research = await self.get_formatted_deep_research(
                prompt,
                sources=["asknews", "google", "x", "wiki"],
                filter_params={"premium": True},
                search_depth=2,
                max_depth=4,
                model=model_name,
            )
        elif "asknews/deep-research/high-depth" in preset:
            research = await self.get_formatted_deep_research(
                prompt,
                sources=["asknews", "google", "x", "wiki"],
                filter_params={"premium": True},
                search_depth=4,
                max_depth=6,
                model=model_name,
            )
        else:
            raise ValueError(f"Preset {preset} not found")
        return research

    async def get_formatted_deep_research(
        self,
        query: str,
        sources: list[str] | None = None,
        model: (
            Literal["deepseek", "deepseek-basic", "claude-3-7-sonnet-latest", "o3-mini"]
            | str
        ) = _default_model,
        search_depth: int = _default_search_depth,
        max_depth: int = _default_max_depth,
        filter_params: dict[str, bool] | None = None,
    ) -> str:
        response = await self.run_deep_research(
            query, sources, model, search_depth, max_depth, filter_params
        )
        text = response.choices[0].message.content

        start_tag = "<final_answer>"
        end_tag = "</final_answer>"
        start_index = text.find(start_tag)

        if start_index != -1:
            start_index += len(start_tag)
            end_index = text.find(end_tag, start_index)
            if end_index != -1:
                return text[start_index:end_index].strip()

        return text

    async def run_deep_research(
        self,
        query: str,
        sources: list[str] | None = None,
        model: (
            Literal["deepseek", "deepseek-basic", "claude-3-7-sonnet-latest", "o3-mini"]
            | str
        ) = _default_model,
        search_depth: int = _default_search_depth,
        max_depth: int = _default_max_depth,
        filter_params: dict[str, bool] | None = None,
    ) -> CreateDeepNewsResponse:
        try:
            from asknews_sdk.dto.deepnews import CreateDeepNewsResponse
        except ImportError:
            raise ImportError(
                "Most recent version of asknews package not installed, deep research will not work. Run `poetry add asknews@0.11.6`"
            )
        async with AsyncAskNewsSDK(
            client_id=self.client_id,
            client_secret=self.client_secret,
            api_key=self.api_key,
            scopes={"chat", "news", "stories", "analytics"},
        ) as sdk:
            response = await sdk.chat.get_deep_news(
                messages=[{"role": "user", "content": query}],
                search_depth=search_depth,
                max_depth=max_depth,
                sources=sources,
                stream=False,
                return_sources=False,
                model=model,
                inline_citations="numbered",
                filter_params=filter_params,
            )
            if not isinstance(response, CreateDeepNewsResponse):
                raise ValueError("Response is not a CreateDeepNewsResponse")

            return response
