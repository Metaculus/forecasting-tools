from __future__ import annotations

import asyncio
import os
from typing import Literal, Optional

from asknews_sdk import AsyncAskNewsSDK

try:
    from asknews_sdk.dto.news import SearchResponseDictItem
except ImportError:
    pass

try:
    from asknews_sdk.dto.deepnews import CreateDeepNewsResponse
except ImportError:
    pass


# NOTE: More information available here:
# https://docs.asknews.app/en/news
# https://docs.asknews.app/en/deepnews


class AskNewsSearcher:
    _default_search_depth = 2
    _default_max_depth = 2

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> None:
        self.client_id = client_id or os.getenv("ASKNEWS_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("ASKNEWS_SECRET")

        if not self.client_id or not self.client_secret:
            raise ValueError("ASKNEWS_CLIENT_ID or ASKNEWS_SECRET is not set")

    def get_formatted_news(self, query: str) -> str:
        return asyncio.run(self.get_formatted_news_async(query))

    async def get_formatted_news_async(self, query: str) -> str:
        """
        Use the AskNews `news` endpoint to get news context for your query.
        The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
        """
        async with AsyncAskNewsSDK(
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=set(["news"]),
        ) as ask:

            # get the latest news related to the query (within the past 48 hours)
            hot_response = await ask.news.search_news(
                query=query,  # your natural language query
                n_articles=6,  # control the number of articles to include in the context, originally 5
                return_type="both",
                strategy="latest news",  # enforces looking at the latest news only
            )

            # get context from the "historical" database that contains a news archive going back to 2023
            historical_response = await ask.news.search_news(
                query=query,
                n_articles=10,
                return_type="both",
                strategy="news knowledge",  # looks for relevant news within the past 60 days
            )

            hot_articles = hot_response.as_dicts
            historical_articles = historical_response.as_dicts
            formatted_articles = "Here are the relevant news articles:\n\n"

            if hot_articles:
                formatted_articles += self._format_articles(hot_articles)
            if historical_articles:
                formatted_articles += self._format_articles(
                    historical_articles
                )
            if not hot_articles and not historical_articles:
                formatted_articles += "No articles were found.\n\n"
                return formatted_articles

            return formatted_articles

    def _format_articles(self, articles: list[SearchResponseDictItem]) -> str:
        formatted_articles = ""
        sorted_articles = sorted(
            articles, key=lambda x: x.pub_date, reverse=True
        )

        for article in sorted_articles:
            pub_date = article.pub_date.strftime("%B %d, %Y %I:%M %p")
            formatted_articles += f"**{article.eng_title}**\n{article.summary}\nOriginal language: {article.language}\nPublish date: {pub_date}\nSource:[{article.source_id}]({article.article_url})\n\n"

        return formatted_articles

    async def get_formatted_deep_research(
        self,
        query: str,
        sources: list[str] | None = None,
        model: Literal[
            "deepseek", "claude-3-7-sonnet-latest", "o3-mini"
        ] = "deepseek",
        search_depth: int = _default_search_depth,
        max_depth: int = _default_max_depth,
    ) -> str:
        response = await self.run_deep_research(
            query, sources, model, search_depth, max_depth
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
        model: Literal[
            "deepseek", "claude-3-7-sonnet-latest", "o3-mini"
        ] = "deepseek",
        search_depth: int = _default_search_depth,
        max_depth: int = _default_max_depth,
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
            )
            if not isinstance(response, CreateDeepNewsResponse):
                raise ValueError("Response is not a CreateDeepNewsResponse")

            return response
