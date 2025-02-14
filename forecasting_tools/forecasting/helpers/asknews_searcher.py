from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union
from urllib.parse import quote, urlencode

import httpx
from httpx import Auth, Request

# NOTE: Until there is more need for asknews endpoints, this is a custom implementation
# That does not use the SDK. As of Feb 1 2025 there were dependency conflicts
# due to asknews dependencies


@dataclass
class SearchResponse:
    as_string: Optional[str] = None
    as_dicts: Optional[List[Dict[str, Any]]] = None
    __content_type__ = "application/json"


def encode_client_secret_basic(client_id: str, client_secret: str) -> str:
    text = f"{quote(client_id)}:{quote(client_secret)}"
    auth = base64.b64encode(text.encode()).decode()
    return f"Basic {auth}"


class OAuth2ClientCredentials(Auth):
    def __init__(
        self, client_id: str, client_secret: str, token_url: str
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.token: Optional[str] = None
        self.token_expires: Optional[float] = None

    def _build_token_request(self) -> Request:
        return Request(
            method="POST",
            url=self.token_url,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": encode_client_secret_basic(
                    self.client_id, self.client_secret
                ),
            },
            content=urlencode(
                {
                    "grant_type": "client_credentials",
                    "scope": "news openid offline",
                }
            ),
        )

    async def async_auth_flow(
        self, request: Request
    ) -> AsyncGenerator[Request, None]:
        if not self.token or (
            self.token_expires
            and datetime.now().timestamp()
            > (self.token_expires - 15)  # 15 seconds before expiration
        ):
            token_request = self._build_token_request()
            async with httpx.AsyncClient() as client:
                response = await client.send(token_request)
                response.raise_for_status()
                data = response.json()
                self.token = data["access_token"]
                self.token_expires = (
                    datetime.now().timestamp() + data["expires_in"]
                )

        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


class AskNewsSearcher:

    def get_formatted_news(self, query: str) -> str:
        """
        Use the AskNews `news` endpoint to get news context for your query.
        The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
        """

        # get the latest news related to the query (within the past 48 hours)
        hot_response = asyncio.run(
            self.search_news(
                query=query,  # your natural language query
                n_articles=6,  # control the number of articles to include in the context, originally 5
                return_type="both",
                strategy="latest news",  # enforces looking at the latest news only
            )
        )

        # get context from the "historical" database that contains a news archive going back to 2023
        historical_response = asyncio.run(
            self.search_news(
                query=query,
                n_articles=10,
                return_type="both",
                strategy="news knowledge",  # looks for relevant news within the past 60 days
            )
        )

        hot_articles = hot_response.as_dicts
        historical_articles = historical_response.as_dicts
        formatted_articles = "Here are the relevant news articles:\n\n"

        if hot_articles:
            # Convert pub_date strings to datetime objects for sorting
            for article in hot_articles:
                article["pub_date"] = datetime.fromisoformat(
                    article["pub_date"].replace("Z", "+00:00")
                )

            hot_articles = sorted(
                hot_articles, key=lambda x: x["pub_date"], reverse=True
            )

            for article in hot_articles:
                pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

        if historical_articles:
            # Convert pub_date strings to datetime objects for sorting
            for article in historical_articles:
                article["pub_date"] = datetime.fromisoformat(
                    article["pub_date"].replace("Z", "+00:00")
                )

            historical_articles = sorted(
                historical_articles, key=lambda x: x["pub_date"], reverse=True
            )

            for article in historical_articles:
                pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

        if not hot_articles and not historical_articles:
            formatted_articles += "No articles were found.\n\n"
            return formatted_articles

        return formatted_articles

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> None:
        self.client_id = client_id or os.getenv("ASKNEWS_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("ASKNEWS_SECRET")
        self.base_url = "https://api.asknews.app/v1"
        self.token_url = "https://auth.asknews.app/oauth2/token"

        if not self.client_id or not self.client_secret:
            raise ValueError("ASKNEWS_CLIENT_ID or ASKNEWS_SECRET is not set")

        self.auth = OAuth2ClientCredentials(
            client_id=self.client_id,
            client_secret=self.client_secret,
            token_url=self.token_url,
        )

    async def search_news(
        self,  # NOSONAR
        query: str = "",
        n_articles: int = 10,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        time_filter: Literal["crawl_date", "pub_date"] = "crawl_date",
        return_type: Literal["string", "dicts", "both"] = "string",
        historical: bool = False,
        method: Literal["nl", "kw", "both"] = "kw",
        similarity_score_threshold: float = 0.5,
        offset: Union[int, str] = 0,
        categories: Optional[
            List[
                Literal[
                    "All",
                    "Business",
                    "Crime",
                    "Politics",
                    "Science",
                    "Sports",
                    "Technology",
                    "Military",
                    "Health",
                    "Entertainment",
                    "Finance",
                    "Culture",
                    "Climate",
                    "Environment",
                    "World",
                ]
            ]
        ] = None,
        strategy: Literal[
            "latest news", "news knowledge", "default"
        ] = "default",
        hours_back: Optional[int] = 24,
        languages: Optional[List[str]] = None,
        premium: Optional[bool] = False,
    ) -> SearchResponse:
        params = {
            "query": query,
            "n_articles": n_articles,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "time_filter": time_filter,
            "return_type": return_type,
            "method": method,
            "historical": historical,
            "offset": offset,
            "categories": categories if categories is not None else ["All"],
            "similarity_score_threshold": similarity_score_threshold,
            "strategy": strategy,
            "hours_back": hours_back,
            "languages": languages,
            "premium": premium,
        }

        params = {k: v for k, v in params.items() if v is not None}

        async with httpx.AsyncClient(auth=self.auth) as client:
            response = await client.get(
                f"{self.base_url}/news/search",
                params=params,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

            return SearchResponse(
                as_string=data.get("as_string"), as_dicts=data.get("as_dicts")
            )
