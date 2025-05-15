from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import aiohttp
from pydantic import BaseModel, Field, model_validator

from forecasting_tools.ai_models.model_interfaces.incurs_cost import IncursCost
from forecasting_tools.ai_models.model_interfaces.request_limited_model import (
    RequestLimitedModel,
)
from forecasting_tools.ai_models.model_interfaces.retryable_model import (
    RetryableModel,
)
from forecasting_tools.ai_models.model_interfaces.time_limited_model import (
    TimeLimitedModel,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class ExaSource(BaseModel, Jsonable):
    original_query: str
    auto_prompt_string: str | None
    title: str | None
    url: str | None
    text: str | None
    author: str | None
    published_date: datetime | None
    score: float | None
    highlights: list[str]
    highlight_scores: list[float]

    @property
    def readable_publish_date(self) -> str:
        return (
            self.published_date.strftime("%Y-%m-%d")
            if self.published_date
            else "unknown date"
        )


class ExaHighlightQuote(BaseModel, Jsonable):
    highlight_text: str
    score: float
    source: ExaSource


class SearchInput(BaseModel, Jsonable):
    web_search_query: str = Field(
        ..., description="The query to search in the search engine"
    )
    highlight_query: str | None = Field(
        description="The query to search within each document using semantic similarity"
    )
    include_domains: list[str] = Field(
        description="List of domains to require in the search results for example: ['youtube.com', 'en.wikipedia.org']. An empty list means no filter. This will constrain search to ONLY results from these domains."
    )
    exclude_domains: list[str] = Field(
        description="List of domains to exclude from the search results: ['youtube.com', 'en.wikipedia.org']. An empty list means no filter. This will constrain search to exclude results from these domains."
    )
    include_text: str | None = Field(
        description="A 1-5 word phrase that must be exactly present in the text of the search results"
    )
    start_published_date: datetime | None = Field(
        description="The earliest publication date for search results"
    )
    end_published_date: datetime | None = Field(
        description="The latest publication date for search results"
    )

    @model_validator(mode="after")
    def validate_times(self: SearchInput) -> SearchInput:
        def to_utc(dt: datetime | None) -> datetime | None:
            if dt is None:
                return None
            if dt.tzinfo is None:
                # Assume local time if naive, convert to UTC
                return dt.astimezone().astimezone(timezone.utc)
            return dt.astimezone(timezone.utc)

        start_date = to_utc(self.start_published_date)
        end_date = to_utc(self.end_published_date)
        now_utc = datetime.now(timezone.utc)

        if start_date and end_date and start_date > end_date:
            raise ValueError(
                "start_published_date must be before or equal to end_published_date"
            )
        if start_date and start_date > now_utc:
            raise ValueError("start_published_date must be before today")
        if end_date and end_date > now_utc + timedelta(days=1):
            raise ValueError("end_published_date must be before end of today")
        if start_date and (now_utc - start_date) <= timedelta(hours=3):
            logger.warning(
                "You are searching for results from within the last 3 hours. This may not return any results."
            )
        return self

    @model_validator(mode="after")
    def validate_include_text(self: SearchInput) -> SearchInput:
        if self.include_text is not None:
            num_words = len(self.include_text.split())
            if num_words < 1 or num_words > 5:
                raise ValueError("include_text must be 1-5 words")
        return self


class ExaSearcher(
    RequestLimitedModel, RetryableModel, TimeLimitedModel, IncursCost
):
    REQUESTS_PER_PERIOD_LIMIT = (
        3  # For rate limits see https://docs.exa.ai/reference/rate-limits
    )
    REQUEST_PERIOD_IN_SECONDS = 1
    TIMEOUT_TIME = 30
    COST_PER_REQUEST = 0.005
    COST_PER_HIGHLIGHT = 0.001
    COST_PER_TEXT = 0.001

    def __init__(
        self,
        *args,
        include_text: bool = False,
        include_highlights: bool = True,
        num_results: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.include_text = include_text
        self.include_highlights = include_highlights
        self.num_highlights_per_url = 10
        self.num_sentences_per_highlight = 4
        self.num_results = num_results

    async def invoke_for_highlights_in_relevance_order(
        self, search_query_or_strategy: str | SearchInput
    ) -> list[ExaHighlightQuote]:
        assert (
            self.include_highlights
        ), "include_highlights must be true to use this method"
        sources = await self.invoke(search_query_or_strategy)
        all_highlights = []
        for source in sources:
            for highlight, score in zip(
                source.highlights, source.highlight_scores
            ):
                all_highlights.append(
                    ExaHighlightQuote(
                        highlight_text=highlight, score=score, source=source
                    )
                )
        sorted_highlights = sorted(
            all_highlights, key=lambda x: x.score, reverse=True
        )
        return sorted_highlights

    async def invoke(
        self, search_query_or_strategy: str | SearchInput
    ) -> list[ExaSource]:
        if isinstance(search_query_or_strategy, str):
            search_strategy = self.string_to_default_search_input(
                search_query_or_strategy
            )
        else:
            search_strategy = search_query_or_strategy
        return await self.__retryable_timed_cost_request_limited_invoke(
            search_strategy
        )

    @RetryableModel._retry_according_to_model_allowed_tries
    @RequestLimitedModel._wait_till_request_capacity_available
    @IncursCost._wrap_in_cost_limiting_and_tracking
    @TimeLimitedModel._wrap_in_model_defined_timeout
    async def __retryable_timed_cost_request_limited_invoke(
        self, search_query_or_strategy: SearchInput
    ) -> list[ExaSource]:
        response = await self._mockable_direct_call_to_model(
            search_query_or_strategy
        )
        return response

    async def _mockable_direct_call_to_model(
        self, search_query: SearchInput
    ) -> list[ExaSource]:
        self._everything_special_to_call_before_direct_call()
        url, headers, payload = self._prepare_request_data(search_query)
        response_data = await self._make_api_request(url, headers, payload)
        exa_sources = self._process_response(response_data, search_query)
        self._log_results(exa_sources)
        return exa_sources

    def _prepare_request_data(
        self, search: SearchInput
    ) -> tuple[str, dict, dict]:
        api_key = self._get_api_key()
        url = "https://api.exa.ai/search"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": api_key,
        }

        payload = {
            "query": search.web_search_query,
            "type": "auto",
            "useAutoprompt": True,
            "numResults": self.num_results,
            "includeDomains": search.include_domains,
            "excludeDomains": search.exclude_domains,
            "livecrawl": "always",
            "contents": {
                "text": (
                    {"includeHtmlTags": True} if self.include_text else False
                ),
                "highlights": (
                    {
                        "query": (
                            search.highlight_query
                            if search.highlight_query
                            else search.web_search_query
                        ),
                        "numSentences": self.num_sentences_per_highlight,
                        "highlightsPerUrl": self.num_highlights_per_url,
                    }
                    if self.include_highlights
                    else False
                ),
            },
        }

        if search.start_published_date:
            payload["startPublishedDate"] = (
                search.start_published_date.isoformat()
            )
        if search.end_published_date:
            payload["endPublishedDate"] = search.end_published_date.isoformat()
        if search.include_text:
            payload["includeText"] = [search.include_text]

        return url, headers, payload

    @classmethod
    def string_to_default_search_input(cls, search_query: str) -> SearchInput:
        return SearchInput(
            web_search_query=search_query,
            highlight_query=search_query,
            include_domains=[],
            exclude_domains=[],
            include_text=None,
            start_published_date=None,
            end_published_date=None,
        )

    def _get_api_key(self) -> str:
        api_key = os.getenv("EXA_API_KEY")
        assert (
            api_key is not None
        ), "EXA_API_KEY is not set in the environment variables"
        return api_key

    async def _make_api_request(
        self, url: str, headers: dict, payload: dict
    ) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                result: dict = await response.json()
                return result

    def _process_response(
        self, response_data: dict, search_query: SearchInput
    ) -> list[ExaSource]:
        exa_sources: list[ExaSource] = []
        auto_prompt_string = response_data.get("autopromptString")

        for result in response_data["results"]:
            assert isinstance(result, dict), "result is not a dict"
            unparsed_publish_date: str | None = result.get("publishedDate")
            if unparsed_publish_date:
                assert isinstance(
                    unparsed_publish_date, str
                ), "unparsed_publish_date is not a str"
                unparsed_publish_date = unparsed_publish_date.strip("Z")
                publish_date = datetime.fromisoformat(unparsed_publish_date)
            else:
                publish_date = None
            exa_source = ExaSource(
                original_query=search_query.web_search_query,
                auto_prompt_string=auto_prompt_string,
                title=result.get("title"),
                url=result.get("url"),
                text=result.get("text"),
                author=result.get("author"),
                published_date=publish_date,
                score=result.get("score"),
                highlights=result.get("highlights", []),
                highlight_scores=result.get("highlightScores", []),
            )
            exa_sources.append(exa_source)
        return exa_sources

    def _log_results(self, exa_sources: list[ExaSource]) -> None:
        logger.debug(
            f"Exa API returned {len(exa_sources)} sources with urls: {[source.url for source in exa_sources]}"
        )

    ##################################### Cost Calculation #####################################

    def _calculate_cost_for_request(self, results: list[ExaSource]) -> float:
        cost = self.COST_PER_REQUEST
        cost += self.COST_PER_TEXT * len(results) if self.include_text else 0
        cost += (
            self.COST_PER_HIGHLIGHT * len(results)
            if self.include_highlights
            else 0
        )
        return cost

    async def _track_cost_in_manager_using_model_response(
        self,
        response_from_direct_call: list[ExaSource],
    ) -> None:
        assert isinstance(
            response_from_direct_call, list
        ), f"response_from_direct_call is not a list, it is a {type(response_from_direct_call)}"
        cost = self._calculate_cost_for_request(response_from_direct_call)
        MonetaryCostManager.increase_current_usage_in_parent_managers(cost)

    ################################### Mocking/Test Functions ###################################
    @staticmethod
    def _get_mock_return_for_direct_call_to_model_using_cheap_input() -> (
        list[ExaSource]
    ):
        return [
            ExaSource(
                original_query="Latest news on AI Research Tools",
                auto_prompt_string="Here is a link to the latest news:",
                title="Latest news on AI Research Tools",
                url="https://www.example.com",
                text="Fake text",
                author=None,
                published_date=None,
                score=0.99,
                highlights=["Fake highlight 1", "Fake highlight 2"],
                highlight_scores=[0.5, 0.6],
            )
        ]

    @classmethod
    def _get_cheap_input_for_invoke(cls) -> SearchInput:
        search_query = "Latest news on AI Research Tools"
        return cls.string_to_default_search_input(search_query)


# Example response from Exa API: (see the Exa playground for more examples)
# ... other data ... below is first source in "results" list
# {
#   "score": 0.19072401523590088,
#   "title": "Assassinated Presidents Archives - History",
#   "id": "https://www.historyonthenet.com/category/assassinated-presidents",
#   "url": "https://www.historyonthenet.com/category/assassinated-presidents",
#   "publishedDate": "2019-01-01T00:00:00.000Z",
#   "author": "None",
#   "text": "<div><div> <div> <p>Scroll down to see articles about the U.S. presidents who died in office, and the backgrounds and motivations of their assassins</p> </div> <p>Scroll down to see articles about the U.S. presidents who died in office, and the backgrounds and motivations of their assassins</p> <hr /> <article> <a href=\"https://www.historyonthenet.com/oscar-ramiro-ortega-hernandez\"> </a> <h2><a href=\"https://www.historyonthenet.com/oscar-ramiro-ortega-hernandez\">Oscar Ramiro Ortega-Hernandez</a></h2> <p>The following article on Oscar Ramiro Ortega-Hernandez is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble. Perhaps the most dangerous threat to President Obama’s life came from twenty-one-year-old Oscar Ramiro Ortega-Hernandez, who had criminal…</p> </article> <article> <a href=\"https://www.historyonthenet.com/copycat-killers\"> </a> <h2><a href=\"https://www.historyonthenet.com/copycat-killers\">Copycat Killers: Becoming Famous by Becoming Infamous</a></h2> <p>The following article on copycat killers is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble. Many assassins and would-be assassins of U.S. presidents were copycat killers obsessed with assassins from the past. Some borrowed…</p> </article> <article> <a href=\"https://www.historyonthenet.com/assassinated-presidents\"> </a> <h2><a href=\"https://www.historyonthenet.com/assassinated-presidents\">Assassinated Presidents: Profiles of Them and Their Killers</a></h2> <p>The following article on assassinated presidents is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble. The list of assassinated presidents receives a new member approximately every 20-40 years. Here are those who were killed while…</p> </article> <article> <a href=\"https://www.historyonthenet.com/isaac-aguigui\"> </a> <h2><a href=\"https://www.historyonthenet.com/isaac-aguigui\">Isaac Aguigui: Militia Leader, Wannabe Presidential Assassin</a></h2> <p>The following article on Isaac Aguigui is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble. In 2012, Barack Obama was targeted by a domestic terror group of soldiers called F.E.A.R. (“Forever Enduring Always Read”),…</p> </article> <article> <a href=\"https://www.historyonthenet.com/khalid-kelly\"> </a> <h2><a href=\"https://www.historyonthenet.com/khalid-kelly\">Khalid Kelly: Irish Would-Be Obama Assassin</a></h2> <p>The following article on Terry Kelly (\"Khalid Kelly\") is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble. In May 2011, Irish Muslim militant Terry “Khalid” Kelly was arrested for threatening to assassinate President…</p> </article> <article> <a href=\"https://www.historyonthenet.com/timothy-ryan-gutierrez-hacker-threatened-obama\"> </a> <h2><a href=\"https://www.historyonthenet.com/timothy-ryan-g",
#   "highlights": [
#     "The list of assassinated presidents receives a new member approximately every 20-40 years. Here are those who were killed while…      Isaac Aguigui: Militia Leader, Wannabe Presidential Assassin  The following article on Isaac Aguigui is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble. In 2012, Barack Obama was targeted by a domestic terror group of soldiers called F.E.A.R.",
#     "In 2012, Barack Obama was targeted by a domestic terror group of soldiers called F.E.A.R. (“Forever Enduring Always Read”),…      Khalid Kelly: Irish Would-Be Obama Assassin  The following article on Terry Kelly (\"Khalid Kelly\") is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble. In May 2011, Irish Muslim militant Terry “Khalid” Kelly was arrested for threatening to assassinate President…",
#     "Here are those who were killed while…      Isaac Aguigui: Militia Leader, Wannabe Presidential Assassin  The following article on Isaac Aguigui is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble. In 2012, Barack Obama was targeted by a domestic terror group of soldiers called F.E.A.R. (“Forever Enduring Always Read”),…      Khalid Kelly: Irish Would-Be Obama Assassin  The following article on Terry Kelly (\"Khalid Kelly\") is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama.",
#     "It is available for order now from Amazon and Barnes &amp; Noble. The list of assassinated presidents receives a new member approximately every 20-40 years. Here are those who were killed while…      Isaac Aguigui: Militia Leader, Wannabe Presidential Assassin  The following article on Isaac Aguigui is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble.",
#     "In May 2011, Irish Muslim militant Terry “Khalid” Kelly was arrested for threatening to assassinate President…",
#     "Perhaps the most dangerous threat to President Obama’s life came from twenty-one-year-old Oscar Ramiro Ortega-Hernandez, who had criminal…      Copycat Killers: Becoming Famous by Becoming Infamous  The following article on copycat killers is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble. Many assassins and would-be assassins of U.S. presidents were copycat killers obsessed with assassins from the past. Some borrowed…      Assassinated Presidents: Profiles of Them and Their Killers  The following article on assassinated presidents is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama.",
#     "It is available for order now from Amazon and Barnes &amp; Noble. In 2012, Barack Obama was targeted by a domestic terror group of soldiers called F.E.A.R. (“Forever Enduring Always Read”),…      Khalid Kelly: Irish Would-Be Obama Assassin  The following article on Terry Kelly (\"Khalid Kelly\") is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble.",
#     "Many assassins and would-be assassins of U.S. presidents were copycat killers obsessed with assassins from the past. Some borrowed…      Assassinated Presidents: Profiles of Them and Their Killers  The following article on assassinated presidents is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble. The list of assassinated presidents receives a new member approximately every 20-40 years.",
#     "It is available for order now from Amazon and Barnes &amp; Noble. Many assassins and would-be assassins of U.S. presidents were copycat killers obsessed with assassins from the past. Some borrowed…      Assassinated Presidents: Profiles of Them and Their Killers  The following article on assassinated presidents is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble.",
#     "Some borrowed…      Assassinated Presidents: Profiles of Them and Their Killers  The following article on assassinated presidents is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama. It is available for order now from Amazon and Barnes &amp; Noble. The list of assassinated presidents receives a new member approximately every 20-40 years. Here are those who were killed while…      Isaac Aguigui: Militia Leader, Wannabe Presidential Assassin  The following article on Isaac Aguigui is an excerpt from Mel Ayton's Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama."
#   ],
#   "highlightScores": [
#     0.10017715394496918,
#     0.0916084423661232,
#     0.08568621426820755,
#     0.08466880768537521,
#     0.08453669399023056,
#     0.08243578672409058,
#     0.08049978315830231,
#     0.08013768494129181,
#     0.0784364566206932,
#     0.07647785544395447
#   ],
#   "summary": "This webpage is a collection of articles about U.S. presidents who died in office, and the backgrounds and motivations of their assassins. It includes excerpts from the book \"Hunting the President: Threats, Plots, and Assassination Attempts—From FDR to Obama\" by Mel Ayton. The articles cover a variety of topics, including the assassination attempt on President Obama by Oscar Ramiro Ortega-Hernandez, the motivations of copycat killers, and the various individuals who have attempted to assassinate U.S. presidents throughout history. \n"
# }
