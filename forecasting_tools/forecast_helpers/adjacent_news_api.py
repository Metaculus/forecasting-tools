from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field

from forecasting_tools.util.misc import raise_for_status_with_additional_info

logger = logging.getLogger(__name__)


class AdjacentQuestion(BaseModel):
    question_text: str
    description: str | None = None
    rules: str | None = None
    status: Literal[
        "active"
    ]  # TODO: Figure out what other statuses are possible
    probability_at_access_time: float | None = None
    num_forecasters: int | None = None
    liquidity: float | None = None
    platform: str
    market_id: str
    market_type: str
    category: str | None = None
    tags: list[str] | None = None
    end_date: datetime | None = None
    created_at: datetime | None = None
    resolution_date: datetime | None = None
    volume: float | None = None
    link: str | None = None
    date_accessed: datetime = Field(default_factory=datetime.now)
    comment_count: int | None = None
    api_json: dict = Field(
        description="The API JSON response used to create the market",
        default_factory=dict,
    )

    @classmethod
    def from_adjacent_api_json(cls, api_json: dict) -> AdjacentQuestion:
        # Parse datetime fields
        end_date = cls._parse_api_date(api_json.get("end_date"))
        created_at = cls._parse_api_date(api_json.get("created_at"))
        resolution_date = cls._parse_api_date(api_json.get("resolution_date"))

        # Map API fields to our model fields
        question = cls(
            question_text=api_json["question"],
            description=api_json.get("description"),
            rules=api_json["rules"],
            status=api_json.get("status", ""),
            probability_at_access_time=api_json.get("probability"),
            num_forecasters=api_json.get("trades_count"),
            liquidity=api_json.get("liquidity"),
            platform=api_json.get("platform", ""),
            market_id=api_json.get("market_id", ""),
            market_type=api_json.get("market_type", ""),
            category=api_json.get("category", ""),
            tags=api_json.get("tags", []),
            end_date=end_date,
            created_at=created_at,
            resolution_date=resolution_date,
            volume=api_json.get("volume"),
            link=api_json.get("link"),
            comment_count=api_json.get("comment_count"),
            api_json=api_json,
        )
        return question

    @classmethod
    def _parse_api_date(
        cls, date_value: str | float | None
    ) -> datetime | None:
        """Parse date from API response."""
        if date_value is None:
            return None

        if isinstance(date_value, float):
            return datetime.fromtimestamp(date_value)

        date_formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
        ]

        assert isinstance(date_value, str)
        for date_format in date_formats:
            try:
                return datetime.strptime(date_value, date_format)
            except ValueError:
                continue

        raise ValueError(f"Unable to parse date: {date_value}")


class AdjacentFilter(BaseModel):
    status: list[Literal["active", "resolved", "closed"]] | None = None
    liquidity_min: float | None = None
    liquidity_max: float | None = None
    num_forecasters_min: int | None = None
    end_date_after: datetime | None = None
    end_date_before: datetime | None = None
    platform: list[str] | None = None
    market_type: list[Literal["binary", "scalar", "categorical"]] | None = None
    keyword: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    volume_min: float | None = None
    volume_max: float | None = None
    include_closed: bool = False
    include_resolved: bool = False
    tag: str | None = None
    category: str | None = None


class AdjacentNewsApi:
    """
    API wrapper for Adjacent News prediction market data.
    Documentation: https://docs.adj.news/
    """

    API_BASE_URL = "https://api.data.adj.news"
    MAX_MARKETS_PER_REQUEST = 500

    @classmethod
    def get_questions_matching_filter(
        cls,
        api_filter: AdjacentFilter,
        num_questions: int | None = None,
        error_if_market_target_missed: bool = True,
        max_pages: int = 10,
    ) -> list[AdjacentQuestion]:
        if num_questions is not None:
            assert num_questions > 0, "Must request at least one market"

        markets = cls._walk_through_pagination(
            api_filter, num_questions, max_pages=max_pages
        )

        if (
            num_questions is not None
            and len(markets) != num_questions
            and error_if_market_target_missed
        ):
            raise ValueError(
                f"Requested number of markets ({num_questions}) does not match number of markets found ({len(markets)})"
            )

        if len(set(m.market_id for m in markets)) != len(markets):
            raise ValueError("Not all markets found are unique")

        if (
            num_questions
            and len(markets) != num_questions
            and error_if_market_target_missed
        ):
            raise ValueError(
                f"Requested number of markets ({num_questions}) does not match number of markets found ({len(markets)})"
            )

        logger.info(
            f"Returning {len(markets)} markets matching the Adjacent News API filter"
        )
        return markets

    @classmethod
    def _get_auth_headers(cls) -> dict[str, dict[str, str]]:
        ADJACENT_NEWS_API_KEY = os.getenv("ADJACENT_NEWS_API_KEY")
        if ADJACENT_NEWS_API_KEY is None:
            raise ValueError(
                "ADJACENT_NEWS_API_KEY environment variable not set"
            )
        return {
            "headers": {
                "Authorization": f"Bearer {ADJACENT_NEWS_API_KEY}",
                "Accept": "application/json",
            }
        }

    @classmethod
    def _walk_through_pagination(
        cls,
        api_filter: AdjacentFilter,
        num_markets: int | None,
        max_pages: int,
    ) -> list[AdjacentQuestion]:
        if num_markets is None:
            markets, _ = cls._grab_filtered_markets_with_offset(api_filter, 0)
            return markets

        markets: list[AdjacentQuestion] = []
        more_markets_available = True
        page_num = 0

        while (
            len(markets) < num_markets
            and more_markets_available
            and page_num < max_pages
        ):
            logger.info(
                f"Getting page {page_num} of markets. Found {len(markets)} markets so far."
            )
            offset = page_num * cls.MAX_MARKETS_PER_REQUEST
            new_markets, more_markets_available = (
                cls._grab_filtered_markets_with_offset(api_filter, offset)
            )
            markets.extend(new_markets)
            page_num += 1

        return markets[:num_markets]

    @classmethod
    def _grab_filtered_markets_with_offset(
        cls,
        api_filter: AdjacentFilter,
        offset: int = 0,
    ) -> tuple[list[AdjacentQuestion], bool]:
        url_params: dict[str, Any] = {
            "limit": cls.MAX_MARKETS_PER_REQUEST,
            "offset": offset,
            "sort_by": "created_at",
            "sort_dir": "desc",
        }

        # Apply API-level filters
        if api_filter.platform:
            url_params["platform"] = ",".join(api_filter.platform)

        if api_filter.status:
            url_params["status"] = ",".join(api_filter.status)
        elif not api_filter.include_closed and not api_filter.include_resolved:
            url_params["status"] = "active"

        if api_filter.market_type:
            url_params["market_type"] = ",".join(api_filter.market_type)

        if api_filter.keyword:
            url_params["keyword"] = api_filter.keyword

        if api_filter.created_after:
            url_params["created_after"] = api_filter.created_after.strftime(
                "%Y-%m-%d"
            )

        if api_filter.created_before:
            url_params["created_before"] = api_filter.created_before.strftime(
                "%Y-%m-%d"
            )

        if api_filter.include_closed:
            url_params["include_closed"] = "true"

        if api_filter.include_resolved:
            url_params["include_resolved"] = "true"

        if api_filter.category:
            url_params["category"] = api_filter.category

        if api_filter.tag:
            url_params["tag"] = api_filter.tag

        markets, more_markets_available = cls._get_markets_from_api(url_params)

        # Apply local filters that aren't supported by the API
        if api_filter.liquidity_min is not None:
            markets = [
                m
                for m in markets
                if m.liquidity is not None
                and m.liquidity >= api_filter.liquidity_min
            ]

        if api_filter.liquidity_max is not None:
            markets = [
                m
                for m in markets
                if m.liquidity is not None
                and m.liquidity <= api_filter.liquidity_max
            ]

        if api_filter.num_forecasters_min is not None:
            markets = [
                m
                for m in markets
                if m.num_forecasters is not None
                and m.num_forecasters >= api_filter.num_forecasters_min
            ]

        if api_filter.volume_min is not None:
            markets = [
                m
                for m in markets
                if m.volume is not None and m.volume >= api_filter.volume_min
            ]

        if api_filter.volume_max is not None:
            markets = [
                m
                for m in markets
                if m.volume is not None and m.volume <= api_filter.volume_max
            ]

        if api_filter.end_date_after:
            markets = [
                m
                for m in markets
                if m.end_date is not None
                and m.end_date >= api_filter.end_date_after
            ]

        if api_filter.end_date_before:
            markets = [
                m
                for m in markets
                if m.end_date is not None
                and m.end_date <= api_filter.end_date_before
            ]

        return markets, more_markets_available

    @classmethod
    def _get_markets_from_api(
        cls, params: dict[str, Any], sleep_time: float = 10
    ) -> tuple[list[AdjacentQuestion], bool]:
        num_requested = params.get("limit")
        assert (
            num_requested is None
            or num_requested <= cls.MAX_MARKETS_PER_REQUEST
        ), f"You cannot get more than {cls.MAX_MARKETS_PER_REQUEST} markets at a time"

        url = f"{cls.API_BASE_URL}/api/markets"
        auth_headers = cls._get_auth_headers()
        response = requests.get(
            url, params=params, headers=auth_headers["headers"]
        )
        raise_for_status_with_additional_info(response)
        data = json.loads(response.content)

        markets = []
        for market_data in data["data"]:
            markets.append(
                AdjacentQuestion.from_adjacent_api_json(market_data)
            )
        more_markets_available = data["meta"]["hasMore"]
        time.sleep(sleep_time)
        return markets, more_markets_available
