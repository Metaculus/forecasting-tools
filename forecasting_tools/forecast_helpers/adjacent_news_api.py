from __future__ import annotations

import json
import logging
import os
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
    status: Literal["open"]
    probability_at_access_time: float | None = None
    num_forecasters: int | None = None
    liquidity: float | None = None
    platform: str
    market_id: str
    market_type: str
    end_date: datetime | None = None
    created_at: datetime | None = None
    volume: float | None = None
    link: str | None = None
    date_accessed: datetime = Field(default_factory=datetime.now)
    api_json: dict = Field(
        description="The API JSON response used to create the market",
        default_factory=dict,
    )

    @classmethod
    def from_adjacent_api_json(cls, api_json: dict) -> AdjacentQuestion:
        # Parse datetime fields
        end_date = cls._parse_api_date(api_json.get("end_date"))
        created_at = cls._parse_api_date(api_json.get("created_at"))

        # Map API fields to our model fields
        return cls(
            question_text=api_json.get("question", ""),
            description=api_json.get("description"),
            rules=api_json.get("rules"),
            status=api_json.get("status", ""),
            probability_at_access_time=api_json.get("probability"),
            num_forecasters=api_json.get("trades_count"),
            liquidity=api_json.get("liquidity"),
            platform=api_json.get("platform", ""),
            market_id=api_json.get("market_id", ""),
            market_type=api_json.get("market_type", ""),
            end_date=end_date,
            created_at=created_at,
            volume=api_json.get("volume"),
            link=api_json.get("link"),
            api_json=api_json,
        )

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


class AdjacentNewsApi:
    """
    API wrapper for Adjacent News prediction market data.
    Documentation: https://docs.adj.news/
    """

    API_BASE_URL = "https://api.data.adj.news"
    MAX_MARKETS_PER_REQUEST = 100

    @classmethod
    def get_questions_matching_filter(
        cls,
        api_filter: AdjacentFilter,
        num_questions: int | None = None,
        error_if_market_target_missed: bool = True,
    ) -> list[AdjacentQuestion]:
        if num_questions is not None:
            assert num_questions > 0, "Must request at least one market"

        markets = cls._filter_sequential_strategy(api_filter, num_questions)

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

        logger.info(
            f"Returning {len(markets)} markets matching the Adjacent News API filter"
        )
        return markets

    @classmethod
    def get_question_by_id(cls, market_id: str) -> AdjacentQuestion:
        logger.info(f"Retrieving market details for market {market_id}")
        url = f"{cls.API_BASE_URL}/api/markets/{market_id}"
        auth_headers = cls._get_auth_headers()
        response = requests.get(url, headers=auth_headers["headers"])
        raise_for_status_with_additional_info(response)
        json_market = json.loads(response.content)
        market = AdjacentQuestion.from_adjacent_api_json(json_market)
        logger.info(f"Retrieved market details for market {market_id}")
        return market

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
    def _get_markets_from_api(
        cls, params: dict[str, Any]
    ) -> list[AdjacentQuestion]:
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
        return markets

    @classmethod
    def _filter_sequential_strategy(
        cls, api_filter: AdjacentFilter, num_markets: int | None
    ) -> list[AdjacentQuestion]:
        if num_markets is None:
            markets = cls._grab_filtered_markets_with_offset(api_filter, 0)
            return markets

        markets: list[AdjacentQuestion] = []
        more_markets_available = True
        page_num = 0

        while len(markets) < num_markets and more_markets_available:
            offset = page_num * cls.MAX_MARKETS_PER_REQUEST
            new_markets = cls._grab_filtered_markets_with_offset(
                api_filter, offset
            )

            if not new_markets:
                more_markets_available = False
            else:
                markets.extend(new_markets)

            page_num += 1

        return markets[:num_markets]

    @classmethod
    def _grab_filtered_markets_with_offset(
        cls,
        api_filter: AdjacentFilter,
        offset: int = 0,
    ) -> list[AdjacentQuestion]:
        url_params: dict[str, Any] = {
            "limit": cls.MAX_MARKETS_PER_REQUEST,
            "offset": offset,
            "sort_by": "updated_at",
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

        markets = cls._get_markets_from_api(url_params)

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

        return markets
