from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import pendulum

from forecasting_tools.util.file_manipulation import add_to_jsonl_file, load_jsonl_file


class AskNewsCache:
    _cache_directory = Path.home() / ".cache" / "forecasting_tools" / "asknews"
    _cache_expiry_hours = 24

    def __init__(
        self,
        cache_mode: Literal[
            "use_cache", "use_cache_with_fallback", "no_cache"
        ] = "no_cache",
        cache_directory: Path | None = None,
    ) -> None:
        self.cache_mode = cache_mode
        if cache_directory:
            self._cache_directory = cache_directory

        if self.cache_mode != "no_cache":
            self._cache_directory.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, query: str) -> str:
        query_hash = hashlib.md5(query.encode()).hexdigest()
        today = pendulum.now(tz="UTC").format("YYYY-MM-DD")
        return f"{today}_{query_hash}"

    def _get_cache_filepath(self, cache_key: str) -> Path:
        return self._cache_directory / f"{cache_key}.jsonl"

    def get(self, query: str) -> str | None:
        if self.cache_mode == "no_cache":
            return None

        cache_key = self._get_cache_key(query)
        cache_file = self._get_cache_filepath(cache_key)

        if not cache_file.exists():
            if self.cache_mode == "use_cache":
                raise ValueError(
                    f"Cache mode is 'use_cache' but no cache found for query: {query[:100]}"
                )
            return None

        try:
            cache_data = load_jsonl_file(str(cache_file))
            assert len(cache_data) == 1, "Cache file should contain only one entry"
            cache_data = cache_data[0]

            cached_time = pendulum.parse(cache_data["timestamp"])
            assert isinstance(cached_time, pendulum.DateTime)
            cached_timestamp = cached_time.timestamp()
            current_timestamp = pendulum.now(tz="UTC").timestamp()
            hours_since_cache = (current_timestamp - cached_timestamp) / 3600

            if hours_since_cache > self._cache_expiry_hours:
                cache_file.unlink()
                if self.cache_mode == "use_cache":
                    raise ValueError(f"Cache expired for query: {query[:100]}")
                return None

            return cache_data["result"]
        except (json.JSONDecodeError, KeyError) as e:
            if self.cache_mode == "use_cache":
                raise ValueError(
                    f"Failed to read cache for query: {query[:100]}"
                ) from e
            return None

    def set(self, query: str, result: str) -> None:
        if self.cache_mode == "no_cache":
            return

        cache_key = self._get_cache_key(query)
        cache_file = self._get_cache_filepath(cache_key)

        if cache_file.exists():
            cache_file.unlink()

        cache_data = {
            "query": query,
            "result": result,
            "timestamp": pendulum.now(tz="UTC").to_iso8601_string(),
        }

        add_to_jsonl_file(str(cache_file), [cache_data])

    def clear_expired_entries(self) -> int:
        if self.cache_mode == "no_cache":
            return 0

        count = 0
        current_timestamp = pendulum.now(tz="UTC").timestamp()

        for cache_file in self._cache_directory.glob("*.jsonl"):
            try:
                cache_data = load_jsonl_file(str(cache_file))
                assert len(cache_data) == 1, "Cache file should contain only one entry"
                cache_data = cache_data[0]

                cached_time = pendulum.parse(cache_data["timestamp"])
                assert isinstance(cached_time, pendulum.DateTime)
                cached_timestamp = cached_time.timestamp()
                hours_since_cache = (current_timestamp - cached_timestamp) / 3600

                if hours_since_cache > self._cache_expiry_hours:
                    cache_file.unlink()
                    count += 1
            except (json.JSONDecodeError, KeyError, OSError):
                cache_file.unlink()
                count += 1

        return count

    def clear_all(self) -> int:
        count = 0
        for cache_file in self._cache_directory.glob("*.jsonl"):
            cache_file.unlink()
            count += 1
        return count
