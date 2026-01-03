import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pendulum
import pytest

from forecasting_tools.helpers.asknews_cache import AskNewsCache

"""
NOTE: AI completely generated this file with no human review.
If something looks stupid or unnecessary in here feel free to delete it.
"""


class TestAskNewsCacheInitialization:
    def test_creates_directory_when_cache_mode_is_use_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "new_cache_dir"
            assert not cache_dir.exists()

            AskNewsCache(cache_mode="use_cache", cache_directory=cache_dir)

            assert cache_dir.exists()

    def test_creates_directory_when_cache_mode_is_use_cache_with_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "new_cache_dir"
            assert not cache_dir.exists()

            AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=cache_dir
            )

            assert cache_dir.exists()

    def test_does_not_create_directory_when_cache_mode_is_no_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "new_cache_dir"
            assert not cache_dir.exists()

            AskNewsCache(cache_mode="no_cache", cache_directory=cache_dir)

            assert not cache_dir.exists()

    def test_uses_custom_cache_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "custom_dir"
            cache = AskNewsCache(cache_mode="use_cache", cache_directory=cache_dir)

            assert cache._cache_directory == cache_dir


class TestGetCacheKey:
    def test_returns_consistent_key_for_same_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "test query"

            key1 = cache._get_cache_key(query)
            key2 = cache._get_cache_key(query)

            assert key1 == key2

    def test_returns_different_keys_for_different_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )

            key1 = cache._get_cache_key("query one")
            key2 = cache._get_cache_key("query two")

            assert key1 != key2

    def test_key_includes_date_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            today = pendulum.now(tz="UTC").format("YYYY-MM-DD")

            key = cache._get_cache_key("test query")

            assert key.startswith(today)

    def test_key_changes_on_different_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )

            with patch(
                "forecasting_tools.helpers.asknews_cache.pendulum"
            ) as mock_pendulum:
                mock_pendulum.now.return_value.format.return_value = "2025-01-01"
                key1 = cache._get_cache_key("test query")

                mock_pendulum.now.return_value.format.return_value = "2025-01-02"
                key2 = cache._get_cache_key("test query")

            assert key1 != key2


class TestGet:
    def test_returns_none_when_cache_mode_is_no_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(cache_mode="no_cache", cache_directory=Path(tmpdir))

            result = cache.get("any query")

            assert result is None

    def test_returns_cached_result_when_valid_cache_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "test query"
            expected_result = "cached result"
            cache.set(query, expected_result)

            result = cache.get(query)

            assert result == expected_result

    def test_raises_error_when_cache_missing_and_mode_is_use_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(cache_mode="use_cache", cache_directory=Path(tmpdir))

            with pytest.raises(ValueError):
                cache.get("nonexistent query")

    def test_returns_none_when_cache_missing_and_mode_is_use_cache_with_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )

            result = cache.get("nonexistent query")

            assert result is None

    def test_raises_error_when_cache_expired_and_mode_is_use_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(cache_mode="use_cache", cache_directory=Path(tmpdir))
            query = "test query"

            cache_key = cache._get_cache_key(query)
            cache_file = cache._get_cache_filepath(cache_key)
            expired_timestamp = pendulum.now(tz="UTC").subtract(hours=25)
            cache_data = {
                "query": query,
                "result": "old result",
                "timestamp": expired_timestamp.to_iso8601_string(),
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            with pytest.raises(ValueError):
                cache.get(query)

            assert not cache_file.exists()

    def test_returns_none_when_cache_expired_and_mode_is_use_cache_with_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "test query"

            cache_key = cache._get_cache_key(query)
            cache_file = cache._get_cache_filepath(cache_key)
            expired_timestamp = pendulum.now(tz="UTC").subtract(hours=25)
            cache_data = {
                "query": query,
                "result": "old result",
                "timestamp": expired_timestamp.to_iso8601_string(),
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            result = cache.get(query)

            assert result is None
            assert not cache_file.exists()

    def test_raises_error_on_corrupted_json_when_mode_is_use_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(cache_mode="use_cache", cache_directory=Path(tmpdir))
            query = "test query"

            cache_key = cache._get_cache_key(query)
            cache_file = cache._get_cache_filepath(cache_key)
            with open(cache_file, "w") as f:
                f.write("not valid json{{{")

            with pytest.raises(ValueError):
                cache.get(query)

    def test_returns_none_on_corrupted_json_when_mode_is_use_cache_with_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "test query"

            cache_key = cache._get_cache_key(query)
            cache_file = cache._get_cache_filepath(cache_key)
            with open(cache_file, "w") as f:
                f.write("not valid json{{{")

            result = cache.get(query)

            assert result is None

    def test_raises_error_on_missing_keys_when_mode_is_use_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(cache_mode="use_cache", cache_directory=Path(tmpdir))
            query = "test query"

            cache_key = cache._get_cache_key(query)
            cache_file = cache._get_cache_filepath(cache_key)
            with open(cache_file, "w") as f:
                json.dump({"query": query}, f)

            with pytest.raises(ValueError):
                cache.get(query)

    def test_returns_none_on_missing_keys_when_mode_is_use_cache_with_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "test query"

            cache_key = cache._get_cache_key(query)
            cache_file = cache._get_cache_filepath(cache_key)
            with open(cache_file, "w") as f:
                json.dump({"query": query}, f)

            result = cache.get(query)

            assert result is None

    def test_returns_result_when_cache_not_yet_expired(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "test query"
            expected_result = "still valid"

            cache_key = cache._get_cache_key(query)
            cache_file = cache._get_cache_filepath(cache_key)
            not_expired_timestamp = pendulum.now(tz="UTC").subtract(hours=23)
            cache_data = {
                "query": query,
                "result": expected_result,
                "timestamp": not_expired_timestamp.to_iso8601_string(),
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            result = cache.get(query)

            assert result == expected_result


class TestSet:
    def test_does_nothing_when_cache_mode_is_no_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache = AskNewsCache(cache_mode="no_cache", cache_directory=cache_dir)

            cache.set("query", "result")

            assert not cache_dir.exists()

    def test_writes_cache_file_with_correct_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "test query"
            result = "test result"

            cache.set(query, result)

            cache_key = cache._get_cache_key(query)
            cache_file = cache._get_cache_filepath(cache_key)
            assert cache_file.exists()

            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            assert cache_data["query"] == query
            assert cache_data["result"] == result
            assert "timestamp" in cache_data

    def test_overwrites_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "test query"

            cache.set(query, "first result")
            cache.set(query, "second result")

            result = cache.get(query)
            assert result == "second result"

    def test_set_creates_retrievable_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "integration test query"
            expected_result = "integration test result"

            cache.set(query, expected_result)
            retrieved_result = cache.get(query)

            assert retrieved_result == expected_result


class TestClearExpiredEntries:
    def test_returns_zero_when_cache_mode_is_no_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(cache_mode="no_cache", cache_directory=Path(tmpdir))

            count = cache.clear_expired_entries()

            assert count == 0

    def test_removes_expired_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=cache_dir
            )

            expired_file = cache_dir / "expired_entry.jsonl"
            expired_timestamp = pendulum.now(tz="UTC").subtract(hours=25)
            with open(expired_file, "w") as f:
                json.dump(
                    {
                        "query": "expired",
                        "result": "old",
                        "timestamp": expired_timestamp.to_iso8601_string(),
                    },
                    f,
                )

            count = cache.clear_expired_entries()

            assert count == 1
            assert not expired_file.exists()

    def test_keeps_valid_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=cache_dir
            )

            valid_file = cache_dir / "valid_entry.jsonl"
            valid_timestamp = pendulum.now(tz="UTC").subtract(hours=12)
            with open(valid_file, "w") as f:
                json.dump(
                    {
                        "query": "valid",
                        "result": "still good",
                        "timestamp": valid_timestamp.to_iso8601_string(),
                    },
                    f,
                )

            count = cache.clear_expired_entries()

            assert count == 0
            assert valid_file.exists()

    def test_removes_corrupted_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=cache_dir
            )

            corrupted_file = cache_dir / "corrupted.jsonl"
            with open(corrupted_file, "w") as f:
                f.write("not valid json{{{")

            count = cache.clear_expired_entries()

            assert count == 1
            assert not corrupted_file.exists()

    def test_removes_files_with_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=cache_dir
            )

            invalid_file = cache_dir / "missing_keys.jsonl"
            with open(invalid_file, "w") as f:
                json.dump({"query": "no timestamp"}, f)

            count = cache.clear_expired_entries()

            assert count == 1
            assert not invalid_file.exists()

    def test_returns_correct_count_with_mixed_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=cache_dir
            )

            valid_timestamp = pendulum.now(tz="UTC").subtract(hours=12)
            valid_file = cache_dir / "valid.jsonl"
            with open(valid_file, "w") as f:
                json.dump(
                    {
                        "query": "valid",
                        "result": "ok",
                        "timestamp": valid_timestamp.to_iso8601_string(),
                    },
                    f,
                )

            expired_timestamp = pendulum.now(tz="UTC").subtract(hours=30)
            expired_file = cache_dir / "expired.jsonl"
            with open(expired_file, "w") as f:
                json.dump(
                    {
                        "query": "expired",
                        "result": "old",
                        "timestamp": expired_timestamp.to_iso8601_string(),
                    },
                    f,
                )

            corrupted_file = cache_dir / "corrupted.jsonl"
            with open(corrupted_file, "w") as f:
                f.write("invalid")

            count = cache.clear_expired_entries()

            assert count == 2
            assert valid_file.exists()
            assert not expired_file.exists()
            assert not corrupted_file.exists()


class TestClearAll:
    def test_removes_all_cache_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=cache_dir
            )

            for i in range(3):
                cache.set(f"query{i}", f"result{i}")

            json_files_before = list(cache_dir.glob("*.jsonl"))
            assert len(json_files_before) == 3

            count = cache.clear_all()

            json_files_after = list(cache_dir.glob("*.jsonl"))
            assert count == 3
            assert len(json_files_after) == 0

    def test_returns_correct_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=cache_dir
            )

            for i in range(5):
                (cache_dir / f"file{i}.jsonl").touch()

            count = cache.clear_all()

            assert count == 5

    def test_returns_zero_when_no_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )

            count = cache.clear_all()

            assert count == 0

    def test_does_not_remove_non_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=cache_dir
            )

            non_json_file = cache_dir / "other.txt"
            non_json_file.touch()
            json_file = cache_dir / "cache.jsonl"
            json_file.touch()

            count = cache.clear_all()

            assert count == 1
            assert non_json_file.exists()
            assert not json_file.exists()


class TestCacheExpiryBoundary:
    def test_cache_valid_just_under_24_hours(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "boundary test"
            expected_result = "still valid at boundary"

            cache_key = cache._get_cache_key(query)
            cache_file = cache._get_cache_filepath(cache_key)
            boundary_timestamp = pendulum.now(tz="UTC").subtract(hours=23, minutes=59)
            cache_data = {
                "query": query,
                "result": expected_result,
                "timestamp": boundary_timestamp.to_iso8601_string(),
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            result = cache.get(query)

            assert result == expected_result

    def test_cache_expired_just_past_24_hours(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "boundary test expired"

            cache_key = cache._get_cache_key(query)
            cache_file = cache._get_cache_filepath(cache_key)
            expired_timestamp = pendulum.now(tz="UTC").subtract(hours=24, minutes=1)
            cache_data = {
                "query": query,
                "result": "should be expired",
                "timestamp": expired_timestamp.to_iso8601_string(),
            }
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            result = cache.get(query)

            assert result is None
            assert not cache_file.exists()


class TestEmptyStrings:
    def test_can_cache_empty_string_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "query with empty result"

            cache.set(query, "")
            result = cache.get(query)

            assert result == ""

    def test_can_cache_with_empty_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            expected_result = "result for empty query"

            cache.set("", expected_result)
            result = cache.get("")

            assert result == expected_result


class TestSpecialCharacters:
    def test_handles_query_with_special_characters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "test query with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?\n\t"
            expected_result = "result with unicode: 日本語 🎉"

            cache.set(query, expected_result)
            result = cache.get(query)

            assert result == expected_result

    def test_handles_very_long_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AskNewsCache(
                cache_mode="use_cache_with_fallback", cache_directory=Path(tmpdir)
            )
            query = "x" * 10000
            expected_result = "result for long query"

            cache.set(query, expected_result)
            result = cache.get(query)

            assert result == expected_result
