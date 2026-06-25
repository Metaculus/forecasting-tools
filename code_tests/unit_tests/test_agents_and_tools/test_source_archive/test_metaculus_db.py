from __future__ import annotations

from forecasting_tools.agents_and_tools.source_archive.ingest.metaculus_db import (
    LOCAL_DEFAULT_DSN,
    MetaculusDbHarvester,
    resolve_dsn,
)


def test_harvest_post_builds_records_with_provenance():
    rows = [
        {
            "comment_id": 1,
            "on_post_id": 42,
            "text": "see https://a.test/x and https://b.test/y",
            "username": "alpha",
            "author_id": 7,
        },
        {
            "comment_id": 2,
            "on_post_id": 42,
            "text": "https://a.test/x again",
            "username": "beta",
            "author_id": 8,
        },
    ]
    seen = {}

    def query(sql, params):
        seen["sql"], seen["params"] = sql, params
        return rows

    records = MetaculusDbHarvester(query).harvest_post(42)

    assert seen["params"] == (42,)
    assert {r.url for r in records} == {"https://a.test/x", "https://b.test/y"}
    r0 = next(r for r in records if r.url == "https://a.test/x")
    assert r0.origin == "metaculus_comment"
    assert r0.question_id == "42"
    assert r0.question_url == "https://www.metaculus.com/questions/42/"
    assert r0.bot in ("alpha", "beta")
    # one record per (URL, comment): a.test/x is cited in both comments
    assert sum(r.url == "https://a.test/x" for r in records) == 2


def test_harvest_recent_passes_days_and_limit():
    seen = {}

    def query(sql, params):
        seen["sql"], seen["params"] = sql, params
        return []

    MetaculusDbHarvester(query).harvest_recent(days=3, limit=50)
    assert seen["params"] == (3, 50)
    assert "limit %s" in seen["sql"]


def test_harvest_recent_uncapped_by_default():
    seen = {}

    def query(sql, params):
        seen["sql"], seen["params"] = sql, params
        return []

    # A daily sweep wants every row from the latest day, not a 1000-row cap.
    MetaculusDbHarvester(query).harvest_recent(days=1)
    assert seen["params"] == (1,)
    assert "limit" not in seen["sql"].lower()


def test_includes_private_bot_comments_by_default():
    seen = {}

    def query(sql, params):
        seen["sql"] = sql
        return []

    # The day-behind replica's value is the now-private bot reasoning, so the
    # default read must NOT filter private rows out.
    MetaculusDbHarvester(query).harvest_recent(days=1)
    assert "is_private" not in seen["sql"]
    assert "u.is_bot" in seen["sql"]


def test_public_only_filters_private_comments():
    seen = {}

    def query(sql, params):
        seen["sql"] = sql
        return []

    MetaculusDbHarvester(query).harvest_post(42, include_private=False)
    assert "not c.is_private" in seen["sql"]


def test_resolve_dsn_prefers_explicit_then_env_then_keychain():
    # explicit flag wins over everything
    assert (
        resolve_dsn(
            "postgresql://flag",
            env={"METACULUS_DB_DSN": "postgresql://env"},
            keychain_reader=lambda: "postgresql://kc",
        )
        == "postgresql://flag"
    )
    # then the env var
    assert (
        resolve_dsn(
            None,
            env={"METACULUS_DB_DSN": "postgresql://env"},
            keychain_reader=lambda: "postgresql://kc",
        )
        == "postgresql://env"
    )
    # then the keychain (the private path)
    assert (
        resolve_dsn(None, env={}, keychain_reader=lambda: "postgresql://kc")
        == "postgresql://kc"
    )


def test_resolve_dsn_falls_back_to_local_default():
    # nothing configured and no keychain item -> local dev DB, not a crash
    assert resolve_dsn(None, env={}, keychain_reader=lambda: None) == LOCAL_DEFAULT_DSN
