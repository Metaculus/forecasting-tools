from __future__ import annotations

from forecasting_tools.agents_and_tools.source_archive.ingest.metaculus_comments import (
    MetaculusCommentHarvester,
)


def _leaderboard():
    return {
        "leaderboard_entries": [
            {"user": {"id": 1, "username": "botA", "is_bot": True}},
            {"user": {"id": 2, "username": "human", "is_bot": False}},
            {"user": {"id": 3, "username": "botB", "is_bot": True}},
        ]
    }


def test_enumerate_bots_filters_non_bots():
    def fetch(path, params):
        assert path == "/leaderboards/project/123/"
        assert params["with_entries"] == "true"
        return _leaderboard()

    h = MetaculusCommentHarvester(fetch_json=fetch)
    bots = h.enumerate_bots(123)
    assert [b["id"] for b in bots] == [1, 3]


def test_harvest_author_builds_records_with_provenance():
    def fetch(path, params):
        assert path == "/comments/"
        if params["offset"] == 0:
            return {
                "results": [{"id": 10, "on_post": 555, "text": "src https://a.test/x"}]
            }
        return {"results": []}

    h = MetaculusCommentHarvester(fetch_json=fetch)
    records = h.harvest_author(1, run_id="r1", bot="botA")
    assert len(records) == 1
    rec = records[0]
    assert rec.url == "https://a.test/x"
    assert rec.bot == "botA"
    assert rec.run_id == "r1"
    assert rec.question_id == "555"
    assert rec.question_url == "https://www.metaculus.com/questions/555/"
    assert rec.trace == "comment:10"
    assert rec.origin == "metaculus_comment"


def test_iter_comments_paginates_until_short_page():
    calls = []

    def fetch(path, params):
        calls.append(params["offset"])
        if params["offset"] == 0:
            return {"results": [{"id": i, "text": ""} for i in range(100)]}
        return {"results": [{"id": 999, "text": ""}]}  # short page -> stop

    h = MetaculusCommentHarvester(fetch_json=fetch)
    comments = list(h.iter_comments(1))
    assert len(comments) == 101
    assert calls == [0, 100]


def test_harvest_project_aggregates_bots():
    def fetch(path, params):
        if path.startswith("/leaderboards/project/"):
            return _leaderboard()
        # one URL per bot, single page each
        if params["offset"] == 0:
            author = params["author"]
            return {
                "results": [
                    {"id": author, "on_post": 1, "text": f"https://bot{author}.test"}
                ]
            }
        return {"results": []}

    h = MetaculusCommentHarvester(fetch_json=fetch)
    records = h.harvest_project(123)
    assert {r.url for r in records} == {"https://bot1.test", "https://bot3.test"}
    assert {r.bot for r in records} == {"botA", "botB"}
    assert all(r.run_id == "metaculus-comments-123" for r in records)


def test_custom_base_url_drives_web_base():
    h = MetaculusCommentHarvester(
        base_url="https://example.org/api", fetch_json=lambda p, q: {"results": []}
    )
    assert h.web_base == "https://example.org"
