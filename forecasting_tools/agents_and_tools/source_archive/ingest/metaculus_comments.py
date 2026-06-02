"""Harvest the URLs bots cite, from their public Metaculus comments.

Both first-party and third-party bots publish their reasoning — with the source
links they used — as comments on the questions they forecast. The public,
no-auth Metaculus API is therefore the one mechanism that works across *every*
bot on the platform, which is why this is the general ingestion path.

Flow:

  1. Enumerate the bots participating in a project (tournament) leaderboard.
  2. Page through each bot's comments.
  3. Extract the URLs from each comment and emit CitationRecords.

The result is a citation manifest you can feed straight to the capture pipeline.

Caveat: comments are length-truncated when posted, so a comment-harvested URL
list can be incomplete versus the bot's full research. For bots you control, an
instrumented trace gives a fuller list; this path is the universal baseline.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any, Callable

from forecasting_tools.agents_and_tools.source_archive.ingest.url_extraction import (
    extract_citation_records,
)
from forecasting_tools.agents_and_tools.source_archive.models import CitationRecord

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://www.metaculus.com/api"
PAGE_LIMIT = 100


def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


class MetaculusCommentHarvester:
    """Reads bot comments via the public Metaculus API.

    HTTP is injectable for testing: pass ``fetch_json=callable(path, params) ->
    dict`` to avoid real network calls.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        session: Any = None,
        timeout: int = 30,
        fetch_json: Callable[[str, dict], dict] | None = None,
    ):
        self.base_url = (
            base_url or os.environ.get("METACULUS_API_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.web_base = (
            self.base_url[:-4] if self.base_url.endswith("/api") else self.base_url
        )
        self.timeout = timeout
        self._session = session
        self._fetch_json = fetch_json

    # --- http --------------------------------------------------------------
    def _get(self, path: str, params: dict) -> dict:
        if self._fetch_json is not None:
            return self._fetch_json(path, params)
        try:
            import requests
        except ImportError as e:  # pragma: no cover - requests is a core dep
            raise ImportError("requests is required for comment harvesting") from e
        if self._session is None:
            self._session = requests.Session()
        resp = self._session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    # --- bots --------------------------------------------------------------
    def enumerate_bots(self, project_id: int | str) -> list[dict]:
        """Return the bot ``user`` records on a project's leaderboard."""
        data = self._get(
            f"/leaderboards/project/{project_id}/", {"with_entries": "true"}
        )
        entries = _first(data, "leaderboard_entries", "entries", "results", default=[])
        bots: list[dict] = []
        seen: set[Any] = set()
        for entry in entries:
            user = entry.get("user") if isinstance(entry, dict) else None
            if not user or not user.get("is_bot"):
                continue
            uid = user.get("id")
            if uid in seen:
                continue
            seen.add(uid)
            bots.append(user)
        return bots

    # --- comments ----------------------------------------------------------
    def iter_comments(
        self, author_id: int | str, post_id: int | str | None = None
    ) -> Iterator[dict]:
        """Yield every comment authored by ``author_id`` (optionally on one post)."""
        offset = 0
        while True:
            params = {"author": author_id, "limit": PAGE_LIMIT, "offset": offset}
            if post_id is not None:
                params["post"] = post_id
            data = self._get("/comments/", params)
            results = (
                _first(data, "results", default=[]) if isinstance(data, dict) else data
            )
            if not results:
                break
            yield from results
            if len(results) < PAGE_LIMIT:
                break
            offset += PAGE_LIMIT

    # --- harvesting --------------------------------------------------------
    def _records_from_comment(
        self, comment: dict, *, run_id: str | None, bot: str | None
    ) -> list[CitationRecord]:
        post_id = _first(comment, "on_post", "post", "post_id")
        post_id_str = str(post_id) if post_id is not None else None
        question_url = (
            f"{self.web_base}/questions/{post_id}/" if post_id is not None else None
        )
        comment_id = comment.get("id")
        return extract_citation_records(
            comment.get("text"),
            run_id=run_id,
            bot=bot,
            question_id=post_id_str,
            metaculus_id=post_id_str,
            question_url=question_url,
            trace=f"comment:{comment_id}" if comment_id is not None else None,
            origin="metaculus_comment",
        )

    def harvest_author(
        self,
        author_id: int | str,
        *,
        run_id: str | None = None,
        bot: str | None = None,
        post_id: int | str | None = None,
    ) -> list[CitationRecord]:
        """All citation records from one bot's comments."""
        records: list[CitationRecord] = []
        for comment in self.iter_comments(author_id, post_id=post_id):
            records.extend(self._records_from_comment(comment, run_id=run_id, bot=bot))
        return records

    def harvest_project(
        self, project_id: int | str, *, run_id: str | None = None
    ) -> list[CitationRecord]:
        """All citation records from every bot on a project's leaderboard.

        Records are kept per-citation (duplicates across bots are preserved as
        distinct provenance); the capture pipeline dedupes URLs before fetching.
        """
        run_id = run_id or f"metaculus-comments-{project_id}"
        records: list[CitationRecord] = []
        bots = self.enumerate_bots(project_id)
        logger.info("project %s: %d bot(s) on leaderboard", project_id, len(bots))
        for user in bots:
            bot_name = user.get("username") or str(user.get("id"))
            bot_records = self.harvest_author(user["id"], run_id=run_id, bot=bot_name)
            logger.info("  bot %s: %d cited URL(s)", bot_name, len(bot_records))
            records.extend(bot_records)
        return records
