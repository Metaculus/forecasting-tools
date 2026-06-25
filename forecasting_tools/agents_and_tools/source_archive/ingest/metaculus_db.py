"""Read a bot's cited URLs from the platform Postgres database (operator tooling).

For operators with database access, this reads the URLs a forecasting bot cited
straight from Postgres (``comments_comment`` joined to ``users_user.is_bot``) and
emits the same :class:`CitationRecord`s as every other ingestion path, so the
catalog / coverage / capture stages downstream are unchanged. By default it reads
all of a bot's comments (``include_private=True``); pass ``include_private=False``
for the public ones only. Only ``u.is_bot`` accounts are ever read.

The DB call is **injected** (``query``) so the core is driver-agnostic and unit
testable; :meth:`from_dsn` wires a psycopg2 connection for real use (a libpq DSN
or a ``postgresql://…`` URL — e.g. a Neon connection string). Reads only; no
secrets are stored — the DSN comes from the caller / environment.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping, Sequence

from forecasting_tools.agents_and_tools.source_archive.ingest.url_extraction import (
    extract_citation_records,
)
from forecasting_tools.agents_and_tools.source_archive.models import CitationRecord

QueryFn = Callable[[str, Sequence[Any]], list[dict]]

# Keychain service name the DSN is stored under (see resolve_dsn / README).
KEYCHAIN_SERVICE = "metaculus-db-dsn"
LOCAL_DEFAULT_DSN = "dbname=metaculus"

_WEB = "https://www.metaculus.com"

# The windowed/post-scoped comment set is computed in a MATERIALIZED CTE so
# Postgres evaluates it FIRST, then joins users_user by primary key. Without the
# CTE the planner's stale stats misjudge the date window at ~300k rows (it is
# really ~2k/day) and pick a join order that times out on the remote pooler.
_OUTER = (
    "select r.id as comment_id, r.on_post_id, r.text, "
    "u.username, r.author_id "
    "from recent r join users_user u on u.id = r.author_id where u.is_bot"
)


def _recent_cte(scope: str, include_private: bool) -> str:
    """A MATERIALIZED ``recent`` CTE of link-bearing, non-deleted comments.

    ``scope`` is the row-narrowing predicate (a post id or a created_at window).
    Private comments are included unless ``include_private`` is False.

    ``strpos(text,'http') > 0`` is a cheap substring pre-filter (a regex `~` scan
    times out on the pooler; ``like`` would need ``%%`` escaping under psycopg2).
    The real URL parsing happens in extract_citation_records, so over-matching
    here just costs a few empty rows.
    """
    clauses = ["not c.is_soft_deleted", "strpos(c.text, 'http') > 0", scope]
    if not include_private:
        clauses.append("not c.is_private")
    where = " and ".join(clauses)
    return (
        "with recent as materialized ("
        "select c.id, c.on_post_id, c.text, c.author_id, c.created_at "
        f"from comments_comment c where {where}) "
    )


def _dsn_from_keychain(service: str = KEYCHAIN_SERVICE) -> str | None:
    """Read the DSN from the macOS login Keychain, or ``None`` if unavailable.

    Uses ``security find-generic-password -w`` so the credential lives only in
    the Keychain — never in ``.env``, a shell rc, or shell history. If the
    Keychain item's ACL is set to confirm on access, this call raises a GUI
    prompt: a human can approve it, an automated agent driving the shell cannot.
    Returns ``None`` off macOS or when the item is absent / access is denied, so
    callers fall through to the next source.
    """
    import shutil
    import subprocess

    if not shutil.which("security"):  # not macOS
        return None
    try:
        proc = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def resolve_dsn(
    explicit: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    keychain_reader: Callable[[], str | None] | None = None,
) -> str:
    """Resolve the DB DSN without ever persisting it to disk.

    Resolution order, first hit wins:
      1. ``explicit`` (e.g. a ``--dsn`` flag — convenient, but lands in shell
         history, so prefer the Keychain for the real secret),
      2. ``$METACULUS_DB_DSN``,
      3. the macOS Keychain item ``metaculus-db-dsn`` (the private path),
      4. the local default ``dbname=metaculus``.
    ``env`` / ``keychain_reader`` are injectable for tests.
    """
    if explicit:
        return explicit
    environ = env if env is not None else os.environ
    from_env = environ.get("METACULUS_DB_DSN")
    if from_env:
        return from_env
    reader = keychain_reader or _dsn_from_keychain
    from_keychain = reader()
    if from_keychain:
        return from_keychain
    return LOCAL_DEFAULT_DSN


class MetaculusDbHarvester:
    """Reads bot comments from Postgres. ``query(sql, params) -> list[dict]``."""

    def __init__(self, query: QueryFn):
        self._query = query

    @classmethod
    def from_dsn(cls, dsn: str = "dbname=metaculus") -> "MetaculusDbHarvester":
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError as e:  # pragma: no cover - optional operator dep
            raise ImportError(
                "psycopg2 is required for DB harvesting "
                "(`pip install psycopg2-binary`)."
            ) from e
        conn = psycopg2.connect(dsn)

        def query(sql: str, params: Sequence[Any]) -> list[dict]:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]

        return cls(query)

    def _records(self, rows: list[dict], run_id: str | None) -> list[CitationRecord]:
        out: list[CitationRecord] = []
        for r in rows:
            post_id = r.get("on_post_id")
            pid = str(post_id) if post_id is not None else None
            comment_id = r.get("comment_id")
            out.extend(
                extract_citation_records(
                    r.get("text"),
                    run_id=run_id,
                    bot=r.get("username") or str(r.get("author_id")),
                    question_id=pid,
                    metaculus_id=pid,
                    question_url=(
                        f"{_WEB}/questions/{post_id}/" if post_id is not None else None
                    ),
                    comment_id=str(comment_id) if comment_id is not None else None,
                    origin="metaculus_comment",
                )
            )
        return out

    def harvest_post(
        self,
        post_id: int | str,
        *,
        run_id: str | None = None,
        include_private: bool = True,
    ) -> list[CitationRecord]:
        """Every bot-cited URL in the comments on one post."""
        run_id = run_id or f"metaculus-db-post-{post_id}"
        sql = (
            _recent_cte("c.on_post_id = %s", include_private)
            + _OUTER
            + " order by r.created_at"
        )
        return self._records(self._query(sql, (post_id,)), run_id)

    def harvest_recent(
        self,
        *,
        days: int = 1,
        limit: int | None = None,
        run_id: str | None = None,
        include_private: bool = True,
    ) -> list[CitationRecord]:
        """Bot-cited URLs from the most recent ``days`` of comments.

        "Recent" is measured against ``max(created_at)`` in the table, not wall
        clock, so a replica that lags real time by a day still returns its latest
        day with ``days=1``. ``limit`` caps the row count; ``None`` (the default)
        is uncapped, which is what a daily sweep wants.
        """
        run_id = run_id or f"metaculus-db-recent-{days}d"
        scope = (
            "c.created_at >= "
            "(select max(created_at) from comments_comment) - (%s * interval '1 day')"
        )
        sql = (
            _recent_cte(scope, include_private) + _OUTER + " order by r.created_at desc"
        )
        params: list[Any] = [days]
        if limit:
            sql += " limit %s"
            params.append(limit)
        return self._records(self._query(sql, tuple(params)), run_id)
