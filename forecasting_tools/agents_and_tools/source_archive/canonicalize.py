"""Canonicalize URLs so trivially-different links collapse to one dedup key.

Every capture of a page is grouped under ``url_hash`` (see :mod:`models`).
Historically that hashed the *raw* URL string, so ``…/x``, ``…/x/``,
``…/x?utm_source=…`` and ``…/x#frag`` were four different "sources" — inflating
both storage and any "how many sources have we covered" count.

This module normalizes away differences that do **not** change *which page* you
get, so the dedup key is stable across those variants:

  - lowercase scheme + host, strip a default port (``:80`` / ``:443``)
  - drop the fragment (``#…``)
  - drop known analytics / click-tracking query params, then sort the rest
  - normalize a trailing slash (``…/x/`` -> ``…/x``; root collapses to no path)

It is deliberately conservative. It does **not** upgrade ``http`` -> ``https`` or
strip ``www.``: those can resolve to genuinely different pages on some hosts, so
collapsing them belongs to a later, opt-in phase (see ``ROADMAP.md``).
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Query params that are analytics/click tracking and never select the page.
# Matched case-insensitively; any key starting with a prefix below is also
# dropped. Bare ``ref`` / ``source`` are intentionally left alone — they are too
# often load-bearing (API refs, content selectors) to drop blindly.
_TRACKING_PARAMS = frozenset(
    {
        "gclid",
        "gclsrc",
        "dclid",
        "gbraid",
        "wbraid",
        "fbclid",
        "msclkid",
        "yclid",
        "twclid",
        "mc_eid",
        "mc_cid",
        "_hsenc",
        "_hsmi",
        "igshid",
        "igsh",
        "vero_id",
        "vero_conv",
        "oly_anon_id",
        "oly_enc_id",
        "spm",
        "scm",
        "ref_src",
        "ref_url",
    }
)
_TRACKING_PREFIXES = ("utm_",)

_DEFAULT_PORTS = {"http": "80", "https": "443"}


def _is_tracking(key: str) -> bool:
    k = key.lower()
    return k in _TRACKING_PARAMS or any(k.startswith(p) for p in _TRACKING_PREFIXES)


def canonicalize_url(url: str) -> str:
    """Return a normalized form of ``url`` to use as a dedup key.

    Idempotent — ``canonicalize_url(canonicalize_url(u)) == canonicalize_url(u)``.
    Non-http(s) or unparsable input is returned stripped but otherwise as-is
    (e.g. ``mailto:``, relative paths), so callers can pass anything safely.
    """
    if not url:
        return url
    raw = url.strip()
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return raw

    scheme = parts.scheme.lower()

    # netloc: lowercase host (bracket IPv6), keep userinfo, strip default port.
    host = (parts.hostname or "").lower()
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    netloc = host
    if parts.username is not None:
        auth = parts.username
        if parts.password is not None:
            auth += f":{parts.password}"
        netloc = f"{auth}@{netloc}"
    if parts.port is not None and str(parts.port) != _DEFAULT_PORTS.get(scheme):
        netloc += f":{parts.port}"

    # path: collapse the bare root to empty; drop a trailing slash otherwise.
    path = parts.path
    if path in ("", "/"):
        path = ""
    elif path.endswith("/"):
        path = path.rstrip("/")

    # query: drop tracking params, then sort so order doesn't matter.
    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking(k)
    ]
    kept.sort()
    query = urlencode(kept)

    # fragment: always dropped.
    return urlunsplit((scheme, netloc, path, query, ""))
