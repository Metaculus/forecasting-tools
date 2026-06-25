"""Streamlit viewer for the source archive.

Browse what the capture pipeline stored in S3: pick a captured URL and see its
**screenshot, markdown, and HTML** side by side, with the question/bot it came
from. Reads provenance from the run manifests and resolves each URL's latest
capture through its per-URL index — no local file wrangling.

Run it::

    # uses the same env as the rest of the archive (WEB_ARCHIVE_S3_BUCKET, etc.)
    AWS_PROFILE=default WEB_ARCHIVE_S3_BUCKET=metaculus-web-archive \\
      streamlit run forecasting_tools/agents_and_tools/source_archive/viewer.py

Nothing here is deployment-specific: bucket/prefix/profile come from
``ArchiveConfig.from_env()``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# `streamlit run <file>` puts only the script's own directory on sys.path, not
# the repo root — so make `import forecasting_tools` work whether the package is
# pip-installed or just checked out. (viewer.py -> source_archive -> agents_and_tools
# -> forecasting_tools -> <repo root>.)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from forecasting_tools.agents_and_tools.source_archive.config import (  # noqa: E402
    ArchiveConfig,
)
from forecasting_tools.agents_and_tools.source_archive.models import (  # noqa: E402
    url_hash,
)

# --- S3 access (cached) ----------------------------------------------------


@st.cache_resource(show_spinner=False)
def _client(profile: str | None, region: str | None):
    import boto3

    return boto3.Session(
        profile_name=profile or None, region_name=region or None
    ).client("s3")


def _cfg() -> ArchiveConfig:
    return ArchiveConfig.from_env()


@st.cache_data(show_spinner=False)
def _list_keys(bucket: str, prefix: str) -> list[str]:
    cfg = _cfg()
    if cfg.local_dir:  # filesystem-backed archive — list matching files
        root = Path(cfg.local_dir)
        if not root.exists():
            return []
        return [
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and p.relative_to(root).as_posix().startswith(prefix)
        ]
    s3 = _client(cfg.aws_profile, cfg.aws_region)
    keys: list[str] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        keys.extend(o["Key"] for o in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


@st.cache_data(show_spinner=False)
def _get_bytes(bucket: str, key: str) -> bytes | None:
    cfg = _cfg()
    if cfg.local_dir:
        p = Path(cfg.local_dir) / key
        return p.read_bytes() if p.exists() else None
    s3 = _client(cfg.aws_profile, cfg.aws_region)
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
        return None


# Metaculus question id -> review URL. Derived at display time (not stored) so
# there's no redundant, drift-prone URL column in S3.
_METACULUS_QUESTION_BASE = "https://www.metaculus.com/questions/"


def _metaculus_url(metaculus_id) -> str:
    if metaculus_id in (None, "", "null"):
        return ""
    return f"{_METACULUS_QUESTION_BASE}{metaculus_id}/"


def _comment_url(metaculus_id, comment_id) -> str:
    """Deep-link to the specific comment the URL was cited in."""
    base = _metaculus_url(metaculus_id)
    if not base or comment_id in (None, "", "null"):
        return ""
    return f"{base}#comment-{comment_id}"


@st.cache_data(show_spinner="Loading manifests…")
def _manifest_rows(bucket: str, prefix: str) -> pd.DataFrame:
    """Every (question, bot, url) the bots cited, from the run manifests."""
    rows = []
    for key in _list_keys(bucket, f"{prefix}/manifests/"):
        body = _get_bytes(bucket, key)
        if not body:
            continue
        for line in body.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(
                {
                    "question": r.get("question_id") or "(none)",
                    "bot": r.get("bot") or "(none)",
                    "run_id": r.get("run_id") or "",
                    "origin": r.get("origin") or "",
                    "query": r.get("query") or "",
                    "metaculus": _metaculus_url(r.get("metaculus_id")),
                    "comment": _comment_url(r.get("metaculus_id"), r.get("comment_id")),
                    "url": r.get("url", ""),
                    "question_url": r.get("question_url") or "",
                    "tool_args": r.get("tool_args"),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        # Keep distinct provenance (a URL cited via two origins/runs = two rows).
        df = df.drop_duplicates(
            subset=["question", "bot", "run_id", "origin", "url"]
        ).reset_index(drop=True)
    return df


def _scrape_report(bucket: str, prefix: str, view: pd.DataFrame):
    """Per-question scraping cost: which backend captured each URL.

    Self-hosted Playwright is free; Firecrawl (the fallback) costs ~1 credit per
    page and is what actually accrues spend once a key is configured. We classify
    each *stored* capture by ``fetcher`` and count Firecrawl pages per question.

    Caveat: only successful captures are recorded in the index, so a Firecrawl
    attempt that failed its quality gate isn't counted here — billed attempts
    aren't yet persisted (see the note in the UI).
    """
    per_q: dict[str, dict] = {}
    for _, row in view.iterrows():
        cap = _index(bucket, prefix, row["url"])
        q = row["question"]
        agg = per_q.setdefault(
            q,
            {
                "question": q,
                "urls": 0,
                "captured": 0,
                "playwright": 0,
                "firecrawl": 0,
                "other": 0,
            },
        )
        agg["urls"] += 1
        if not cap:
            continue
        agg["captured"] += 1
        fetcher = (cap.get("fetcher") or "").lower()
        if fetcher in ("playwright", "firecrawl"):
            agg[fetcher] += 1
        else:
            agg["other"] += 1
    return per_q


@st.cache_data(show_spinner=False)
def _index(bucket: str, prefix: str, url: str) -> dict | None:
    """Latest stored capture for a URL (keys + metadata), or None if uncaptured."""
    body = _get_bytes(bucket, f"{prefix}/index/{url_hash(url)}.json")
    if not body:
        return None
    idx = json.loads(body.decode("utf-8"))
    ch = idx.get("latest_content_hash")
    cap = (idx.get("captures") or {}).get(ch)
    return cap


# --- UI --------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Source Archive Viewer", layout="wide")
    cfg = _cfg()
    st.title("📚 Source Archive Viewer")

    location = cfg.local_dir or cfg.s3_bucket
    if not location:
        st.error(
            "No archive configured. Set WEB_ARCHIVE_LOCAL_DIR (a local capture "
            "directory) or WEB_ARCHIVE_S3_BUCKET (S3), then reload."
        )
        st.stop()
    if cfg.local_dir:
        st.caption(f"📂 local: {cfg.local_dir}/{cfg.s3_prefix}")
    else:
        st.caption(
            f"s3://{cfg.s3_bucket}/{cfg.s3_prefix}  ·  "
            f"profile={cfg.aws_profile or 'default'}"
        )

    with st.sidebar:
        st.header("Filters")
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    df = _manifest_rows(location, cfg.s3_prefix)
    if df.empty:
        st.warning("No manifests found under this prefix yet. Run a capture first.")
        st.stop()

    with st.sidebar:
        bots = sorted(df["bot"].unique())
        qs = sorted(df["question"].unique())
        sel_bots = st.multiselect("Bot", bots, default=bots)
        sel_qs = st.multiselect("Question", qs, default=qs)
        search = st.text_input("URL contains")

    view = df[df["bot"].isin(sel_bots) & df["question"].isin(sel_qs)]
    if search:
        view = view[view["url"].str.contains(search, case=False, na=False)]
    view = view.reset_index(drop=True)

    st.subheader(f"{len(view)} cited URL(s)")

    # Resolve capture status for the filtered rows (cached per-URL).
    if len(view) > 300:
        st.info(
            "Showing 300 of %d — narrow with the filters for capture details."
            % len(view)
        )
    table = []
    for _, row in view.head(300).iterrows():
        cap = _index(location, cfg.s3_prefix, row["url"])
        table.append(
            {
                "question": row["question"],
                "bot": row["bot"],
                "run_id": row["run_id"],
                "origin": row["origin"],
                "captured": "✅" if cap else "—",
                "fetcher": (cap or {}).get("fetcher", ""),
                "captured_at": (cap or {}).get("captured_at", "")[:19],
                "metaculus": row["metaculus"],
                "comment": row["comment"],
                "url": row["url"],
            }
        )
    st.dataframe(
        pd.DataFrame(table),
        use_container_width=True,
        hide_index=True,
        column_config={
            # Show the full link address as the clickable text (not a label).
            "url": st.column_config.LinkColumn("url"),
            "metaculus": st.column_config.LinkColumn(
                "metaculus", display_text="question ↗"
            ),
            "comment": st.column_config.LinkColumn("comment", display_text="comment ↗"),
        },
    )

    if st.sidebar.checkbox("💸 Show scraping cost"):
        st.subheader("💸 Scraping cost (filtered set)")
        rate = st.number_input(
            "Firecrawl cost per page ($)",
            min_value=0.0,
            value=0.001,
            step=0.0005,
            format="%.4f",
            help="Self-hosted Playwright is free; this prices the Firecrawl "
            "fallback. Adjust to your plan's credit rate.",
        )
        per_q = _scrape_report(location, cfg.s3_prefix, view.head(300))
        rows, t_fc, t_pw, t_cap, t_url = [], 0, 0, 0, 0
        for agg in sorted(per_q.values(), key=lambda a: a["question"]):
            rows.append(
                {
                    "question": agg["question"],
                    "urls": agg["urls"],
                    "captured": agg["captured"],
                    "playwright (free)": agg["playwright"],
                    "firecrawl (paid)": agg["firecrawl"],
                    "firecrawl $": round(agg["firecrawl"] * rate, 4),
                }
            )
            t_fc += agg["firecrawl"]
            t_pw += agg["playwright"]
            t_cap += agg["captured"]
            t_url += agg["urls"]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        a, b, c = st.columns(3)
        a.metric("Captured", f"{t_cap}/{t_url}")
        b.metric("Firecrawl pages", t_fc, help="Playwright pages are free")
        c.metric("Est. Firecrawl cost", f"${t_fc * rate:.4f}")
        st.caption(
            f"Playwright (free): {t_pw} · Firecrawl (paid): {t_fc}.  "
            "⚠️ Only **successful** captures carry a fetcher in the index, so "
            "Firecrawl attempts that failed the quality gate aren't counted — "
            "billed-attempt tracking needs the pipeline to persist fetch attempts."
        )

    st.divider()
    st.subheader("Inspect a capture")
    labels = [f"[{r['question']}] {r['url']}" for _, r in view.iterrows()]
    if not labels:
        st.stop()
    choice = st.selectbox("URL", range(len(labels)), format_func=lambda i: labels[i])
    row = view.iloc[choice]
    url = row["url"]
    cap = _index(location, cfg.s3_prefix, url)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**URL:** [{url}]({url})")
        st.markdown(
            f"**Question:** `{row['question']}` · **Bot:** `{row['bot']}` · "
            f"**Origin:** `{row['origin'] or '—'}`"
        )
        st.markdown(f"**Run:** `{row['run_id'] or '—'}`")
        review = row["metaculus"] or row["question_url"]
        if review:
            st.markdown(f"**Metaculus question:** [{review}]({review})")
        if row["comment"]:
            st.markdown(f"**Cited in comment:** [{row['comment']}]({row['comment']})")
        if row["query"]:
            st.markdown(f"**Search query:** `{row['query']}`")
        if row.get("tool_args"):
            st.markdown(f"**Tool args:** `{row['tool_args']}`")
    with c2:
        if cap:
            st.markdown(
                f"**Captured:** {cap.get('captured_at','')[:19]}  ·  "
                f"**Fetcher:** {cap.get('fetcher','')}  ·  "
                f"**HTTP:** {cap.get('status_code','?')}"
            )

    if not cap:
        st.warning(
            "No stored capture for this URL — it failed the quality gate / errored, "
            "or hasn't been captured yet."
        )
        st.stop()

    tab_shot, tab_md, tab_html = st.tabs(["🖼 Screenshot", "📝 Markdown", "🌐 HTML"])

    with tab_shot:
        key = cap.get("screenshot_key")
        data = _get_bytes(location, key) if key else None
        if data:
            st.download_button("Download .webp", data, file_name="screenshot.webp")
            st.image(data, use_container_width=True)
        else:
            st.info("No screenshot stored.")

    with tab_md:
        key = cap.get("markdown_key")
        data = _get_bytes(location, key) if key else None
        if data:
            text = data.decode("utf-8", "replace")
            st.download_button("Download .md", data, file_name="page.md")
            st.caption(f"{len(text):,} chars")
            st.markdown(text)
        else:
            st.info("No markdown stored.")

    with tab_html:
        key = cap.get("html_key")
        data = _get_bytes(location, key) if key else None
        if data:
            html = data.decode("utf-8", "replace")
            st.download_button("Download .html", data, file_name="page.html")
            st.caption(
                f"{len(html):,} chars · rendered below (CSS/images load from the "
                "original site and may not all resolve — the screenshot is the "
                "faithful visual record)."
            )
            st.components.v1.html(html, height=800, scrolling=True)
        else:
            st.info("No HTML stored.")


if __name__ == "__main__":
    main()
