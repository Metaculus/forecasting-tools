"""Capture pipeline: turn a list of cited URLs into archived captures.

For each unique URL:

  1. :meth:`ContentStore.lookup` — within the TTL? cache hit, skip the fetch.
  2. ``fetcher.fetch``           — tiered Playwright -> Firecrawl, quality-gated.
  3. quality gate                — junk (404 / block / thin) is not archived.
  4. :meth:`ContentStore.store`  — write blobs (deduped by content hash).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.source_archive.content_store import ContentStore
from forecasting_tools.agents_and_tools.source_archive.fetchers.base import (
    Fetcher,
    FetchError,
)
from forecasting_tools.agents_and_tools.source_archive.manifest import unique_urls
from forecasting_tools.agents_and_tools.source_archive.models import (
    CitationRecord,
    StoredCapture,
)
from forecasting_tools.agents_and_tools.source_archive.quality import evaluate

logger = logging.getLogger(__name__)

# "cache_hit" | "stored" | "deduped" | "quality_failed" | "error"
Status = str
_STATUSES = ("cache_hit", "stored", "deduped", "quality_failed", "error")


class CaptureOutcome(BaseModel):
    url: str
    status: Status
    stored: StoredCapture | None = None
    reason: str = ""


class PipelineSummary(BaseModel):
    outcomes: list[CaptureOutcome] = []

    def count(self, status: Status) -> int:
        return sum(1 for o in self.outcomes if o.status == status)

    @property
    def captures(self) -> dict[str, StoredCapture]:
        return {o.url: o.stored for o in self.outcomes if o.stored is not None}

    def __str__(self) -> str:
        body = ", ".join(f"{s}={self.count(s)}" for s in _STATUSES)
        return f"PipelineSummary(total={len(self.outcomes)}, {body})"


class CapturePipeline:
    def __init__(self, fetcher: Fetcher, content_store: ContentStore):
        self.fetcher = fetcher
        self.content_store = content_store

    def capture_url(self, url: str) -> CaptureOutcome:
        cached = self.content_store.lookup(url)
        if cached is not None:
            return CaptureOutcome(url=url, status="cache_hit", stored=cached)

        try:
            result = self.fetcher.fetch(url)
        except FetchError as e:
            logger.info("fetch error for %s: %s", url, e)
            return CaptureOutcome(url=url, status="error", reason=str(e))
        except Exception as e:  # never let one bad URL abort the whole run
            logger.warning("unexpected error capturing %s: %s", url, e)
            return CaptureOutcome(url=url, status="error", reason=f"unexpected: {e}")

        # Gate here so any fetcher is covered; the tiered fetcher also gates
        # internally to decide fallback, but this is the authoritative check.
        verdict = evaluate(result)
        if not verdict.passed:
            return CaptureOutcome(
                url=url, status="quality_failed", reason=verdict.reason
            )

        store_result = self.content_store.store(result)
        status = "stored" if store_result.created else "deduped"
        return CaptureOutcome(url=url, status=status, stored=store_result.capture)

    def run(self, urls: Iterable[str]) -> PipelineSummary:
        summary = PipelineSummary()
        for url in urls:
            summary.outcomes.append(self.capture_url(url))
        return summary

    def run_manifest(self, records: Iterable[CitationRecord]) -> PipelineSummary:
        return self.run(unique_urls(records))


# An outcome whose error reason contains one of these means the browser itself
# died (crash, OOM, or the machine slept and severed the CDP pipe) — not a
# problem with the URL. Without recovery, every later URL in that worker's shard
# would error against the dead browser, so we rebuild the browser and retry.
_DEAD_BROWSER_MARKERS = (
    "has been closed",
    "Target page, context or browser",
    "Browser.new_context",
    "Connection closed",
    "browser has been closed",
)


def _browser_died(reason: str | None) -> bool:
    return any(m in (reason or "") for m in _DEAD_BROWSER_MARKERS)


def _close_quietly(cm, timeout_s: float = 15.0) -> None:
    """Tear down a fetcher context manager, but never block on it: a wedged
    browser's ``close()`` can itself hang, so run it in a daemon thread and give
    up after ``timeout_s`` (the leftover process is reaped at the end of the run).
    """
    done = threading.Event()

    def _close() -> None:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=_close, daemon=True).start()
    done.wait(timeout_s)


def _running_loop():
    import asyncio

    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _restore_thread_loop_state(baseline) -> None:
    """Un-poison a worker thread's asyncio state after abandoning a dead
    sync-Playwright instance.

    Sync Playwright drives its asyncio loop *on the calling thread* (via a
    greenlet), so while an instance is alive the thread is marked as being
    inside a running event loop. A clean ``stop()`` clears that mark — but a
    SIGKILLed browser can't be closed cleanly: ``close()``/``stop()`` fail or
    hang, and both are thread-affine, so :func:`_close_quietly`'s helper thread
    can't reach them either. The abandoned loop then stays registered as
    running, and the next ``sync_playwright().start()`` on this thread refuses
    with "Playwright Sync API inside the asyncio loop", killing the rebuild.

    Resetting the thread-local marker to its pre-fetcher ``baseline`` lets the
    rebuilt fetcher start a fresh loop. The old loop object is leaked on
    purpose (its browser processes are swept by the reaper); that is the price
    of recovering without touching thread-affine Playwright internals.
    """
    import asyncio

    if _running_loop() is baseline:
        return
    try:
        asyncio.events._set_running_loop(baseline)
    except Exception:
        pass


def _reap_browser_descendants() -> None:
    """Best-effort: kill automation Chromium descending from this process. Used
    both to recover a wedged worker (kill its browser so the blocked sync call
    errors out) and to sweep leftovers at end of run. No-op without psutil so it
    never becomes a hard dependency.
    """
    try:
        import os

        import psutil
    except Exception:
        return
    try:
        for child in psutil.Process(os.getpid()).children(recursive=True):
            try:
                if "chrom" in (child.name() or "").lower():
                    child.kill()
            except Exception:
                pass
    except Exception:
        pass


def capture_urls_concurrent(
    urls: Iterable[str],
    store: ContentStore,
    config,
    fetcher_factory,
    per_url_timeout: float | None = None,
    reaper=_reap_browser_descendants,
) -> PipelineSummary:
    """Capture ``urls`` across ``config.concurrency`` worker threads.

    Headless Chromium's sync API is **thread-affine** — a browser must be used on
    the thread that created it — so each worker opens its **own** browser via
    ``fetcher_factory(config)`` and runs all captures inline on its own thread.
    The content store is shared (writes are keyed by URL hash and idempotent, so
    shards never collide). Order of outcomes is not preserved.

    Hang protection runs *out of band*: a supervisor thread watches each worker's
    heartbeat and, if one is stuck on a single URL past ``per_url_timeout`` (a
    wedged sync call whose Playwright timeout never fires — e.g. the machine
    slept and severed the CDP pipe), it **kills the browser processes**. That is
    an OS-level action (safe across threads, unlike touching Playwright objects),
    so the blocked call errors out and the worker rebuilds via the same
    dead-browser path — no single stuck worker can freeze the whole run.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    url_list = list(urls)
    workers = max(1, int(getattr(config, "concurrency", 1) or 1))
    if per_url_timeout is None:
        nav_s = float(getattr(config, "nav_timeout_ms", 30000)) / 1000.0
        per_url_timeout = max(90.0, nav_s * 4)

    # worker index -> monotonic start of its current URL (None when between URLs)
    heartbeats: dict[int, float | None] = {}
    hb_lock = threading.Lock()
    stop = threading.Event()

    def supervisor() -> None:
        interval = max(0.5, min(per_url_timeout / 2, 30.0))
        while not stop.wait(interval):
            now = time.monotonic()
            with hb_lock:
                stalled = [
                    w
                    for w, t in heartbeats.items()
                    if t is not None and now - t > per_url_timeout
                ]
            if stalled:
                logger.warning(
                    "worker(s) %s stuck > %.0fs on one URL; killing browsers to recover",
                    stalled,
                    per_url_timeout,
                )
                reaper()
                with hb_lock:  # grace: don't reap again before workers rebuild
                    for w in list(heartbeats):
                        if heartbeats[w] is not None:
                            heartbeats[w] = now

    def work(idx: int, shard: list[str]) -> list[CaptureOutcome]:
        outcomes: list[CaptureOutcome] = []
        baseline_loop = _running_loop()  # almost always None; see _restore_...
        cm = fetcher_factory(config)
        pipeline = CapturePipeline(cm.__enter__(), store)
        try:
            for url in shard:
                with hb_lock:
                    heartbeats[idx] = time.monotonic()
                outcome = pipeline.capture_url(url)
                if outcome.status == "error" and _browser_died(outcome.reason):
                    logger.warning(
                        "browser died; rebuilding worker %d, retrying %s", idx, url
                    )
                    _close_quietly(cm)
                    _restore_thread_loop_state(baseline_loop)
                    try:
                        cm = fetcher_factory(config)
                        pipeline = CapturePipeline(cm.__enter__(), store)
                    except Exception as e:
                        # A failed rebuild must never kill the run: keep this
                        # URL's error outcome and move on — the still-dead
                        # pipeline makes the next URL error with a dead-browser
                        # reason, which retries the rebuild.
                        logger.warning(
                            "worker %d rebuild failed (%s); keeping error outcome",
                            idx,
                            e,
                        )
                    else:
                        with hb_lock:
                            heartbeats[idx] = time.monotonic()
                        # one retry on a fresh browser
                        outcome = pipeline.capture_url(url)
                outcomes.append(outcome)
                with hb_lock:
                    heartbeats[idx] = None
        finally:
            _close_quietly(cm)
            _restore_thread_loop_state(baseline_loop)
        return outcomes

    supervisor_thread = threading.Thread(target=supervisor, daemon=True)
    supervisor_thread.start()
    try:
        if workers == 1:
            heartbeats[0] = None
            return PipelineSummary(outcomes=work(0, url_list))

        shards = [url_list[i::workers] for i in range(workers)]
        for i in range(workers):
            heartbeats[i] = None
        summary = PipelineSummary()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, i, shards[i]) for i in range(workers)]
            for future in futures:
                summary.outcomes.extend(future.result())
        return summary
    finally:
        stop.set()
        reaper()
