"""Configuration for the source archive, read from environment variables.

No bucket names, credentials, or other deployment-specific values are baked in
here, so this module is safe to publish. Operators set the bucket via
``WEB_ARCHIVE_S3_BUCKET`` (see ``.env.template``).
"""

from __future__ import annotations

import os

from pydantic import BaseModel


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class ArchiveConfig(BaseModel):
    """Runtime configuration. Construct directly in tests, or ``from_env()``."""

    s3_bucket: str | None = None
    s3_prefix: str = "source-archive"
    # Local archive directory. When set, the viewer reads captures from here
    # instead of S3 — handy for inspecting a local capture run with no AWS.
    local_dir: str | None = None
    aws_profile: str | None = None
    aws_region: str | None = None
    firecrawl_api_key: str | None = None
    # Firecrawl proxy mode for the anti-bot path: "basic" (1 credit) | "auto"
    # (1 credit, escalates to 5 on fallback) | "stealth"/"enhanced" (5 credits).
    # Only the fallback Firecrawl tier pays this; basic is the default.
    firecrawl_proxy: str = "basic"
    hyperbrowser_api_key: str | None = None
    # Hyperbrowser session knobs for the anti-bot path. Proxy turns a 1-credit
    # scrape into a 10-credit one, so leave it on only for the Cloudflare tier.
    hyperbrowser_use_proxy: bool = True
    hyperbrowser_use_stealth: bool = True
    hyperbrowser_solve_captchas: bool = True
    # CloakBrowser exposes ``cloakbrowser.launch() -> Browser``; the module name
    # is overridable in case the package is renamed.
    cloakbrowser_import: str = "cloakbrowser"
    pdf_max_pages: int = 50  # cap PDF parsing so a huge report can't blow latency/cost
    ttl_days: int = 14
    screenshot_format: str = "webp"  # webp | jpeg | png
    screenshot_max_height: int = 16_000  # px; safety cap (under WebP's 16383 limit)
    nav_timeout_ms: int = 30_000
    concurrency: int = 5

    @classmethod
    def from_env(cls) -> "ArchiveConfig":
        return cls(
            s3_bucket=os.environ.get("WEB_ARCHIVE_S3_BUCKET"),
            s3_prefix=os.environ.get("WEB_ARCHIVE_S3_PREFIX", "source-archive"),
            local_dir=os.environ.get("WEB_ARCHIVE_LOCAL_DIR"),
            aws_profile=os.environ.get("WEB_ARCHIVE_AWS_PROFILE"),
            aws_region=os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION"),
            firecrawl_api_key=os.environ.get("FIRECRAWL_API_KEY"),
            firecrawl_proxy=os.environ.get("WEB_ARCHIVE_FIRECRAWL_PROXY", "basic"),
            hyperbrowser_api_key=os.environ.get("HYPERBROWSER_API_KEY"),
            hyperbrowser_use_proxy=_get_bool("WEB_ARCHIVE_HYPERBROWSER_PROXY", True),
            hyperbrowser_use_stealth=_get_bool(
                "WEB_ARCHIVE_HYPERBROWSER_STEALTH", True
            ),
            hyperbrowser_solve_captchas=_get_bool(
                "WEB_ARCHIVE_HYPERBROWSER_CAPTCHA", True
            ),
            cloakbrowser_import=os.environ.get(
                "WEB_ARCHIVE_CLOAKBROWSER_IMPORT", "cloakbrowser"
            ),
            pdf_max_pages=_get_int("WEB_ARCHIVE_PDF_MAX_PAGES", 50),
            ttl_days=_get_int("WEB_ARCHIVE_TTL_DAYS", 14),
            screenshot_format=os.environ.get("WEB_ARCHIVE_SCREENSHOT_FORMAT", "webp"),
            screenshot_max_height=_get_int("WEB_ARCHIVE_SCREENSHOT_MAX_HEIGHT", 16_000),
            nav_timeout_ms=_get_int("WEB_ARCHIVE_NAV_TIMEOUT_MS", 30_000),
            concurrency=_get_int("WEB_ARCHIVE_CONCURRENCY", 5),
        )
