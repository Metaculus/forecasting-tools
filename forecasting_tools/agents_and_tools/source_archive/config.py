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


class ArchiveConfig(BaseModel):
    """Runtime configuration. Construct directly in tests, or ``from_env()``."""

    s3_bucket: str | None = None
    s3_prefix: str = "source-archive"
    aws_profile: str | None = None
    aws_region: str | None = None
    firecrawl_api_key: str | None = None
    ttl_days: int = 14
    screenshot_format: str = "webp"  # webp | jpeg | png
    screenshot_max_height: int = 4000  # px; cap full-page captures
    nav_timeout_ms: int = 30_000
    concurrency: int = 5

    @classmethod
    def from_env(cls) -> "ArchiveConfig":
        return cls(
            s3_bucket=os.environ.get("WEB_ARCHIVE_S3_BUCKET"),
            s3_prefix=os.environ.get("WEB_ARCHIVE_S3_PREFIX", "source-archive"),
            aws_profile=os.environ.get("WEB_ARCHIVE_AWS_PROFILE"),
            aws_region=os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION"),
            firecrawl_api_key=os.environ.get("FIRECRAWL_API_KEY"),
            ttl_days=_get_int("WEB_ARCHIVE_TTL_DAYS", 14),
            screenshot_format=os.environ.get("WEB_ARCHIVE_SCREENSHOT_FORMAT", "webp"),
            screenshot_max_height=_get_int("WEB_ARCHIVE_SCREENSHOT_MAX_HEIGHT", 4000),
            nav_timeout_ms=_get_int("WEB_ARCHIVE_NAV_TIMEOUT_MS", 30_000),
            concurrency=_get_int("WEB_ARCHIVE_CONCURRENCY", 5),
        )
