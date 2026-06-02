"""Quality gate for captures.

A headless browser will happily "succeed" at screenshotting a 404 or a bot-block
interstitial. Gate captures on HTTP status, content length, and block-page
signatures before archiving, so junk is neither stored nor counted as a success
(and so the tiered fetcher knows when to fall back to another backend).
"""

from __future__ import annotations

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.source_archive.models import CaptureResult

# Substrings that strongly indicate a block / interstitial rather than the real
# page. Matched case-insensitively against extracted text.
BLOCK_SIGNATURES = (
    "verify you are a human",
    "are you a human",
    "checking your browser before",
    "enable javascript and cookies to continue",
    "please enable javascript",
    "access to this page has been denied",
    "access denied",
    "request unsuccessful. incapsula",
    "attention required! | cloudflare",
    "ddos protection by cloudflare",
    "ray id:",
    "captcha",
    "unusual traffic from your computer",
)

MIN_TEXT_LEN = 200


class QualityVerdict(BaseModel):
    passed: bool
    reason: str = ""


def evaluate(
    result: CaptureResult, *, min_text_len: int = MIN_TEXT_LEN
) -> QualityVerdict:
    if result.status_code is not None and result.status_code >= 400:
        return QualityVerdict(passed=False, reason=f"http_status={result.status_code}")

    text = (result.markdown or result.html or "").strip()
    if len(text) < min_text_len:
        return QualityVerdict(passed=False, reason=f"thin_content len={len(text)}")

    lowered = text.lower()
    for sig in BLOCK_SIGNATURES:
        if sig in lowered:
            return QualityVerdict(passed=False, reason=f"block_signature={sig!r}")

    return QualityVerdict(passed=True)
