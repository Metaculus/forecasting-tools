"""Tests for screenshot encoding + the height cap.

Regression guard for a silent truncation bug: the height cap used to be applied
via Playwright's ``clip`` *without* ``full_page``, which is bounded by the
viewport and chopped tall pages down to a single screen. The cap is now enforced
by cropping the full-page render in Pillow — these tests pin that behavior.
"""

from __future__ import annotations

import io

import pytest

from forecasting_tools.agents_and_tools.source_archive.fetchers.playwright_fetcher import (
    _encode_screenshot,
)

Image = pytest.importorskip("PIL.Image")


def _png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (255, 0, 0)).save(out, format="PNG")
    return out.getvalue()


def test_tall_page_cropped_to_max_height():
    data, ct = _encode_screenshot(_png(1280, 12000), "webp", max_height=4000)
    assert ct == "image/webp"
    img = Image.open(io.BytesIO(data))
    assert img.size == (1280, 4000)  # cropped to the cap, full width preserved


def test_short_page_not_cropped():
    data, _ = _encode_screenshot(_png(1280, 3000), "webp", max_height=20000)
    assert Image.open(io.BytesIO(data)).size == (1280, 3000)  # untouched


def test_webp_clamped_to_format_limit_even_without_cap():
    # WebP cannot exceed 16383px; an over-tall page must crop, not crash.
    data, _ = _encode_screenshot(_png(1280, 25000), "webp", max_height=0)
    assert Image.open(io.BytesIO(data)).size == (1280, 16383)


def test_webp_cap_above_format_limit_is_clamped():
    # A configured cap above WebP's limit still degrades safely to 16383.
    data, _ = _encode_screenshot(_png(1280, 18000), "webp", max_height=16000)
    assert Image.open(io.BytesIO(data)).height == 16000


def test_png_keeps_full_height_uncapped():
    # PNG has no such limit, so max_height=0 preserves the whole render.
    data, _ = _encode_screenshot(_png(1280, 20000), "png", max_height=0)
    assert Image.open(io.BytesIO(data)).size == (1280, 20000)


def test_webp_is_real_webp():
    data, ct = _encode_screenshot(_png(800, 600), "webp")
    assert ct == "image/webp"
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def test_jpeg_format():
    data, ct = _encode_screenshot(_png(800, 600), "jpeg")
    assert ct == "image/jpeg"
    assert Image.open(io.BytesIO(data)).format == "JPEG"
