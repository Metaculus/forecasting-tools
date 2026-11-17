"""
Checks every active bot's OpenRouter provider pin against OpenRouter's live endpoint list.
Pins disallow fallbacks, so a pinned endpoint that OpenRouter stops listing makes the bot's calls fail;
the script exits non-zero when any pinned endpoint is no longer listed.

Run from the repo root: `uv run python -m scripts.check_openrouter_provider_pins`
"""

import logging
import sys
from dataclasses import dataclass

import requests

from forecasting_tools.ai_models.general_llm import GeneralLlm
from run_bots import get_default_bot_dict

logger = logging.getLogger(__name__)

OPENROUTER_MODEL_PREFIX = "openrouter/"
OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"


@dataclass(frozen=True)
class ProviderPin:
    bot_name: str
    llm_purpose: str
    model: str
    endpoint_slugs: tuple[str, ...]


@dataclass(frozen=True)
class PinCheckResult:
    pin: ProviderPin
    missing_slugs: list[str]
    down_slugs: list[str]
    listed_endpoints: list[dict]


def collect_provider_pins() -> list[ProviderPin]:
    pins: list[ProviderPin] = []
    for bot_name, config in get_default_bot_dict().items():
        if not config.tournaments or config.bot is None:
            continue
        for purpose, llm_config in config.bot.make_llm_dict().items():
            if llm_config is None:
                continue
            llm = config.bot.get_llm(purpose)
            if not isinstance(llm, GeneralLlm):
                continue
            provider = llm.litellm_kwargs.get("extra_body", {}).get("provider", {})
            endpoint_slugs = provider.get("order")
            if not endpoint_slugs:
                continue
            if not llm.model.startswith(OPENROUTER_MODEL_PREFIX):
                raise ValueError(
                    f"{bot_name} ({purpose}) pins providers for non-OpenRouter model {llm.model}"
                )
            pins.append(
                ProviderPin(
                    bot_name=bot_name,
                    llm_purpose=purpose,
                    model=llm.model.removeprefix(OPENROUTER_MODEL_PREFIX),
                    endpoint_slugs=tuple(endpoint_slugs),
                )
            )
    return pins


def endpoints_matching_slug(slug: str, endpoints: list[dict]) -> list[dict]:
    """
    OpenRouter matches a provider slug to endpoint tags by prefix, so a bare provider slug
    like "minimax" also covers "minimax/highspeed".
    """
    return [
        endpoint
        for endpoint in endpoints
        if endpoint["tag"] == slug or endpoint["tag"].startswith(f"{slug}/")
    ]


def check_pin(pin: ProviderPin, endpoints: list[dict]) -> PinCheckResult:
    missing_slugs: list[str] = []
    down_slugs: list[str] = []
    for slug in pin.endpoint_slugs:
        matching = endpoints_matching_slug(slug, endpoints)
        if not matching:
            missing_slugs.append(slug)
        elif all(endpoint["status"] < 0 for endpoint in matching):
            down_slugs.append(slug)
    return PinCheckResult(
        pin=pin,
        missing_slugs=missing_slugs,
        down_slugs=down_slugs,
        listed_endpoints=endpoints,
    )


def fetch_endpoints(model: str) -> list[dict]:
    response = requests.get(OPENROUTER_ENDPOINTS_URL.format(model=model), timeout=30)
    if response.status_code == 404:
        logger.error(f"OpenRouter no longer lists model {model}")
        return []
    response.raise_for_status()
    return response.json()["data"]["endpoints"]


def describe_endpoints(endpoints: list[dict]) -> str:
    if not endpoints:
        return "none"
    return ", ".join(
        f"{endpoint['tag']} ({'up' if endpoint['status'] >= 0 else 'down'})"
        for endpoint in endpoints
    )


def check_all_pins() -> list[PinCheckResult]:
    pins = collect_provider_pins()
    models = sorted({pin.model for pin in pins})
    logger.info(f"Checking {len(pins)} provider pins across {len(models)} models")
    endpoints_by_model = {model: fetch_endpoints(model) for model in models}
    return [check_pin(pin, endpoints_by_model[pin.model]) for pin in pins]


def main() -> int:
    results = check_all_pins()
    broken_results: list[PinCheckResult] = []
    for result in results:
        pin = result.pin
        label = f"{pin.bot_name} ({pin.llm_purpose}, {pin.model}, pinned to {', '.join(pin.endpoint_slugs)})"
        if result.missing_slugs:
            broken_results.append(result)
            logger.error(
                f"{label}: no longer listed: {', '.join(result.missing_slugs)}. "
                f"Listed endpoints: {describe_endpoints(result.listed_endpoints)}"
            )
        elif result.down_slugs:
            logger.warning(f"{label}: currently down: {', '.join(result.down_slugs)}")
        else:
            logger.info(f"{label}: OK")

    if broken_results:
        logger.error(
            f"{len(broken_results)} of {len(results)} provider pins point at endpoints OpenRouter no longer lists. "
            "Re-pin each with make_openrouter_provider_pin in run_bots.py."
        )
        return 1
    logger.info(f"All {len(results)} provider pins are listed on OpenRouter")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    # Building every bot config logs cost-tracking and missing-token warnings that don't concern pins
    logging.getLogger("forecasting_tools").setLevel(logging.ERROR)
    sys.exit(main())
