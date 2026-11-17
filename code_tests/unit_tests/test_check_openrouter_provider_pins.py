from run_bots import get_default_bot_dict
from scripts.check_openrouter_provider_pins import (
    ProviderPin,
    check_pin,
    collect_provider_pins,
    endpoints_matching_slug,
)

UP = 0
DOWN = -5


def make_endpoint(tag: str, status: int = UP) -> dict:
    return {"tag": tag, "status": status}


def make_pin(*endpoint_slugs: str) -> ProviderPin:
    return ProviderPin(
        bot_name="METAC_TEST",
        llm_purpose="default",
        model="lab/model",
        endpoint_slugs=endpoint_slugs,
    )


def test_bare_provider_slug_matches_all_of_that_providers_endpoints() -> None:
    endpoints = [
        make_endpoint("minimax"),
        make_endpoint("minimax/highspeed"),
        make_endpoint("minimaxi"),
    ]
    matching = endpoints_matching_slug("minimax", endpoints)
    assert [endpoint["tag"] for endpoint in matching] == [
        "minimax",
        "minimax/highspeed",
    ]


def test_precision_slug_does_not_match_other_precisions() -> None:
    endpoints = [make_endpoint("deepinfra/fp4"), make_endpoint("baseten/fp4/us")]
    assert endpoints_matching_slug("deepinfra/fp8", endpoints) == []
    assert endpoints_matching_slug("baseten/fp4", endpoints) == [
        make_endpoint("baseten/fp4/us")
    ]


def test_check_pin_reports_delisted_and_down_slugs_separately() -> None:
    endpoints = [make_endpoint("crusoe/bf16"), make_endpoint("deepinfra/bf16", DOWN)]
    result = check_pin(
        make_pin("akashml/bf16", "crusoe/bf16", "deepinfra/bf16"), endpoints
    )
    assert result.missing_slugs == ["akashml/bf16"]
    assert result.down_slugs == ["deepinfra/bf16"]


def test_check_pin_passes_when_every_slug_is_listed_and_up() -> None:
    endpoints = [make_endpoint("crusoe/bf16"), make_endpoint("dekallm/bf16")]
    result = check_pin(make_pin("crusoe/bf16", "dekallm/bf16"), endpoints)
    assert result.missing_slugs == []
    assert result.down_slugs == []


def test_check_pin_treats_unlisted_model_as_every_slug_missing() -> None:
    result = check_pin(make_pin("deepinfra/fp8", "together"), [])
    assert result.missing_slugs == ["deepinfra/fp8", "together"]


def test_collect_provider_pins_only_includes_active_bots() -> None:
    bot_dict = get_default_bot_dict()
    pins = collect_provider_pins()

    assert pins
    for pin in pins:
        assert bot_dict[pin.bot_name].tournaments
        assert not pin.model.startswith("openrouter/")
        assert pin.endpoint_slugs
