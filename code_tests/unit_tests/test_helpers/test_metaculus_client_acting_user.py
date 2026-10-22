import json
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import requests

from forecasting_tools import ApiFilter, BinaryQuestion, MetaculusClient

MODULE = "forecasting_tools.helpers.metaculus_client"
BOT = "metac-agent-x"
A_FORECAST = {
    "start_time": 1760000000.0,
    "end_time": None,
    "forecast_values": [0.4, 0.6],
}


@pytest.fixture(autouse=True)
def no_sleep() -> Iterator[None]:
    with patch.object(MetaculusClient, "_sleep_between_requests"):
        yield


def _client(acting_user: str | int | None = None) -> MetaculusClient:
    return MetaculusClient(base_url="http://x", token="t", acting_user=acting_user)


def _response(payload: dict | None = None, status_code: int = 201) -> MagicMock:
    return MagicMock(
        status_code=status_code, content=json.dumps(payload or {}).encode()
    )


def _binary_post(my_forecasts_history: list[dict]) -> dict:
    return {
        "id": 1,
        "question": {
            "id": 2,
            "title": "Will it happen?",
            "type": "binary",
            "status": "open",
            "aggregations": {"recency_weighted": {"latest": None, "history": None}},
            "my_forecasts": {"history": my_forecasts_history},
        },
    }


def test_prediction_with_acting_user_posts_to_bulk_endpoint() -> None:
    with patch(f"{MODULE}.requests.post", return_value=_response()) as post_mock:
        _client(BOT).post_binary_question_prediction(5, 0.3)

    assert post_mock.call_args.args[0] == "http://x/questions/bulk-forecast-comment/"
    assert post_mock.call_args.kwargs["json"] == {
        "acting_user": BOT,
        "forecasts": [{"question": 5, "probability_yes": 0.3}],
        "comments": [],
    }


def test_prediction_without_acting_user_posts_to_forecast_endpoint() -> None:
    with patch(f"{MODULE}.requests.post", return_value=_response()) as post_mock:
        _client().post_binary_question_prediction(5, 0.3)

    assert post_mock.call_args.args[0] == "http://x/questions/forecast/"
    assert post_mock.call_args.kwargs["json"] == [
        {"question": 5, "source": "api", "probability_yes": 0.3}
    ]


def test_private_comment_with_acting_user_posts_to_bulk_endpoint() -> None:
    with patch(f"{MODULE}.requests.post", return_value=_response()) as post_mock:
        _client(BOT).post_question_comment(9, "the rationale")

    assert post_mock.call_args.args[0] == "http://x/questions/bulk-forecast-comment/"
    assert post_mock.call_args.kwargs["json"] == {
        "acting_user": BOT,
        "forecasts": [],
        "comments": [
            {
                "on_post": 9,
                "text": "the rationale",
                "is_private": True,
                "included_forecast": True,
            }
        ],
    }


def test_public_comment_with_acting_user_raises_without_a_request() -> None:
    with patch(f"{MODULE}.requests.post") as post_mock:
        with pytest.raises(ValueError, match="Public comments"):
            _client(BOT).post_question_comment(9, "the rationale", is_private=False)

    post_mock.assert_not_called()


def test_public_comment_without_acting_user_posts_to_comment_endpoint() -> None:
    with patch(f"{MODULE}.requests.post", return_value=_response()) as post_mock:
        _client().post_question_comment(9, "the rationale", is_private=False)

    assert post_mock.call_args.args[0] == "http://x/comments/create/"
    assert post_mock.call_args.kwargs["json"]["is_private"] is False


def test_question_by_post_id_passes_acting_user_to_post_detail() -> None:
    with (
        patch(
            f"{MODULE}.requests.get",
            return_value=_response(_binary_post([A_FORECAST]), 200),
        ) as get_mock,
        patch(f"{MODULE}.requests.post") as post_mock,
    ):
        question = _client(BOT).get_question_by_post_id(1)

    assert get_mock.call_args.args[0] == "http://x/posts/1/"
    assert get_mock.call_args.kwargs["params"] == {"acting_user": BOT}
    post_mock.assert_not_called()
    assert isinstance(question, BinaryQuestion)
    assert question.already_forecasted is True
    assert question.previous_forecasts is not None
    assert question.previous_forecasts[0].prediction_in_decimal == 0.6


def test_question_by_post_id_without_acting_user_sends_no_acting_user() -> None:
    with (
        patch(
            f"{MODULE}.requests.get",
            return_value=_response(_binary_post([A_FORECAST]), 200),
        ) as get_mock,
        patch(f"{MODULE}.requests.post") as post_mock,
    ):
        question = _client().get_question_by_post_id(1)

    sent_url = (
        requests.Request(
            "GET",
            get_mock.call_args.args[0],
            params=get_mock.call_args.kwargs["params"],
        )
        .prepare()
        .url
    )
    assert sent_url == "http://x/posts/1/"
    post_mock.assert_not_called()
    assert question.already_forecasted is True


def test_my_forecasts_field_set_on_group_and_conditional_subquestions() -> None:
    group_post = {
        "id": 1,
        "group_of_questions": {"questions": [{"id": 10}, {"id": 11}]},
    }
    conditional_post = {
        "id": 2,
        "conditional": {
            "condition": {"id": 20},
            "condition_child": {"id": 21},
            "question_yes": {"id": 22},
            "question_no": {"id": 23},
        },
    }
    client = _client(BOT)
    with patch.object(
        client,
        "_get_acting_user_forecasts",
        return_value={11: [A_FORECAST], 22: [A_FORECAST]},
    ) as read_mock:
        client._set_my_forecasts_field_to_acting_users([group_post, conditional_post])

    read_mock.assert_called_once_with([10, 11, 20, 21, 22, 23])
    group_questions = group_post["group_of_questions"]["questions"]
    assert group_questions[0]["my_forecasts"] == {
        "history": [],
        "latest": None,
        "score_data": {},
    }
    assert group_questions[1]["my_forecasts"]["latest"] == A_FORECAST
    assert conditional_post["conditional"]["question_yes"]["my_forecasts"][
        "history"
    ] == [A_FORECAST]
    assert (
        conditional_post["conditional"]["question_no"]["my_forecasts"]["history"] == []
    )


def test_previously_forecasted_filter_with_acting_user_is_local() -> None:
    client = _client(BOT)
    api_filter = ApiFilter(is_previously_forecasted_by_user=True)
    with patch.object(client, "get_current_user_id") as get_id_mock:
        url_params = client._create_url_params_for_search(api_filter)

    get_id_mock.assert_not_called()
    assert "forecaster_id" not in url_params

    forecasted = BinaryQuestion(question_text="forecasted", already_forecasted=True)
    not_forecasted = BinaryQuestion(question_text="new", already_forecasted=False)
    filtered = client._apply_local_filters([forecasted, not_forecasted], api_filter)
    assert [question.question_text for question in filtered] == ["forecasted"]


def test_previously_forecasted_filter_without_acting_user_uses_forecaster_id() -> None:
    client = _client()
    api_filter = ApiFilter(is_previously_forecasted_by_user=True)
    with patch.object(client, "get_current_user_id", return_value=7):
        url_params = client._create_url_params_for_search(api_filter)

    assert url_params["forecaster_id"] == 7


def test_create_question_defaults_to_clients_acting_user() -> None:
    client = _client("BenWilson")
    created = BinaryQuestion(
        question_text="Will X happen?",
        id_of_post=123,
        api_json={"author_username": "BenWilson"},
    )
    with (
        patch(f"{MODULE}.requests.post", return_value=_response()) as post_mock,
        patch.object(
            client,
            "_post_json_to_questions_while_handling_groups",
            return_value=[created],
        ),
        patch.object(client, "get_question_by_post_id", return_value=created),
    ):
        client.create_question(BinaryQuestion(question_text="Will X happen?"))

    assert post_mock.call_args.kwargs["json"]["acting_user"] == "BenWilson"
