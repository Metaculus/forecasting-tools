from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from forecasting_tools import AuthorMismatchError, BinaryQuestion, MetaculusClient

OPEN_TIME = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


def _make_question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will X happen?",
        resolution_criteria="Resolves YES if X.",
        fine_print="",
        background_info="bg",
        open_time=OPEN_TIME,
        close_time=OPEN_TIME + timedelta(days=2),
        scheduled_resolution_time=OPEN_TIME + timedelta(days=5),
        includes_bots_in_aggregates=True,
        question_weight=1.0,
        custom_metadata={"short_name": "Short"},
    )


def _created_question_with_author(author: str, author_id: int = 42) -> BinaryQuestion:
    created = _make_question()
    created.id_of_post = 123
    created.api_json = {"author_username": author, "author_id": author_id}
    return created


def _create_with_mocked_api(
    question: BinaryQuestion,
    acting_user: str | int | None,
    server_author: str,
    server_author_id: int = 42,
) -> tuple[BinaryQuestion, MagicMock]:
    client = MetaculusClient(base_url="http://x", token="t")
    created = _created_question_with_author(server_author, server_author_id)
    response = MagicMock(status_code=201, content=b"{}")
    with (
        patch(
            "forecasting_tools.helpers.metaculus_client.requests.post",
            return_value=response,
        ) as post_mock,
        patch.object(client, "_sleep_between_requests"),
        patch.object(
            client,
            "_post_json_to_questions_while_handling_groups",
            return_value=[created],
        ),
        patch.object(client, "get_question_by_post_id", return_value=created),
    ):
        result = client.create_question(question, acting_user=acting_user)
    return result, post_mock


def test_post_create_data_includes_acting_user_when_given() -> None:
    question = _make_question()
    data = MetaculusClient._get_post_create_data(question, acting_user="johnnycaffeine")
    assert data["acting_user"] == "johnnycaffeine"
    assert "is_staff_override" not in data
    assert "author_username" not in data

    data_without_acting_user = {
        key: value for key, value in data.items() if key != "acting_user"
    }
    assert data_without_acting_user == MetaculusClient._get_post_create_data(question)


def test_post_create_data_has_no_acting_user_without_one() -> None:
    data = MetaculusClient._get_post_create_data(_make_question())
    assert "acting_user" not in data


def test_create_question_sends_acting_user_to_the_api() -> None:
    _, post_mock = _create_with_mocked_api(
        _make_question(), acting_user="BenWilson", server_author="BenWilson"
    )
    sent_payload = post_mock.call_args.kwargs["json"]
    assert sent_payload["acting_user"] == "BenWilson"


def test_create_question_sends_no_acting_user_without_one() -> None:
    _, post_mock = _create_with_mocked_api(
        _make_question(), acting_user=None, server_author="BenWilson"
    )
    sent_payload = post_mock.call_args.kwargs["json"]
    assert "acting_user" not in sent_payload


def test_create_question_returns_when_author_matches_username() -> None:
    created, _ = _create_with_mocked_api(
        _make_question(), acting_user="BenWilson", server_author="BenWilson"
    )
    assert created.id_of_post == 123


@pytest.mark.parametrize("acting_user", [7, "7"])
def test_create_question_returns_when_author_matches_user_id(
    acting_user: str | int,
) -> None:
    created, _ = _create_with_mocked_api(
        _make_question(),
        acting_user=acting_user,
        server_author="BenWilson",
        server_author_id=7,
    )
    assert created.id_of_post == 123


def test_create_question_raises_when_server_ignored_username() -> None:
    with pytest.raises(
        AuthorMismatchError, match="'johnnycaffeine' instead of 'BenWilson'"
    ):
        _create_with_mocked_api(
            _make_question(),
            acting_user="BenWilson",
            server_author="johnnycaffeine",
        )


def test_create_question_raises_when_server_ignored_user_id() -> None:
    with pytest.raises(AuthorMismatchError, match="42 instead of 7"):
        _create_with_mocked_api(
            _make_question(),
            acting_user=7,
            server_author="johnnycaffeine",
            server_author_id=42,
        )


def test_create_question_skips_author_check_without_acting_user() -> None:
    created, _ = _create_with_mocked_api(
        _make_question(), acting_user=None, server_author="johnnycaffeine"
    )
    assert created.id_of_post == 123
