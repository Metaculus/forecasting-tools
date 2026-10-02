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


def _created_question_with_author(author: str) -> BinaryQuestion:
    created = _make_question()
    created.id_of_post = 123
    created.api_json = {"author_username": author}
    return created


def _create_with_mocked_api(
    question: BinaryQuestion,
    author_username: str | None,
    server_author: str,
) -> tuple[BinaryQuestion, MagicMock]:
    client = MetaculusClient(base_url="http://x", token="t")
    created = _created_question_with_author(server_author)
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
        result = client.create_question(question, author_username=author_username)
    return result, post_mock


def test_post_create_data_includes_staff_override_when_author_given() -> None:
    question = _make_question()
    data = MetaculusClient._get_post_create_data(
        question, author_username="johnnycaffeine"
    )
    assert data["author_username"] == "johnnycaffeine"
    assert data["is_staff_override"] is True

    data_without_override = {
        key: value
        for key, value in data.items()
        if key not in ("author_username", "is_staff_override")
    }
    assert data_without_override == MetaculusClient._get_post_create_data(question)


def test_post_create_data_has_no_override_fields_without_author() -> None:
    data = MetaculusClient._get_post_create_data(_make_question())
    assert "author_username" not in data
    assert "is_staff_override" not in data


def test_create_question_sends_override_fields_to_the_api() -> None:
    _, post_mock = _create_with_mocked_api(
        _make_question(), author_username="BenWilson", server_author="BenWilson"
    )
    sent_payload = post_mock.call_args.kwargs["json"]
    assert sent_payload["author_username"] == "BenWilson"
    assert sent_payload["is_staff_override"] is True


def test_create_question_sends_no_override_fields_without_author() -> None:
    _, post_mock = _create_with_mocked_api(
        _make_question(), author_username=None, server_author="BenWilson"
    )
    sent_payload = post_mock.call_args.kwargs["json"]
    assert "author_username" not in sent_payload
    assert "is_staff_override" not in sent_payload


def test_create_question_returns_when_author_matches() -> None:
    created, _ = _create_with_mocked_api(
        _make_question(), author_username="BenWilson", server_author="BenWilson"
    )
    assert created.id_of_post == 123


def test_create_question_raises_when_server_ignored_author() -> None:
    with pytest.raises(
        AuthorMismatchError, match="'johnnycaffeine' instead of 'BenWilson'"
    ):
        _create_with_mocked_api(
            _make_question(),
            author_username="BenWilson",
            server_author="johnnycaffeine",
        )


def test_create_question_skips_author_check_without_author() -> None:
    created, _ = _create_with_mocked_api(
        _make_question(), author_username=None, server_author="johnnycaffeine"
    )
    assert created.id_of_post == 123
