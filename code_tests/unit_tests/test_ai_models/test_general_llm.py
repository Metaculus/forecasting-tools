import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from forecasting_tools.ai_models.general_llm import GeneralLlm

MUSE_SPARK_MODEL = "openrouter/meta/muse-spark-1.1"


def _model_response(
    content: str | None, reasoning_content: str | None = None
) -> ModelResponse:
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content=content,
                    role="assistant",
                    reasoning_content=reasoning_content,
                ),
            )
        ],
    )
    response.usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)  # type: ignore
    return response


def _llm_returning(
    monkeypatch: pytest.MonkeyPatch, response: ModelResponse
) -> GeneralLlm:
    llm = GeneralLlm(model=MUSE_SPARK_MODEL, temperature=0.3, allowed_tries=1)

    async def fake_call(prompt: str) -> ModelResponse:
        return response

    monkeypatch.setattr(llm, "_call_litellm_dropping_deprecated_temperature", fake_call)
    return llm


async def test_error_names_model_and_finish_reason_when_nothing_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _llm_returning(monkeypatch, _model_response(content=None))

    with pytest.raises(RuntimeError) as error:
        await llm.invoke("Whats 2+2?")

    error_message = str(error.value)
    assert MUSE_SPARK_MODEL in error_message
    assert "stop" in error_message
    assert "no text" in error_message


async def test_reasoning_content_is_used_when_content_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _llm_returning(
        monkeypatch,
        _model_response(content=None, reasoning_content="The answer is 4"),
    )

    answer = await llm.invoke("Whats 2+2?")

    assert answer == "The answer is 4"


async def test_normal_content_is_returned_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _llm_returning(monkeypatch, _model_response(content="4"))

    answer = await llm.invoke("Whats 2+2?")

    assert answer == "4"
