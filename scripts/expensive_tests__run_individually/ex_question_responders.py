import asyncio
from unittest.mock import Mock

import pytest

from forecasting_tools.agents_and_tools.base_rates.base_rate_researcher import (
    BaseRateResearcher,
)
from forecasting_tools.agents_and_tools.deprecated.general_researcher import (
    GeneralResearcher,
)
from forecasting_tools.agents_and_tools.deprecated.question_responder import (
    QuestionResponder,
)
from forecasting_tools.agents_and_tools.deprecated.question_router import QuestionRouter
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.deprecated_model_classes.gpt4o import Gpt4o


#################################### HELPERS ####################################
def mock_question_responder__answer_with_markdown(
    mocker: Mock, subclass: type[QuestionResponder]
) -> Mock:
    mock_function = mocker.patch(
        f"{subclass.respond_with_markdown.__module__}.{subclass.respond_with_markdown.__qualname__}"
    )
    mock_function.return_value = "This is a test answer"
    return mock_function


def mock_question_responder__answer_with_markdown__with_error(
    mocker: Mock, subclass: type[QuestionResponder]
) -> Mock:
    mock_function = mocker.patch(
        f"{subclass.respond_with_markdown.__module__}.{subclass.respond_with_markdown.__qualname__}"
    )
    mock_function.side_effect = Exception("This is a test error")
    return mock_function


RESPONDERS_WITH_TEST_QUESTIONS = [
    (
        BaseRateResearcher,
        "What are the chances per year that a person will win the lottery given they buy a ticket every week?",
    ),
    (GeneralResearcher, "Who is Abraham Lincoln?"),
]

#################################### TESTS ####################################


@pytest.mark.parametrize("responder_class, question", RESPONDERS_WITH_TEST_QUESTIONS)
async def test_responders_give_answer(
    responder_class: type[QuestionResponder], question: str
) -> None:
    responder_instance = responder_class(question)
    answer = asyncio.run(responder_instance.respond_with_markdown())
    assert answer != ""
    assert answer is not None
    assert isinstance(answer, str)
    model = Gpt4o(temperature=0.3)
    prompt = clean_indents(
        f"""
        Are the two below criteria met?
        1) The below text is a reasonable answer to the below question
        2) If the text is in markdown, that the markdown is properly formatted with no level 1 or level 2 headers (level 3 and below are fine)

        Question:
        {responder_instance.question}

        Text: (in <><> tags)
        <><><><><><><><><><><><><><>
        {answer}
        <><><><><><><><><><><><><><>

        Lets take this step by step.
        1) Examine each criteria
        2) Answer with YES or NO at the end and do not include "YES" or "NO" elsewhere in the text. There should only be one instance of YES or NO in the text.
        """
    )
    answer_1_passes = await model.invoke_and_check_for_boolean_keyword(prompt)
    answer_2_passes = await model.invoke_and_check_for_boolean_keyword(prompt)
    at_least_one_passes = answer_1_passes or answer_2_passes
    assert (
        at_least_one_passes
    ), f"GPT thought that the answer was not in proper markdown or plaintext or was not reasonable.\n GPT's Answer: {answer_1_passes}\n Responder's Answer: {answer}"


@pytest.mark.parametrize("responder_class", QuestionRouter.AVAILABLE_REPONDERS)
def test_responders_error_if_empty_question(
    responder_class: type[QuestionResponder],
) -> None:
    with pytest.raises(ValueError, match="Question cannot be empty"):
        responder_class("")


@pytest.mark.parametrize("responder_class", QuestionRouter.AVAILABLE_REPONDERS)
def test_responders_error_if_question_too_long(
    responder_class: type[QuestionResponder],
) -> None:
    long_question = "a" * 1001  # Create a string that's 1001 characters long
    with pytest.raises(ValueError, match="Question is too long"):
        responder_class(long_question)


@pytest.mark.parametrize(
    "correct_responder, router_question", RESPONDERS_WITH_TEST_QUESTIONS
)
def test_question_router_calls_correct_responder(
    mocker: Mock,
    correct_responder: type[QuestionResponder],
    router_question: str,
) -> None:
    mocked_functions: list[Mock] = []
    for responder in QuestionRouter.AVAILABLE_REPONDERS:
        mocked_functions.append(
            mock_question_responder__answer_with_markdown(mocker, responder)
        )

    router = QuestionRouter()
    asyncio.run(router.answer_question_with_markdown_using_routing(router_question))

    for responder, mocked_function in zip(
        QuestionRouter.AVAILABLE_REPONDERS, mocked_functions
    ):
        if responder == correct_responder:
            mocked_function.assert_called_once()
        else:
            mocked_function.assert_not_called()


def test_question_router_raises_error_if_responder_errors(
    mocker: Mock,
) -> None:
    for responder in QuestionRouter.AVAILABLE_REPONDERS:
        mock_question_responder__answer_with_markdown__with_error(mocker, responder)

    router = QuestionRouter()
    with pytest.raises(Exception):
        asyncio.run(
            router.answer_question_with_markdown_using_routing(
                "Will the world end in 2100?"
            )
        )
