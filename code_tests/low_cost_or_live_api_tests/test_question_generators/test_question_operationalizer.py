from forecasting_tools.agents_and_tools.question_generators.question_operationalizer import (
    QuestionOperationalizer,
)


async def test_question_operationalizer_runs() -> None:
    question_operationalizer = QuestionOperationalizer(
        model="openrouter/openai/gpt-4o-mini:online"
    )
    question = await question_operationalizer.operationalize_question(
        question_title="Will humanity go extinct before 2100?",
        related_research=None,
        additional_context=None,
    )
    assert question is not None
    assert question.question_text is not None
    assert question.resolution_criteria is not None
    assert question.background_information is not None
    assert question.expected_resolution_date is not None
