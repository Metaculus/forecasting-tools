import asyncio
import logging

from forecasting_tools.agents_and_tools.question_generators.question_decomposer import (
    QuestionDecomposer,
)

logger = logging.getLogger(__name__)


def test_question_decomposer_runs() -> None:
    result = asyncio.run(
        QuestionDecomposer().decompose_into_questions_fast(
            "Will humanity go extinct before 2100?",
            related_research=None,
            additional_context=None,
            model="openrouter/openai/gpt-4o-mini:online",
        )
    )
    logger.info(f"result: {result}")
    assert len(result.questions) == 5
