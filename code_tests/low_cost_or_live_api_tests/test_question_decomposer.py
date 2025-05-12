import asyncio
import logging

from forecasting_tools.agents_and_tools.question_generators.question_decomposer import (
    QuestionDecomposer,
)

logger = logging.getLogger(__name__)


def test_question_decomposer_runs() -> None:
    question_decomposer = QuestionDecomposer()
    result = asyncio.run(
        question_decomposer.decompose_into_questions(
            "Will humanity go extinct before 2100?", 5
        )
    )
    logger.info(f"result: {result}")
    assert len(result.questions) == 5
