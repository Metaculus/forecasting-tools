import logging
from datetime import datetime, timedelta

import pytest

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.research_agents.question_generator import (
    QuestionGenerator,
    SimpleQuestion,
)

logger = logging.getLogger(__name__)


async def test_question_generator_returns_necessary_number_and_stays_within_cost() -> (
    None
):
    number_of_questions_to_generate = 3
    cost_threshold = 0.5
    topic = "Lithuania"
    model = GeneralLlm(model="gpt-4o-mini")
    before_date = datetime.now() + timedelta(days=14)
    before_date = before_date.replace(
        hour=23, minute=59, second=59, microsecond=999999
    )
    after_date = datetime.now()
    after_date = after_date.replace(hour=0, minute=0, second=0, microsecond=0)
    with MonetaryCostManager(cost_threshold) as cost_manager:
        generator = QuestionGenerator(model=model)
        questions = await generator.generate_questions(
            number_of_questions=number_of_questions_to_generate,
            topic=f"Generate questions about {topic}",
            resolve_before_date=before_date,
            resolve_after_date=after_date,
        )

        assert (
            len(questions) == number_of_questions_to_generate
        ), f"Expected {number_of_questions_to_generate} questions, got {len(questions)}"

        for question in questions:
            assert isinstance(question, SimpleQuestion)
            assert question.question_text is not None
            assert question.resolution_criteria is not None
            assert question.background_information is not None
            assert question.expected_resolution_date is not None
            assert (
                topic.lower() in str(question).lower()
            ), f"Expected topic {topic} to be in question {question}"
            # assert (
            #     before_date > question.expected_resolution_date > after_date
            # ), f"Expected resolution date {question.expected_resolution_date} to be between {before_date} and {after_date}"

        final_cost = cost_manager.current_usage
        logger.info(f"Cost: ${final_cost:.4f}")
        assert (
            final_cost < cost_threshold
        ), f"Cost exceeded threshold: ${final_cost:.4f} > ${cost_threshold:.4f}"
        assert final_cost > 0, "Cost should be greater than 0"


async def test_question_generator_raises_on_invalid_dates() -> None:
    model = GeneralLlm(model="gpt-4o-mini")
    generator = QuestionGenerator(model=model)

    current_time = datetime.now()
    before_date = current_time
    after_date = current_time + timedelta(days=1)

    with pytest.raises(ValueError):
        await generator.generate_questions(
            number_of_questions=3,
            topic="Lithuania",
            resolve_before_date=before_date,
            resolve_after_date=after_date,
        )


async def test_question_generator_raises_on_invalid_question_count() -> None:
    model = GeneralLlm(model="gpt-4o-mini")
    generator = QuestionGenerator(model=model)

    before_date = datetime.now() + timedelta(days=14)
    after_date = datetime.now()

    with pytest.raises(ValueError):
        await generator.generate_questions(
            number_of_questions=0,
            topic="Lithuania",
            resolve_before_date=before_date,
            resolve_after_date=after_date,
        )

    with pytest.raises(ValueError):
        await generator.generate_questions(
            number_of_questions=-1,
            topic="Lithuania",
            resolve_before_date=before_date,
            resolve_after_date=after_date,
        )


async def test_question_generator_works_with_empty_topic() -> None:
    model = GeneralLlm(model="gpt-4o-mini")
    generator = QuestionGenerator(model=model)

    before_date = datetime.now() + timedelta(days=14)
    after_date = datetime.now()

    questions = await generator.generate_questions(
        number_of_questions=1,
        topic="",
        resolve_before_date=before_date,
        resolve_after_date=after_date,
    )

    assert len(questions) == 1
    assert isinstance(questions[0], SimpleQuestion)
