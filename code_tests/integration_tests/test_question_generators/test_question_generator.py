import logging
from datetime import datetime, timedelta

import pytest

from forecasting_tools.agents_and_tools.deprecated.question_generator import (
    QuestionGenerator,
)
from forecasting_tools.agents_and_tools.question_generators.generated_question import (
    GeneratedQuestion,
)
from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

logger = logging.getLogger(__name__)


@pytest.mark.skip(reason="Skipping question generator test since its expensive")
async def test_question_generator_returns_necessary_number_and_stays_within_cost() -> (
    None
):
    # TODO: Move this to 'expensive' tests and also test small/large date ranges

    number_of_questions_to_generate = 2
    cost_threshold = 2
    topic = "Lithuania"
    model = GeneralLlm(model="gpt-4o-mini")
    before_date = datetime.now() + timedelta(days=14)
    before_date = before_date.replace(hour=23, minute=59, second=59, microsecond=999999)
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

        for question in questions:
            assert isinstance(question, GeneratedQuestion)
            assert question.question_text is not None
            assert question.resolution_criteria is not None
            assert question.background_information is not None
            assert question.expected_resolution_date is not None
            assert (
                topic.lower() in str(question).lower()
            ), f"Expected topic {topic} to be in question {question}"
            assert question.forecast_report is not None
            if question.question_type == "numeric":
                assert question.min_value is not None
                assert question.max_value is not None
                assert question.open_lower_bound is not None
                assert question.open_upper_bound is not None
            if question.question_type == "multiple_choice":
                assert question.options is not None
                assert len(question.options) > 0

        assert (
            len(questions) >= number_of_questions_to_generate
        ), f"Expected {number_of_questions_to_generate} questions, got {len(questions)}"
        assert (
            len([question for question in questions if question.is_uncertain])
            >= number_of_questions_to_generate
        ), f"Expected {number_of_questions_to_generate} uncertain questions, got {len([question for question in questions if question.is_uncertain])}. Questions: {questions}"
        assert (
            len(
                [
                    question
                    for question in questions
                    if before_date > question.expected_resolution_date > after_date
                ]
            )
            >= number_of_questions_to_generate
        ), f"Expected {number_of_questions_to_generate} questions to be resolved between {before_date} and {after_date}, got {len([question for question in questions if before_date > question.expected_resolution_date > after_date])}"

        final_cost = cost_manager.current_usage
        logger.info(f"Cost: ${final_cost:.4f}")
        assert final_cost > 0, "Cost should be greater than 0"
        assert (
            final_cost < cost_threshold
        ), f"Cost exceeded threshold: ${final_cost:.4f} > ${cost_threshold:.4f}"


@pytest.mark.skip(reason="Skipping question generator test since its expensive")
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
