import logging

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

    with MonetaryCostManager(cost_threshold) as cost_manager:
        generator = QuestionGenerator(model=model)
        questions = await generator.generate_questions(
            number_of_questions=number_of_questions_to_generate,
            topic=f"Generate questions about {topic}",
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
            assert topic.lower() in str(question).lower()

        final_cost = cost_manager.current_usage
        logger.info(f"Cost: ${final_cost:.4f}")
        assert (
            final_cost < cost_threshold
        ), f"Cost exceeded threshold: ${final_cost:.4f} > ${cost_threshold:.4f}"
        assert final_cost > 0, "Cost should be greater than 0"
