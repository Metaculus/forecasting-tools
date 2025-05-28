from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.benchmarking.prompt_evaluator import PromptEvaluator
from forecasting_tools.benchmarking.prompt_optimizer import PromptOptimizer
from forecasting_tools.benchmarking.question_research_snapshot import (
    QuestionResearchSnapshot,
    ResearchItem,
    ResearchType,
)


async def test_prompt_optimizer() -> None:
    question = ForecastingTestManager.get_fake_binary_question()
    research_snapshot = QuestionResearchSnapshot(
        question=question,
        research_items=[
            ResearchItem(
                research="Something will happen!",
                type=ResearchType.ASK_NEWS_SUMMARIES,
            )
        ],
    )
    evaluator = PromptEvaluator(
        evaluation_questions=[research_snapshot],
        research_type=ResearchType.ASK_NEWS_SUMMARIES,
        concurrent_evaluation_batch_size=10,
        file_or_folder_to_save_benchmarks=None,
    )
    prompt_optimizer = PromptOptimizer(
        evaluator=evaluator,
        num_prompts_to_try=1,
        forecast_llm=GeneralLlm(model="gpt-4.1-nano"),
        ideation_llm_name="gpt-4.1-nano",
    )
    with MonetaryCostManager(1):
        optimized_result = await prompt_optimizer.create_optimized_prompt()
    assert optimized_result is not None
    assert optimized_result.best_prompt_text is not None
