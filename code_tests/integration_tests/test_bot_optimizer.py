import logging

from code_tests.unit_tests.forecasting_test_manager import ForecastingTestManager
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.auto_optimizers.bot_optimizer import BotOptimizer
from forecasting_tools.auto_optimizers.control_prompt import ControlPrompt
from forecasting_tools.auto_optimizers.prompt_data_models import ResearchTool, ToolName
from forecasting_tools.auto_optimizers.prompt_optimizer import ScoredPrompt

logger = logging.getLogger(__name__)


async def test_bot_optimizer() -> None:
    # NOTE: This test costs around 10c
    question = ForecastingTestManager.get_fake_binary_question(
        question_text="Will the world end in 2100?"
    )
    with MonetaryCostManager(1) as cost_manager:
        initial_prompt_population_size = 4
        survivors_per_iteration = 2
        mutated_prompts_per_survivor = 2
        breeded_prompts_per_iteration = 1
        iterations_per_run = 2

        optimized_result = (
            await BotOptimizer.optimize_a_combined_research_and_reasoning_prompt(
                evaluation_questions=[question],
                research_tools_bot_can_use=[
                    ResearchTool(tool_name=ToolName.MOCK_TOOL, max_calls=1)
                ],
                research_agent_llm_name="gpt-4.1-mini",
                reasoning_llm=GeneralLlm(model="gpt-4.1-mini"),
                batch_size_for_question_evaluation=5,
                num_iterations_per_run=iterations_per_run,
                ideation_llm_name="o4-mini",
                remove_background_info_from_questions=True,
                folder_to_save_benchmarks=None,
                initial_prompt_population_size=initial_prompt_population_size,
                survivors_per_iteration=survivors_per_iteration,
                mutated_prompts_per_survivor=mutated_prompts_per_survivor,
                breeded_prompts_per_iteration=breeded_prompts_per_iteration,
            )
        )
        logger.info(f"Cost: {cost_manager.current_usage}")

    assert optimized_result is not None
    assert optimized_result.best_prompt.prompt.text is not None
    total_number_of_prompts = initial_prompt_population_size + (
        (survivors_per_iteration * mutated_prompts_per_survivor)
        + (breeded_prompts_per_iteration)
    ) * (iterations_per_run - 1)
    assert len(optimized_result.scored_prompts) == total_number_of_prompts

    control_prompt: ScoredPrompt | None = None
    for scored_prompt in optimized_result.scored_prompts:
        if (
            scored_prompt.score.metadata["benchmark"].forecast_bot_class_name
            == "Initial_Seed"
        ):
            control_prompt = scored_prompt
            break
    assert control_prompt is not None

    expected_control_prompt = (
        ControlPrompt.get_control_combined_prompt()
        .replace("\n", "")
        .replace(" ", "")
        .strip()
    )
    observed_control_prompt = (
        control_prompt.prompt.text.replace("\n", "").replace(" ", "").strip()
    )
    benchmark_prompt = (
        control_prompt.score.metadata["benchmark"]
        .bot_prompt.replace("\n", "")
        .replace(" ", "")
        .strip()
    )
    logger.info(f"Expected control prompt: \n{expected_control_prompt}")
    logger.info(f"Observed control prompt: \n{observed_control_prompt}")
    logger.info(f"Benchmark prompt: \n{benchmark_prompt}")

    assert observed_control_prompt == expected_control_prompt
    assert benchmark_prompt == expected_control_prompt
