import asyncio
import logging

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.auto_optimizers.bot_optimizer import BotOptimizer
from forecasting_tools.auto_optimizers.prompt_data_models import ResearchTool, ToolName
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def run_optimizer() -> None:
    # ----- Settings for the optimizer -----
    metaculus_question_path = (
        "logs/forecasts/benchmarks/questions_v4.0.train__50qs.json"
    )
    questions = DataOrganizer.load_questions_from_file_path(metaculus_question_path)
    questions_batch_size = 25
    research_tools = [
        ResearchTool(
            tool_name=ToolName.PERPLEXITY_LOW_COST,
            max_calls=7,
        ),
        ResearchTool(
            tool_name=ToolName.ASKNEWS,
            max_calls=2,
        ),
        ResearchTool(
            tool_name=ToolName.DATA_ANALYZER,
            max_calls=1,
        ),
        ResearchTool(
            tool_name=ToolName.PERPLEXITY_REASONING_PRO_SEARCH,
            max_calls=1,
        ),
    ]
    ideation_llm = "openrouter/google/gemini-2.5-pro"
    research_coordination_llm = "openrouter/openai/gpt-4.1-nano"
    reasoning_llm = GeneralLlm(model="openrouter/openai/gpt-4.1-nano", temperature=0.3)
    folder_to_save_benchmarks = "logs/forecasts/benchmarks/"
    num_iterations_per_run = 3
    remove_background_info = True
    initial_prompt_population_size = 20
    survivors_per_iteration = 5
    mutated_prompts_per_survivor = 3
    breeded_prompts_per_iteration = 5

    # ------ Run the optimizer -----
    await BotOptimizer.optimize_a_combined_research_and_reasoning_prompt(
        evaluation_questions=questions,
        research_tools_bot_can_use=research_tools,
        research_agent_llm_name=research_coordination_llm,
        reasoning_llm=reasoning_llm,
        batch_size_for_question_evaluation=questions_batch_size,
        num_iterations_per_run=num_iterations_per_run,
        ideation_llm_name=ideation_llm,
        remove_background_info_from_questions=remove_background_info,
        folder_to_save_benchmarks=folder_to_save_benchmarks,
        initial_prompt_population_size=initial_prompt_population_size,
        survivors_per_iteration=survivors_per_iteration,
        mutated_prompts_per_survivor=mutated_prompts_per_survivor,
        breeded_prompts_per_iteration=breeded_prompts_per_iteration,
    )


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(run_optimizer())
