import asyncio
import logging

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.benchmarking.prompt_optimizer import PromptOptimizer
from forecasting_tools.benchmarking.question_research_snapshot import (
    QuestionResearchSnapshot,
)
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def run_optimizer() -> None:
    evaluation_questions = QuestionResearchSnapshot.load_json_from_file_path(
        "logs/forecasts/question_snapshots_v1_500qs.json"
    )
    logger.info(f"Loaded {len(evaluation_questions)} evaluation questions")

    forecast_llm = GeneralLlm(
        model="gpt-4o-mini",
        temperature=0.3,
    )
    ideation_llm = "openrouter/google/gemini-2.5-pro-preview"
    optimizer = PromptOptimizer(
        evaluation_questions=evaluation_questions,
        num_prompts_to_try=10,
        forecast_llm=forecast_llm,
        ideation_llm_name=ideation_llm,
    )
    await optimizer.create_optimized_prompt()


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(run_optimizer())
