import logging

from code_tests.unit_tests.forecasting_test_manager import ForecastingTestManager
from forecasting_tools.auto_optimizers.control_prompt import ControlPrompt
from forecasting_tools.auto_optimizers.customizable_bot import CustomizableBot
from forecasting_tools.auto_optimizers.prompt_data_models import (
    MockToolTracker,
    PromptIdea,
    ResearchTool,
    ToolName,
)

logger = logging.getLogger(__name__)


async def test_customizable_bot_runs_control_bot() -> None:
    bot = CustomizableBot(
        reasoning_prompt=ControlPrompt.get_reasoning_prompt(),
        research_prompt=ControlPrompt.get_control_research_prompt(),  # This prompt is probably be a preset research strategy
        research_tools=[
            ResearchTool(tool_name=ToolName.PERPLEXITY_LOW_COST, max_calls=2)
        ],
        cached_research=[],
        cached_research_type=None,
        llms={"default": "gpt-4.1-mini", "researcher": "gpt-4.1-mini"},
        originating_idea=PromptIdea(
            short_name="Test idea",
            full_text="Test idea process",
        ),
    )

    fake_question_1 = ForecastingTestManager.get_fake_binary_question(
        question_text="Will Hungary win the next World Cup?"
    )
    research_result = await bot.run_research(fake_question_1)
    assert research_result is not None
    assert "Hungary" in research_result
    assert (
        "Here are the relevant news articles" in research_result
    ), "The catch phrase that starts AskNews research is not present"
    assert (
        "Original language" in research_result
    ), "The original language of the articles is not present which it should be if the AskNews formatted news is used"

    forecast_result = await bot.forecast_question(fake_question_1)
    assert forecast_result is not None


async def test_customizable_bot_respects_max_tool_calls_limit() -> None:
    starting_mock_tool_calls = MockToolTracker.global_mock_tool_calls
    bot = CustomizableBot(
        reasoning_prompt=f"Give me a probability of {{question_text}} happening. {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES}",
        research_prompt="Research the internet for {question_text}. If a tool fails, try it at least 4 more times.",
        research_tools=[ResearchTool(tool_name=ToolName.MOCK_TOOL, max_calls=2)],
        cached_research=[],
        cached_research_type=None,
        llms={"default": "gpt-4.1-mini", "researcher": "gpt-4.1-mini"},
        originating_idea=PromptIdea(
            short_name="Test idea",
            full_text="Test idea process",
        ),
    )

    fake_question_1 = ForecastingTestManager.get_fake_binary_question(
        question_text="Will the world end in 2025?"
    )
    await bot.run_research(fake_question_1)
    ending_mock_tool_calls = MockToolTracker.global_mock_tool_calls
    assert ending_mock_tool_calls - starting_mock_tool_calls == 2
