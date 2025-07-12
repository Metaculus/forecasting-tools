from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_tests.unit_tests.forecasting_test_manager import ForecastingTestManager
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.auto_optimizers.control_prompt import ControlPrompt
from forecasting_tools.auto_optimizers.customizable_bot import (
    CustomizableBot,
    PresetResearchStrategy,
    ResearchTool,
)
from forecasting_tools.auto_optimizers.prompt_data_models import (
    MockToolTracker,
    PromptIdea,
    ToolName,
)
from forecasting_tools.auto_optimizers.question_plus_research import (
    QuestionPlusResearch,
    ResearchItem,
    ResearchType,
)
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot

fake_question_1 = ForecastingTestManager.get_fake_binary_question(question_text="Q1")
fake_question_2 = ForecastingTestManager.get_fake_binary_question(question_text="Q2")


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock(spec=GeneralLlm)
    llm.invoke = AsyncMock(return_value="Probability: 50%")
    return llm


mock_research_tool = ResearchTool(tool_name=ToolName.MOCK_TOOL, max_calls=2)


@pytest.fixture
def research_snapshots() -> list[QuestionPlusResearch]:
    return [
        QuestionPlusResearch(
            question=fake_question_1,
            research_items=[
                ResearchItem(research="q1_news", type=ResearchType.ASK_NEWS_SUMMARIES)
            ],
        ),
        QuestionPlusResearch(
            question=fake_question_2,
            research_items=[
                ResearchItem(research="q2_news", type=ResearchType.ASK_NEWS_SUMMARIES)
            ],
        ),
    ]


@pytest.fixture
def customizable_bot(
    mock_llm: MagicMock,
    research_snapshots: list[QuestionPlusResearch],
) -> CustomizableBot:
    return CustomizableBot(
        reasoning_prompt=f"Test prompt with {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES}",
        research_prompt=f"Research prompt with {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES}",
        research_tools=[mock_research_tool],
        cached_research=research_snapshots,
        cached_research_type=ResearchType.ASK_NEWS_SUMMARIES,
        llms={"default": mock_llm, "researcher": "gpt-4o-mini"},
        originating_idea=PromptIdea(
            short_name="Test idea",
            full_text="Test idea process",
        ),
    )


async def test_customizable_bot_run_research_success(
    customizable_bot: CustomizableBot,
) -> None:
    question = fake_question_1
    research = await customizable_bot.run_research(question)
    assert research == "q1_news"


async def test_customizable_bot_run_research_question_not_found(
    customizable_bot: CustomizableBot,
) -> None:
    question = ForecastingTestManager.get_fake_binary_question(
        question_text="Q3 - not in snapshots"
    )

    with patch.object(
        customizable_bot,
        customizable_bot._run_research_with_tools.__name__,
        new_callable=AsyncMock,
    ) as mock_run_research:
        mock_run_research.return_value = "Research result from tools"

        result = await customizable_bot.run_research(question)

        assert result == "Research result from tools"
        mock_run_research.assert_called_once_with(question)


async def test_customizable_bot_run_research_duplicate_questions_in_snapshots(
    mock_llm: MagicMock,
) -> None:
    q1 = ForecastingTestManager.get_fake_binary_question(question_text="Q1")
    snapshots = [
        QuestionPlusResearch(
            question=q1,
            research_items=[
                ResearchItem(research="q1_news1", type=ResearchType.ASK_NEWS_SUMMARIES)
            ],
        ),
        QuestionPlusResearch(
            question=q1,  # Duplicate question
            research_items=[
                ResearchItem(research="q1_news2", type=ResearchType.ASK_NEWS_SUMMARIES)
            ],
        ),
    ]
    with pytest.raises(ValueError):
        CustomizableBot(
            reasoning_prompt="Test prompt",
            research_prompt="Research prompt",
            research_tools=[],
            cached_research=snapshots,
            cached_research_type=ResearchType.ASK_NEWS_SUMMARIES,
            llms={"default": mock_llm},
            originating_idea=PromptIdea(
                short_name="Test idea",
                full_text="Test idea process",
            ),
        )


async def test_customizable_bot_raises_error_when_no_researcher_llm_configured(
    mock_llm: MagicMock,
    research_snapshots: list[QuestionPlusResearch],
) -> None:
    with pytest.raises(ValueError, match="LLM is undefined"):
        bot = CustomizableBot(
            reasoning_prompt="Test prompt",
            research_prompt="Research prompt",
            research_tools=[mock_research_tool],
            cached_research=research_snapshots,
            cached_research_type=ResearchType.ASK_NEWS_SUMMARIES,
            llms={"default": mock_llm},
            originating_idea=PromptIdea(
                short_name="Test idea",
                full_text="Test idea process",
            ),
        )

        question = ForecastingTestManager.get_fake_binary_question(
            question_text="Q3 - not in snapshots"
        )

        await bot.run_research(question)


async def test_config_generation(customizable_bot: CustomizableBot) -> None:
    config = customizable_bot.get_config()
    assert config is not None
    assert str(customizable_bot.reasoning_prompt) in config["reasoning_prompt"]
    assert str(customizable_bot.research_prompt) in config["research_prompt"]


@pytest.mark.parametrize(
    "test_name, prompt, should_raise",
    [
        (
            "Happy path",
            f"Test prompt with {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES} {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING} {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES}",
            False,
        ),
        (
            "Happy path but with option variables included",
            f"Research: {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES} {CustomizableBot.OPTIONAL_RESEARCH_PROMPT_VARIABLES} {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING} Reasoning: {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES}",
            False,
        ),
        (
            "Unrecognized variables (research) (This is fine)",
            f"Research: {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES} {{invalid_variable}}{CustomizableBot.RESEARCH_REASONING_SPLIT_STRING} Reasoning: {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES} ",
            False,
        ),
        (
            "Unrecognized variables (reasoning) (This is fine)",
            f"Research: {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES} {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING} Reasoning: {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES} {{invalid_variable}}",
            False,
        ),
        ("Sad path", "None of them", True),
        (
            "Missing split string",
            f"Research: {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES} Reasoning: {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES}",
            True,
        ),
        (
            "Missing required research variables",
            f"Research: ...  {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING} Reasoning: {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES}",
            True,
        ),
        (
            "Missing required reasoning variables",
            f"Research: {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES} {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING} Reasoning: ...",
            True,
        ),
        (
            "Reasoning and research switched",
            f"Reasoning: {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES} {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING} Research: {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES}",
            True,
        ),
        (
            "Reasoning only variable included in research prompt",
            f"Research: {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES} {{research}} {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING} Reasoning: {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES}",
            True,
        ),
        (
            "Multiple research variables included in reasoning prompt",
            f"Research: {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES} {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING} Reasoning: {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES} {{research}}",
            True,
        ),
        (
            "Multiple split strings",
            f"Research: {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES} {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING} Reasoning: {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES} {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING}",
            True,
        ),
        (
            "Multiple of all research variables (This is fine)",
            f"Research: {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES} {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES} {CustomizableBot.OPTIONAL_RESEARCH_PROMPT_VARIABLES} {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING} Reasoning: {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES}",
            False,
        ),
    ],
)
def test_validate_combined_research_reasoning_prompt(
    test_name: str, prompt: str, should_raise: bool
) -> None:
    if should_raise:
        with pytest.raises(ValueError):
            CustomizableBot.validate_combined_research_reasoning_prompt(prompt)
    else:
        CustomizableBot.validate_combined_research_reasoning_prompt(prompt)


def test_split_and_combine_prompts_generate_equivalent_strings() -> None:
    research_prompt = f"Research: {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES}"
    reasoning_prompt = (
        f"Reasoning: {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES}"
    )
    combined_prompt = CustomizableBot.combine_research_reasoning_prompt(
        research_prompt, reasoning_prompt
    )

    new_research_prompt, new_reasoning_prompt = (
        CustomizableBot.split_combined_research_reasoning_prompt(combined_prompt)
    )
    assert new_research_prompt == research_prompt
    assert new_reasoning_prompt == reasoning_prompt


def test_get_benchmark_config_dict() -> None:
    research_prompt = (
        f"Research prompt with {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES}"
    )
    reasoning_prompt = (
        f"Test prompt with {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES}"
    )
    customizable_bot = CustomizableBot(
        research_prompt=research_prompt,
        reasoning_prompt=reasoning_prompt,
        research_tools=[mock_research_tool],
        cached_research=[],
        cached_research_type=None,
        llms={"default": mock_llm, "researcher": "gpt-4o-mini"},
        originating_idea=PromptIdea(
            short_name="Test idea",
            full_text="Test idea process",
        ),
    )
    config = customizable_bot.get_config()
    assert config is not None

    benchmark = BenchmarkForBot.initialize_benchmark_for_bot(
        customizable_bot,
        10,
    )
    benchmark_config = benchmark.forecast_bot_config
    assert benchmark_config is not None

    assert len(benchmark.research_tools_used) == 1
    assert benchmark.research_tools_used[0].tool_name == mock_research_tool.tool_name

    assert benchmark.bot_prompt == CustomizableBot.combine_research_reasoning_prompt(
        research_prompt, reasoning_prompt
    )


async def test_preset_research_strategies_work() -> None:
    before_hand_mock_calls = MockToolTracker.global_mock_tool_calls
    bot = CustomizableBot(
        research_prompt=PresetResearchStrategy.SEARCH_ASKNEWS_WITH_QUESTION_TEXT.value,
        reasoning_prompt=ControlPrompt.get_reasoning_prompt(),
        research_tools=[mock_research_tool],
        cached_research=[],
        cached_research_type=None,
        originating_idea=PromptIdea(
            short_name="Test idea",
            full_text="Test idea process",
        ),
        llms={"default": mock_llm, "researcher": mock_llm},
    )
    expected_research = "Here are the relevant news articles"
    original_function = (
        PresetResearchStrategy.SEARCH_ASKNEWS_WITH_QUESTION_TEXT.run_research
    )
    PresetResearchStrategy.SEARCH_ASKNEWS_WITH_QUESTION_TEXT.run_research = AsyncMock(
        return_value=expected_research
    )

    research = await bot.run_research(fake_question_1)
    assert research == expected_research
    assert (
        MockToolTracker.global_mock_tool_calls == before_hand_mock_calls
    ), "A call to the mock tool was made when it should not have been"

    PresetResearchStrategy.SEARCH_ASKNEWS_WITH_QUESTION_TEXT.run_research = (
        original_function
    )
