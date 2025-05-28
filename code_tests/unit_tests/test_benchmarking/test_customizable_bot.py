from unittest.mock import AsyncMock, MagicMock

import pytest

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.benchmarking.customizable_bot import CustomizableBot
from forecasting_tools.benchmarking.prompt_data_models import PromptIdea
from forecasting_tools.benchmarking.question_research_snapshot import (
    QuestionResearchSnapshot,
    ResearchItem,
    ResearchType,
)

fake_question_1 = ForecastingTestManager.get_fake_binary_question(
    question_text="Q1"
)
fake_question_2 = ForecastingTestManager.get_fake_binary_question(
    question_text="Q2"
)


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock(spec=GeneralLlm)
    llm.invoke = AsyncMock(return_value="Probability: 50%")
    return llm


@pytest.fixture
def research_snapshots() -> list[QuestionResearchSnapshot]:
    return [
        QuestionResearchSnapshot(
            question=fake_question_1,
            research_items=[
                ResearchItem(
                    research="q1_news", type=ResearchType.ASK_NEWS_SUMMARIES
                )
            ],
        ),
        QuestionResearchSnapshot(
            question=fake_question_2,
            research_items=[
                ResearchItem(
                    research="q2_news", type=ResearchType.ASK_NEWS_SUMMARIES
                )
            ],
        ),
    ]


@pytest.fixture
def customizable_bot(
    mock_llm: MagicMock, research_snapshots: list[QuestionResearchSnapshot]
) -> CustomizableBot:
    return CustomizableBot(
        prompt="Test prompt with {question_text} and {research}",
        research_snapshots=research_snapshots,
        research_type=ResearchType.ASK_NEWS_SUMMARIES,
        llms={"default": mock_llm},
        originating_idea=PromptIdea(
            short_name="Test idea",
            idea="Test idea process",
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
    with pytest.raises(ValueError):
        await customizable_bot.run_research(question)


async def test_customizable_bot_run_research_duplicate_questions_in_snapshots(
    mock_llm: MagicMock,
) -> None:
    q1 = ForecastingTestManager.get_fake_binary_question(question_text="Q1")
    snapshots = [
        QuestionResearchSnapshot(
            question=q1,
            research_items=[
                ResearchItem(
                    research="q1_news1", type=ResearchType.ASK_NEWS_SUMMARIES
                )
            ],
        ),
        QuestionResearchSnapshot(
            question=q1,  # Duplicate question
            research_items=[
                ResearchItem(
                    research="q1_news2", type=ResearchType.ASK_NEWS_SUMMARIES
                )
            ],
        ),
    ]
    with pytest.raises(ValueError):
        CustomizableBot(
            prompt="Test prompt",
            research_snapshots=snapshots,
            research_type=ResearchType.ASK_NEWS_SUMMARIES,
            llms={"default": mock_llm},
            originating_idea=PromptIdea(
                short_name="Test idea",
                idea="Test idea process",
            ),
        )
