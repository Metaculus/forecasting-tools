import pytest

from code_tests.unit_tests.forecasting_test_manager import ForecastingTestManager
from forecasting_tools.auto_optimizers.question_plus_research import (
    QuestionPlusResearch,
    ResearchItem,
    ResearchType,
)


def test_get_research_for_type_success() -> None:
    question = ForecastingTestManager.get_fake_binary_question()
    research_items = [
        ResearchItem(research="news data", type=ResearchType.ASK_NEWS_SUMMARIES),
        ResearchItem(research="other data", type=ResearchType.ASK_NEWS_DEEP_RESEARCH),
    ]
    snapshot = QuestionPlusResearch(question=question, research_items=research_items)

    news_research = snapshot.get_research_for_type(ResearchType.ASK_NEWS_SUMMARIES)
    assert news_research == "news data"


def test_get_research_for_type_not_found() -> None:
    question = ForecastingTestManager.get_fake_binary_question()
    research_items = [
        ResearchItem(research="other data", type=ResearchType.ASK_NEWS_DEEP_RESEARCH),
    ]
    snapshot = QuestionPlusResearch(question=question, research_items=research_items)

    with pytest.raises(ValueError):
        snapshot.get_research_for_type(ResearchType.ASK_NEWS_SUMMARIES)


def test_get_research_for_type_multiple_found() -> None:
    question = ForecastingTestManager.get_fake_binary_question()
    research_items = [
        ResearchItem(research="news data 1", type=ResearchType.ASK_NEWS_SUMMARIES),
        ResearchItem(research="news data 2", type=ResearchType.ASK_NEWS_SUMMARIES),
    ]
    snapshot = QuestionPlusResearch(question=question, research_items=research_items)

    with pytest.raises(ValueError):
        snapshot.get_research_for_type(ResearchType.ASK_NEWS_SUMMARIES)
