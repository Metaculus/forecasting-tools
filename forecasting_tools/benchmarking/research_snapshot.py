from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusQuestion


class ResearchSnapshot(BaseModel):
    question: MetaculusQuestion
    ask_news_summaries: str
    time_stamp: datetime = Field(default_factory=datetime.now)

    @classmethod
    async def question_to_research_snapshot(
        cls, question: MetaculusQuestion
    ) -> ResearchSnapshot:
        ask_news_summaries = await AskNewsSearcher().get_formatted_news_async(
            question.question_text
        )
        return cls(question=question, ask_news_summaries=ask_news_summaries)
