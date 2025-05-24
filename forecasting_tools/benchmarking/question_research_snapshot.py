from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusQuestion
from forecasting_tools.util.jsonable import Jsonable


class ResearchType(Enum):
    ASK_NEWS_SUMMARIES = "ask_news_summaries"


class ResearchItem(BaseModel):
    research: str
    type: ResearchType


class QuestionResearchSnapshot(BaseModel, Jsonable):
    question: MetaculusQuestion
    research_items: list[ResearchItem]
    time_stamp: datetime = Field(default_factory=datetime.now)

    @classmethod
    async def create_snapshot_of_question(
        cls, question: MetaculusQuestion
    ) -> QuestionResearchSnapshot:
        ask_news_summaries = await AskNewsSearcher().get_formatted_news_async(
            question.question_text
        )
        return cls(
            question=question,
            research_items=[
                ResearchItem(
                    research=ask_news_summaries,
                    type=ResearchType.ASK_NEWS_SUMMARIES,
                )
            ],
        )

    def get_research_for_type(self, research_type: ResearchType) -> str:
        items = [
            item.research
            for item in self.research_items
            if item.type == research_type
        ]
        if len(items) != 1:
            raise ValueError(
                f"Expected 1 research item for type {research_type}, got {len(items)}"
            )
        return items[0]
