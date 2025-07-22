from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import AliasChoices, BaseModel, Field, field_validator

from forecasting_tools.data_models.data_organizer import QuestionTypes
from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher
from forecasting_tools.util.jsonable import Jsonable


class ResearchType(Enum):
    ASK_NEWS_SUMMARIES = "ask_news_summaries"
    ASK_NEWS_DEEP_RESEARCH = "ask_news_deep_research"
    # TODO: Add other research types
    # PERPLEXITY_SEARCHES = "perplexity_searches"
    # EXA_SEARCHES = "exa_searches"
    # SMART_SEARCHES = "smart_searches"
    # GOOGLE_GROUNDING = "google_grounding"
    # ONE_SENTENCE_BACKGROUND_INFO = "one_sentence_background_info"


class ResearchItem(BaseModel):
    research: str
    type: ResearchType

    @field_validator("research")
    @classmethod
    def validate_research_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError(
                "Research field cannot be empty or contain only whitespace"
            )
        return value


class QuestionPlusResearch(BaseModel, Jsonable):
    question: QuestionTypes
    research_items: list[ResearchItem]
    research_timestamp: datetime = Field(
        default_factory=datetime.now,
        validation_alias=AliasChoices("research_timestamp", "time_stamp"),
    )

    @classmethod
    async def create_snapshot_of_question(
        cls, question: QuestionTypes
    ) -> QuestionPlusResearch:
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
            item.research for item in self.research_items if item.type == research_type
        ]
        if len(items) != 1:
            raise ValueError(
                f"Expected 1 research item for type {research_type}, got {len(items)}"
            )
        return items[0]
