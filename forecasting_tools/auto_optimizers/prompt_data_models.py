from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.minor_tools import (
    perplexity_quick_search_low_context,
    perplexity_reasoning_pro_search,
    query_asknews,
)
from forecasting_tools.agents_and_tools.other.data_analyzer import DataAnalyzer
from forecasting_tools.ai_models.agent_wrappers import AgentTool, agent_tool
from forecasting_tools.ai_models.general_llm import GeneralLlm


class PromptIdea(BaseModel):
    short_name: str
    full_text: str


class ToolName(str, Enum):
    ASKNEWS = "query_asknews"
    PERPLEXITY_LOW_COST = "perplexity_quick_search_low_context"
    PERPLEXITY_REASONING_PRO_SEARCH = "perplexity_pro_search"
    MOCK_TOOL = "mock_tool"
    DATA_ANALYZER = "data_analyzer"

    @staticmethod
    def name_to_tool_map() -> dict[ToolName, AgentTool]:
        return {
            ToolName.ASKNEWS: query_asknews,
            ToolName.PERPLEXITY_LOW_COST: perplexity_quick_search_low_context,
            ToolName.PERPLEXITY_REASONING_PRO_SEARCH: perplexity_reasoning_pro_search,
            ToolName.MOCK_TOOL: mock_tool,
            ToolName.DATA_ANALYZER: DataAnalyzer.data_analysis_tool,
        }

    def get_agent_tool(self) -> AgentTool:
        return self.name_to_tool_map()[self]


class MockToolTracker:
    global_mock_tool_calls = 0


@agent_tool
def mock_tool(query: str) -> str:
    """
    Mock tool that returns a research result
    """
    MockToolTracker.global_mock_tool_calls += 1
    return f"No search result found for {query}. However based on previous information, the most likely forecast you are looking for is 50%"


class ResearchTool(BaseModel):
    tool_name: ToolName
    max_calls: int | None

    @property
    def tool(self) -> AgentTool:
        return self.tool_name.get_agent_tool()


@dataclass
class BotConfig:
    reasoning_prompt_template: str
    research_prompt_template: str
    research_tools: list[ResearchTool]
    reasoning_llm: GeneralLlm | str
    research_llm: GeneralLlm | str
    originating_idea: PromptIdea
    research_reports_per_question: int
    predictions_per_research_report: int
