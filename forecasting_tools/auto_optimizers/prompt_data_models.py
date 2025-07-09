from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.minor_tools import (
    perplexity_pro_search,
    perplexity_quick_search_low_context,
    query_asknews,
)
from forecasting_tools.ai_models.agent_wrappers import AgentTool, agent_tool
from forecasting_tools.ai_models.general_llm import GeneralLlm


class PromptIdea(BaseModel):
    short_name: str
    full_text: str


class ToolName(str, Enum):
    ASKNEWS = "query_asknews"
    PERPLEXITY_LOW_COST = "perplexity_quick_search_low_context"
    PERPLEXITY_PRO_SEARCH = "perplexity_pro_search"
    MOCK_TOOL = "mock_tool"

    @staticmethod
    def name_to_tool_map() -> dict[ToolName, AgentTool]:
        return {
            ToolName.ASKNEWS: query_asknews,
            ToolName.PERPLEXITY_LOW_COST: perplexity_quick_search_low_context,
            ToolName.PERPLEXITY_PRO_SEARCH: perplexity_pro_search,
            ToolName.MOCK_TOOL: mock_tool,
        }

    def get_agent_tool(self) -> AgentTool:
        return self.name_to_tool_map()[self]


@agent_tool
def mock_tool(query: str) -> str:
    """
    Mock tool that returns a research result
    """
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
