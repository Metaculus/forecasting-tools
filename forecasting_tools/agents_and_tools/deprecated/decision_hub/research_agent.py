from __future__ import annotations

import logging

from agents.result import RunResultStreaming

from forecasting_tools.agents_and_tools.decision_hub.data_models import ResearchReport
from forecasting_tools.agents_and_tools.minor_tools import (
    perplexity_reasoning_pro_search,
    query_asknews,
    smart_searcher_search,
)
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AiAgent,
    general_trace_or_span,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openrouter/anthropic/claude-sonnet-4"

RESEARCH_SYSTEM_PROMPT = clean_indents(
    """
    You are an expert policy researcher. Your job is to produce a comprehensive
    background research report on a given policy question.

    Instructions:
    - Use your search tools to gather information from multiple perspectives.
    - Run at least 3-5 searches on different aspects of the question.
    - Include key facts, statistics, stakeholder positions, historical context,
      and recent developments.
    - Cite all sources with URLs inline.
    - Structure your report with clear markdown headings.
    - Be objective and balanced -- present all sides.
    - End with a "Key Uncertainties" section listing the main unknowns.

    Format your output as a well-structured markdown report.
    """
)


def _build_research_agent(model: str = DEFAULT_MODEL) -> AiAgent:
    return AiAgent(
        name="Research Agent",
        instructions=RESEARCH_SYSTEM_PROMPT,
        model=AgentSdkLlm(model=model),
        tools=[
            perplexity_reasoning_pro_search,
            query_asknews,
            smart_searcher_search,
        ],
    )


def run_research_streamed(
    policy_question: str,
    custom_query: str | None = None,
    model: str = DEFAULT_MODEL,
) -> RunResultStreaming:
    agent = _build_research_agent(model)
    user_prompt = f"Research the following policy question:\n\n{policy_question}"
    if custom_query:
        user_prompt += f"\n\nFocus specifically on: {custom_query}"
    messages = [{"role": "user", "content": user_prompt}]
    return AgentRunner.run_streamed(agent, messages, max_turns=15)


async def run_research_to_completion(
    policy_question: str,
    session_id: str,
    username: str,
    custom_query: str | None = None,
    model: str = DEFAULT_MODEL,
) -> ResearchReport:
    agent = _build_research_agent(model)
    user_prompt = f"Research the following policy question:\n\n{policy_question}"
    if custom_query:
        user_prompt += f"\n\nFocus specifically on: {custom_query}"
    messages = [{"role": "user", "content": user_prompt}]

    with general_trace_or_span("Decision Hub Research"):
        with MonetaryCostManager():
            result = await AgentRunner.run(agent, messages, max_turns=15)
            report_text = result.final_output or ""

    return ResearchReport(
        session_id=session_id,
        username=username,
        query=custom_query or policy_question,
        report_markdown=report_text,
    )
