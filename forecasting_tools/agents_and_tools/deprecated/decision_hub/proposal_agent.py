from __future__ import annotations

import logging

from agents.result import RunResultStreaming

from forecasting_tools.agents_and_tools.decision_hub.data_models import (
    Forecast,
    ForecastQuestion,
    PolicyProposal,
    ScenarioSet,
)
from forecasting_tools.agents_and_tools.minor_tools import (
    perplexity_reasoning_pro_search,
    query_asknews,
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
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openrouter/anthropic/claude-sonnet-4"

PROPOSAL_SYSTEM_PROMPT = clean_indents(
    """
    You are an expert policy analyst. Your job is to write a detailed,
    actionable policy proposal based on research, scenario analysis, and
    forecasts.

    Instructions:
    - Write a comprehensive policy proposal in markdown format.
    - Structure it with these sections:
      ## Executive Summary
      ## Problem Analysis
      ## Proposed Policy
      ## Key Recommendations
      (numbered list of specific, actionable recommendations)
      ## Implementation Considerations
      ## Risks and Mitigation
      ## Contingency Plans
      (what to do if specific scenarios materialize)
      ## Expected Outcomes
    - Reference the scenarios and forecasts throughout.
    - Be specific and actionable -- avoid vague platitudes.
    - Include contingency plans tied to scenarios (e.g., "If [scenario X]
      materializes, then [action Y]").
    - Cite sources inline where relevant.
    """
)


def _build_proposal_agent(model: str = DEFAULT_MODEL) -> AiAgent:
    return AiAgent(
        name="Proposal Agent",
        instructions=PROPOSAL_SYSTEM_PROMPT,
        model=AgentSdkLlm(model=model),
        tools=[perplexity_reasoning_pro_search, query_asknews],
    )


def _format_forecasts_for_prompt(
    questions: list[ForecastQuestion], forecasts: list[Forecast]
) -> str:
    forecast_by_qid = {f.question_id: f for f in forecasts}
    lines = []
    for q in questions:
        f = forecast_by_qid.get(q.question_id)
        pred = f.prediction if f else "No forecast"
        reasoning = f.reasoning[:200] if f and f.reasoning else ""
        cond = (
            f" [Under: {q.conditional_on_scenario}]"
            if q.conditional_on_scenario
            else ""
        )
        lines.append(f"- {q.question_text}{cond}: **{pred}** -- {reasoning}")
    return "\n".join(lines)


def generate_proposal_streamed(
    policy_question: str,
    research_context: str,
    scenarios: ScenarioSet,
    questions: list[ForecastQuestion],
    forecasts: list[Forecast],
    model: str = DEFAULT_MODEL,
) -> RunResultStreaming:
    agent = _build_proposal_agent(model)
    from forecasting_tools.agents_and_tools.decision_hub.forecast_agent import (
        _format_scenarios_for_prompt,
    )

    scenarios_text = _format_scenarios_for_prompt(scenarios)
    forecasts_text = _format_forecasts_for_prompt(questions, forecasts)
    user_prompt = clean_indents(
        f"""
        Policy Question: {policy_question}

        Background Research:
        {research_context[:4000]}

        Scenarios:
        {scenarios_text}

        Forecasts:
        {forecasts_text}

        Write a detailed policy proposal addressing this question.
        """
    )
    messages = [{"role": "user", "content": user_prompt}]
    return AgentRunner.run_streamed(agent, messages, max_turns=10)


async def generate_proposal_to_completion(
    policy_question: str,
    research_context: str,
    scenarios: ScenarioSet,
    questions: list[ForecastQuestion],
    forecasts: list[Forecast],
    session_id: str,
    username: str,
    model: str = DEFAULT_MODEL,
) -> PolicyProposal:
    agent = _build_proposal_agent(model)
    from forecasting_tools.agents_and_tools.decision_hub.forecast_agent import (
        _format_scenarios_for_prompt,
    )

    scenarios_text = _format_scenarios_for_prompt(scenarios)
    forecasts_text = _format_forecasts_for_prompt(questions, forecasts)
    user_prompt = clean_indents(
        f"""
        Policy Question: {policy_question}

        Background Research:
        {research_context[:4000]}

        Scenarios:
        {scenarios_text}

        Forecasts:
        {forecasts_text}

        Write a detailed policy proposal addressing this question.
        """
    )
    messages = [{"role": "user", "content": user_prompt}]

    with general_trace_or_span("Decision Hub Proposal"):
        with MonetaryCostManager():
            result = await AgentRunner.run(agent, messages, max_turns=10)
            raw_text = result.final_output or ""

    structured = await structure_output(
        raw_text,
        PolicyProposal,
        additional_instructions=clean_indents(
            f"""
            Extract the policy proposal from the text.
            Set session_id to "{session_id}" and username to "{username}".
            Put the full markdown text in proposal_markdown.
            Extract key_recommendations as a list of strings.
            Extract contingency_plans as a list of strings.
            """
        ),
    )
    structured.session_id = session_id
    structured.username = username
    return structured
