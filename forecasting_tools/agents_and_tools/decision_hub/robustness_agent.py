from __future__ import annotations

import logging

from agents.result import RunResultStreaming

from forecasting_tools.agents_and_tools.decision_hub.data_models import (
    Forecast,
    ForecastQuestion,
    PolicyProposal,
    RobustnessReport,
    ScenarioSet,
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

ROBUSTNESS_SYSTEM_PROMPT = clean_indents(
    """
    You are an expert decision analyst specializing in robustness analysis.
    Your job is to evaluate how well a policy proposal performs across
    different scenarios.

    Instructions:
    1. Build a Robustness Matrix: For each scenario, summarize expected
       outcomes under (a) status quo and (b) the proposed policy.
       Format as a markdown table.
    2. Identify Robust Recommendations: Actions that help across ALL
       scenarios (low-regret moves).
    3. Identify Scenario-Dependent Recommendations: Actions that only
       make sense under specific scenarios, with trigger conditions.
    4. Propose a Hedging Strategy: What to do to protect against the
       worst scenarios.
    5. Identify Key Decision Drivers: "The decision hinges on whether
       Driver X goes high or low."

    Format your output with these markdown sections:
    ## Robustness Matrix
    (markdown table: Scenario | Status Quo Outcome | With Policy Outcome | Net Impact)

    ## Key Decision Drivers
    ## Robust Recommendations
    ## Scenario-Dependent Recommendations
    ## Hedging Strategy
    """
)


def _build_robustness_agent(model: str = DEFAULT_MODEL) -> AiAgent:
    return AiAgent(
        name="Robustness Agent",
        instructions=ROBUSTNESS_SYSTEM_PROMPT,
        model=AgentSdkLlm(model=model),
        tools=[],
    )


def analyze_robustness_streamed(
    policy_question: str,
    scenarios: ScenarioSet,
    proposal: PolicyProposal,
    questions: list[ForecastQuestion],
    forecasts: list[Forecast],
    model: str = DEFAULT_MODEL,
) -> RunResultStreaming:
    agent = _build_robustness_agent(model)
    from forecasting_tools.agents_and_tools.decision_hub.forecast_agent import (
        _format_scenarios_for_prompt,
    )
    from forecasting_tools.agents_and_tools.decision_hub.proposal_agent import (
        _format_forecasts_for_prompt,
    )

    user_prompt = clean_indents(
        f"""
        Policy Question: {policy_question}

        Scenarios:
        {_format_scenarios_for_prompt(scenarios)}

        Policy Proposal:
        {proposal.proposal_markdown[:4000]}

        Key Recommendations:
        {chr(10).join(f"- {r}" for r in proposal.key_recommendations)}

        Forecasts:
        {_format_forecasts_for_prompt(questions, forecasts)}

        Perform a robustness analysis of this proposal across the scenarios.
        """
    )
    messages = [{"role": "user", "content": user_prompt}]
    return AgentRunner.run_streamed(agent, messages, max_turns=5)


async def analyze_robustness_to_completion(
    policy_question: str,
    scenarios: ScenarioSet,
    proposal: PolicyProposal,
    questions: list[ForecastQuestion],
    forecasts: list[Forecast],
    session_id: str,
    username: str,
    model: str = DEFAULT_MODEL,
) -> RobustnessReport:
    agent = _build_robustness_agent(model)
    from forecasting_tools.agents_and_tools.decision_hub.forecast_agent import (
        _format_scenarios_for_prompt,
    )
    from forecasting_tools.agents_and_tools.decision_hub.proposal_agent import (
        _format_forecasts_for_prompt,
    )

    user_prompt = clean_indents(
        f"""
        Policy Question: {policy_question}

        Scenarios:
        {_format_scenarios_for_prompt(scenarios)}

        Policy Proposal:
        {proposal.proposal_markdown[:4000]}

        Key Recommendations:
        {chr(10).join(f"- {r}" for r in proposal.key_recommendations)}

        Forecasts:
        {_format_forecasts_for_prompt(questions, forecasts)}

        Perform a robustness analysis of this proposal across the scenarios.
        """
    )
    messages = [{"role": "user", "content": user_prompt}]

    with general_trace_or_span("Decision Hub Robustness"):
        with MonetaryCostManager():
            result = await AgentRunner.run(agent, messages, max_turns=5)
            raw_text = result.final_output or ""

    structured = await structure_output(
        raw_text,
        RobustnessReport,
        additional_instructions=clean_indents(
            f"""
            Extract the robustness analysis from the text.
            Set session_id to "{session_id}" and username to "{username}".
            Put the robustness matrix (markdown table) in matrix_markdown.
            Put robust recommendations in robust_recommendations.
            Put scenario-dependent recommendations in scenario_dependent_recs.
            Put hedging strategy in hedging_strategy.
            """
        ),
    )
    structured.session_id = session_id
    structured.username = username
    return structured
