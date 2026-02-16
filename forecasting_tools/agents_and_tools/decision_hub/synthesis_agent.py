from __future__ import annotations

import logging

from agents.result import RunResultStreaming

from forecasting_tools.agents_and_tools.decision_hub.data_models import (
    Forecast,
    ForecastQuestion,
    PolicyProposal,
    ResearchReport,
    RobustnessReport,
    ScenarioSet,
    SynthesisReport,
)
from forecasting_tools.agents_and_tools.minor_tools import roll_dice
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

SYNTHESIS_SYSTEM_PROMPT = clean_indents(
    """
    You are an expert policy synthesis writer. Your job is to produce a final
    comprehensive report that ties together all the analysis into a coherent
    narrative.

    Instructions:
    - Produce a report with these sections:
      ## Executive Summary
      (2-3 paragraph overview of findings and recommendations)

      ## Full Analysis
      (Comprehensive writeup integrating research, scenarios, forecasts,
      proposals, and robustness analysis)

      ## Blog Post
      (A ~1500 word accessible blog-style version for a general audience.
      Should be engaging and informative.)

      ## Picture of the Future
      (A vivid narrative of what the world might look like in 2-5 years
      under the most likely scenario if recommendations are followed.
      Use the roll_dice tool to simulate uncertain outcomes based on
      forecast probabilities.)

    - Cite sources throughout.
    - Be specific and grounded in the analysis done.
    - The blog post should be self-contained and readable without the full report.
    """
)


def _build_synthesis_agent(model: str = DEFAULT_MODEL) -> AiAgent:
    return AiAgent(
        name="Synthesis Agent",
        instructions=SYNTHESIS_SYSTEM_PROMPT,
        model=AgentSdkLlm(model=model),
        tools=[roll_dice],
    )


def _build_context_prompt(
    policy_question: str,
    research_reports: list[ResearchReport],
    scenarios: ScenarioSet | None,
    questions: list[ForecastQuestion],
    forecasts: list[Forecast],
    proposal: PolicyProposal | None,
    robustness: RobustnessReport | None,
) -> str:
    from forecasting_tools.agents_and_tools.decision_hub.forecast_agent import (
        _format_scenarios_for_prompt,
    )
    from forecasting_tools.agents_and_tools.decision_hub.proposal_agent import (
        _format_forecasts_for_prompt,
    )

    research_text = "\n\n---\n\n".join(
        r.report_markdown[:2000] for r in research_reports
    )
    scenarios_text = (
        _format_scenarios_for_prompt(scenarios)
        if scenarios
        else "No scenarios generated."
    )
    forecasts_text = (
        _format_forecasts_for_prompt(questions, forecasts)
        if questions
        else "No forecasts."
    )
    proposal_text = proposal.proposal_markdown[:3000] if proposal else "No proposal."
    robustness_text = ""
    if robustness:
        robustness_text = f"""
Robustness Matrix:
{robustness.matrix_markdown}

Robust Recommendations:
{robustness.robust_recommendations}

Hedging Strategy:
{robustness.hedging_strategy}
"""

    return clean_indents(
        f"""
        Policy Question: {policy_question}

        Research:
        {research_text[:4000]}

        Scenarios:
        {scenarios_text}

        Forecasts:
        {forecasts_text}

        Policy Proposal:
        {proposal_text}

        Robustness Analysis:
        {robustness_text or "No robustness analysis."}

        Please synthesize all of the above into a final comprehensive report.
        """
    )


def synthesize_report_streamed(
    policy_question: str,
    research_reports: list[ResearchReport],
    scenarios: ScenarioSet | None,
    questions: list[ForecastQuestion],
    forecasts: list[Forecast],
    proposal: PolicyProposal | None,
    robustness: RobustnessReport | None,
    model: str = DEFAULT_MODEL,
) -> RunResultStreaming:
    agent = _build_synthesis_agent(model)
    user_prompt = _build_context_prompt(
        policy_question,
        research_reports,
        scenarios,
        questions,
        forecasts,
        proposal,
        robustness,
    )
    messages = [{"role": "user", "content": user_prompt}]
    return AgentRunner.run_streamed(agent, messages, max_turns=10)


async def synthesize_report_to_completion(
    policy_question: str,
    research_reports: list[ResearchReport],
    scenarios: ScenarioSet | None,
    questions: list[ForecastQuestion],
    forecasts: list[Forecast],
    proposal: PolicyProposal | None,
    robustness: RobustnessReport | None,
    session_id: str,
    username: str,
    model: str = DEFAULT_MODEL,
) -> SynthesisReport:
    agent = _build_synthesis_agent(model)
    user_prompt = _build_context_prompt(
        policy_question,
        research_reports,
        scenarios,
        questions,
        forecasts,
        proposal,
        robustness,
    )
    messages = [{"role": "user", "content": user_prompt}]

    with general_trace_or_span("Decision Hub Synthesis"):
        with MonetaryCostManager():
            result = await AgentRunner.run(agent, messages, max_turns=10)
            raw_text = result.final_output or ""

    structured = await structure_output(
        raw_text,
        SynthesisReport,
        additional_instructions=clean_indents(
            f"""
            Extract the synthesis report sections from the text.
            Set session_id to "{session_id}" and username to "{username}".
            - executive_summary: the Executive Summary section
            - full_report_markdown: the Full Analysis section
            - blog_post: the Blog Post section
            - future_snapshot: the Picture of the Future section
            """
        ),
    )
    structured.session_id = session_id
    structured.username = username
    return structured
