from __future__ import annotations

import logging

from agents.result import RunResultStreaming

from forecasting_tools.agents_and_tools.decision_hub.data_models import ScenarioSet
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

SCENARIO_SYSTEM_PROMPT = clean_indents(
    """
    You are an expert scenario planner. Your job is to identify key external
    drivers of uncertainty and construct plausible future scenarios for a
    policy question.

    Instructions:
    1. Review the policy question and background research provided.
    2. Identify 2-3 key EXTERNAL drivers of uncertainty. These must be:
       - External (not policy choices the decision-maker controls)
       - Uncertain (reasonable people disagree on which way they'll go)
       - Impactful (they significantly change which policy is best)
    3. For each driver, describe its high and low states.
    4. Construct 3-4 named scenarios from combinations of the top 2 drivers.
    5. For each scenario, write a 2-3 sentence narrative.
    6. Optionally assign rough probability estimates.

    Format your output as structured markdown with these sections:
    ## Rationale
    (Why these drivers matter)

    ## Drivers
    For each driver:
    ### Driver: [Name]
    - Description: ...
    - High state: ...
    - Low state: ...

    ## Scenarios
    For each scenario:
    ### Scenario: [Name]
    - Driver states: ...
    - Narrative: ...
    - Estimated probability: ...
    """
)


def _build_scenario_agent(model: str = DEFAULT_MODEL) -> AiAgent:
    return AiAgent(
        name="Scenario Agent",
        instructions=SCENARIO_SYSTEM_PROMPT,
        model=AgentSdkLlm(model=model),
        tools=[perplexity_reasoning_pro_search, query_asknews],
    )


def generate_scenarios_streamed(
    policy_question: str,
    research_context: str,
    model: str = DEFAULT_MODEL,
) -> RunResultStreaming:
    agent = _build_scenario_agent(model)
    user_prompt = clean_indents(
        f"""
        Policy Question: {policy_question}

        Background Research:
        {research_context}

        Please identify the key external drivers of uncertainty and construct
        plausible future scenarios.
        """
    )
    messages = [{"role": "user", "content": user_prompt}]
    return AgentRunner.run_streamed(agent, messages, max_turns=10)


async def generate_scenarios_to_completion(
    policy_question: str,
    research_context: str,
    session_id: str,
    username: str,
    model: str = DEFAULT_MODEL,
) -> ScenarioSet:
    agent = _build_scenario_agent(model)
    user_prompt = clean_indents(
        f"""
        Policy Question: {policy_question}

        Background Research:
        {research_context}

        Please identify the key external drivers of uncertainty and construct
        plausible future scenarios.
        """
    )
    messages = [{"role": "user", "content": user_prompt}]

    with general_trace_or_span("Decision Hub Scenarios"):
        with MonetaryCostManager():
            result = await AgentRunner.run(agent, messages, max_turns=10)
            raw_text = result.final_output or ""

    structured = await structure_output(
        raw_text,
        ScenarioSet,
        additional_instructions=clean_indents(
            f"""
            Extract scenario drivers and scenarios from the text.
            Set session_id to "{session_id}" and username to "{username}".
            For each driver, extract name, description, high_state, low_state.
            For each scenario, extract name, narrative, driver_states (dict
            mapping driver name to its state), and probability if given.
            Also extract the rationale text.
            """
        ),
    )
    structured.session_id = session_id
    structured.username = username
    return structured
