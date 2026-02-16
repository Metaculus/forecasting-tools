from __future__ import annotations

import logging

from agents.result import RunResultStreaming

from forecasting_tools.agents_and_tools.decision_hub.data_models import (
    Forecast,
    ForecastQuestion,
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
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openrouter/anthropic/claude-sonnet-4"

QUESTION_GEN_PROMPT = clean_indents(
    """
    You are an expert forecaster. Your job is to generate precise, measurable
    forecasting questions that would help evaluate a policy decision.

    Instructions:
    - Generate 5-8 forecasting questions total:
      - 2-3 baseline questions (about the status quo, not conditional on any scenario)
      - 2-5 scenario-conditional questions (how outcomes differ under each scenario)
    - Each question must have:
      - Clear, unambiguous question text
      - Specific resolution criteria (how we'd know if it resolved yes/no)
      - A question_type (baseline or scenario_conditional)
      - If scenario_conditional, specify which scenario it is conditional on
    - Questions should be binary (yes/no resolvable) with a specific time horizon.
    - Focus on questions where the answer meaningfully differs across scenarios.
    """
)

FORECAST_SYSTEM_PROMPT = clean_indents(
    """
    You are an expert superforecaster. Your job is to produce calibrated
    probability estimates for forecasting questions.

    Instructions:
    - For each question, provide:
      - A probability estimate (e.g., "65%" or "0.65")
      - Detailed reasoning explaining your estimate
      - Key sources or reference classes you considered
    - Use outside view (base rates) and inside view (specific factors).
    - Consider each scenario's context when answering scenario-conditional questions.
    - Be well-calibrated: don't cluster around 50%, use extreme probabilities
      when warranted.
    - Cite sources inline where possible.

    Format each forecast as:
    ## Question: [question text]
    **Prediction:** [probability]
    **Reasoning:** [your reasoning]
    **Key Sources:** [sources]
    """
)


async def generate_forecast_questions(
    policy_question: str,
    research_context: str,
    scenarios: ScenarioSet,
    session_id: str,
    username: str,
    model: str = DEFAULT_MODEL,
) -> list[ForecastQuestion]:
    scenarios_text = _format_scenarios_for_prompt(scenarios)
    prompt = clean_indents(
        f"""
        Policy Question: {policy_question}

        Background Research (summary):
        {research_context[:3000]}

        Scenarios:
        {scenarios_text}

        Generate forecasting questions for this policy analysis.
        Return them as a list of ForecastQuestion objects.
        """
    )

    llm = GeneralLlm(model=model, temperature=0.3)
    with general_trace_or_span("Decision Hub Forecast Questions"):
        questions = await llm.invoke_and_return_verified_type(
            f"{QUESTION_GEN_PROMPT}\n\n{prompt}",
            list[ForecastQuestion],
        )

    for q in questions:
        q.session_id = session_id
        q.username = username
    return questions


def _build_forecast_agent(model: str = DEFAULT_MODEL) -> AiAgent:
    return AiAgent(
        name="Forecast Agent",
        instructions=FORECAST_SYSTEM_PROMPT,
        model=AgentSdkLlm(model=model),
        tools=[perplexity_reasoning_pro_search, query_asknews],
    )


def _format_scenarios_for_prompt(scenarios: ScenarioSet) -> str:
    lines = []
    for s in scenarios.scenarios:
        driver_info = ", ".join(f"{k}: {v}" for k, v in s.driver_states.items())
        prob_str = f" (p={s.probability})" if s.probability else ""
        lines.append(f"- **{s.name}**{prob_str}: {s.narrative} [{driver_info}]")
    return "\n".join(lines)


def _format_questions_for_prompt(questions: list[ForecastQuestion]) -> str:
    lines = []
    for i, q in enumerate(questions, 1):
        cond = ""
        if q.conditional_on_scenario:
            cond = f" [Conditional on: {q.conditional_on_scenario}]"
        lines.append(
            f"{i}. ({q.question_type}) {q.question_text}{cond}\n"
            f"   Resolution: {q.resolution_criteria}"
        )
    return "\n".join(lines)


def run_forecasts_streamed(
    questions: list[ForecastQuestion],
    policy_question: str,
    research_context: str,
    scenarios: ScenarioSet,
    model: str = DEFAULT_MODEL,
) -> RunResultStreaming:
    agent = _build_forecast_agent(model)
    scenarios_text = _format_scenarios_for_prompt(scenarios)
    questions_text = _format_questions_for_prompt(questions)
    user_prompt = clean_indents(
        f"""
        Policy Question: {policy_question}

        Key Research Context:
        {research_context[:3000]}

        Scenarios:
        {scenarios_text}

        Please forecast the following questions:

        {questions_text}
        """
    )
    messages = [{"role": "user", "content": user_prompt}]
    return AgentRunner.run_streamed(agent, messages, max_turns=10)


async def run_forecasts_to_completion(
    questions: list[ForecastQuestion],
    policy_question: str,
    research_context: str,
    scenarios: ScenarioSet,
    session_id: str,
    username: str,
    model: str = DEFAULT_MODEL,
) -> list[Forecast]:
    agent = _build_forecast_agent(model)
    scenarios_text = _format_scenarios_for_prompt(scenarios)
    questions_text = _format_questions_for_prompt(questions)
    user_prompt = clean_indents(
        f"""
        Policy Question: {policy_question}

        Key Research Context:
        {research_context[:3000]}

        Scenarios:
        {scenarios_text}

        Please forecast the following questions:

        {questions_text}
        """
    )
    messages = [{"role": "user", "content": user_prompt}]

    with general_trace_or_span("Decision Hub Forecasts"):
        with MonetaryCostManager():
            result = await AgentRunner.run(agent, messages, max_turns=10)
            raw_text = result.final_output or ""

    forecasts = await structure_output(
        raw_text,
        list[Forecast],
        additional_instructions=clean_indents(
            f"""
            Extract forecasts from the text. For each forecast:
            - Match it to a question_id from this list:
            {chr(10).join(f'  question_id="{q.question_id}" -> "{q.question_text}"' for q in questions)}
            - Set session_id to "{session_id}" and username to "{username}"
            - Extract prediction (probability string), reasoning, and key_sources
            """
        ),
    )
    for f in forecasts:
        f.session_id = session_id
        f.username = username
    return forecasts
