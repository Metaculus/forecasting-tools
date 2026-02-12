from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from forecasting_tools.agents_and_tools.situation_simulator.data_models import (
    SimulationState,
    Situation,
)
from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.data_models import (
    ForecastCategory,
    InterventionForecast,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)

OPERATOR_MAP: dict[str, callable] = {
    ">=": lambda actual, threshold: actual >= threshold,
    ">": lambda actual, threshold: actual > threshold,
    "<=": lambda actual, threshold: actual <= threshold,
    "<": lambda actual, threshold: actual < threshold,
    "==": lambda actual, threshold: actual == threshold,
    "!=": lambda actual, threshold: actual != threshold,
}


class QualitativeResolution(BaseModel, Jsonable):
    resolved_yes: bool = Field(
        description="True if the event described in the question DID occur, False otherwise"
    )
    reasoning: str = Field(
        description="Brief explanation of why this resolved yes or no based on the evidence"
    )


def resolve_hard_metric_forecast(
    forecast: InterventionForecast,
    final_state: SimulationState,
) -> InterventionForecast:
    if forecast.hard_metric_criteria is None:
        logger.warning(
            f"Hard metric forecast '{forecast.question_title}' has no structured criteria, "
            "falling back to unresolved"
        )
        return forecast

    criteria = forecast.hard_metric_criteria
    agent_inventory = final_state.inventories.get(criteria.agent_name, {})
    actual_value = agent_inventory.get(criteria.item_name, 0)

    comparator = OPERATOR_MAP.get(criteria.operator)
    if comparator is None:
        logger.warning(
            f"Unknown operator '{criteria.operator}' in forecast '{forecast.question_title}'"
        )
        return forecast

    resolution = comparator(actual_value, criteria.threshold)
    brier_score = calculate_brier_score(forecast.prediction, resolution)

    logger.info(
        f"Hard metric resolved: '{forecast.question_title}' - "
        f"{criteria.agent_name}.{criteria.item_name} = {actual_value} "
        f"{criteria.operator} {criteria.threshold} -> {resolution} "
        f"(predicted {forecast.prediction:.2f}, brier={brier_score:.4f})"
    )

    return forecast.model_copy(
        update={
            "resolved": True,
            "resolution": resolution,
            "brier_score": brier_score,
        }
    )


async def resolve_qualitative_forecast(
    forecast: InterventionForecast,
    final_state: SimulationState,
    situation: Situation,
    resolution_model: GeneralLlm | None = None,
) -> InterventionForecast:
    if resolution_model is None:
        resolution_model = GeneralLlm(
            "openrouter/openai/gpt-4.1-mini",
            temperature=0.1,
            timeout=120,
        )

    transcript = _build_simulation_transcript(final_state)

    resolution_prompt = clean_indents(
        f"""
        You are a judge resolving a forecast question based on a simulation transcript.

        ## Question
        **{forecast.question_title}**
        {forecast.question_text}

        ## Resolution Criteria
        {forecast.resolution_criteria}

        ## Simulation Transcript

        {transcript}

        ## Your Task

        Based on the simulation transcript above, determine whether the event
        described in the question DID occur (resolved_yes = true) or DID NOT
        occur (resolved_yes = false).

        Be objective and base your judgment strictly on evidence in the transcript.
        If the evidence is ambiguous, lean toward the interpretation that is most
        supported by concrete actions, messages, and inventory changes.
        """
    )

    raw_output = await resolution_model.invoke(resolution_prompt)
    result = await structure_output(
        raw_output,
        QualitativeResolution,
        additional_instructions="Extract whether the event occurred (resolved_yes) and the reasoning.",
    )

    resolution = result.resolved_yes
    brier_score = calculate_brier_score(forecast.prediction, resolution)

    logger.info(
        f"Qualitative resolved: '{forecast.question_title}' -> {resolution} "
        f"(predicted {forecast.prediction:.2f}, brier={brier_score:.4f}) "
        f"Reason: {result.reasoning[:100]}"
    )

    return forecast.model_copy(
        update={
            "resolved": True,
            "resolution": resolution,
            "brier_score": brier_score,
        }
    )


async def resolve_all_forecasts(
    forecasts: list[InterventionForecast],
    status_quo_final_state: SimulationState,
    intervention_final_state: SimulationState,
    situation: Situation,
    resolution_model: GeneralLlm | None = None,
) -> list[InterventionForecast]:
    resolved: list[InterventionForecast] = []
    for forecast in forecasts:
        final_state = (
            intervention_final_state
            if forecast.is_conditional
            else status_quo_final_state
        )
        if forecast.category == ForecastCategory.HARD_METRIC:
            resolved_forecast = resolve_hard_metric_forecast(forecast, final_state)
        else:
            resolved_forecast = await resolve_qualitative_forecast(
                forecast, final_state, situation, resolution_model
            )
        resolved.append(resolved_forecast)
    return resolved


def calculate_brier_score(prediction: float, resolution: bool) -> float:
    outcome = 1.0 if resolution else 0.0
    return (prediction - outcome) ** 2


def _build_simulation_transcript(state: SimulationState) -> str:
    sections: list[str] = []
    _append_inventories(sections, state)
    _append_messages(sections, state)
    _append_actions(sections, state)
    _append_trades(sections, state)
    return "\n".join(sections)


def _append_inventories(sections: list[str], state: SimulationState) -> None:
    sections.append("### Final Inventories")
    for agent_name, inventory in state.inventories.items():
        items_text = ", ".join(f"{k}: {v}" for k, v in inventory.items())
        sections.append(f"- {agent_name}: {items_text}")
    env_items = ", ".join(f"{k}: {v}" for k, v in state.environment_inventory.items())
    if env_items:
        sections.append(f"- Environment: {env_items}")


def _append_messages(sections: list[str], state: SimulationState) -> None:
    sections.append("\n### Message History")
    for msg in state.message_history:
        if msg.channel:
            sections.append(
                f"[Step {msg.step}] #{msg.channel} | {msg.sender}: {msg.content}"
            )
        else:
            recipients = [r for r in msg.recipients if r != msg.sender]
            dm_target = recipients[0] if recipients else "unknown"
            sections.append(
                f"[Step {msg.step}] DM {msg.sender} -> {dm_target}: {msg.content}"
            )


def _append_actions(sections: list[str], state: SimulationState) -> None:
    sections.append("\n### Action Log")
    for action in state.action_log:
        if action.action_name == "no_action":
            sections.append(f"- {action.agent_name}: no_action")
        else:
            params_text = ", ".join(f"{k}={v}" for k, v in action.parameters.items())
            sections.append(
                f"- {action.agent_name}: {action.action_name}({params_text})"
            )


def _append_trades(sections: list[str], state: SimulationState) -> None:
    sections.append("\n### Trade History")
    if state.trade_history:
        for trade in state.trade_history:
            sections.append(
                f"- Step {trade.step}: {trade.from_agent} -> {trade.to_agent}: "
                f"{trade.quantity} {trade.item_name}"
            )
    else:
        sections.append("No completed trades.")
