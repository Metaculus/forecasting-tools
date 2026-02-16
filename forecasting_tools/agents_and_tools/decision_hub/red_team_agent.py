from __future__ import annotations

import logging
from typing import Literal

from agents.result import RunResultStreaming

from forecasting_tools.agents_and_tools.decision_hub.data_models import RedTeamResult
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

DEVILS_ADVOCATE_PROMPT = clean_indents(
    """
    You are a contrarian adversarial critic -- a "Devil's Advocate."
    Your job is to find weaknesses, counterarguments, and failure modes
    in the text provided.

    Instructions:
    - Assume the text's conclusions are WRONG and argue why.
    - Steel-man the opposing view -- present the strongest possible
      counterarguments.
    - Identify hidden assumptions that may not hold.
    - Find logical fallacies or weak reasoning.
    - Point out what evidence is missing or cherry-picked.
    - Identify potential unintended consequences.
    - Suggest what would need to be true for the opposite conclusion.
    - Be thorough but constructive -- the goal is to strengthen the
      analysis, not tear it down gratuitously.

    Structure your critique as:
    ## Key Vulnerabilities
    ## Strongest Counterarguments
    ## Hidden Assumptions
    ## Missing Evidence
    ## Unintended Consequences
    ## What Would Change Your Mind
    """
)

BIAS_DETECTOR_PROMPT = clean_indents(
    """
    You are a cognitive bias expert and reasoning analyst. Your job is to
    identify cognitive biases, logical errors, and systematic reasoning
    patterns in the text provided.

    Instructions:
    - Scan for common cognitive biases including but not limited to:
      - Anchoring bias
      - Availability bias
      - Confirmation bias
      - Groupthink
      - Base rate neglect
      - Scope insensitivity
      - Status quo bias
      - Sunk cost fallacy
      - Narrative fallacy
      - Overconfidence
      - Dunning-Kruger effect
      - Planning fallacy
      - Survivorship bias
    - For each bias found, explain:
      - What the bias is
      - Where it appears in the text (with quotes)
      - How it might distort the conclusions
      - How to correct for it
    - Also identify structural reasoning issues:
      - Circular reasoning
      - False dichotomies
      - Hasty generalizations
      - Appeal to authority without evidence
      - Correlation vs causation confusion
    - Rate the overall reasoning quality (Strong / Moderate / Weak)

    Structure your analysis as:
    ## Overall Reasoning Quality: [rating]
    ## Cognitive Biases Detected
    (for each bias found)
    ### [Bias Name]
    - **Where:** [quote from text]
    - **Impact:** [how it distorts conclusions]
    - **Correction:** [how to fix]
    ## Structural Reasoning Issues
    ## Recommendations for Improvement
    """
)


def _get_system_prompt(mode: Literal["devils_advocate", "bias_detector"]) -> str:
    if mode == "devils_advocate":
        return DEVILS_ADVOCATE_PROMPT
    return BIAS_DETECTOR_PROMPT


def _build_red_team_agent(
    mode: Literal["devils_advocate", "bias_detector"],
    model: str = DEFAULT_MODEL,
) -> AiAgent:
    return AiAgent(
        name=f"Red Team Agent ({mode})",
        instructions=_get_system_prompt(mode),
        model=AgentSdkLlm(model=model),
        tools=[],
    )


def red_team_streamed(
    text: str,
    mode: Literal["devils_advocate", "bias_detector"],
    model: str = DEFAULT_MODEL,
) -> RunResultStreaming:
    agent = _build_red_team_agent(mode, model)
    messages = [
        {
            "role": "user",
            "content": f"Please analyze the following text:\n\n{text}",
        }
    ]
    return AgentRunner.run_streamed(agent, messages, max_turns=3)


async def red_team_to_completion(
    text: str,
    mode: Literal["devils_advocate", "bias_detector"],
    session_id: str | None,
    username: str,
    model: str = DEFAULT_MODEL,
) -> RedTeamResult:
    agent = _build_red_team_agent(mode, model)
    messages = [
        {
            "role": "user",
            "content": f"Please analyze the following text:\n\n{text}",
        }
    ]

    with general_trace_or_span(f"Decision Hub Red Team ({mode})"):
        with MonetaryCostManager():
            result = await AgentRunner.run(agent, messages, max_turns=3)
            raw_text = result.final_output or ""

    return RedTeamResult(
        session_id=session_id,
        username=username,
        input_text=text[:500],
        mode=mode,
        critique_markdown=raw_text,
    )
