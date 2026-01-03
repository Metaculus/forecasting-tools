import asyncio
import logging
from datetime import datetime

from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.questions import BinaryQuestion, MetaculusQuestion
from forecasting_tools.forecast_bots.official_bots.template_bot_2025_fall import (
    FallTemplateBot2025,
)
from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher

logger = logging.getLogger(__name__)


class GPT41OptimizedBot(FallTemplateBot2025):
    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")
            if researcher != "asknews/news-summaries":
                logger.warning(
                    "This bot was optimized on AskNews search, using other search providers may result in lower performance"
                )
            research = await AskNewsSearcher(
                cache_mode="use_cache_with_fallback"
            ).get_formatted_news_async(question.question_text)
            return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = create_binary_prompt(question, research)
        return await self._binary_prompt_to_forecast(
            question, prompt, double_check_extraction=False
        )


def create_binary_prompt(question: BinaryQuestion, research: str) -> str:
    question_text = question.question_text
    background_info = question.background_info
    resolution_criteria = question.resolution_criteria
    fine_print = question.fine_print
    today = datetime.now().strftime("%Y-%m-%d")

    # NOTE: the research (and other) fields are intentionally included twice since this is how Gemini designed the prompt when the automated prompt optimization iterations were run.
    return f"""
**Question to Forecast:**
{question_text}

**Background Information:**
{background_info}

**Resolution Criteria:**
{resolution_criteria}

**Fine Print:**
{fine_print}

**Today's Date:** {today}

**Available Research:**
{research}

---

**Forecasting Process: Weighted Critique & Sensitivity-Driven Revision**

You will engage in a three-phase forecasting process to determine the probability of the question stated above resolving as "Yes". Your goal is to produce a well-reasoned forecast incorporating rigorous critique and revision.

**Phase 1: Initial Scenario Development & Probability Assessment (Blue Team Perspective)**

Based on the provided information ({question_text}, {background_info}, {resolution_criteria}, {fine_print}) and the {research} as of {today}, please perform the following:

1.  **Develop Three Scenarios:**
    *   **Optimistic Scenario (event occurs / question resolves "Yes"):** Describe a plausible sequence of events, key drivers, and conditions under which the event is most likely to occur.
    *   **Pessimistic Scenario (event does not occur / question resolves "No"):** Describe a plausible sequence of events, key drivers, and conditions under which the event is most likely not to occur.
    *   **Most Likely Scenario:** Describe what you believe is the most probable sequence of events and conditions. Clearly state whether this scenario leads to a "Yes" or "No" resolution for the question.

2.  **Initial Probability Assessment:**
    *   Provide an initial probability estimate (as a decimal float, e.g., 0.65) that the question will resolve as "Yes".
    *   Briefly explain the reasoning, key factors, and assumptions supporting this initial assessment.

**Phase 2: Red Team Critique with Impact Assessment**

Now, adopt the role of a Red Team. Your objective is to critically evaluate the Blue Team's Phase 1 output with the aim of identifying weaknesses, biases, and overlooked perspectives.

1.  **Critique Each Scenario (Optimistic, Pessimistic, Most Likely):**
    *   For each scenario, systematically assess:
        *   Plausibility of the narrative and timelines.
        *   Identified flawed assumptions, logical inconsistencies, or unsupported claims.
        *   Potential alternative interpretations of data or events.
        *   Overlooked critical factors or counterarguments.
    *   **NEW - Impact Scoring for Critiques:** For *each specific point of critique* you raise for *each scenario*:
        *   Assign a **Vulnerability Score (1-5):** How vulnerable is the original scenario/assumption to this critique point? (1=low vulnerability, 5=critically vulnerable).
        *   Assign a **Plausibility of Alternative (1-5):** If you offer an alternative interpretation or factor as part of your critique, how plausible is your alternative? (1=not very plausible, 5=highly plausible). If no specific alternative is offered for a critique point, this can be N/A.
        *   Present this clearly, for example:
            *   "Critique for Optimistic Scenario's assumption that X will happen: This overlooks recent policy change Y. Vulnerability Score: 4. Alternative: Policy Y will significantly delay X. Plausibility of Alternative: 5."

2.  **Critique the Initial Probability Assessment's Method:**
    *   Evaluate the stated reasoning for the initial probability.
    *   Identify any potential cognitive biases (e.g., anchoring, confirmation bias, availability heuristic) evident in the Blue Team's assessment or scenario construction.
    *   Challenge the perceived importance or weighting of key factors mentioned by the Blue Team.

3.  **NEW - Identify Most Critical Assumptions:**
    *   Based on your critique, identify the 2-3 assumptions (explicit or implicit in the Blue Team's scenarios or probability reasoning) whose failure or incorrectness would most drastically alter the forecast outcome. List these clearly.

**Phase 3: Synthesis, Sensitivity-Driven Revision, and Final Forecast (Synthesizer/Blue Team Perspective)**

Revert to the role of the Synthesizer (or original Blue Team). Your task is to carefully consider the Red Team's critique and integrate it to produce a revised, more robust forecast.

1.  **Review Red Team Critique & Impact Scores:**
    *   Thoroughly analyze all points raised by the Red Team, paying particular attention to critiques with high Vulnerability Scores and alternatives with high Plausibility Scores.

2.  **Revise Scenarios:**
    *   Based on the Red Team's feedback, revise your Optimistic, Pessimistic, and Most Likely scenarios.
    *   **Explicitly state how specific Red Team critiques, especially those with high impact scores, led to changes in your scenarios.** For example: "The Red Team highlighted that our Pessimistic Scenario's reliance on factor Z was vulnerable (Score: 5) due to evidence A. We have revised this scenario to de-emphasize Z and incorporate factor B, which the Red Team suggested as a plausible alternative (Score: 4)."

3.  **NEW - Qualitative Sensitivity Analysis:**
    *   For each of the "Most Critical Assumptions" identified by the Red Team in Phase 2.3:
        *   Briefly discuss how your forecast (both the probability and the core logic of your scenarios) would likely change if that specific assumption were invalid or turned out to be significantly different from what was initially posited.

4.  **Final Rationale and Forecast:**
    *   Provide a comprehensive final rationale for your forecast.
    *   **Explain how the Red Team's critiques, their assigned impact scores (Vulnerability and Plausibility of Alternatives), and your qualitative sensitivity analysis specifically influenced the revisions to your scenarios and, ultimately, your final probability estimate.** This section should demonstrate a clear link between the critique and the refined forecast.
    *   **State your final probability forecast that the question will resolve as "Yes". This must be a single binary float (e.g., 0.72).**

---
**Output Instructions:**
Please structure your entire response clearly, labeling each Phase (1, 2, 3) and its sub-components as outlined above. Ensure all requested elements, particularly the impact scores in Phase 2, the explicit links between critique and revision in Phase 3, the qualitative sensitivity analysis, and the final binary float probability, are included.
"""
