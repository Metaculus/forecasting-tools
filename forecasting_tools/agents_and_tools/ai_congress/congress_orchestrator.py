from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from forecasting_tools.agents_and_tools.ai_congress.congress_member_agent import (
    CongressMemberAgent,
)
from forecasting_tools.agents_and_tools.ai_congress.data_models import (
    CongressMember,
    CongressSession,
    PolicyProposal,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)

LONG_TIMEOUT = 480  # 8 minutes for long-running LLM calls


class CongressOrchestrator:
    def __init__(
        self,
        aggregation_model: str = "openrouter/anthropic/claude-sonnet-4",
    ):
        self.aggregation_model = aggregation_model

    async def run_session(
        self,
        prompt: str,
        members: list[CongressMember],
    ) -> CongressSession:
        logger.info(
            f"Starting congress session with {len(members)} members on: {prompt[:100]}..."
        )

        agents = [CongressMemberAgent(m) for m in members]

        results = await asyncio.gather(
            *[self._run_member_with_error_handling(a, prompt) for a in agents],
            return_exceptions=False,
        )

        proposals: list[PolicyProposal] = []
        errors: list[str] = []

        for result in results:
            if isinstance(result, PolicyProposal):
                proposals.append(result)
            elif isinstance(result, Exception):
                errors.append(str(result))
            else:
                errors.append(f"Unexpected result type: {type(result)}")

        logger.info(f"Completed {len(proposals)} proposals with {len(errors)} errors")

        aggregated_report = ""
        twitter_posts: list[str] = []

        if proposals:
            aggregated_report = await self._aggregate_proposals(prompt, proposals)
            twitter_posts = await self._generate_twitter_posts(prompt, proposals)

        return CongressSession(
            prompt=prompt,
            members_participating=members,
            proposals=proposals,
            aggregated_report_markdown=aggregated_report,
            twitter_posts=twitter_posts,
            timestamp=datetime.now(timezone.utc),
            errors=errors,
        )

    async def _run_member_with_error_handling(
        self,
        agent: CongressMemberAgent,
        prompt: str,
    ) -> PolicyProposal | Exception:
        try:
            logger.info(f"Starting deliberation for {agent.member.name}")
            proposal = await agent.deliberate(prompt)
            logger.info(f"Completed deliberation for {agent.member.name}")
            return proposal
        except Exception as e:
            logger.error(f"Error in {agent.member.name}'s deliberation: {e}")
            return e

    async def _aggregate_proposals(
        self,
        prompt: str,
        proposals: list[PolicyProposal],
    ) -> str:
        llm = GeneralLlm(self.aggregation_model, timeout=LONG_TIMEOUT)

        proposals_text = "\n\n---\n\n".join(
            [
                f"## {p.member.name} ({p.member.role})\n\n{p.get_full_markdown_with_footnotes()}"
                for p in proposals
                if p.member
            ]
        )

        aggregation_prompt = clean_indents(
            f"""
            # AI Forecasting Congress: Synthesis Report

            You are synthesizing the proposals from multiple AI congress members
            deliberating on the following policy question:

            "{prompt}"

            ## Individual Proposals

            {proposals_text}

            ---

            ## Your Task

            Write a comprehensive synthesis report that helps readers understand the
            full range of perspectives and find actionable insights. Structure your
            report as follows:

            ### Executive Summary

            A 3-4 sentence overview of:
            - The key areas of agreement across members
            - The most significant disagreements
            - The most important forecasts that inform the debate

            ### Consensus Recommendations

            What policies do multiple members support? For each consensus area:
            - State the recommendation
            - List which members support it
            - Include the relevant forecasts (use footnotes [^N] referencing the
              Combined Forecast Appendix below)
            - Note any caveats or conditions members attached

            ### Key Disagreements

            Where do members diverge and why? For each major disagreement:
            - State the issue
            - Summarize each side's position and which members hold it
            - Explain how different forecasts, criteria, or values lead to different
              conclusions
            - Assess the crux of the disagreement

            ### Forecast Comparison

            Create a summary of how forecasts differed across members:
            - Note where forecasts converged (similar probabilities)
            - Highlight where forecasts diverged significantly
            - Discuss what might explain the differences (different information,
              different priors, different interpretations)

            ### Integrated Recommendations

            Your synthesis of the best policy path forward:
            - Draw on the strongest arguments from each perspective
            - Identify low-regret actions that most members would support
            - Note high-uncertainty areas where more caution is warranted
            - Be specific and actionable

            ### Combined Forecast Appendix

            Compile all unique forecasts from all members into a single appendix.
            When members made similar forecasts, group them and note the range of
            predictions.

            Format each forecast as:

            [^1] **[Question Title]** (from [Member Name])
            - Question: [Full question]
            - Resolution: [Resolution criteria]
            - Prediction: [Probability]
            - Reasoning: [Summary of reasoning]

            Number the footnotes sequentially [^1], [^2], [^3], etc.

            ---

            Be balanced but not wishy-washy. Identify which arguments are strongest
            and why. Your goal is to help decision-makers, so be clear about what
            the analysis supports.
            """
        )

        return await llm.invoke(aggregation_prompt)

    async def _generate_twitter_posts(
        self,
        prompt: str,
        proposals: list[PolicyProposal],
    ) -> list[str]:
        llm = GeneralLlm(self.aggregation_model, timeout=LONG_TIMEOUT)

        proposals_summary = "\n\n".join(
            [
                f"**{p.member.name}** ({p.member.role}, {p.member.political_leaning}):\n"
                f"Key recommendations: {', '.join(p.key_recommendations[:3])}\n"
                f"Key forecasts: {'; '.join([f'{f.question_title}: {f.prediction}' for f in p.forecasts[:3]])}"
                for p in proposals
                if p.member
            ]
        )

        twitter_prompt = clean_indents(
            f"""
            Based on this AI Forecasting Congress session on "{prompt}", generate
            8-12 tweet-length excerpts (max 280 characters each) highlighting
            interesting patterns for a policy/tech audience on Twitter/X.

            ## Proposals Summary

            {proposals_summary}

            ## Categories to Cover

            Generate tweets in these categories:

            **THE GOOD** (2-3 tweets):
            - Surprising areas of consensus across different ideologies
            - Innovative ideas that emerged from the deliberation
            - Forecasts that challenge conventional wisdom

            **THE BAD** (2-3 tweets):
            - Concerning blind spots that multiple members missed
            - Problematic reasoning patterns you noticed
            - Important questions that weren't addressed

            **THE UGLY** (2-3 tweets):
            - Stark disagreements that reveal deep value differences
            - Uncomfortable tradeoffs that the analysis surfaced
            - Forecasts with wide uncertainty that matter a lot

            **THE INTERESTING** (2-3 tweets):
            - Unexpected forecasts or counter-intuitive findings
            - Surprising agreement between unlikely allies
            - Questions where the forecasts diverged most

            ## Tweet Guidelines

            Each tweet should:
            - Be self-contained and intriguing (people should want to click through)
            - Reference specific forecasts when relevant (e.g., "65% probability of X")
            - Attribute to the relevant congress member when applicable
            - Use hooks like "Surprising:" or "The [Member] vs [Member] split:"
            - Be under 280 characters
            - Not include hashtags

            Return a JSON list of strings, one per tweet.
            """
        )

        try:
            posts = await llm.invoke_and_return_verified_type(twitter_prompt, list[str])
            return [p[:280] for p in posts]
        except Exception as e:
            logger.error(f"Failed to generate twitter posts: {e}")
            return []
