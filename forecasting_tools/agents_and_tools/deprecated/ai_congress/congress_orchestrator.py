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
from forecasting_tools.agents_and_tools.minor_tools import (
    perplexity_reasoning_pro_search,
    roll_dice,
)
from forecasting_tools.ai_models.agent_wrappers import AgentRunner, AgentSdkLlm, AiAgent
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)

LONG_TIMEOUT = 480  # 8 minutes for long-running LLM calls


class CongressOrchestrator:
    def __init__(
        self,
        aggregation_model: str = "openrouter/anthropic/claude-sonnet-4.5",
        num_delphi_rounds: int = 1,
    ):
        self.aggregation_model = aggregation_model
        self.num_delphi_rounds = max(1, num_delphi_rounds)

    async def run_session(
        self,
        prompt: str,
        members: list[CongressMember],
    ) -> CongressSession:
        logger.info(
            f"Starting congress session with {len(members)} members, "
            f"{self.num_delphi_rounds} Delphi round(s) on: {prompt[:100]}..."
        )

        with MonetaryCostManager() as session_cost_manager:
            agents = [CongressMemberAgent(m) for m in members]
            proposals, errors = await self._run_initial_round(agents, prompt)

            initial_proposals = list(proposals)
            proposals, delphi_errors = await self._run_delphi_revisions(
                agents, prompt, proposals
            )
            errors.extend(delphi_errors)

            aggregated_report, blog_post, future_snapshot, twitter_posts = (
                await self._generate_outputs(prompt, proposals, members)
            )

            total_cost = session_cost_manager.current_usage

        proposal_costs = sum(
            p.price_estimate for p in proposals if p.price_estimate is not None
        )
        logger.info(
            f"Completed congress session. Total cost: ${total_cost:.4f}, "
            f"Proposal costs: ${proposal_costs:.4f}"
        )

        return CongressSession(
            prompt=prompt,
            members_participating=members,
            proposals=proposals,
            aggregated_report_markdown=aggregated_report,
            blog_post=blog_post,
            future_snapshot=future_snapshot,
            twitter_posts=twitter_posts,
            timestamp=datetime.now(timezone.utc),
            errors=errors,
            total_price_estimate=total_cost,
            num_delphi_rounds=self.num_delphi_rounds,
            initial_proposals=initial_proposals if self.num_delphi_rounds > 1 else [],
        )

    async def _run_initial_round(
        self,
        agents: list[CongressMemberAgent],
        prompt: str,
    ) -> tuple[list[PolicyProposal], list[str]]:
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

        logger.info(
            f"Round 1 completed: {len(proposals)} proposals with {len(errors)} errors"
        )
        return proposals, errors

    async def _run_delphi_revisions(
        self,
        agents: list[CongressMemberAgent],
        prompt: str,
        proposals: list[PolicyProposal],
    ) -> tuple[list[PolicyProposal], list[str]]:
        errors: list[str] = []
        for delphi_round in range(2, self.num_delphi_rounds + 1):
            if len(proposals) < 2:
                logger.warning(
                    f"Skipping Delphi round {delphi_round}: need at least 2 "
                    f"proposals but only have {len(proposals)}"
                )
                break

            logger.info(
                f"Starting Delphi round {delphi_round} with {len(proposals)} proposals"
            )
            revised, round_errors = await self._run_single_delphi_round(
                agents, prompt, proposals, delphi_round
            )
            errors.extend(round_errors)

            if revised:
                proposals = revised
            else:
                logger.warning(
                    f"Delphi round {delphi_round} produced no proposals, "
                    f"keeping previous round's proposals"
                )
        return proposals, errors

    async def _run_single_delphi_round(
        self,
        agents: list[CongressMemberAgent],
        prompt: str,
        proposals: list[PolicyProposal],
        delphi_round: int,
    ) -> tuple[list[PolicyProposal], list[str]]:
        revision_tasks = []
        for agent in agents:
            own_proposal = next(
                (
                    p
                    for p in proposals
                    if p.member and p.member.name == agent.member.name
                ),
                None,
            )
            if own_proposal is None:
                continue
            other_proposals = [
                p for p in proposals if p.member and p.member.name != agent.member.name
            ]
            revision_tasks.append(
                self._run_revision_with_error_handling(
                    agent, prompt, own_proposal, other_proposals, delphi_round
                )
            )

        revision_results = await asyncio.gather(
            *revision_tasks, return_exceptions=False
        )

        revised_proposals: list[PolicyProposal] = []
        errors: list[str] = []
        for result in revision_results:
            if isinstance(result, PolicyProposal):
                revised_proposals.append(result)
            elif isinstance(result, Exception):
                errors.append(f"Delphi round {delphi_round}: {result}")

        logger.info(
            f"Delphi round {delphi_round} completed: "
            f"{len(revised_proposals)} revised proposals"
        )
        return revised_proposals, errors

    async def _generate_outputs(
        self,
        prompt: str,
        proposals: list[PolicyProposal],
        members: list[CongressMember],
    ) -> tuple[str, str, str, list[str]]:
        if not proposals:
            return "", "", "", []

        aggregated_report = await self._aggregate_proposals(prompt, proposals)
        blog_post = await self._generate_blog_post(prompt, proposals, members)
        future_snapshot = await self._generate_future_snapshot(
            prompt, proposals, aggregated_report
        )
        twitter_posts = await self._generate_twitter_posts(prompt, proposals)
        return aggregated_report, blog_post, future_snapshot, twitter_posts

    async def _run_member_with_error_handling(
        self,
        agent: CongressMemberAgent,
        prompt: str,
    ) -> PolicyProposal | Exception:
        try:
            logger.info(f"Starting deliberation for {agent.member.name}")
            with MonetaryCostManager() as member_cost_manager:
                proposal = await agent.deliberate(prompt)
                member_cost = member_cost_manager.current_usage
            proposal.price_estimate = member_cost
            logger.info(
                f"Completed deliberation for {agent.member.name}, cost: ${member_cost:.4f}"
            )
            return proposal
        except Exception as e:
            logger.error(f"Error in {agent.member.name}'s deliberation: {e}")
            return e

    async def _run_revision_with_error_handling(
        self,
        agent: CongressMemberAgent,
        prompt: str,
        own_proposal: PolicyProposal,
        other_proposals: list[PolicyProposal],
        delphi_round: int,
    ) -> PolicyProposal | Exception:
        try:
            logger.info(
                f"Starting Delphi round {delphi_round} revision for {agent.member.name}"
            )
            with MonetaryCostManager() as revision_cost_manager:
                proposal = await agent.revise_proposal(
                    prompt, own_proposal, other_proposals, delphi_round
                )
                revision_cost = revision_cost_manager.current_usage
            previous_cost = own_proposal.price_estimate or 0
            proposal.price_estimate = previous_cost + revision_cost
            logger.info(
                f"Completed Delphi round {delphi_round} revision for {agent.member.name}, "
                f"revision cost: ${revision_cost:.4f}"
            )
            return proposal
        except Exception as e:
            logger.error(
                f"Error in {agent.member.name}'s Delphi round {delphi_round} revision: {e}"
            )
            return e

    async def _aggregate_proposals(
        self,
        prompt: str,
        proposals: list[PolicyProposal],
    ) -> str:
        logger.info(f"Aggregating proposals for congress session: {prompt}")
        llm = GeneralLlm(self.aggregation_model, timeout=LONG_TIMEOUT)

        proposals_text = "\n\n---\n\n".join(
            [
                f"## {p.member.name} ({p.member.role})\n\n```markdown\n{p.get_full_markdown_with_footnotes()}\n```"
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

            # Individual Proposals

            {proposals_text}

            ---

            # Your Task

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

            ### Baseline Forecast Comparison

            Create a summary of how BASELINE forecasts (status quo predictions) differed
            across members:
            - Note where forecasts converged (similar probabilities)
            - Highlight where forecasts diverged significantly
            - Discuss what might explain the differences (different information,
              different priors, different interpretations)

            ### Conditional Forecast Comparison

            Create a summary of how CONDITIONAL forecasts (predictions conditional on
            policy being implemented) differed across members:
            - Compare members' predictions about what their policies would achieve
            - Highlight the most optimistic vs pessimistic conditional forecasts
            - Note which members forecast significant unintended consequences
            - Compare baseline vs conditional forecasts on similar topics to show
              the expected policy impact

            ### Integrated Recommendations

            Your synthesis of the best policy path forward:
            - Draw on the strongest arguments from each perspective
            - Use both baseline and conditional forecasts to assess expected policy impact
            - Identify low-regret actions that most members would support
            - Note high-uncertainty areas where more caution is warranted
            - Be specific and actionable

            ### Combined Baseline Forecast Appendix

            Compile all baseline (status quo) forecasts from all members into a single
            appendix. When members made similar forecasts, group them and note the range
            of predictions.

            Format each forecast as:

            [^1] **[Question Title]** (from [Member Name])
            - Question: [Full question]
            - Resolution: [Resolution criteria]
            - Prediction: [Probability]
            - Reasoning: [Summary of reasoning]

            Number the footnotes sequentially [^1], [^2], [^3], etc.

            ### Combined Conditional Forecast Appendix

            Compile all conditional (policy-dependent) forecasts from all members into
            a separate appendix. Continue the footnote numbering from the baseline
            appendix.

            Format each forecast as:

            [^N] **[Question Title]** *(Conditional on policy)* (from [Member Name])
            - Question: [Full question]
            - Resolution: [Resolution criteria]
            - Prediction: [Probability]
            - Reasoning: [Summary of reasoning]

            ---

            Be balanced but not wishy-washy. Identify which arguments are strongest
            and why. Your goal is to help decision-makers, so be clear about what
            the analysis supports.
            """
        )

        result = await llm.invoke(aggregation_prompt)
        logger.info("Completed aggregation of proposals")
        return result

    async def _generate_blog_post(
        self,
        prompt: str,
        proposals: list[PolicyProposal],
        members: list[CongressMember],
    ) -> str:
        logger.info(f"Generating blog post for congress session: {prompt}")
        llm = GeneralLlm(self.aggregation_model, timeout=LONG_TIMEOUT)

        ai_model_members = [
            m
            for m in members
            if "behaves as" in m.political_leaning.lower()
            or "naturally" in m.political_leaning.lower()
        ]
        has_ai_model_comparison = len(ai_model_members) >= 2

        proposals_summary = "\n\n".join(
            [
                f"### {p.member.name} ({p.member.role})\n"
                f"**Political Leaning:** {p.member.political_leaning}\n"
                f"**AI Model:** {p.member.ai_model}\n\n"
                f"**Key Recommendations:**\n"
                + "\n".join(f"- {rec}" for rec in p.key_recommendations[:5])
                + "\n\n**Key Forecasts:**\n"
                + "\n".join(
                    f"- {f.question_title}: {f.prediction}" for f in p.forecasts[:5]
                )
                + f"\n\n**Proposal Text:**\n"
                f"```markdown\n"
                f"{p.get_full_markdown_with_footnotes()}\n"
                f"```\n\n"
                for p in proposals
                if p.member
            ]
        )

        ai_comparison_section = ""
        if has_ai_model_comparison:
            ai_comparison_section = clean_indents(
                """
                ## Special Section: AI Model Comparison

                Since this congress included multiple AI models acting naturally (without
                assigned political personas), include a dedicated analysis section:

                ### How the Models Compared

                For each AI model participant, analyze:
                - What was their overall approach and tone?
                - What priorities or values seemed most salient to them?
                - How did their forecasts compare to other models on similar questions?
                - Did they show any distinctive reasoning patterns?

                ### Unexpected Behaviors

                Highlight anything surprising:
                - Did any model take a position you wouldn't expect?
                - Were there cases where models with similar training diverged significantly?
                - Did any model show unusual certainty or uncertainty?
                - Were there any reasoning patterns that seemed distinctive to one model?

                ### Model Personality Insights

                What does this session reveal about each model's "personality"?
                - Risk tolerance (cautious vs bold)
                - Epistemic style (hedging vs confident)
                - Value emphasis (efficiency, equity, security, etc.)
                - Reasoning style (data-driven, principled, pragmatic)
                """
            )

        blog_prompt = clean_indents(
            f"""
            # Write a Blog Post About This AI Congress Session

            You are writing an engaging blog post about an AI Forecasting Congress
            session where AI agents deliberated on the following policy question:

            "{prompt}"

            ## Proposals Summary

            {proposals_summary}

            ## Blog Post Requirements

            Write a ~1500-2000 word blog post that would be engaging for a tech/policy
            audience interested in AI capabilities and policy analysis. The post should:

            ### Structure

            1. **Hook** (1 paragraph): Start with the most surprising or interesting
               finding from the session. Make readers want to continue.

            2. **Context** (1-2 paragraphs): Briefly explain what the AI Forecasting
               Congress is and what question was being deliberated.

            3. **Key Insights** (3-5 paragraphs): The most important takeaways from
               the session. What did the AI congress conclude? Where did they agree
               and disagree? What forecasts matter most? How do the conditional
               forecasts (what happens IF policies are implemented) compare to the
               baseline forecasts (status quo predictions)?

            4. **The Good, Bad, and Ugly** (2-3 paragraphs): Highlight:
               - The Good: Surprising consensus, innovative ideas, strong reasoning
               - The Bad: Blind spots, weak arguments, missed considerations
               - The Ugly: Uncomfortable tradeoffs, unresolved tensions

            5. **Implications** (1-2 paragraphs): What does this mean for policymakers
               or the public? What actions might follow from these insights?

            {ai_comparison_section}

            6. **Conclusion** (1 paragraph): End with a thought-provoking takeaway
               about what this exercise reveals about AI policy analysis capabilities.

            ### Style Guidelines

            - Write in an engaging, accessible style (not academic)
            - Use specific examples and quotes from the proposals
            - Include specific forecasts with probabilities
            - Be analytical but not dry
            - Feel free to express opinions about which arguments were strongest
            - Use markdown formatting with headers, bullet points, and bold text
            - Include a catchy title at the start

            Write the blog post now.
            """
        )

        try:
            logger.info(f"Generating blog post for congress session: {prompt}")
            return await llm.invoke(blog_prompt)
        except Exception as e:
            logger.error(f"Failed to generate blog post: {e}")
            return ""

    @staticmethod
    def _collect_forecast_dicts(
        proposals: list[PolicyProposal],
    ) -> tuple[list[dict], list[dict]]:
        baseline = []
        conditional = []
        for proposal in proposals:
            member_name = proposal.member.name if proposal.member else "Unknown"
            for forecast in proposal.forecasts:
                forecast_dict = {
                    "member": member_name,
                    "title": forecast.question_title,
                    "question": forecast.question_text,
                    "prediction": forecast.prediction,
                    "resolution_criteria": forecast.resolution_criteria,
                    "reasoning": forecast.reasoning,
                }
                if forecast.is_conditional:
                    conditional.append(forecast_dict)
                else:
                    baseline.append(forecast_dict)
        return baseline, conditional

    @staticmethod
    def _collect_recommendations(proposals: list[PolicyProposal]) -> list[dict]:
        recommendations = []
        for proposal in proposals:
            if proposal.member:
                for rec in proposal.key_recommendations:
                    recommendations.append(
                        {"member": proposal.member.name, "recommendation": rec}
                    )
        return recommendations

    async def _generate_future_snapshot(
        self,
        prompt: str,
        proposals: list[PolicyProposal],
        aggregated_report: str,
    ) -> str:
        logger.info(f"Generating future snapshot for congress session: {prompt}")

        all_baseline, all_conditional = self._collect_forecast_dicts(proposals)
        all_recommendations = self._collect_recommendations(proposals)

        baseline_forecasts_text = "\n".join(
            f"- **{f['title']}** ({f['member']}): {f['prediction']}\n"
            f"  - Question: {f['question']}\n"
            f"  - Resolution: {f['resolution_criteria']}"
            for f in all_baseline
        )

        conditional_forecasts_text = "\n".join(
            f"- **{f['title']}** *(Conditional on policy)* ({f['member']}): {f['prediction']}\n"
            f"  - Question: {f['question']}\n"
            f"  - Resolution: {f['resolution_criteria']}"
            for f in all_conditional
        )

        recommendations_text = "\n".join(
            f"- [{r['member']}] {r['recommendation']}" for r in all_recommendations
        )

        snapshot_prompt = clean_indents(
            f"""
            # Picture of the Future: AI Congress Scenario Generator

            You are a journalist writing a retrospective "Year in Review" article from the
            future, looking back at what happened after the AI Congress's recommendations
            were either implemented or rejected.

            ## Original Policy Question

            "{prompt}"

            ## Aggregate Policy Report

            ```markdown
            {aggregated_report}
            ```

            ## Baseline Forecasts (Status Quo Predictions)

            {baseline_forecasts_text}

            ## Conditional Forecasts (If Policy Is Implemented)

            {conditional_forecasts_text}

            ## All Policy Recommendations

            {recommendations_text}

            ---

            ## Your Task

            Write TWO compelling newspaper-style narratives:

            ### PART 1: "THE WORLD WITH THE RECOMMENDATIONS" (Recommendations Implemented)

            Start with: "The date is <date you pick>..."

            Write a flowing narrative in the style of a newspaper giving an annual review
            of the biggest news of the last two years. Assume:

            1. The AI Congress's aggregate recommendations were implemented.
               The date is now one you choose that would give enough time
               for the effects of the policies to be known.

            2. For each forecast, you will ROLL THE DICE to determine if it happened:
               - Use the roll_forecast_dice tool for EACH forecast
               - Pass the probability from the forecast (e.g., 35 for "35%")
               - The tool returns whether that event occurred based on the probability
               - Incorporate the outcome naturally into your narrative
               - For the "WITH recommendations" narrative, use the CONDITIONAL forecasts
                 (since the policy is implemented) for policy-dependent outcomes,
                 and baseline forecasts for background events
               - For the "WITHOUT recommendations" narrative, use the BASELINE forecasts
                 (since the policy is NOT implemented)

            3. For any gaps in the forecasts, create your own probabilistic predictions
               marked with asterisks (*). For example: "The unemployment rate dropped to
               4.2%* (*AI-generated estimate based on historical policy impacts)."

            4. Reference the original forecasts inline using this format "(X% [^1])".
               Make sure X% is the probability for the event that happened (so you may need to invert).
               In the footnote, include the full forecast details including the question, resolution, prediction,
               reasoning, sources, and outcome like this:
                [^1] **[Question Title]**
                - Question: [Full question]
                - Resolution: [Resolution criteria]
                - Prediction: [Probability]
                - Reasoning: [Summary of reasoning]
                - Sources: [Key sources used, can be URLs or source names]
                - Outcome: [OCCURRED/DID NOT OCCUR]

            5. You MUST incorporate the majority of the policy recommendations as
               concrete events or policy changes in the timeline.

            6. Consider any new forecasting questions/forecasts that would help fill in the narrative or old forecasts that would
               now be different given the policy was enacted. If appropriate make new questions and forecasts of your own.
               If you do mark the forecasts inline with a single asterisk and include your forecasts in a special section at
               the bottom with an explanation that they were made by you.

            ### PART 2: "THE WORLD WITHOUT THE RECOMMENDATIONS" (Recommendations Rejected)

            After completing Part 1, write a contrasting narrative showing what the world
            looks like if the recommendations were NOT implemented. Use the same dice
            rolls for forecasts but show how the lack of policy action changed outcomes.

            Start with: "In an alternate timeline where the AI Congress recommendations
            were rejected..."

            ---

            ## Important Guidelines

            - Make the narrative vivid and engaging, like real journalism
            - Include specific dates, names of real world people where relevant
              (or fake names if they would not be known yet) and concrete details
            - If you make up any fake people or orgs, mark these with † and then explain this in the footnotes.
            - Show cause-and-effect relationships between policies and outcomes
            - Your own estimates marked with * should be plausible extrapolations
            - The tone should be neutral/journalistic, not promotional
            - Include both positive and negative consequences where realistic
            - Each forecast should be explicitly mentioned with its dice roll outcome
            - Ground speculation in research where possible
            - Use the aggregate policy as the source of truth for what policy is taken
            - You are writing for an audience that may not be familiar with the subject area.
              Make sure to include the events of the forecasts, but write in a way that they
              will understand as much as possible.

            ## Format

            Use markdown formatting with clear section headers. Aim for 1500-2500 words
            total across both parts.
            """
        )

        try:
            llm_wrapper = AgentSdkLlm("openrouter/openai/gpt-5.2")

            snapshot_agent = AiAgent(
                name="Future Snapshot Writer",
                instructions=snapshot_prompt,
                model=llm_wrapper,
                tools=[roll_dice, perplexity_reasoning_pro_search],
            )

            result = await AgentRunner.run(
                snapshot_agent, "Generate the future snapshot now.", max_turns=25
            )
            return result.final_output

        except Exception as e:
            logger.error(f"Failed to generate future snapshot: {e}")
            return ""

    async def _generate_twitter_posts(
        self,
        prompt: str,
        proposals: list[PolicyProposal],
    ) -> list[str]:
        logger.info(f"Generating twitter posts for congress session: {prompt}")
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
            logger.info(f"Generated {len(posts)} twitter posts")
            return [p[:280] for p in posts]
        except Exception as e:
            logger.error(f"Failed to generate twitter posts: {e}")
            return []
