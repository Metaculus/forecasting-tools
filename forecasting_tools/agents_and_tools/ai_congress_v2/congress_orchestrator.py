from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from forecasting_tools.agents_and_tools.ai_congress_v2.congress_member_agent import (
    CongressMemberAgent,
)
from forecasting_tools.agents_and_tools.ai_congress_v2.data_models import (
    CongressMember,
    CongressSession,
    PolicyProposal,
)
from forecasting_tools.agents_and_tools.ai_congress_v2.tools import (
    create_search_tool,
    roll_dice,
    roll_multiple_dice,
)
from forecasting_tools.ai_models.agent_wrappers import AgentRunner, AgentSdkLlm, AiAgent
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)

LONG_TIMEOUT = 480


class CongressOrchestrator:
    def __init__(
        self,
        aggregation_model: str = "openrouter/anthropic/claude-sonnet-4",
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
            f"Starting congress v2 session with {len(members)} members, "
            f"{self.num_delphi_rounds} Delphi round(s) on: {prompt[:100]}..."
        )

        with MonetaryCostManager() as session_cost_manager:
            agents = [CongressMemberAgent(m) for m in members]
            raw_outputs, proposals, errors = await self._run_initial_round(
                agents, prompt
            )

            initial_proposals = list(proposals)

            if self.num_delphi_rounds > 1:
                raw_outputs, proposals, delphi_errors = (
                    await self._run_delphi_revisions(
                        agents, prompt, raw_outputs, proposals
                    )
                )
                errors.extend(delphi_errors)

            (
                aggregated_report,
                scenario_report,
                blog_post,
                future_snapshot,
                twitter_posts,
            ) = await self._generate_outputs(prompt, proposals, members)

            total_cost = session_cost_manager.current_usage

        proposal_costs = sum(
            p.price_estimate for p in proposals if p.price_estimate is not None
        )
        logger.info(
            f"Completed congress v2 session. Total cost: ${total_cost:.4f}, "
            f"Proposal costs: ${proposal_costs:.4f}"
        )

        return CongressSession(
            prompt=prompt,
            members_participating=members,
            proposals=proposals,
            aggregated_report_markdown=aggregated_report,
            scenario_report=scenario_report,
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
    ) -> tuple[list[str], list[PolicyProposal], list[str]]:
        results = await asyncio.gather(
            *[self._run_member_with_error_handling(a, prompt) for a in agents],
            return_exceptions=False,
        )

        raw_outputs: list[str] = []
        proposals: list[PolicyProposal] = []
        errors: list[str] = []
        for result in results:
            if isinstance(result, tuple):
                raw_output, proposal = result
                raw_outputs.append(raw_output)
                proposals.append(proposal)
            elif isinstance(result, Exception):
                raw_outputs.append("")
                errors.append(str(result))
            else:
                raw_outputs.append("")
                errors.append(f"Unexpected result type: {type(result)}")

        logger.info(
            f"Round 1 completed: {len(proposals)} proposals with {len(errors)} errors"
        )
        return raw_outputs, proposals, errors

    async def _run_delphi_revisions(
        self,
        agents: list[CongressMemberAgent],
        prompt: str,
        raw_outputs: list[str],
        proposals: list[PolicyProposal],
    ) -> tuple[list[str], list[PolicyProposal], list[str]]:
        errors: list[str] = []
        for delphi_round in range(2, self.num_delphi_rounds + 1):
            if len(proposals) < 2:
                logger.warning(
                    f"Skipping Delphi round {delphi_round}: need at least 2 "
                    f"proposals but only have {len(proposals)}"
                )
                break

            new_raw_outputs, new_proposals, round_errors = (
                await self._run_single_delphi_round(
                    agents, prompt, raw_outputs, proposals, delphi_round
                )
            )
            errors.extend(round_errors)

            if new_proposals:
                raw_outputs = new_raw_outputs
                proposals = new_proposals

        return raw_outputs, proposals, errors

    async def _run_single_delphi_round(
        self,
        agents: list[CongressMemberAgent],
        prompt: str,
        _raw_outputs: list[str],
        proposals: list[PolicyProposal],
        delphi_round: int,
    ) -> tuple[list[str], list[PolicyProposal], list[str]]:
        logger.info(
            f"Starting Delphi round {delphi_round} with {len(proposals)} proposals"
        )

        final_reports_by_member = self._build_final_reports_map(proposals)
        revision_tasks = self._build_revision_tasks(
            agents, prompt, proposals, final_reports_by_member, delphi_round
        )

        revision_results = await asyncio.gather(
            *revision_tasks, return_exceptions=False
        )

        new_raw_outputs: list[str] = []
        new_proposals: list[PolicyProposal] = []
        errors: list[str] = []
        for result in revision_results:
            if isinstance(result, tuple):
                raw_output, proposal = result
                new_raw_outputs.append(raw_output)
                new_proposals.append(proposal)
            elif isinstance(result, Exception):
                errors.append(f"Delphi round {delphi_round}: {result}")

        if not new_proposals:
            logger.warning(
                f"Delphi round {delphi_round} produced no proposals, "
                f"keeping previous round's proposals"
            )
        else:
            logger.info(
                f"Delphi round {delphi_round} completed: "
                f"{len(new_proposals)} revised proposals"
            )

        return new_raw_outputs, new_proposals, errors

    @staticmethod
    def _build_final_reports_map(
        proposals: list[PolicyProposal],
    ) -> dict[str, str]:
        reports: dict[str, str] = {}
        for proposal in proposals:
            member_name = proposal.member.name if proposal.member else "Unknown"
            reports[member_name] = proposal.get_full_markdown_with_footnotes()
        return reports

    def _build_revision_tasks(
        self,
        agents: list[CongressMemberAgent],
        prompt: str,
        proposals: list[PolicyProposal],
        final_reports_by_member: dict[str, str],
        delphi_round: int,
    ) -> list:
        revision_tasks = []
        for i, agent in enumerate(agents):
            if i >= len(proposals):
                continue

            own_report = final_reports_by_member.get(agent.member.name, "")
            other_reports = [
                (name, report)
                for name, report in final_reports_by_member.items()
                if name != agent.member.name
            ]

            revision_tasks.append(
                self._run_delphi_with_error_handling(
                    agent, prompt, own_report, other_reports, delphi_round
                )
            )
        return revision_tasks

    async def _run_member_with_error_handling(
        self,
        agent: CongressMemberAgent,
        prompt: str,
    ) -> tuple[str, PolicyProposal] | Exception:
        try:
            logger.info(f"Starting deliberation for {agent.member.name}")
            with MonetaryCostManager() as member_cost_manager:
                raw_output, proposal = await agent.deliberate(prompt)
                member_cost = member_cost_manager.current_usage
            proposal.price_estimate = member_cost
            logger.info(
                f"Completed deliberation for {agent.member.name}, "
                f"cost: ${member_cost:.4f}"
            )
            return raw_output, proposal
        except Exception as e:
            logger.error(f"Error in {agent.member.name}'s deliberation: {e}")
            return e

    async def _run_delphi_with_error_handling(
        self,
        agent: CongressMemberAgent,
        prompt: str,
        own_report: str,
        other_reports: list[tuple[str, str]],
        delphi_round: int,
    ) -> tuple[str, PolicyProposal] | Exception:
        try:
            logger.info(f"Starting Delphi round {delphi_round} for {agent.member.name}")
            with MonetaryCostManager() as revision_cost_manager:
                raw_output, proposal = await agent.continue_with_delphi(
                    prompt, own_report, other_reports, delphi_round
                )
                revision_cost = revision_cost_manager.current_usage
            previous_cost = proposal.price_estimate or 0
            proposal.price_estimate = previous_cost + revision_cost
            logger.info(
                f"Completed Delphi round {delphi_round} for {agent.member.name}, "
                f"revision cost: ${revision_cost:.4f}"
            )
            return raw_output, proposal
        except Exception as e:
            logger.error(
                f"Error in {agent.member.name}'s Delphi round {delphi_round}: {e}"
            )
            return e

    async def _generate_outputs(
        self,
        prompt: str,
        proposals: list[PolicyProposal],
        members: list[CongressMember],
    ) -> tuple[str, str, str, str, list[str]]:
        if not proposals:
            return "", "", "", "", []

        aggregated_report = await self._aggregate_proposals(prompt, proposals)
        scenario_report = await self._generate_scenario_report(
            prompt, proposals, aggregated_report
        )
        blog_post = await self._generate_blog_post(prompt, proposals, members)
        future_snapshot = await self._generate_future_snapshot(
            prompt, proposals, aggregated_report
        )
        twitter_posts = await self._generate_twitter_posts(prompt, proposals)
        return (
            aggregated_report,
            scenario_report,
            blog_post,
            future_snapshot,
            twitter_posts,
        )

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    async def _aggregate_proposals(
        self,
        prompt: str,
        proposals: list[PolicyProposal],
    ) -> str:
        logger.info(f"Aggregating proposals for congress v2 session: {prompt}")
        llm = GeneralLlm(self.aggregation_model, timeout=LONG_TIMEOUT)

        proposals_text = "\n\n---\n\n".join(
            [
                f"## {p.member.name} ({p.member.role})\n\n"
                f"```markdown\n{p.get_full_markdown_with_footnotes()}\n```"
                for p in proposals
                if p.member
            ]
        )

        all_scenarios = self._collect_scenarios_across_members(proposals)
        scenarios_text = self._format_scenarios_for_aggregation(
            all_scenarios, proposals
        )

        aggregation_prompt = clean_indents(
            f"""
            # AI Forecasting Congress V2: Scenario-Focused Synthesis Report

            You are synthesizing the proposals from multiple AI congress members
            deliberating on the following policy question:

            "{prompt}"

            # Scenarios Identified Across Members

            {scenarios_text}

            # Individual Proposals

            {proposals_text}

            ---

            # Your Task

            Write a comprehensive synthesis report that integrates scenario
            planning with policy analysis. Structure as follows:

            ### Executive Summary

            A 3-4 sentence overview of:
            - The key scenarios identified and their implications
            - The areas of agreement and disagreement across members
            - The most important forecasts that inform the debate

            ### Scenario Consensus

            Synthesize the scenarios across members:
            - Where do members agree on likely scenarios?
            - What key drivers do multiple members identify?
            - Are there scenarios some members missed that others caught?
            - What is the "consensus scenario map" — the 2-4 scenarios that
              best capture the range of plausible futures?

            ### Consensus Recommendations

            What policies do multiple members support? For each:
            - State the recommendation
            - List which members support it
            - Include relevant forecasts with footnotes [^N]
            - Note which scenarios this recommendation is best suited for

            ### Key Disagreements

            Where do members diverge and why? For each:
            - State the issue
            - Summarize each side's position
            - Explain how different scenarios, forecasts, or values lead to
              different conclusions
            - Assess the crux of the disagreement

            ### Cross-Scenario Robustness Matrix

            Create a summary matrix showing how different proposals perform
            across the consensus scenarios:
            - Which proposals are robust (work well across all scenarios)?
            - Which proposals are fragile (only work in specific scenarios)?
            - What are the key decision drivers — which external factors does
              the decision depend on most?

            ### Baseline Forecast Comparison

            Compare BASELINE forecasts across members:
            - Where forecasts converged (similar probabilities)
            - Where forecasts diverged significantly
            - What explains the differences

            ### Scenario-Conditional Forecast Comparison

            Compare SCENARIO-CONDITIONAL forecasts across members:
            - How do members' scenario probabilities differ?
            - Under each scenario, where do outcome predictions align or diverge?
            - Which scenarios show the most forecast uncertainty?

            ### Proposal-Conditional Forecast Comparison

            Compare PROPOSAL-CONDITIONAL forecasts:
            - Which proposals show the largest expected improvement over status quo?
            - Which proposals have the most uncertain outcomes?
            - Where do members predict unintended consequences?

            ### Integrated Recommendations

            Your synthesis of the best policy path forward:
            - Robust actions that work across all scenarios (no-regret moves)
            - Scenario-contingent actions with clear trigger conditions
            - Key hedging strategies
            - Early warning indicators to monitor
            - Be specific and actionable

            ### Combined Forecast Appendices

            Compile all forecasts by type (baseline, scenario indicator,
            scenario-conditional, proposal-conditional) from all members.
            Group similar forecasts and note the range of predictions.

            Format each forecast as:

            [^1] **[Question Title]** (from [Member Name])
            - Question: [Full question]
            - Resolution: [Resolution criteria]
            - Prediction: [Probability]
            - Reasoning: [Summary]

            Number the footnotes sequentially across all appendices.

            ---

            Be balanced but not wishy-washy. Identify which arguments are
            strongest and why. Your goal is to help decision-makers understand
            which policies are robust across scenarios and which are risky bets.
            """
        )

        result = await llm.invoke(aggregation_prompt)
        logger.info("Completed aggregation of proposals")
        return result

    # =========================================================================
    # SCENARIO REPORT
    # =========================================================================

    async def _generate_scenario_report(
        self,
        prompt: str,
        proposals: list[PolicyProposal],
        aggregated_report: str,
    ) -> str:
        logger.info(f"Generating scenario report for session: {prompt}")
        llm = GeneralLlm(self.aggregation_model, timeout=LONG_TIMEOUT)

        all_scenarios = self._collect_scenarios_across_members(proposals)
        all_drivers = self._collect_drivers_across_members(proposals)
        all_recommendations = self._collect_recommendations(proposals)

        scenarios_text = "\n\n".join(
            f"### {s['name']} (Prob: {s['probability']})\n"
            f"**Source members:** {', '.join(s['members'])}\n"
            f"**Narrative:** {s['narrative']}\n"
            f"**Criteria:**\n"
            + "\n".join(
                f"- {c['criterion_text']} (by {c.get('target_date', 'TBD')})"
                for c in s.get("criteria", [])
            )
            for s in all_scenarios
        )

        drivers_text = "\n".join(
            f"- **{d['name']}**: {d['description']} (identified by: {', '.join(d['members'])})"
            for d in all_drivers
        )

        recommendations_text = "\n".join(
            f"- [{r['member']}] {r['recommendation']}" for r in all_recommendations
        )

        scenario_report_prompt = clean_indents(
            f"""
            # Scenario Report: Formal Scenario Planning Document

            You are writing a formal scenario planning report in the style of
            Shell's scenario planning documents or the National Intelligence
            Council's Global Trends reports. This should be a polished,
            professional document suitable for senior decision-makers.

            ## Policy Question

            "{prompt}"

            ## Aggregated Analysis

            ```markdown
            {aggregated_report}
            ```

            ## Scenarios Identified

            {scenarios_text}

            ## Key Drivers

            {drivers_text}

            ## Policy Recommendations Across Members

            {recommendations_text}

            ---

            ## Your Task

            Write a formal Scenario Report with the following structure:

            # Scenario Report: [Descriptive Title]

            ## Executive Summary

            A 1-page executive summary of the strategic question, the scenario
            framework, and the key implications for decision-makers.
            (4-6 paragraphs)

            ## The Strategic Landscape

            Context-setting section describing the current situation, key
            trends, and why scenario planning is valuable for this question.
            (3-4 paragraphs)

            ## The Scenario Framework

            ### Key Drivers of Uncertainty

            Describe the 2-3 most important drivers that shape the future.
            Explain why these were selected and what makes them uncertain.

            ### The Scenario Matrix

            Present the scenario framework. If 2 drivers dominate, describe
            the 2x2 matrix. Otherwise, explain the logic behind the
            scenario set.

            ## Scenario Narratives

            For EACH scenario (2-4), write a 1-2 page narrative:

            ### [Scenario Name]

            **Probability:** [X%]

            **The Story:** A vivid, engaging narrative describing how this
            scenario unfolds. Written as if telling the story of the next
            5-10 years. Include specific developments, turning points,
            and cause-and-effect chains.

            **Key Indicators:** Early warning signs that this scenario is
            materializing (drawn from the scenario criteria).

            **Implications for Policy:** What this scenario means for the
            policy question. Which approaches work and which fail.

            ## Strategic Implications

            ### Robust Strategies

            Actions that perform well across ALL scenarios. These are the
            "no-regret" moves decision-makers should pursue regardless of
            which scenario unfolds.

            ### Contingent Strategies

            Actions that should be triggered only if specific scenarios
            begin to materialize. Include clear trigger conditions.

            ### Hedging Strategies

            Actions designed to reduce downside risk in adverse scenarios,
            even if they have some cost in favorable scenarios.

            ## Early Warning System

            A monitoring framework:
            - For each scenario, list the key indicators to watch
            - For each indicator, specify what data to track
            - Define trigger thresholds for activating contingent strategies

            ## Conclusion

            Closing thoughts on the strategic outlook and the most important
            actions decision-makers should take now.

            ---

            ## Guidelines

            - Write for senior decision-makers who need actionable insight
            - Be vivid and concrete in scenario narratives
            - Use specific forecasts and probabilities from the analysis
            - The tone should be professional but engaging, not academic
            - Scenario narratives should feel like plausible stories, not
              dry descriptions
            - Total length: 3000-5000 words
            - Use markdown formatting with clear headers
            """
        )

        try:
            result = await llm.invoke(scenario_report_prompt)
            logger.info("Completed scenario report generation")
            return result
        except Exception as e:
            logger.error(f"Failed to generate scenario report: {e}")
            return ""

    # =========================================================================
    # FUTURE SNAPSHOT
    # =========================================================================

    async def _generate_future_snapshot(
        self,
        prompt: str,
        proposals: list[PolicyProposal],
        aggregated_report: str,
    ) -> str:
        logger.info(f"Generating future snapshot for session: {prompt}")

        all_scenarios = self._collect_scenarios_across_members(proposals)
        all_forecasts = self._collect_all_forecasts(proposals)
        all_recommendations = self._collect_recommendations(proposals)

        baseline_text = self._format_forecasts_by_type(all_forecasts, "baseline")
        scenario_indicators_text = self._format_forecasts_by_type(
            all_forecasts, "scenario_indicator"
        )
        scenario_conditional_text = self._format_forecasts_by_type(
            all_forecasts, "scenario_conditional"
        )
        proposal_conditional_text = self._format_forecasts_by_type(
            all_forecasts, "proposal_conditional", "proposal_scenario_conditional"
        )

        scenarios_text = "\n\n".join(
            f"### {s['name']} (Prob: {s['probability']})\n{s['narrative']}"
            for s in all_scenarios
        )

        recommendations_text = "\n".join(
            f"- [{r['member']}] {r['recommendation']}" for r in all_recommendations
        )

        snapshot_prompt = clean_indents(
            f"""
            # Picture of the Future: Scenario-Aware Narrative Generator

            You are a journalist writing retrospective "Year in Review" articles
            from the future, looking back at what happened after the AI Congress's
            recommendations were either implemented or rejected — under different
            scenarios.

            ## Original Policy Question

            "{prompt}"

            ## Aggregate Policy Report

            ```markdown
            {aggregated_report}
            ```

            ## Scenarios

            {scenarios_text}

            ## Baseline Forecasts (Status Quo)

            {baseline_text}

            ## Scenario Indicator Forecasts

            {scenario_indicators_text}

            ## Scenario-Conditional Forecasts

            {scenario_conditional_text}

            ## Proposal-Conditional Forecasts

            {proposal_conditional_text}

            ## All Policy Recommendations

            {recommendations_text}

            ---

            ## Your Task

            Write 2-3 compelling scenario-specific newspaper narratives.

            ### Approach

            1. First, use the roll_multiple_dice tool to determine which scenario
               materializes. Roll dice for each scenario indicator forecast.

            2. Pick the 2 most contrasting scenarios to write about.

            3. For each selected scenario, write a newspaper narrative:

            #### NARRATIVE: "[Scenario Name]" — Recommendations Implemented

            Start with: "The date is <date you pick>..."

            Write a flowing narrative assuming:
            - The AI Congress's recommendations were implemented
            - This specific scenario materialized
            - For each relevant forecast, ROLL THE DICE using the
              roll_multiple_dice tool to determine outcomes
            - Use CONDITIONAL forecasts for policy-dependent outcomes
            - Use BASELINE forecasts for background events
            - For gaps, create plausible estimates marked with *

            Reference forecasts inline: "(X% [^N])"

            #### NARRATIVE: "[Scenario Name]" — Recommendations Rejected

            A contrasting narrative showing what the world looks like if
            recommendations were NOT implemented under this scenario.
            Use BASELINE forecasts for all outcomes.

            ---

            ## Guidelines

            - Make narratives vivid and engaging, like real journalism
            - Include specific dates, names (real where relevant, fake
              marked with †), and concrete details
            - Show cause-and-effect relationships
            - Your own estimates marked with * should be plausible
            - Neutral/journalistic tone
            - Include both positive and negative consequences
            - Each forecast should be explicitly mentioned with dice outcome
            - Ground speculation in research where possible
            - 1500-2500 words total across all narratives

            Include footnotes at the end of each narrative with full
            forecast details and outcomes.
            """
        )

        try:
            llm_wrapper = AgentSdkLlm("openrouter/openai/gpt-5.2")

            snapshot_agent = AiAgent(
                name="Future Snapshot Writer",
                instructions=snapshot_prompt,
                model=llm_wrapper,
                tools=[
                    roll_dice,
                    roll_multiple_dice,
                    create_search_tool("openrouter/perplexity/sonar-reasoning-pro"),
                ],
            )

            result = await AgentRunner.run(
                snapshot_agent,
                "Generate the scenario-aware future snapshot now.",
                max_turns=25,
            )
            return result.final_output

        except Exception as e:
            logger.error(f"Failed to generate future snapshot: {e}")
            return ""

    # =========================================================================
    # BLOG POST
    # =========================================================================

    async def _generate_blog_post(
        self,
        prompt: str,
        proposals: list[PolicyProposal],
        members: list[CongressMember],
    ) -> str:
        logger.info(f"Generating blog post for session: {prompt}")
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
                f"**Scenarios Identified:** {', '.join(s.name for s in p.scenarios)}\n\n"
                f"**Selected Proposal:** {p.selected_proposal_name}\n\n"
                f"**Key Recommendations:**\n"
                + "\n".join(f"- {rec}" for rec in p.key_recommendations[:5])
                + "\n\n**Key Forecasts:**\n"
                + "\n".join(
                    f"- {f.question_title}: {f.prediction}" for f in p.forecasts[:5]
                )
                + "\n\n**Contingency Plans:**\n"
                + "\n".join(f"- {c}" for c in p.contingency_plans[:3])
                + f"\n\n**Full Proposal:**\n"
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

                Since this congress included multiple AI models acting naturally,
                include a dedicated analysis section:

                ### How the Models Compared
                For each AI model participant, analyze:
                - What scenarios did they identify? How did their scenario
                  frameworks differ?
                - What priorities or values seemed most salient?
                - How did their forecasts compare on similar questions?
                - Did they show distinctive reasoning patterns?

                ### Scenario Thinking Differences
                How did the models approach scenario planning differently?
                - Which models identified the most creative scenarios?
                - Which models had the most rigorous scenario criteria?
                - How did their robustness analyses differ?
                """
            )

        blog_prompt = clean_indents(
            f"""
            # Write a Blog Post About This AI Congress V2 Session

            Write an engaging blog post about an AI Forecasting Congress V2
            session where AI agents used scenario planning to deliberate on:

            "{prompt}"

            ## Proposals Summary

            {proposals_summary}

            ## Blog Post Requirements

            Write a ~1500-2000 word blog post for a tech/policy audience.

            ### Structure

            1. **Hook** (1 paragraph): Start with the most surprising finding.

            2. **Context** (1-2 paragraphs): What is the AI Congress V2 and
               how does it differ from traditional policy analysis? Emphasize
               the scenario planning approach.

            3. **The Scenarios** (2-3 paragraphs): What scenarios did the AI
               members identify? Where did they agree and disagree on how the
               future might unfold?

            4. **Key Insights** (3-5 paragraphs): Most important takeaways.
               How do proposals perform across different scenarios? What are
               the robust vs fragile strategies?

            5. **The Good, Bad, and Ugly** (2-3 paragraphs):
               - Good: Surprising consensus, innovative scenarios, strong reasoning
               - Bad: Blind spots, weak arguments, missed scenarios
               - Ugly: Uncomfortable tradeoffs, unresolvable tensions

            6. **Implications** (1-2 paragraphs): What should decision-makers do?

            {ai_comparison_section}

            7. **Conclusion** (1 paragraph): Thought-provoking takeaway.

            ### Style Guidelines

            - Engaging, accessible style (not academic)
            - Use specific examples and quotes
            - Include specific forecasts with probabilities
            - Be analytical but not dry
            - Use markdown formatting
            - Include a catchy title
            """
        )

        try:
            return await llm.invoke(blog_prompt)
        except Exception as e:
            logger.error(f"Failed to generate blog post: {e}")
            return ""

    # =========================================================================
    # TWITTER POSTS
    # =========================================================================

    async def _generate_twitter_posts(
        self,
        prompt: str,
        proposals: list[PolicyProposal],
    ) -> list[str]:
        logger.info(f"Generating twitter posts for session: {prompt}")
        llm = GeneralLlm(self.aggregation_model, timeout=LONG_TIMEOUT)

        proposals_summary = "\n\n".join(
            [
                f"**{p.member.name}** ({p.member.role}, {p.member.political_leaning}):\n"
                f"Scenarios: {', '.join(s.name for s in p.scenarios)}\n"
                f"Selected proposal: {p.selected_proposal_name}\n"
                f"Key recommendations: {', '.join(p.key_recommendations[:3])}\n"
                f"Key forecasts: {'; '.join([f'{f.question_title}: {f.prediction}' for f in p.forecasts[:3]])}"
                for p in proposals
                if p.member
            ]
        )

        twitter_prompt = clean_indents(
            f"""
            Based on this AI Forecasting Congress V2 session on "{prompt}",
            generate 8-12 tweet-length excerpts (max 280 characters each)
            for a policy/tech audience on Twitter/X.

            ## Proposals Summary

            {proposals_summary}

            ## Categories to Cover

            **THE SCENARIOS** (2-3 tweets):
            - Most interesting or creative scenarios identified
            - Surprising scenario probability estimates
            - Where members' scenario frameworks diverged

            **THE GOOD** (2-3 tweets):
            - Robust strategies that work across all scenarios
            - Innovative contingency plans
            - Surprising areas of consensus

            **THE BAD** (2-3 tweets):
            - Scenarios where popular policies fail
            - Blind spots in scenario thinking
            - Fragile strategies masquerading as robust ones

            **THE INTERESTING** (2-3 tweets):
            - Counter-intuitive findings
            - Unexpected agreement between unlikely allies
            - Forecasts that diverged most across scenarios

            ## Tweet Guidelines

            Each tweet should:
            - Be self-contained and intriguing
            - Reference specific forecasts when relevant
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

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @staticmethod
    def _collect_scenarios_across_members(
        proposals: list[PolicyProposal],
    ) -> list[dict]:
        scenario_map: dict[str, dict] = {}
        for proposal in proposals:
            member_name = proposal.member.name if proposal.member else "Unknown"
            for scenario in proposal.scenarios:
                if scenario.name not in scenario_map:
                    scenario_map[scenario.name] = {
                        "name": scenario.name,
                        "narrative": scenario.narrative,
                        "probability": scenario.probability,
                        "is_status_quo": scenario.is_status_quo,
                        "members": [member_name],
                        "criteria": [
                            {
                                "criterion_text": c.criterion_text,
                                "target_date": c.target_date,
                                "resolution_criteria": c.resolution_criteria,
                            }
                            for c in scenario.criteria
                        ],
                    }
                else:
                    if member_name not in scenario_map[scenario.name]["members"]:
                        scenario_map[scenario.name]["members"].append(member_name)
        return list(scenario_map.values())

    @staticmethod
    def _collect_drivers_across_members(
        proposals: list[PolicyProposal],
    ) -> list[dict]:
        driver_map: dict[str, dict] = {}
        for proposal in proposals:
            member_name = proposal.member.name if proposal.member else "Unknown"
            for driver in proposal.drivers:
                if driver.name not in driver_map:
                    driver_map[driver.name] = {
                        "name": driver.name,
                        "description": driver.description,
                        "members": [member_name],
                    }
                else:
                    if member_name not in driver_map[driver.name]["members"]:
                        driver_map[driver.name]["members"].append(member_name)
        return list(driver_map.values())

    @staticmethod
    def _collect_all_forecasts(
        proposals: list[PolicyProposal],
    ) -> list[dict]:
        forecasts = []
        for proposal in proposals:
            member_name = proposal.member.name if proposal.member else "Unknown"
            for forecast in proposal.forecasts:
                forecasts.append(
                    {
                        "member": member_name,
                        "title": forecast.question_title,
                        "question": forecast.question_text,
                        "prediction": forecast.prediction,
                        "resolution_criteria": forecast.resolution_criteria,
                        "reasoning": forecast.reasoning,
                        "forecast_type": forecast.forecast_type,
                        "conditional_on_scenario": forecast.conditional_on_scenario,
                        "conditional_on_proposal": forecast.conditional_on_proposal,
                    }
                )
        return forecasts

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

    @staticmethod
    def _format_scenarios_for_aggregation(
        all_scenarios: list[dict],
        proposals: list[PolicyProposal],
    ) -> str:
        if not all_scenarios:
            return "No scenarios identified."

        lines = []
        for s in all_scenarios:
            members_str = ", ".join(s["members"])
            status_quo_note = " (Status Quo)" if s.get("is_status_quo") else ""
            lines.append(
                f"### {s['name']}{status_quo_note} (Prob: {s['probability']})\n"
                f"**Identified by:** {members_str}\n"
                f"**Narrative:** {s['narrative']}"
            )
            if s.get("criteria"):
                lines.append("**Criteria:**")
                for c in s["criteria"]:
                    date_str = (
                        f" (by {c['target_date']})" if c.get("target_date") else ""
                    )
                    lines.append(f"- {c['criterion_text']}{date_str}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_forecasts_by_type(
        all_forecasts: list[dict],
        *forecast_types: str,
    ) -> str:
        filtered = [f for f in all_forecasts if f["forecast_type"] in forecast_types]
        if not filtered:
            return "None"

        lines = []
        for f in filtered:
            scenario_note = ""
            if f.get("conditional_on_scenario"):
                scenario_note = f" *(Under scenario: {f['conditional_on_scenario']})*"
            proposal_note = ""
            if f.get("conditional_on_proposal"):
                proposal_note = " *(Conditional on proposal)*"

            lines.append(
                f"- **{f['title']}**{scenario_note}{proposal_note} "
                f"({f['member']}): {f['prediction']}\n"
                f"  - Question: {f['question']}\n"
                f"  - Resolution: {f['resolution_criteria']}"
            )
        return "\n".join(lines)
