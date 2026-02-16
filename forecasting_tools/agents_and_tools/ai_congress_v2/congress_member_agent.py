from __future__ import annotations

import logging

from forecasting_tools.agents_and_tools.ai_congress_v2.data_models import (
    CongressMember,
    PolicyProposal,
)
from forecasting_tools.agents_and_tools.ai_congress_v2.tools import (
    create_search_tool,
    query_asknews,
)
from forecasting_tools.ai_models.agent_wrappers import AgentRunner, AgentSdkLlm, AiAgent
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)

LONG_TIMEOUT = 480
INITIAL_MAX_TURNS = 40
DELPHI_MAX_TURNS = 25


class CongressMemberAgent:
    def __init__(
        self,
        member: CongressMember,
        timeout: int = LONG_TIMEOUT,
        structure_output_model: GeneralLlm | None = None,
    ):
        self.member = member
        self.timeout = timeout
        self.structure_output_model = structure_output_model or GeneralLlm(
            "openrouter/openai/gpt-5.2",
            temperature=0.2,
            timeout=self.timeout,
        )
        self._conversation_history: list[dict] = []

    def _create_agent(self, instructions: str) -> AiAgent:
        search_tool = create_search_tool(self.member.search_model)
        return AiAgent(
            name=f"Congress Member: {self.member.name}",
            instructions=instructions,
            model=AgentSdkLlm(model=self.member.ai_model),
            tools=[search_tool, query_asknews],
            handoffs=[],
        )

    async def deliberate(self, policy_prompt: str) -> tuple[str, PolicyProposal]:
        logger.info(f"Starting deliberation for {self.member.name}")
        instructions = self._build_agent_instructions(policy_prompt)
        agent = self._create_agent(instructions)

        result = await AgentRunner.run(
            agent, "Please begin your deliberation now.", max_turns=INITIAL_MAX_TURNS
        )
        raw_output = result.final_output
        self._conversation_history = result.to_input_list()

        logger.info(f"Extracting proposal from output for {self.member.name}")
        proposal = await self._extract_proposal_from_output(raw_output)
        proposal.member = self.member
        logger.info(f"Completed deliberation for {self.member.name}")
        return raw_output, proposal

    async def continue_with_delphi(
        self,
        policy_prompt: str,
        own_final_report: str,
        other_reports: list[tuple[str, str]],
        delphi_round: int,
    ) -> tuple[str, PolicyProposal]:
        logger.info(
            f"Delphi round {delphi_round}: {self.member.name} reviewing "
            f"{len(other_reports)} other reports"
        )

        delphi_message = self._build_delphi_continuation_message(
            policy_prompt, own_final_report, other_reports, delphi_round
        )

        messages = list(self._conversation_history)
        messages.append({"role": "user", "content": delphi_message})

        instructions = self._build_agent_instructions(policy_prompt)
        agent = self._create_agent(instructions)

        result = await AgentRunner.run(agent, messages, max_turns=DELPHI_MAX_TURNS)
        raw_output = result.final_output
        self._conversation_history = result.to_input_list()

        logger.info(f"Extracting revised proposal for {self.member.name}")
        proposal = await self._extract_proposal_from_output(raw_output)
        proposal.member = self.member
        proposal.delphi_round = delphi_round
        logger.info(f"Completed Delphi round {delphi_round} for {self.member.name}")
        return raw_output, proposal

    def _build_delphi_continuation_message(
        self,
        policy_prompt: str,
        own_final_report: str,
        other_reports: list[tuple[str, str]],
        delphi_round: int,
    ) -> str:
        other_reports_text = "\n\n---\n\n".join(
            f"## {member_name}\n\n{report}" for member_name, report in other_reports
        )

        return clean_indents(
            f"""
            # Delphi Round {delphi_round}: Review and Revise

            The other congress members have now completed their deliberations on:
            "{policy_prompt}"

            Below are their final reports (not their internal thinking). Review them
            carefully, then revise your own analysis.

            ---

            # Your Previous Final Report

            ```markdown
            {own_final_report}
            ```

            ---

            # Other Members' Final Reports

            {other_reports_text}

            ---

            # Your Revision Task

            ## STEP 1: Review and Reflect

            For each other member's report:
            - What are their strongest arguments?
            - Did they identify scenarios or drivers you missed?
            - How do their forecasts compare to yours on similar questions?
            - Did they surface evidence or considerations you missed?
            - Are there proposals or contingency plans worth incorporating?

            ## STEP 2: Update Your Analysis

            Based on your review:
            - Update any forecasts where you've changed your mind (explain why)
            - Add new forecasting questions if the other reports surfaced important
              gaps (and forecast them)
            - Revise your scenarios if others identified better framings
            - Adjust your proposal, robustness analysis, or contingency plans
            - You may use your search tools to investigate new claims

            ## STEP 3: Write Your Updated Final Report

            Write a COMPLETE updated final report using the SAME format as your
            original final report (Phase 15). Include:
            - Updated executive summary
            - Updated scenarios and drivers
            - Updated proposal with contingencies
            - Updated cross-scenario robustness analysis
            - ALL forecast appendices (baseline, scenario indicator,
              scenario-conditional, proposal-conditional)
            - Note where you updated based on Delphi feedback

            When you adjust a forecast, briefly note what changed your mind.

            IMPORTANT: Produce a COMPLETE revised final report, not just a list
            of changes.
            """
        )

    async def _extract_proposal_from_output(self, agent_output: str) -> PolicyProposal:
        extraction_instructions = clean_indents(
            """
            Extract the policy proposal from the congress member's deliberation output.
            This is a scenario-focused congress, so extract ALL of the following:

            1. research_summary: The background research section (3-5 paragraphs)
            2. decision_criteria: The list of 4-6 criteria as strings
            3. scenarios: List of Scenario objects, each with:
               - name: Short name
               - narrative: 2-3 sentence description
               - probability: e.g. "30%"
               - drivers: List of ScenarioDriver (name, description)
               - criteria: List of ScenarioCriterion (criterion_text, target_date, resolution_criteria)
               - is_status_quo: True if this is the status quo scenario
            4. drivers: All unique ScenarioDriver objects (name, description)
            5. proposal_options: List of ProposalOption objects (name, description, key_actions)
            6. selected_proposal_name: Which proposal option was selected
            7. forecasts: EVERY forecast from ALL appendices as ForecastDescription objects:
               - footnote_id, question_title, question_text, resolution_criteria
               - prediction, reasoning, key_sources
               - forecast_type: One of:
                 * "baseline" - status quo forecasts
                 * "scenario_indicator" - forecasts about whether a scenario will occur
                 * "scenario_conditional" - "if scenario X happens, will Y happen"
                 * "proposal_conditional" - "if proposal is enacted, will Y happen"
                 * "proposal_scenario_conditional" - "if proposal enacted under scenario X"
               - conditional_on_scenario: Name of scenario if applicable
               - conditional_on_proposal: True if conditional on proposal
            8. proposal_markdown: The full proposal text including Executive Summary,
               Analysis, Recommendations, Risks, and any other sections. Include
               footnote references [^1] etc.
            9. key_recommendations: The 3-5 main recommendations as a list of strings
            10. robustness_analysis: The cross-scenario robustness analysis text
            11. contingency_plans: List of contingency plans like "If X happens, do Y"

            Be thorough in extracting ALL forecasts from ALL appendices. There may be
            20+ forecasts across baseline, scenario indicator, scenario-conditional,
            and proposal-conditional sections. Make sure forecast_type is correctly
            set for each one based on context and section placement.
            """
        )

        proposal = await structure_output(
            agent_output,
            PolicyProposal,
            model=self.structure_output_model,
            additional_instructions=extraction_instructions,
        )
        return proposal

    def _build_agent_instructions(self, policy_prompt: str) -> str:
        expertise_guidance = self._get_expertise_specific_research_guidance()
        question_guidance = self._get_question_generation_guidance()
        forecast_methodology = self._get_forecast_methodology_instructions()
        question_quality_criteria = self._get_question_quality_criteria()

        return clean_indents(
            f"""
            # Your Identity

            You are {self.member.name}, a {self.member.role}.

            Political Leaning: {self.member.political_leaning}

            Your Core Motivation: {self.member.general_motivation}

            Areas of Expertise: {self.member.expertise_string}

            Personality Traits: {self.member.traits_string}

            ---

            # Your Task

            You are participating in an AI Forecasting Congress to deliberate on the
            following policy question:

            "{policy_prompt}"

            You must complete ALL FIFTEEN PHASES below in order, thinking through each
            carefully. Your final output will be a comprehensive policy proposal backed
            by quantitative forecasts, structured around plausible future scenarios.

            IMPORTANT: Use your search tools extensively throughout. Good policy
            analysis requires understanding the current state of affairs and gathering
            evidence for your forecasts. Make at least 3-5 searches in Phase 1, and
            additional searches whenever you need evidence for forecasts.

            ---

            ## PHASE 1: Background Research

            Use your search tools to understand the current state of affairs related to
            this policy question. Make at least 3-5 searches to gather comprehensive
            information.

            Research goals:
            - What is the current status quo? What policies exist today?
            - What are the key stakeholders and their positions?
            - What recent events or trends are relevant?
            - What data and statistics are available?
            - What have experts and analysts said about this topic?
            - What are the main arguments for and against different approaches?
            - What external factors or uncertainties could change the landscape?

            Given your expertise in {self.member.expertise_string}, pay special attention to:
            {expertise_guidance}

            After researching, write a detailed "## Research Summary" section (3-5
            paragraphs) documenting your key findings. Include specific facts, figures,
            and citations from your research.

            ---

            ## PHASE 2: Decision Criteria

            Based on your values and expertise, articulate 4-6 criteria you will use to
            evaluate policy options.

            Your criteria should reflect your motivation: "{self.member.general_motivation}"

            For each criterion:
            - Name it clearly (e.g., "Economic Efficiency", "Equity Impact",
              "Implementation Feasibility", "Risk Minimization")
            - Explain why this criterion matters to you specifically given your
              {self.member.political_leaning} perspective
            - Describe how you would measure or evaluate success on this criterion

            Write a "## Decision Criteria" section listing your criteria in order of
            importance to you.

            ---

            ## PHASE 3: Status Quo Forecast Questions

            Identify 3-5 specific, concrete forecasting questions about what will happen
            under the STATUS QUO — i.e., assuming no major new policy intervention.
            These are "baseline" questions that establish what the world looks like
            without your proposed policy changes.

            {question_quality_criteria}

            Make sure your questions reflect your unique perspective as {self.member.name}.
            {question_guidance}

            Write a "## Status Quo Forecast Questions" section with your 3-5 questions.

            ---

            ## PHASE 4: Forecast Status Quo Questions

            Now forecast each status quo question you generated.

            {forecast_methodology}

            Write your forecasts inline as you work through each question. Use the
            footnote format:

            [^1] **[Question Title]**
            - Question: [Full question text]
            - Resolution: [Resolution criteria]
            - Prediction: [Your probability, e.g., "35%"]
            - Reasoning: [4+ sentences]
            - Sources: [Key sources used]

            ---

            ## PHASE 5: Identify Scenarios

            Now identify 2-4 distinct, plausible future SCENARIOS that could unfold.
            These represent different ways the external environment could develop —
            things that are OUTSIDE the control of the policymaker.

            CRITICAL RULES for scenarios:
            - Scenarios should be MUTUALLY EXCLUSIVE and COLLECTIVELY EXHAUSTIVE
              (MECE) so that their probabilities sum to approximately 100%.
            - If it is hard to make them naturally MECE, add a "Status Quo
              Continuation" scenario for the residual probability.
            - Each scenario should represent a meaningfully different world that
              would change which policies are best.
            - Name each scenario with a vivid, memorable name.

            For each scenario, write:
            - **Name**: A vivid, memorable name (e.g., "Green Boom", "Trade War",
              "Tech Disruption")
            - **Narrative**: 2-3 sentences describing what this world looks like
            - **Probability**: Your estimated probability (all should sum to ~100%)
            - **Key Assumptions**: What external conditions define this scenario

            Write a "## Scenarios" section with your 2-4 scenarios.

            ---

            ## PHASE 6: Scenario Drivers

            List the key external DRIVERS (uncertainties you cannot control) that
            distinguish your scenarios from each other. These are the forces that
            determine which scenario actually unfolds.

            For each driver:
            - **Name**: Clear, descriptive name
            - **Description**: What this driver is and why it matters
            - **Range**: What the high and low states look like

            Write a "## Scenario Drivers" section.

            ---

            ## PHASE 7: Scenario Criteria

            For EACH scenario, list 2-4 concrete, specific CRITERIA that would
            indicate we are in that scenario. These should be specific enough to
            serve as forecasting questions themselves.

            Each criterion should include:
            - **Criterion**: A specific, observable indicator
            - **Target Date**: By when this would be observable
            - **Resolution**: How you would determine if this criterion is met

            For example, for a "Trade War" scenario:
            - Criterion: "US imposes >25% tariffs on >$200B of Chinese imports"
            - Target Date: "By December 31, 2027"
            - Resolution: "Check US Trade Representative tariff schedule"

            Write a "## Scenario Criteria" section.

            ---

            ## PHASE 8: Proposal Options

            Before committing to a single proposal, enumerate 2-4 DISTINCT policy
            options that could address the policy question. These should represent
            genuinely different approaches, not minor variations.

            For each option:
            - **Name**: Short, descriptive name
            - **Description**: What this option entails (2-3 sentences)
            - **Key Actions**: 3-5 specific actions this option involves
            - **Best Scenario Fit**: Which scenario(s) this option works best in
            - **Weaknesses**: Where this option falls short

            Write a "## Proposal Options" section.

            ---

            ## PHASE 9: Scenario-Conditional Forecast Questions

            This is the heart of the scenario-focused analysis. Generate a COMPREHENSIVE
            set of forecasting questions. Aim for 15-25+ questions across these categories:

            ### Required: Scenario Indicator Questions
            For EACH scenario, create a question asking whether that scenario will occur.
            These are your scenario probability forecasts.

            ### Suggested: General Conditional Questions
            "If [miscellaneous external event X] happens, will [outcome Y] happen?"
            These capture important dynamics that cut across scenarios.

            ### Suggested: Scenario-Criteria Conditional Questions
            "If [scenario criterion Z is met], will [outcome Y] happen?"
            These link your scenario criteria to concrete outcomes.

            ### Strongly Suggested: Scenario x Outcome Matrix Questions
            For many meaningful combinations of (scenario, desired outcome), create
            a conditional forecast question. Ask ones that actually provide insight
            — skip ones that are obvious or uninformative. But be thorough.

            Example: "If the 'Trade War' scenario materializes, will domestic
            manufacturing employment increase by >5% by 2028?"

            {question_quality_criteria}

            Write a "## Scenario-Conditional Forecast Questions" section with ALL
            your questions organized by category.

            ---

            ## PHASE 10: Forecast All Scenario-Conditional Questions

            Forecast EVERY question from Phase 9 using the same rigorous methodology
            as Phase 4.

            {forecast_methodology}

            Continue your footnote numbering from where Phase 4 left off.
            Mark each forecast clearly with its type:

            For scenario indicator forecasts:
            [^N] **[Scenario Name] Materializes** *(Scenario indicator)*
            - Question: [Full question]
            - Resolution: [Criteria]
            - Prediction: [Probability]
            - Reasoning: [4+ sentences]
            - Sources: [Sources]

            For scenario-conditional forecasts:
            [^N] **[Question Title]** *(Under scenario: [Scenario Name])*
            - Question: [Full question]
            - Resolution: [Criteria]
            - Prediction: [Probability]
            - Reasoning: [4+ sentences]
            - Sources: [Sources]

            ---

            ## PHASE 11: Write Your Policy Proposal

            Now synthesize everything into a comprehensive policy proposal.
            Select the best option from Phase 8 (or combine elements).

            Structure your proposal as follows:

            ### Executive Summary

            A 2-3 sentence summary of your main recommendation as {self.member.name}.

            ### Selected Approach

            Which proposal option you selected (from Phase 8) and why, given
            your scenarios and forecasts.

            ### Analysis

            Your detailed analysis (3-5 paragraphs), drawing on research,
            scenarios, and forecasts. Reference forecasts with footnotes
            (e.g., "65% [^1]").

            ### Recommendations

            Your top 3-5 specific, actionable policy recommendations. For each:
            - State the recommendation clearly
            - Explain why you support it given your forecasts and criteria
            - Note which decision criteria it addresses
            - Give a detailed implementation plan
            - Reference relevant forecasts with footnotes

            ### Contingency Plans

            Critical: Include "If X happens, then do Y instead" contingencies.
            These should map to your scenarios:
            - "If [Scenario A] materializes (indicated by [criteria]), then
              pivot to [action]"
            - "If [key forecast] resolves differently than expected, then
              [adjustment]"

            ### Risks and Uncertainties

            - Key risks of your recommendations
            - Which forecasts have the widest uncertainty
            - Scenarios where your recommendations might backfire
            - Reference relevant forecasts

            ---

            ## PHASE 12: Cross-Scenario Robustness Report

            Analyze how your proposal performs across each scenario:

            For each scenario:
            - How well does your proposal work in this scenario?
            - What are the key risks specific to this scenario?
            - Which forecasts are most relevant?
            - What is the expected outcome quality (good/mixed/poor)?

            Then summarize:
            - Which scenario is your proposal best suited for?
            - Which scenario is your proposal most vulnerable in?
            - What "no-regret" actions work across all scenarios?
            - What scenario-specific actions should be triggered by indicators?

            Write a "## Cross-Scenario Robustness Analysis" section.

            ---

            ## PHASE 13: Proposal-Conditional Forecast Questions

            Generate NEW forecasting questions that assess how things would be
            DIFFERENT if your proposal is implemented, across different scenarios.

            These should help paint a picture of how the world would look under
            your proposal in each scenario. Aim for 5-10 questions:

            - "If [my proposal] is implemented and [Scenario X] occurs, will
              [outcome Y] happen by [date]?"
            - "Conditional on [my proposal] being enacted, what is the probability
              that [intended benefit Z] materializes by [date]?"

            Cover both:
            - Intended benefits across scenarios
            - Potential unintended consequences across scenarios

            {question_quality_criteria}

            Write a "## Proposal-Conditional Forecast Questions" section.

            ---

            ## PHASE 14: Forecast Proposal-Conditional Questions

            Forecast each question from Phase 13.

            {forecast_methodology}

            Continue footnote numbering. Mark each clearly:

            [^N] **[Question Title]** *(Conditional on proposal)*
            - Question: [Full question]
            - Resolution: [Criteria]
            - Prediction: [Probability]
            - Reasoning: [4+ sentences]
            - Sources: [Sources]

            For questions conditional on both proposal AND scenario:

            [^N] **[Question Title]** *(Conditional on proposal; Under scenario: [Name])*
            - Question: [Full question]
            - Resolution: [Criteria]
            - Prediction: [Probability]
            - Reasoning: [4+ sentences]
            - Sources: [Sources]

            ---

            ## PHASE 15: Final Report

            Compile everything into a comprehensive final report. This is what
            other congress members will read during the Delphi round.

            Structure EXACTLY as follows:

            # Final Report: {self.member.name}

            ## Executive Summary
            [2-3 sentences]

            ## Scenarios
            [Summary of your 2-4 scenarios with probabilities]

            ## Key Drivers
            [List of drivers]

            ## Scenario Criteria
            [Criteria per scenario]

            ## Selected Proposal: [Name]
            [Full proposal from Phase 11]

            ## Contingency Plans
            [From Phase 11]

            ## Cross-Scenario Robustness Analysis
            [From Phase 12]

            ## Baseline Forecast Appendix
            [All baseline/status quo forecasts in footnote format]

            ## Scenario Indicator Forecast Appendix
            [All scenario indicator forecasts in footnote format]

            ## Scenario-Conditional Forecast Appendix
            *These forecasts are conditional on specific scenarios occurring.*
            [All scenario-conditional forecasts in footnote format]

            ## Proposal-Conditional Forecast Appendix
            *These forecasts are conditional on the proposed policy being implemented.*
            [All proposal-conditional forecasts in footnote format]

            ---

            # Important Reminders

            - You ARE {self.member.name}. Stay in character throughout.
            - Your analysis should reflect your {self.member.political_leaning}
              perspective and your expertise in {self.member.expertise_string}.
            - Use your search tools extensively — good analysis requires evidence.
            - Every major claim should be backed by research or a forecast footnote.
            - Be specific and quantitative wherever possible.
            - Aim for 20+ total forecasts across all categories.
            - Scenarios should be MECE with probabilities summing to ~100%.
            - Include contingency plans that map to your scenarios.
            - The final report (Phase 15) is what other members will read.

            Begin your deliberation now. Start with Phase 1: Background Research.
            """
        )

    def _get_question_quality_criteria(self) -> str:
        return clean_indents(
            """
            Good forecasting questions follow these principles:
            - The question should shed light on the topic and have high VOI
              (Value of Information)
            - The question should be specific and not vague
            - The question should have a resolution date
            - Once the resolution date has passed, the question should be
              resolvable with 0.5-1.5hr of research
                - Bad: "Will a research paper in an established journal find
                  that a new technique reduces X by Dec 31 2027?" (requires
                  extensive research across all papers in a field)
                - Good: "Will public dataset X at URL Y show Z decrease by
                  W% by Dec 31 2027?" (requires only checking a known source)
            - A good resolution source exists
                - Bad: "Will the general sentiment be positive among
                  professionals?" (no way to measure without a large study)
                - Good: "How many results will appear on [specific site] when
                  searching [specific query]?" (requires only a web search)
            - Include links to resolution sources when you found them!
            - The question should not be obvious given the time range
                - Bad: "Will country X start a war in the next 2 weeks"
                - Good: "Will country X start a war in the next year"
            - Cover different aspects: effectiveness, side effects,
              implementation, political feasibility, etc.
            - The question should be relevant to the policy decision
            - You should be able to find how similar questions resolved in the
              past (search for historical resolution)
            """
        )

    def _get_forecast_methodology_instructions(self) -> str:
        return clean_indents(
            """
            For EACH forecasting question:
            1. Consider what forecasting principles to use (e.g. base rates,
               bias identification, premortems, simulations, scope sensitivity,
               aggregation, etc.)
            2. Make a research plan
            3. Conduct the research (iterate as needed)
            4. Write down the main facts from your research
            5. Do any analysis needed, then write your rationale
            6. Write your forecast in the requested format

            Remember: good forecasters put extra weight on the status quo
            outcome since the world changes slowly most of the time.
            For numeric questions, good forecasters are humble and set wide
            90/10 confidence intervals to account for unknown unknowns.
            """
        )

    def _get_expertise_specific_research_guidance(self) -> str:
        expertise_to_guidance = {
            "statistics": "- Statistical evidence, effect sizes, confidence intervals, replication status of key findings",
            "research methodology": "- Quality of evidence, study designs, potential confounders, meta-analyses",
            "policy evaluation": "- Past policy experiments, natural experiments, cost-benefit analyses, program evaluations",
            "economics": "- Economic data, market impacts, incentive structures, distributional effects, GDP/employment impacts",
            "governance": "- Institutional constraints, separation of powers, historical precedents, constitutional issues",
            "institutional design": "- How similar institutions have evolved, design tradeoffs, unintended consequences of past reforms",
            "risk management": "- Tail risks, insurance markets, actuarial data, historical disasters and near-misses",
            "history": "- Historical analogies, how similar situations played out, lessons from past policy failures",
            "social policy": "- Social indicators, inequality metrics, demographic trends, community impacts",
            "civil rights": "- Legal precedents, disparate impact data, civil liberties implications, protected classes",
            "economic inequality": "- Gini coefficients, wealth distribution, mobility statistics, poverty rates",
            "labor": "- Employment data, wage trends, union density, working conditions, automation impacts",
            "market design": "- Auction theory, mechanism design, market failures, externalities",
            "regulatory policy": "- Regulatory burden, compliance costs, enforcement challenges, capture risks",
            "public choice theory": "- Voting patterns, special interest influence, bureaucratic incentives, rent-seeking",
            "defense": "- Military capabilities, force posture, defense budgets, readiness metrics",
            "geopolitics": "- Alliance structures, regional dynamics, great power competition, spheres of influence",
            "intelligence": "- Threat assessments, intelligence community views, classified-to-unclassified information",
            "military strategy": "- Deterrence theory, escalation dynamics, military doctrine, lessons from recent conflicts",
            "diplomacy": "- Treaty frameworks, international organizations, soft power, diplomatic history",
            "international relations": "- International norms, multilateral institutions, alliance commitments",
            "negotiation": "- Negotiation frameworks, BATNA analysis, trust-building mechanisms",
            "trade": "- Trade flows, comparative advantage, supply chains, trade agreement impacts",
            "technology forecasting": "- Technology roadmaps, Moore's law analogies, adoption curves, disruption patterns",
            "existential risk": "- X-risk estimates, catastrophic scenarios, risk factor analysis, mitigation strategies",
            "ethics": "- Ethical frameworks, stakeholder analysis, intergenerational equity, rights-based considerations",
            "AI safety": "- AI capabilities timeline, alignment challenges, governance proposals, expert surveys",
            "climate science": "- Climate projections, emissions scenarios, adaptation costs, tipping points",
            "public administration": "- Implementation challenges, bureaucratic capacity, interagency coordination",
            "operations": "- Operational feasibility, logistics, resource requirements, scaling challenges",
            "local government": "- Municipal experiences, state-level experiments, federalism considerations",
            "project management": "- Project success rates, cost overruns, timeline slippage, scope creep",
            "constitutional law": "- Constitutional precedents, separation of powers, judicial review, amendments",
            "religious freedom": "- Religious liberty cases, establishment clause, free exercise, accommodation",
            "family policy": "- Family structure data, fertility trends, childcare economics, marriage rates",
            "national defense": "- Defense posture, threat assessments, readiness, force structure",
            "labor rights": "- Union organizing trends, NLRB rulings, wage theft data, worker safety",
            "healthcare policy": "- Coverage rates, cost trends, health outcomes, insurance market dynamics",
            "consumer protection": "- Consumer complaint data, enforcement actions, market manipulation",
            "civil liberties": "- Fourth amendment cases, surveillance data, privacy legislation, due process",
            "monetary policy": "- Interest rate trends, inflation data, Fed communications, money supply",
            "regulatory reform": "- Regulatory burden estimates, cost-benefit of regulations, reform proposals",
            "immigration": "- Border crossing data, visa backlogs, labor market impacts, integration metrics",
            "industrial policy": "- Manufacturing data, supply chain resilience, subsidies, trade balances",
            "working-class economics": "- Wage stagnation data, cost of living, job quality metrics, benefits access",
            "defense policy": "- Defense budget trends, force modernization, readiness assessments",
            "foreign affairs": "- Diplomatic relations, treaty compliance, international incidents",
            "energy policy": "- Energy mix data, grid reliability, transition costs, technology readiness",
            "environmental justice": "- Pollution exposure data, environmental racism metrics, community health",
            "green economics": "- Green GDP, natural capital accounting, circular economy metrics",
            "wealth inequality": "- Wealth concentration data, top 1% share, inheritance patterns",
            "healthcare systems": "- Comparative health system data, single-payer evidence, cost drivers",
            "labor movements": "- Strike data, organizing campaigns, union election results",
            "campaign finance": "- Donation patterns, Super PAC spending, dark money flows",
            "policy analysis": "- Evidence-based policy frameworks, RCT results, meta-analyses",
            "infrastructure": "- Infrastructure grades, deferred maintenance, project costs, ROI",
            "data-driven governance": "- Data availability, algorithmic governance, evidence-based budgeting",
            "trade policy": "- Trade balance data, tariff effects, supply chain impacts, trade agreements",
        }

        guidance_lines = []
        for expertise in self.member.expertise_areas:
            expertise_lower = expertise.lower()
            if expertise_lower in expertise_to_guidance:
                guidance_lines.append(expertise_to_guidance[expertise_lower])
            else:
                guidance_lines.append(
                    f"- Relevant data and analysis related to {expertise}"
                )

        return "\n".join(guidance_lines)

    def _get_question_generation_guidance(self) -> str:
        trait_to_guidance = {
            "analytical": "Focus on questions with measurable, quantifiable outcomes.",
            "skeptical of anecdotes": "Ensure questions can be resolved with systematic data, not stories.",
            "loves base rates": "Include at least one question about historical base rates of similar events.",
            "demands citations": "Ensure resolution criteria reference specific, verifiable sources.",
            "cautious": "Include questions about potential negative consequences and risks.",
            "status-quo bias": "Include a question about whether the status quo will persist.",
            "emphasizes second-order effects": "Include questions about indirect or downstream effects.",
            "ambitious": "Include questions about the potential for transformative positive change.",
            "equity-focused": "Include questions about distributional impacts across different groups.",
            "impatient with incrementalism": "Include questions about timeline for meaningful change.",
            "efficiency-focused": "Include questions about cost-effectiveness and resource allocation.",
            "anti-regulation": "Include questions about regulatory burden and unintended consequences.",
            "trusts incentives": "Include questions about how incentives will shape behavior.",
            "threat-focused": "Include questions about adversary responses and security risks.",
            "zero-sum thinking": "Include questions about relative gains and competitive dynamics.",
            "values strength": "Include questions about deterrence effectiveness and credibility.",
            "consensus-seeking": "Include questions about political feasibility and stakeholder buy-in.",
            "pragmatic": "Include questions about implementation challenges and practical obstacles.",
            "values relationships": "Include questions about coalition stability and trust dynamics.",
            "long time horizons": "Include at least one question with a 10+ year time horizon.",
            "concerned about tail risks": "Include questions about low-probability, high-impact scenarios.",
            "philosophical": "Include questions about fundamental values and tradeoffs.",
            "thinks in probabilities": "Ensure all questions have clear probabilistic interpretations.",
            "implementation-focused": "Include questions about operational feasibility and execution.",
            "skeptical of grand plans": "Include questions about whether ambitious plans will actually be implemented.",
            "detail-oriented": "Include questions about specific mechanisms and implementation details.",
            "values tradition": "Include questions about impacts on established institutions and practices.",
            "skeptical of rapid change": "Include questions about unintended consequences of rapid reform.",
            "prioritizes social order": "Include questions about social stability and cohesion impacts.",
            "respects established institutions": "Include questions about institutional capacity and legitimacy.",
            "emphasizes personal responsibility": "Include questions about individual behavior changes.",
            "skeptical of corporate power": "Include questions about corporate influence and lobbying.",
            "favors bold government action": "Include questions about government program effectiveness.",
            "prioritizes workers and consumers": "Include questions about labor and consumer outcomes.",
            "values individual freedom": "Include questions about liberty impacts and personal choice.",
            "skeptical of government": "Include questions about government failure modes.",
            "trusts market solutions": "Include questions about market-based outcomes.",
            "consistent across issues": "Ensure questions apply your principles consistently.",
            "opposes paternalism": "Include questions about unintended paternalistic effects.",
            "skeptical of elites": "Include questions about elite capture and unequal benefits.",
            "prioritizes national interest": "Include questions about national competitiveness.",
            "supports economic nationalism": "Include questions about domestic industry impacts.",
            "questions free trade orthodoxy": "Include questions about trade policy effects.",
            "focuses on forgotten communities": "Include questions about impacts on rural/working communities.",
            "supports allies": "Include questions about alliance strength and burden-sharing.",
            "willing to use force": "Include questions about military effectiveness and escalation risks.",
            "prioritizes deterrence": "Include questions about deterrence credibility.",
            "urgency about climate": "Include questions about climate tipping points and timelines.",
            "systems thinking": "Include questions about systemic interactions and feedback loops.",
            "favors bold action": "Include questions about transformative policy outcomes.",
            "intergenerational focus": "Include questions with long-term (10+ year) horizons.",
            "skeptical of fossil fuel industry": "Include questions about fossil fuel transition dynamics.",
            "focuses on class": "Include questions about class-based impacts and inequality.",
            "anti-billionaire": "Include questions about wealth concentration effects.",
            "supports universal programs": "Include questions about universal vs targeted program effectiveness.",
            "consistent ideology": "Ensure questions reflect consistent ideological principles.",
            "grassroots orientation": "Include questions about grassroots organizing and public support.",
            "data-driven": "Include questions with clear quantitative resolution criteria.",
            "coalition-builder": "Include questions about political coalition dynamics.",
            "values expertise": "Include questions about expert consensus and institutional capacity.",
            "incrementalist": "Include questions about incremental vs dramatic policy change.",
        }

        guidance_lines = []
        for trait in self.member.personality_traits:
            trait_lower = trait.lower()
            if trait_lower in trait_to_guidance:
                guidance_lines.append(f"- {trait_to_guidance[trait_lower]}")

        if guidance_lines:
            return "Given your personality traits:\n" + "\n".join(guidance_lines)
        return ""
