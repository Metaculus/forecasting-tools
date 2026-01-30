from __future__ import annotations

import logging

from forecasting_tools.agents_and_tools.ai_congress.data_models import (
    CongressMember,
    PolicyProposal,
)
from forecasting_tools.agents_and_tools.minor_tools import (
    perplexity_reasoning_pro_search,
    query_asknews,
)
from forecasting_tools.ai_models.agent_wrappers import AgentRunner, AgentSdkLlm, AiAgent
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)

LONG_TIMEOUT = 480  # 8 minutes for long-running LLM calls


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

    async def deliberate(self, policy_prompt: str) -> PolicyProposal:
        logger.info(f"Deliberating on policy question: {policy_prompt[:100]}...")
        instructions = self._build_agent_instructions(policy_prompt)

        agent = AiAgent(
            name=f"Congress Member: {self.member.name}",
            instructions=instructions,
            model=AgentSdkLlm(model=self.member.ai_model),
            tools=[
                perplexity_reasoning_pro_search,
                query_asknews,
            ],
            handoffs=[],
        )

        result = await AgentRunner.run(
            agent, "Please begin your deliberation now.", max_turns=20
        )

        logger.info(f"Extracting proposal from output for {self.member.name}")
        proposal = await self._extract_proposal_from_output(result.final_output)
        proposal.member = self.member
        logger.info(f"Completed deliberation for {self.member.name}")
        return proposal

    async def _extract_proposal_from_output(self, agent_output: str) -> PolicyProposal:
        extraction_instructions = clean_indents(
            """
            Extract the policy proposal from the congress member's deliberation output.

            You must extract:
            1. research_summary: The background research section (3-5 paragraphs)
            2. decision_criteria: The list of 4-6 criteria as strings
            3. forecasts: Each forecast from the appendix as a ForecastDescription object
               - footnote_id: The number (1, 2, 3, etc.)
               - question_title: Short title
               - question_text: Full question
               - resolution_criteria: How it resolves
               - prediction: The probability (e.g., "35%" or "70% Option A, 20% Option B, 10% Option C" or "10% chance less than X units, ... ,90% chance less than Y units")
               - reasoning: The reasoning explanation
               - key_sources: List of sources mentioned
            4. proposal_markdown: The full proposal section including Executive Summary,
               Analysis, Recommendations, Risks, and any other section you see. Include footnote references [^1] etc.
            5. key_recommendations: The 3-5 main recommendations as a list of strings

            Be thorough in extracting all forecasts from the Forecast Appendix section.
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

            You must complete ALL FIVE PHASES below in order, thinking through each
            carefully. Your final output will be a comprehensive policy proposal backed
            by quantitative forecasts.

            IMPORTANT: Use your search tools extensively in Phases 1 and 4. Good policy
            analysis requires understanding the current state of affairs and gathering
            evidence for your forecasts.

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

            ## PHASE 3: Generate Forecasting Questions

            Identify 3-5 specific, concrete forecasting questions that would help inform
            this policy decision. These questions should be ones where the answer
            genuinely matters for deciding what to do.

            Good forecasting questions follow these principles:
            - The question should shed light on the topic and have high VOI (Value of Information)
            - The question should be specific and not vague
            - The question should have a resolution date
            - Once the resolution date has passed, the question should be resolvable with 0.5-1.5hr of research
                - Bad: "Will a research paper in an established journal find that a new knee surgery technique reduces follow up surgery with significance by Dec 31 2023?" (To resolve this you have to do extensive research into all new research in a field)
                - Good: "Will public dataset X at URL Y show the number of follow ups to knee surgeries decrease by Z% by Dec 31 2023?" (requires only some math on a few data points at a known URL)
            - A good resolution source exists
                - Bad: "On 15 January 2026, will the general sentiment be generally positive for knee surgery professionals with at least 10 years of experience concerning ACL reconstruction research?" (There is no way to research this online. You would have to run a large study on knee professionals)
                - Good: "As of 15 January 2026, how many 'recruiting study' search results will there be on ClinicalTrials.gov when searching 'ACL reconstruction' in 'intervention/treatment'?" (requires only a search on a known website)
            - Don't forget to INCLUDE Links if you found any! Copy the links IN FULL especially to resolution sources!
            - The questions should match any additional criteria that the superforecaster/client has given you
            - The question should not be obvious. Consider the time range when determining this (short time ranges means things are less likely).
                - Bad: "Will country X start a war in the next 2 weeks" (Probably not, especially if they have not said anything about this)
                - Good: "Will country X start a war in the next year" (Could be possible, especially if there are risk factors)
            - Cover different aspects: policy effectiveness, side effects, implementation,
              political feasibility, etc.
            - Are relevant to the policy decision at hand
            - You can find how this question resolved in the past (search for a past resolution, and consider iterating the question if you cannot find how to resolve it)


            For each question, write:
            - **Question Title**: A short descriptive title
            - **Full Question**: The complete, unambiguous question
            - **Resolution Criteria**: Exactly what would make this resolve YES vs NO,
              or how a numeric value would be measured. Be very specific.
            - **Time Horizon**: When will we know the answer?
            - **Why It Matters**: How does this question inform the policy decision?

            Make sure your questions reflect your unique perspective as {self.member.name}.
            {question_guidance}

            Write a "## Forecasting Questions" section with your 3-5 questions.

            ---

            ## PHASE 4: Forecast Each Question

            Now forecast each question you generated. This is the most important phase.

            For EACH forecasting question:
            1. Consider what principles associated with good forecasting you plan to use in this situation, if any (e.g. base rates, bias identification, premortems, simulations, scope sensitivity, aggregation, etc)
            2. Make a research plan
            3. Conduct the research (iterate as needed)
            4. Write down the main facts from the research you conducted that you will consider in your forecast
            5. Do any analysis you need to do, and then write down your rationale for the forecast
            6. Write down your forecast in accordance with the format requested of you

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            For numeric questions, you remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            Write your forecasts inline as you work through each question.

            ---

            ## PHASE 5: Write Your Policy Proposal

            Now synthesize everything into a comprehensive policy proposal. This is
            your final output.

            Structure your proposal EXACTLY as follows:

            ### Executive Summary

            A 2-3 sentence summary of your main recommendation as {self.member.name}.
            What is the single most important thing policymakers should do?

            ### Analysis

            Your detailed analysis of the policy question (3-5 paragraphs), drawing on
            your research and forecasts.

            CRITICAL: When you reference forecasts, use footnote format:
            - In the text: "This approach has a significant chance of success (65% [^1])"
            - Or: "The risk of unintended consequences is moderate (25% probability [^2])"

            The footnote number [^1], [^2], etc. corresponds to the forecast in your
            appendix below.

            ### Recommendations

            Your top 3-5 specific, actionable policy recommendations. For each:
            - State the recommendation clearly
            - Explain why you support it given your forecasts and criteria
            - Note which of your decision criteria it addresses
            - Give a detailed implementation plan for the recommendation. What would this actually look like on the ground?
            - Reference relevant forecasts with footnotes

            ### Risks and Uncertainties

            What could go wrong? What are you most uncertain about?
            - Identify the key risks of your recommendations
            - Note which forecasts have the widest uncertainty
            - Describe scenarios where your recommendations might backfire
            - Reference relevant forecasts

            ### Forecast Appendix

            At the end, provide a structured appendix with ALL your forecasts in this
            EXACT format:

            [^1] **[Question Title]**
            - Question: [Full question text]
            - Resolution: [Resolution criteria]
            - Prediction: [Your probability, e.g., "35%"]
            - Reasoning: [4+ sentences explaining your reasoning, key evidence, and
              considerations]
            - Sources: [Key sources used, can be URLs or source names]

            [^2] **[Question Title]**
            - Question: [Full question text]
            - Resolution: [Resolution criteria]
            - Prediction: [Your probability]
            - Reasoning: [4+ sentences]
            - Sources: [Sources]

            ... continue for all forecasts ...

            ---

            # Important Reminders

            - You ARE {self.member.name}. Stay in character throughout.
            - Your analysis should reflect your {self.member.political_leaning}
              perspective and your expertise in {self.member.expertise_string}.
            - Use your search tools extensively - good analysis requires evidence.
            - Every major claim in your proposal should be backed by either research
              or a forecast with a footnote.
            - Be specific and quantitative wherever possible.

            Begin your deliberation now. Start with Phase 1: Background Research.
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
        }

        guidance_lines = []
        for trait in self.member.personality_traits:
            trait_lower = trait.lower()
            if trait_lower in trait_to_guidance:
                guidance_lines.append(f"- {trait_to_guidance[trait_lower]}")

        if guidance_lines:
            return "Given your personality traits:\n" + "\n".join(guidance_lines)
        return ""
