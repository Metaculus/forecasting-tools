from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from forecasting_tools.agents_and_tools.research.computer_use import ComputerUse
from forecasting_tools.agents_and_tools.research.smart_searcher import SmartSearcher
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AgentTool,
    AiAgent,
    agent_tool,
    general_trace_or_span,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.data_models.questions import BinaryQuestion, MetaculusQuestion
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)


class ResolutionDecision(str, Enum):
    YES = "yes"
    NO = "no"
    AMBIGUOUS = "ambiguous"
    ANNULLED = "annulled"


class ResolutionReport(BaseModel):
    question_text: str
    question_id: int | None
    resolution: ResolutionDecision
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the resolution decision"
    )
    reasoning: str
    evidence_summary: str
    sources_consulted: list[str] = Field(default_factory=list)


class BinaryAutoResolver:
    """
    Auto-resolver for binary (yes/no) questions.

    Takes a BinaryQuestion as input and uses research tools to determine
    whether the question should resolve to "yes" or "no".
    """

    def __init__(
        self,
        research_model: str | GeneralLlm = "openrouter/openai/o4-mini",
        decision_model: str | GeneralLlm = "openrouter/openai/o4-mini",
        use_computer_use: bool = False,
        num_searches: int = 3,
    ) -> None:
        """
        Initialize the auto-resolver.

        Args:
            research_model: Model to use for research tasks
            decision_model: Model to use for final decision making
            use_computer_use: Whether to enable browser-based research (slower but more thorough)
            num_searches: Number of search queries to run
        """
        self.research_model = GeneralLlm.to_llm(research_model)
        self.decision_model = GeneralLlm.to_llm(decision_model)
        self.use_computer_use = use_computer_use
        self.num_searches = num_searches

    async def resolve(self, question: MetaculusQuestion) -> ResolutionReport:
        """
        Resolve a binary question by researching and determining the outcome.

        Args:
            question: The Metaculus question to resolve

        Returns:
            ResolutionReport containing the resolution decision and supporting evidence
        """
        if not isinstance(question, BinaryQuestion):
            raise TypeError(
                f"Expected BinaryQuestion, got {type(question).__name__}. "
                "This resolver currently only supports binary questions."
            )

        with general_trace_or_span(
            "BinaryAutoResolver.resolve",
            data={"question_id": question.id_of_post},
        ):
            with MonetaryCostManager() as cost_manager:
                report = await self._resolve_binary_question(question)
                logger.info(
                    f"Resolution complete. Cost: ${cost_manager.current_usage:.4f}"
                )
                return report

    async def _resolve_binary_question(
        self, question: BinaryQuestion
    ) -> ResolutionReport:
        """Internal method to resolve binary questions."""
        logger.info(f"Starting resolution for question: {question.question_text[:100]}...")

        # Step 1: Gather evidence through research
        evidence = await self._gather_evidence(question)

        # Step 2: Make resolution decision
        report = await self._make_resolution_decision(question, evidence)

        return report

    async def _gather_evidence(self, question: BinaryQuestion) -> ResearchEvidence:
        """Gather evidence about the question's resolution."""
        question_details = question.give_question_details_as_markdown()

        # Generate research queries
        research_queries = await self._generate_research_queries(question_details)

        # Run searches in parallel
        search_tasks = []

        # SmartSearcher for detailed research
        for query in research_queries[: self.num_searches]:
            search_tasks.append(self._run_smart_search(query))

        # Perplexity for quick factual lookups
        search_tasks.append(self._run_perplexity_search(question))

        # Optionally use computer use for complex research
        if self.use_computer_use:
            search_tasks.append(self._run_computer_use_research(question))

        # Gather all results
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Process results
        search_results = []
        sources = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Search failed: {result}")
            elif isinstance(result, ResearchResult):
                search_results.append(result)
                sources.extend(result.sources)

        combined_evidence = "\n\n---\n\n".join(
            [r.content for r in search_results if r.content]
        )

        return ResearchEvidence(
            raw_evidence=combined_evidence,
            sources=list(set(sources)),
            num_searches_completed=len(search_results),
        )

    async def _generate_research_queries(
        self, question_details: str
    ) -> list[str]:
        """Generate research queries for the question."""
        prompt = clean_indents(
            f"""
            You are a research assistant helping to determine if a prediction question has resolved.

            Given the following question details, generate {self.num_searches + 2} search queries
            that would help determine if this question has resolved and what the resolution is.

            Focus on:
            1. Finding official announcements or confirmations
            2. Finding news articles about the outcome
            3. Finding data sources that would show the resolution
            4. Finding any official statements from relevant parties

            Question Details:
            {question_details}

            Return the queries as a JSON list of strings. Return only the JSON list.
            Example: ["query 1", "query 2", "query 3"]
            """
        )

        queries = await self.research_model.invoke_and_return_verified_type(
            prompt, list[str]
        )
        logger.info(f"Generated {len(queries)} research queries")
        return queries

    async def _run_smart_search(self, query: str) -> ResearchResult:
        """Run a SmartSearcher query."""
        try:
            searcher = SmartSearcher(
                include_works_cited_list=True,
                use_brackets_around_citations=True,
                num_searches_to_run=2,
                num_sites_per_search=5,
                model=self.research_model,
            )
            result = await searcher.invoke(query)
            return ResearchResult(
                query=query,
                content=result,
                sources=[query],  # SmartSearcher includes sources in the result
                source_type="smart_search",
            )
        except Exception as e:
            logger.warning(f"SmartSearcher failed for query '{query}': {e}")
            return ResearchResult(
                query=query,
                content="",
                sources=[],
                source_type="smart_search",
            )

    async def _run_perplexity_search(
        self, question: BinaryQuestion
    ) -> ResearchResult:
        """Run a Perplexity search for quick facts."""
        try:
            llm = GeneralLlm(
                model="openrouter/perplexity/sonar-reasoning-pro",
                reasoning_effort="high",
                web_search_options={"search_context_size": "high"},
                populate_citations=True,
            )

            prompt = clean_indents(
                f"""
                I need to determine if the following prediction question has resolved and what the outcome is.

                Question: {question.question_text}

                Resolution Criteria:
                {question.resolution_criteria}

                Please search for the most recent and authoritative information about:
                1. Has this event/outcome occurred?
                2. What official sources confirm the outcome?
                3. What is the current status?

                Provide specific dates, sources, and evidence.
                """
            )

            result = await llm.invoke(prompt)
            return ResearchResult(
                query="perplexity_resolution_search",
                content=result,
                sources=["Perplexity AI Search"],
                source_type="perplexity",
            )
        except Exception as e:
            logger.warning(f"Perplexity search failed: {e}")
            return ResearchResult(
                query="perplexity_resolution_search",
                content="",
                sources=[],
                source_type="perplexity",
            )

    async def _run_computer_use_research(
        self, question: BinaryQuestion
    ) -> ResearchResult:
        """Use browser automation for complex research tasks."""
        try:
            computer = ComputerUse()
            prompt = clean_indents(
                f"""
                Research the following prediction question to determine if it has resolved:

                Question: {question.question_text}

                Resolution Criteria:
                {question.resolution_criteria}

                Please:
                1. Search for official sources and announcements
                2. Check relevant websites mentioned in the question
                3. Look for news articles about the outcome
                4. Document what you find with specific dates and sources

                {f"Start by checking: {question.page_url}" if question.page_url else ""}
                """
            )

            result = await computer.answer_prompt(prompt)
            return ResearchResult(
                query="computer_use_research",
                content=result.as_string,
                sources=[result.recording_url or "Browser research session"],
                source_type="computer_use",
            )
        except Exception as e:
            logger.warning(f"Computer use research failed: {e}")
            return ResearchResult(
                query="computer_use_research",
                content="",
                sources=[],
                source_type="computer_use",
            )

    async def _make_resolution_decision(
        self,
        question: BinaryQuestion,
        evidence: ResearchEvidence,
    ) -> ResolutionReport:
        """Make the final resolution decision based on gathered evidence."""
        question_details = question.give_question_details_as_markdown()

        prompt = clean_indents(
            f"""
            You are a resolution analyst for Metaculus, a prediction platform.
            Your task is to determine if a binary question should resolve to "yes" or "no".

            ## Question Details
            {question_details}

            ## Research Evidence
            The following evidence was gathered from multiple sources:

            {evidence.raw_evidence}

            ## Your Task
            Based on the resolution criteria and the evidence gathered, determine:
            1. Has the question resolved? If the outcome is unclear or hasn't happened yet, indicate "ambiguous"
            2. If resolved, should it resolve to "yes" or "no"?
            3. What is your confidence level (0.0 to 1.0)?
            4. Provide reasoning for your decision
            5. Summarize the key evidence that supports your decision

            Important:
            - Follow the resolution criteria exactly as written
            - Only resolve "yes" or "no" if there is clear evidence
            - If the evidence is inconclusive or the event hasn't occurred, use "ambiguous"
            - If the question should be annulled (e.g., the resolution criteria can never be met), use "annulled"

            Return your answer as a JSON object with the following structure:
            {{
                "resolution": "yes" or "no" or "ambiguous" or "annulled",
                "confidence": 0.0 to 1.0,
                "reasoning": "Your detailed reasoning here",
                "evidence_summary": "Summary of key evidence supporting the decision"
            }}

            Return only the JSON object.
            """
        )

        result = await self.decision_model.invoke_and_return_verified_type(
            prompt, dict
        )

        resolution_str = result["resolution"].lower()
        try:
            resolution = ResolutionDecision(resolution_str)
        except ValueError:
            logger.warning(
                f"Invalid resolution value '{resolution_str}', defaulting to AMBIGUOUS"
            )
            resolution = ResolutionDecision.AMBIGUOUS

        return ResolutionReport(
            question_text=question.question_text,
            question_id=question.id_of_post,
            resolution=resolution,
            confidence=float(result.get("confidence", 0.5)),
            reasoning=result.get("reasoning", "No reasoning provided"),
            evidence_summary=result.get("evidence_summary", "No evidence summary"),
            sources_consulted=evidence.sources,
        )


class ResearchResult(BaseModel):
    """Result from a single research source."""

    query: str
    content: str
    sources: list[str]
    source_type: Literal["smart_search", "perplexity", "computer_use", "other"]


class ResearchEvidence(BaseModel):
    """Combined evidence from all research sources."""

    raw_evidence: str
    sources: list[str]
    num_searches_completed: int


# Agent tools for use with the OpenAI Agents SDK


@agent_tool
async def resolve_binary_question(question_url_or_id: str | int) -> str:
    """
    Automatically resolve a binary Metaculus question by researching its outcome.

    Takes a Metaculus question URL or ID and returns the resolution ("yes" or "no")
    along with supporting evidence.

    Use this when you need to determine if a binary prediction question has resolved.
    """
    from forecasting_tools.helpers.metaculus_api import MetaculusApi

    if isinstance(question_url_or_id, str):
        try:
            question_url_or_id = int(question_url_or_id)
        except ValueError:
            pass

    if isinstance(question_url_or_id, int):
        question = MetaculusApi.get_question_by_post_id(question_url_or_id)
    else:
        question = MetaculusApi.get_question_by_url(question_url_or_id)

    if not isinstance(question, BinaryQuestion):
        return f"Error: Question is not a binary question. Type: {type(question).__name__}"

    resolver = BinaryAutoResolver()
    report = await resolver.resolve(question)

    return clean_indents(
        f"""
        ## Resolution Report

        **Question**: {report.question_text}
        **Resolution**: {report.resolution.value}
        **Confidence**: {report.confidence:.0%}

        ### Reasoning
        {report.reasoning}

        ### Evidence Summary
        {report.evidence_summary}

        ### Sources Consulted
        {chr(10).join(f"- {source}" for source in report.sources_consulted[:10])}
        """
    )


def create_auto_resolver_agent(
    model: str = "openrouter/openai/o4-mini",
) -> AiAgent:
    """
    Create an agent specialized for auto-resolving prediction questions.

    Returns an AiAgent configured with research tools for question resolution.
    """
    from forecasting_tools.agents_and_tools.minor_tools import (
        grab_question_details_from_metaculus,
        perplexity_reasoning_pro_search,
        smart_searcher_search,
    )
    from forecasting_tools.agents_and_tools.research.computer_use import ComputerUse

    tools: list[AgentTool] = [
        resolve_binary_question,
        grab_question_details_from_metaculus,
        perplexity_reasoning_pro_search,
        smart_searcher_search,
        ComputerUse.computer_use_tool,
    ]

    agent = AiAgent(
        name="AutoResolverAgent",
        instructions=clean_indents(
            """
            You are an expert at resolving prediction questions on Metaculus.

            Your task is to determine if binary questions have resolved and what
            the resolution should be based on the resolution criteria.

            When given a question to resolve:
            1. First grab the question details to understand the resolution criteria
            2. Use the research tools to find evidence about the outcome
            3. Use the resolve_binary_question tool to make the final determination
            4. Report your findings with confidence level and supporting evidence

            Always follow the resolution criteria exactly as written.
            Only resolve "yes" or "no" if there is clear evidence.
            If uncertain, indicate that the question cannot be resolved yet.
            """
        ),
        model=AgentSdkLlm(model=model),
        tools=tools,
    )

    return agent


async def run_auto_resolver_agent(
    question_url_or_id: str | int,
    model: str = "openrouter/openai/o4-mini",
) -> str:
    """
    Run the auto-resolver agent on a question.

    Args:
        question_url_or_id: Metaculus question URL or post ID
        model: Model to use for the agent

    Returns:
        The agent's resolution report as a string
    """
    agent = create_auto_resolver_agent(model=model)

    prompt = f"Please resolve this Metaculus question: {question_url_or_id}"

    result = await AgentRunner.run(agent, prompt)

    return result.final_output


if __name__ == "__main__":
    # Example usage
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        question_id = sys.argv[1]
    else:
        question_id = "31866"  # Example question ID

    async def main():
        result = await run_auto_resolver_agent(question_id)
        print(result)

    asyncio.run(main())
