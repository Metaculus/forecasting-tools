"""
Agentic question resolver using OpenAI Agents.

This module implements a multi-agent architecture for resolving Metaculus
forecasting questions. It uses a pluggable researcher pattern with a default
Perplexity-based implementation.

Architecture:
    0. Question rephraser (standalone LLM call) converts past-deadline
       questions from future tense to past tense for better research
    1. Orchestrator (minimal) coordinates handoffs between agents
    2. Researcher agent performs multiple strategic searches
    3. Resolver agent analyzes research and determines resolution
    4. Structured output parsing converts to typed resolution
"""

import logging
from typing import AsyncGenerator, Optional, Callable

from openai.types.responses import ResponseTextDeltaEvent

from forecasting_tools.data_models.questions import (
    ResolutionType,
    CanceledResolution,
    BinaryResolution,
)
from forecasting_tools import MetaculusQuestion, BinaryQuestion
from forecasting_tools.agents_and_tools.auto_resolver.agentic.instructions import *
from forecasting_tools.agents_and_tools.auto_resolver.resolution_models import BinaryResolutionResult
from forecasting_tools.agents_and_tools.auto_resolver import AutoResolver
from forecasting_tools.agents_and_tools.minor_tools import (
    create_date_filtered_asknews_tool,
    perplexity_reasoning_pro_search,
)
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AiAgent,
    event_to_tool_message,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)




class StructuredOutputParsingError(Exception):
    """Raised when structured output parsing fails.

    This exception preserves the raw unparsed output for debugging purposes.

    Attributes:
        raw_output: The unparsed output that failed parsing
        original_error: The original exception that caused the failure
    """

    def __init__(self, raw_output: str, original_error: Exception):
        self.raw_output = raw_output
        self.original_error = original_error
        super().__init__(
            f"Failed to parse structured output: {original_error}\n"
            f"Raw output available in exception.raw_output"
        )


class AgenticResolver(AutoResolver):
    """
    Agentic resolver using agents SDK.
    """

    def __init__(self,
        model_for_supervisor: str = "openrouter/anthropic/claude-sonnet-4.6",
        model_for_resolver: str = "openrouter/anthropic/claude-sonnet-4.6",
        model_for_output_structure: str = "openrouter/anthropic/claude-sonnet-4.6",
        model_for_researcher: str = "openrouter/anthropic/claude-sonnet-4.6",
        model_for_rephraser: str = "openrouter/anthropic/claude-sonnet-4.6",
        timeout: int = 480
    ):
       self.model_for_supervisor = model_for_supervisor
       self.model_for_resolver = model_for_resolver
       self.model_for_output_structure = model_for_output_structure
       self.model_for_researcher = model_for_researcher
       self.model_for_rephraser = model_for_rephraser
       self.timeout = timeout
       
           
    async def resolve_question(
        self, question: MetaculusQuestion
    ) -> Optional[ResolutionType]:
        if isinstance(question, BinaryQuestion):
            return await self._resolve_binary(question)
        else:
            return NotImplemented

    async def _resolve_binary(
        self, question: BinaryQuestion
    ) -> Optional[BinaryResolution]:

        # Rephrase question if its time context has passed
        question = await self._rephrase_question_if_needed(question)

        # Create agents
        researcher = self._create_researcher(question)
        annulled_ambiguous_resolver = self._create_annulled_ambiguous_resolver_agent(question)
        resolver = self._create_resolver_agent(question, annulled_ambiguous_resolver)
        orchestrator = self._create_orchestrator_agent(researcher, resolver)

        # Run the workflow (non-streaming)
        result = await AgentRunner.run(
            orchestrator, "Please begin the resolution process.", max_turns=10
        )

        # Parse structured output with error handling
        try:
            resolution_result = await structure_output(
                result.final_output,
                BinaryResolutionResult,
                model=self.model_for_output_structure,
            )
            logger.info(
                f"Successfully parsed resolution: {resolution_result.resolution_status}"
            )
        except Exception as e:
            logger.error(f"Failed to parse structured output: {e}", exc_info=True)
            raise StructuredOutputParsingError(
                raw_output=result.final_output, original_error=e
            ) from e

        # Store metadata for later retrieval
        self._last_resolution_metadata = {
            "reasoning": resolution_result.reasoning,
            "key_evidence": resolution_result.key_evidence,
        }

        # Convert to typed resolution
        typed_resolution = resolution_result.convert_to_binary_resolution()
        logger.info(f"Final resolution: {typed_resolution}")

        return typed_resolution

    async def _rephrase_question_if_needed(
        self, question: BinaryQuestion
    ) -> BinaryQuestion:
        """Rephrase the question into past tense if its time context has passed.

        Uses a lightweight LLM call to determine whether the question's deadline
        has already passed and, if so, rephrases it from future tense to past
        tense. This makes downstream research searches more effective.

        Args:
            question: The original question to potentially rephrase.

        Returns:
            A copy of the question with question_text updated if rephrasing
            was needed, or the original question unchanged.
        """
        prompt = question_rephraser_instructions(question)
        llm = GeneralLlm(model=self.model_for_rephraser, temperature=0.0)

        try:
            rephrased_text = await llm.invoke(prompt)
            rephrased_text = rephrased_text.strip().strip('"').strip("'")

            if rephrased_text and rephrased_text != question.question_text:
                logger.info(
                    f"Question rephrased:\n"
                    f"  Original:  {question.question_text}\n"
                    f"  Rephrased: {rephrased_text}"
                )
                question = question.model_copy(deep=True)
                question.question_text = rephrased_text
            else:
                logger.info("Question rephraser: no rephrasing needed")
        except Exception as e:
            logger.warning(
                f"Question rephrasing failed, proceeding with original: {e}",
                exc_info=True,
            )

        return question

    def _create_researcher(self, question: BinaryQuestion) -> AiAgent:
        instructions = researcher_instructions(question)
        asknews_tool = create_date_filtered_asknews_tool(question.close_time)
        return AiAgent(
            name="Resolution Researcher",
            instructions=instructions,
            model=AgentSdkLlm(model=self.model_for_researcher),
            tools=[perplexity_reasoning_pro_search, asknews_tool],
            handoffs=[],
        )

    def _create_resolver_agent(self, question: BinaryQuestion, annulled_ambiguous_resolver: AiAgent) -> AiAgent:
        instructions = binary_resolver_instructions(question)
        return AiAgent(
            name="resolver",
            instructions=instructions,
            model=AgentSdkLlm(model=self.model_for_resolver),
            tools=[],  # No tools - only analyzes research
            handoffs=[annulled_ambiguous_resolver],  # Can hand off to specialized agent
        )

    def _create_annulled_ambiguous_resolver_agent(self, question: BinaryQuestion) -> AiAgent:
        instructions = annulled_ambiguous_resolver_instructions(question)
        return AiAgent(
            name="annulled_ambiguous_resolver",
            instructions=instructions,
            model=AgentSdkLlm(model=self.model_for_resolver),
            tools=[],  # No tools - only analyzes research
            handoffs=[],  # Terminal agent
        )

    def _create_orchestrator_agent(
        self, researcher: AiAgent, resolver: AiAgent
    ) -> AiAgent:
        logger.debug("Creating minimal orchestrator agent")

        instructions = clean_indents(
            """
            You are coordinating a question resolution process.

            Your task is simple:
            1. Hand off to the Resolution Researcher to gather information
            2. The researcher will hand off to the resolver when ready
            3. The resolver will provide the final resolution

            Begin by handing off to the researcher.
            """
        )

        return AiAgent(
            name="Resolution Orchestrator",
            instructions=instructions,
            model=AgentSdkLlm(model=self.model_for_output_structure),
            tools=[],
            handoffs=[researcher, resolver],
        )

    async def resolve_question_streamed(
        self, question: MetaculusQuestion
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Resolve a question with streaming, yielding (event_type, message) tuples.

        This is the streaming counterpart to resolve_question(). It yields
        intermediate events as they occur during the agent resolution process,
        allowing a TUI or other consumer to display live progress.

        Event types:
            "status"  -- lifecycle status updates (rephrasing, agent creation, etc.)
            "text"    -- streamed text delta from the agent's response
            "tool"    -- tool call / handoff / reasoning events
            "result"  -- final resolution result (yielded once at the end)
            "error"   -- an error occurred during resolution

        After all events are yielded, the resolver's internal metadata is updated
        (same as resolve_question), so get_last_resolution_metadata() will work.

        Args:
            question: The Metaculus question to resolve.

        Yields:
            Tuples of (event_type, message_text).
        """
        if not isinstance(question, BinaryQuestion):
            yield ("error", f"Unsupported question type: {type(question).__name__}")
            return

        # Step 1: Rephrase if needed
        yield ("status", "Checking if question needs rephrasing...")
        question = await self._rephrase_question_if_needed(question)
        yield ("status", f"Question text: {question.question_text}")

        # Step 2: Create agents
        yield ("status", "Creating resolution agents...")
        researcher = self._create_researcher(question)
        annulled_ambiguous_resolver = self._create_annulled_ambiguous_resolver_agent(question)
        resolver = self._create_resolver_agent(question, annulled_ambiguous_resolver)
        orchestrator = self._create_orchestrator_agent(researcher, resolver)

        # Step 3: Run streamed workflow
        yield ("status", "Starting resolution process...")
        streamed_text = ""

        result = AgentRunner.run_streamed(
            orchestrator, "Please begin the resolution process.", max_turns=10
        )

        async for event in result.stream_events():
            # Capture text deltas
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                streamed_text += event.data.delta
                yield ("text", event.data.delta)

            # Capture tool/handoff/reasoning events
            tool_msg = event_to_tool_message(event)
            if tool_msg:
                yield ("tool", tool_msg)

        # Step 4: Parse structured output
        yield ("status", "Parsing resolution output...")
        final_output = result.final_output

        try:
            resolution_result = await structure_output(
                final_output,
                BinaryResolutionResult,
                model=self.model_for_output_structure,
            )
            logger.info(
                f"Successfully parsed resolution: {resolution_result.resolution_status}"
            )
        except Exception as e:
            logger.error(f"Failed to parse structured output: {e}", exc_info=True)
            yield ("error", f"Failed to parse structured output: {e}")
            self._last_resolution_metadata = None
            return

        # Step 5: Store metadata
        self._last_resolution_metadata = {
            "reasoning": resolution_result.reasoning,
            "key_evidence": resolution_result.key_evidence,
        }

        typed_resolution = resolution_result.convert_to_binary_resolution()
        logger.info(f"Final resolution: {typed_resolution}")

        yield (
            "result",
            f"Resolution: {resolution_result.resolution_status}\n"
            f"Reasoning: {resolution_result.reasoning}\n"
            f"Key Evidence:\n"
            + "\n".join(f"  - {e}" for e in resolution_result.key_evidence),
        )
