import logging
from datetime import datetime

import pytest

from forecasting_tools.agents_and_tools.auto_resolver import (
    BinaryAutoResolver,
    ResearchEvidence,
    ResearchResult,
    ResolutionDecision,
    ResolutionReport,
)
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    NumericQuestion,
)

logger = logging.getLogger(__name__)


class TestResolutionDecision:
    def test_resolution_decision_values(self) -> None:
        assert ResolutionDecision.YES.value == "yes"
        assert ResolutionDecision.NO.value == "no"
        assert ResolutionDecision.AMBIGUOUS.value == "ambiguous"
        assert ResolutionDecision.ANNULLED.value == "annulled"

    def test_resolution_decision_from_string(self) -> None:
        assert ResolutionDecision("yes") == ResolutionDecision.YES
        assert ResolutionDecision("no") == ResolutionDecision.NO
        assert ResolutionDecision("ambiguous") == ResolutionDecision.AMBIGUOUS
        assert ResolutionDecision("annulled") == ResolutionDecision.ANNULLED


class TestResolutionReport:
    def test_resolution_report_creation(self) -> None:
        report = ResolutionReport(
            question_text="Will X happen by Y date?",
            question_id=12345,
            resolution=ResolutionDecision.YES,
            confidence=0.85,
            reasoning="Based on evidence, X has happened.",
            evidence_summary="Event X occurred on date Z.",
            sources_consulted=["source1", "source2"],
        )

        assert report.question_text == "Will X happen by Y date?"
        assert report.question_id == 12345
        assert report.resolution == ResolutionDecision.YES
        assert report.confidence == 0.85
        assert len(report.sources_consulted) == 2

    def test_resolution_report_confidence_validation(self) -> None:
        # Valid confidence
        report = ResolutionReport(
            question_text="Test",
            question_id=None,
            resolution=ResolutionDecision.NO,
            confidence=0.5,
            reasoning="Test reasoning",
            evidence_summary="Test evidence",
        )
        assert report.confidence == 0.5

        # Invalid confidence (too high)
        with pytest.raises(ValueError):
            ResolutionReport(
                question_text="Test",
                question_id=None,
                resolution=ResolutionDecision.NO,
                confidence=1.5,
                reasoning="Test reasoning",
                evidence_summary="Test evidence",
            )

        # Invalid confidence (negative)
        with pytest.raises(ValueError):
            ResolutionReport(
                question_text="Test",
                question_id=None,
                resolution=ResolutionDecision.NO,
                confidence=-0.1,
                reasoning="Test reasoning",
                evidence_summary="Test evidence",
            )


class TestResearchResult:
    def test_research_result_creation(self) -> None:
        result = ResearchResult(
            query="test query",
            content="test content",
            sources=["source1", "source2"],
            source_type="smart_search",
        )

        assert result.query == "test query"
        assert result.content == "test content"
        assert len(result.sources) == 2
        assert result.source_type == "smart_search"

    def test_research_result_source_types(self) -> None:
        valid_types = ["smart_search", "perplexity", "computer_use", "other"]
        for source_type in valid_types:
            result = ResearchResult(
                query="test",
                content="test",
                sources=[],
                source_type=source_type,
            )
            assert result.source_type == source_type


class TestResearchEvidence:
    def test_research_evidence_creation(self) -> None:
        evidence = ResearchEvidence(
            raw_evidence="Combined evidence from multiple sources",
            sources=["source1", "source2", "source3"],
            num_searches_completed=3,
        )

        assert "Combined evidence" in evidence.raw_evidence
        assert len(evidence.sources) == 3
        assert evidence.num_searches_completed == 3


class TestBinaryAutoResolver:
    def test_resolver_initialization_default(self) -> None:
        resolver = BinaryAutoResolver()

        assert resolver.use_computer_use is False
        assert resolver.num_searches == 3

    def test_resolver_initialization_custom(self) -> None:
        resolver = BinaryAutoResolver(
            research_model="gpt-4o",
            decision_model="gpt-4o",
            use_computer_use=True,
            num_searches=5,
        )

        assert resolver.use_computer_use is True
        assert resolver.num_searches == 5

    def test_resolver_rejects_non_binary_questions(self) -> None:
        resolver = BinaryAutoResolver()

        # Create a non-binary question (NumericQuestion)
        numeric_question = NumericQuestion(
            question_text="What will be the temperature?",
            upper_bound=100.0,
            lower_bound=0.0,
            open_upper_bound=True,
            open_lower_bound=True,
        )

        with pytest.raises(TypeError) as exc_info:
            import asyncio

            asyncio.run(resolver.resolve(numeric_question))

        assert "Expected BinaryQuestion" in str(exc_info.value)
        assert "NumericQuestion" in str(exc_info.value)

    def test_create_binary_question_for_testing(self) -> None:
        """Test helper to verify we can create valid BinaryQuestion objects."""
        question = BinaryQuestion(
            question_text="Will SpaceX launch Starship in 2024?",
            id_of_post=12345,
            resolution_criteria="Resolves YES if SpaceX successfully launches Starship.",
            fine_print="Must be a full stack launch.",
            background_info="SpaceX has been developing Starship for years.",
        )

        assert question.question_type == "binary"
        assert question.id_of_post == 12345
        assert "SpaceX" in question.question_text


class TestAutoResolverTools:
    def test_resolve_binary_question_tool_exists(self) -> None:
        from forecasting_tools.agents_and_tools.auto_resolver import (
            resolve_binary_question,
        )

        # Verify the tool is callable
        assert callable(resolve_binary_question)

    def test_create_auto_resolver_agent_function_exists(self) -> None:
        from forecasting_tools.agents_and_tools.auto_resolver import (
            create_auto_resolver_agent,
        )

        assert callable(create_auto_resolver_agent)

    def test_run_auto_resolver_agent_function_exists(self) -> None:
        from forecasting_tools.agents_and_tools.auto_resolver import (
            run_auto_resolver_agent,
        )

        assert callable(run_auto_resolver_agent)
