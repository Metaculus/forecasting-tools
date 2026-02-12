from __future__ import annotations

import logging

from forecasting_tools.agents_and_tools.ai_congress.congress_member_agent import (
    CongressMemberAgent,
)
from forecasting_tools.agents_and_tools.ai_congress.data_models import (
    CongressMember,
    ForecastDescription,
    PolicyProposal,
)

logger = logging.getLogger(__name__)


def _make_member(
    name: str = "Test Member", ai_model: str = "test-model"
) -> CongressMember:
    return CongressMember(
        name=name,
        role="Test Role",
        political_leaning="Center",
        general_motivation="Test motivation",
        expertise_areas=["economics", "policy evaluation"],
        personality_traits=["analytical", "pragmatic"],
        ai_model=ai_model,
    )


def _make_forecast(
    footnote_id: int = 1,
    title: str = "Test Forecast",
    prediction: str = "65%",
    is_conditional: bool = False,
) -> ForecastDescription:
    return ForecastDescription(
        footnote_id=footnote_id,
        question_title=title,
        question_text="Will X happen by 2027?",
        resolution_criteria="Resolves YES if X happens",
        prediction=prediction,
        reasoning="Based on historical data and trends",
        key_sources=["source1.com", "source2.com"],
        is_conditional=is_conditional,
    )


def _make_proposal(
    member_name: str = "Member A",
    recommendations: list[str] | None = None,
    forecasts: list[ForecastDescription] | None = None,
    delphi_round: int = 1,
) -> PolicyProposal:
    member = _make_member(name=member_name)
    if recommendations is None:
        recommendations = [
            f"Recommendation 1 from {member_name}",
            f"Recommendation 2 from {member_name}",
        ]
    if forecasts is None:
        forecasts = [_make_forecast(footnote_id=1, title=f"{member_name} Forecast")]
    return PolicyProposal(
        member=member,
        research_summary=f"Research summary from {member_name}",
        decision_criteria=["Criterion 1", "Criterion 2"],
        forecasts=forecasts,
        proposal_markdown=f"# Proposal from {member_name}\n\nSome analysis text.",
        key_recommendations=recommendations,
        delphi_round=delphi_round,
    )


class TestFormatOtherProposalsForReview:
    def test_formats_multiple_proposals(self) -> None:
        proposals = [
            _make_proposal("Alice"),
            _make_proposal("Bob"),
        ]
        result = CongressMemberAgent._format_other_proposals_for_review(proposals)
        assert "Alice" in result
        assert "Bob" in result
        assert "Recommendation 1 from Alice" in result
        assert "Recommendation 1 from Bob" in result

    def test_includes_forecast_titles_and_predictions(self) -> None:
        forecasts = [
            _make_forecast(footnote_id=1, title="GDP Growth", prediction="40%"),
            _make_forecast(footnote_id=2, title="Unemployment", prediction="25%"),
        ]
        proposals = [_make_proposal("Carol", forecasts=forecasts)]
        result = CongressMemberAgent._format_other_proposals_for_review(proposals)
        assert "GDP Growth" in result
        assert "40%" in result
        assert "Unemployment" in result
        assert "25%" in result

    def test_skips_proposals_without_member(self) -> None:
        proposal_without_member = PolicyProposal(
            member=None,
            research_summary="No member",
            decision_criteria=["C1"],
            forecasts=[],
            proposal_markdown="Text",
            key_recommendations=["Rec"],
        )
        result = CongressMemberAgent._format_other_proposals_for_review(
            [proposal_without_member]
        )
        assert result == ""

    def test_separates_proposals_with_dividers(self) -> None:
        proposals = [_make_proposal("Alice"), _make_proposal("Bob")]
        result = CongressMemberAgent._format_other_proposals_for_review(proposals)
        assert "---" in result


class TestFormatOwnProposalForReview:
    def test_formats_own_recommendations_and_forecasts(self) -> None:
        proposal = _make_proposal(
            "Self",
            recommendations=["My Rec 1", "My Rec 2"],
            forecasts=[_make_forecast(title="My Forecast", prediction="70%")],
        )
        result = CongressMemberAgent._format_own_proposal_for_review(proposal)
        assert "My Rec 1" in result
        assert "My Rec 2" in result
        assert "My Forecast" in result
        assert "70%" in result


class TestBuildRevisionInstructions:
    def test_includes_member_identity(self) -> None:
        member = _make_member("TestAgent")
        agent = CongressMemberAgent(member)
        own_proposal = _make_proposal("TestAgent")
        other_proposals = [_make_proposal("Other1")]
        instructions = agent._build_revision_instructions(
            "Should we regulate AI?", own_proposal, other_proposals, delphi_round=2
        )
        assert "TestAgent" in instructions
        assert "Test Role" in instructions

    def test_includes_policy_prompt(self) -> None:
        member = _make_member("TestAgent")
        agent = CongressMemberAgent(member)
        own_proposal = _make_proposal("TestAgent")
        other_proposals = [_make_proposal("Other1")]
        policy_prompt = "How should we address climate change?"
        instructions = agent._build_revision_instructions(
            policy_prompt, own_proposal, other_proposals, delphi_round=2
        )
        assert policy_prompt in instructions

    def test_includes_delphi_round_number(self) -> None:
        member = _make_member("TestAgent")
        agent = CongressMemberAgent(member)
        own_proposal = _make_proposal("TestAgent")
        other_proposals = [_make_proposal("Other1")]
        instructions = agent._build_revision_instructions(
            "Test prompt", own_proposal, other_proposals, delphi_round=3
        )
        assert "Delphi Round 3" in instructions

    def test_includes_other_members_proposals(self) -> None:
        member = _make_member("TestAgent")
        agent = CongressMemberAgent(member)
        own_proposal = _make_proposal("TestAgent")
        other_proposals = [
            _make_proposal("Rival1", recommendations=["Impose tariffs"]),
            _make_proposal("Rival2", recommendations=["Remove tariffs"]),
        ]
        instructions = agent._build_revision_instructions(
            "Trade policy", own_proposal, other_proposals, delphi_round=2
        )
        assert "Rival1" in instructions
        assert "Rival2" in instructions
        assert "Impose tariffs" in instructions
        assert "Remove tariffs" in instructions

    def test_includes_own_proposal_content(self) -> None:
        member = _make_member("TestAgent")
        agent = CongressMemberAgent(member)
        own_proposal = _make_proposal(
            "TestAgent", recommendations=["My special recommendation"]
        )
        other_proposals = [_make_proposal("Other")]
        instructions = agent._build_revision_instructions(
            "Test prompt", own_proposal, other_proposals, delphi_round=2
        )
        assert "My special recommendation" in instructions


class TestOrchestratorProposalFiltering:
    def test_other_proposals_exclude_own_member(self) -> None:
        proposals = [
            _make_proposal("Alice"),
            _make_proposal("Bob"),
            _make_proposal("Carol"),
        ]
        alice_others = [p for p in proposals if p.member and p.member.name != "Alice"]
        assert len(alice_others) == 2
        assert all(p.member.name != "Alice" for p in alice_others)
        assert {p.member.name for p in alice_others} == {"Bob", "Carol"}

    def test_own_proposal_found_by_member_name(self) -> None:
        proposals = [
            _make_proposal("Alice"),
            _make_proposal("Bob"),
            _make_proposal("Carol"),
        ]
        bob_own = next(
            (p for p in proposals if p.member and p.member.name == "Bob"),
            None,
        )
        assert bob_own is not None
        assert bob_own.member.name == "Bob"

    def test_own_proposal_none_when_member_missing(self) -> None:
        proposals = [
            _make_proposal("Alice"),
            _make_proposal("Carol"),
        ]
        bob_own = next(
            (p for p in proposals if p.member and p.member.name == "Bob"),
            None,
        )
        assert bob_own is None
