from __future__ import annotations

import json
import logging

import pytest

from forecasting_tools.agents_and_tools.ai_congress_v2.congress_member_agent import (
    CongressMemberAgent,
)
from forecasting_tools.agents_and_tools.ai_congress_v2.data_models import (
    CongressMember,
    CongressSession,
    ForecastDescription,
    PolicyProposal,
    ProposalOption,
    Scenario,
    ScenarioCriterion,
    ScenarioDriver,
)
from forecasting_tools.agents_and_tools.ai_congress_v2.member_profiles import (
    AVAILABLE_MEMBERS,
    get_member_by_name,
    get_members_by_names,
)
from forecasting_tools.agents_and_tools.ai_congress_v2.tools import (
    roll_multiple_dice_raw,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FACTORIES
# =============================================================================


def _make_member(
    name: str = "Test Member",
    ai_model: str = "test-model",
    search_model: str = "test-search-model",
) -> CongressMember:
    return CongressMember(
        name=name,
        role="Test Role",
        political_leaning="Center",
        general_motivation="Test motivation",
        expertise_areas=["economics", "policy evaluation"],
        personality_traits=["analytical", "pragmatic"],
        ai_model=ai_model,
        search_model=search_model,
    )


def _make_driver(
    name: str = "Tech Progress",
    description: str = "Rate of technology advancement",
) -> ScenarioDriver:
    return ScenarioDriver(name=name, description=description)


def _make_criterion(
    criterion_text: str = "AI systems pass ARC evals",
    target_date: str = "2027-12-31",
) -> ScenarioCriterion:
    return ScenarioCriterion(
        criterion_text=criterion_text,
        target_date=target_date,
        resolution_criteria="Check ARC eval results",
    )


def _make_scenario(
    name: str = "Tech Boom",
    probability: str = "40%",
    is_status_quo: bool = False,
) -> Scenario:
    return Scenario(
        name=name,
        narrative="Technology advances rapidly, enabling new policy options.",
        probability=probability,
        drivers=[_make_driver()],
        criteria=[_make_criterion()],
        is_status_quo=is_status_quo,
    )


def _make_forecast(
    footnote_id: int = 1,
    title: str = "Test Forecast",
    prediction: str = "65%",
    forecast_type: str = "baseline",
    conditional_on_scenario: str | None = None,
    conditional_on_proposal: bool = False,
) -> ForecastDescription:
    return ForecastDescription(
        footnote_id=footnote_id,
        question_title=title,
        question_text="Will X happen by 2027?",
        resolution_criteria="Resolves YES if X happens",
        prediction=prediction,
        reasoning="Based on historical data and trends",
        key_sources=["source1.com", "source2.com"],
        forecast_type=forecast_type,
        conditional_on_scenario=conditional_on_scenario,
        conditional_on_proposal=conditional_on_proposal,
    )


def _make_proposal(
    member_name: str = "Member A",
    recommendations: list[str] | None = None,
    forecasts: list[ForecastDescription] | None = None,
    scenarios: list[Scenario] | None = None,
) -> PolicyProposal:
    member = _make_member(name=member_name)
    if recommendations is None:
        recommendations = [
            f"Recommendation 1 from {member_name}",
            f"Recommendation 2 from {member_name}",
        ]
    if forecasts is None:
        forecasts = [
            _make_forecast(
                footnote_id=1, title=f"{member_name} Baseline", forecast_type="baseline"
            ),
            _make_forecast(
                footnote_id=2,
                title=f"{member_name} Scenario",
                forecast_type="scenario_indicator",
            ),
            _make_forecast(
                footnote_id=3,
                title=f"{member_name} Conditional",
                forecast_type="scenario_conditional",
                conditional_on_scenario="Tech Boom",
            ),
            _make_forecast(
                footnote_id=4,
                title=f"{member_name} Proposal",
                forecast_type="proposal_conditional",
                conditional_on_proposal=True,
            ),
        ]
    if scenarios is None:
        scenarios = [
            _make_scenario("Tech Boom", "60%"),
            _make_scenario("Slow Burn", "40%"),
        ]

    return PolicyProposal(
        member=member,
        research_summary=f"Research summary from {member_name}",
        decision_criteria=["Criterion 1", "Criterion 2"],
        scenarios=scenarios,
        drivers=[_make_driver()],
        proposal_options=[
            ProposalOption(
                name="Option A",
                description="A bold approach",
                key_actions=["Action 1", "Action 2"],
            ),
        ],
        selected_proposal_name="Option A",
        forecasts=forecasts,
        proposal_markdown=f"# Proposal from {member_name}\n\nSome analysis [^1].",
        key_recommendations=recommendations,
        robustness_analysis="Works well in Tech Boom, less so in Slow Burn.",
        contingency_plans=["If Tech Boom fails, pivot to cautious approach"],
    )


# =============================================================================
# DATA MODEL TESTS
# =============================================================================


class TestCongressMember:
    def test_has_search_model_field(self) -> None:
        member = _make_member()
        assert member.search_model == "test-search-model"

    def test_default_search_model(self) -> None:
        member = CongressMember(
            name="Default",
            role="Role",
            political_leaning="Center",
            general_motivation="Motivation",
            expertise_areas=["economics"],
            personality_traits=["analytical"],
        )
        assert member.search_model == "openrouter/perplexity/sonar-reasoning-pro"


class TestScenarioModels:
    def test_scenario_has_criteria(self) -> None:
        scenario = _make_scenario()
        assert len(scenario.criteria) == 1
        assert scenario.criteria[0].target_date == "2027-12-31"

    def test_scenario_has_drivers(self) -> None:
        scenario = _make_scenario()
        assert len(scenario.drivers) == 1
        assert scenario.drivers[0].name == "Tech Progress"

    def test_status_quo_scenario(self) -> None:
        scenario = _make_scenario(is_status_quo=True)
        assert scenario.is_status_quo is True


class TestForecastDescription:
    def test_baseline_forecast_footnote(self) -> None:
        forecast = _make_forecast(forecast_type="baseline")
        footnote = forecast.as_footnote_markdown()
        assert "[^1]:" in footnote
        assert "Test Forecast" in footnote
        assert "65%" in footnote

    def test_scenario_conditional_forecast_footnote(self) -> None:
        forecast = _make_forecast(
            forecast_type="scenario_conditional",
            conditional_on_scenario="Tech Boom",
        )
        footnote = forecast.as_footnote_markdown()
        assert "Under scenario: Tech Boom" in footnote

    def test_proposal_conditional_forecast_footnote(self) -> None:
        forecast = _make_forecast(
            forecast_type="proposal_conditional",
            conditional_on_proposal=True,
        )
        footnote = forecast.as_footnote_markdown()
        assert "Conditional on proposal" in footnote

    def test_combined_conditional_forecast_footnote(self) -> None:
        forecast = _make_forecast(
            forecast_type="proposal_scenario_conditional",
            conditional_on_scenario="Trade War",
            conditional_on_proposal=True,
        )
        footnote = forecast.as_footnote_markdown()
        assert "Conditional on proposal" in footnote
        assert "Under scenario: Trade War" in footnote


class TestPolicyProposal:
    def test_baseline_forecasts_property(self) -> None:
        proposal = _make_proposal()
        baseline = proposal.baseline_forecasts
        assert len(baseline) == 1
        assert baseline[0].forecast_type == "baseline"

    def test_scenario_indicator_forecasts_property(self) -> None:
        proposal = _make_proposal()
        indicators = proposal.scenario_indicator_forecasts
        assert len(indicators) == 1
        assert indicators[0].forecast_type == "scenario_indicator"

    def test_scenario_conditional_forecasts_property(self) -> None:
        proposal = _make_proposal()
        conditional = proposal.scenario_conditional_forecasts
        assert len(conditional) == 1
        assert conditional[0].conditional_on_scenario == "Tech Boom"

    def test_proposal_conditional_forecasts_property(self) -> None:
        proposal = _make_proposal()
        prop_cond = proposal.proposal_conditional_forecasts
        assert len(prop_cond) == 1
        assert prop_cond[0].conditional_on_proposal is True

    def test_forecasts_by_scenario(self) -> None:
        proposal = _make_proposal()
        by_scenario = proposal.forecasts_by_scenario
        assert "Tech Boom" in by_scenario
        assert "General" in by_scenario

    def test_full_markdown_includes_all_appendices(self) -> None:
        proposal = _make_proposal()
        markdown = proposal.get_full_markdown_with_footnotes()
        assert "Baseline Forecast Appendix" in markdown
        assert "Scenario Indicator Forecast Appendix" in markdown
        assert "Scenario-Conditional Forecast Appendix" in markdown
        assert "Proposal-Conditional Forecast Appendix" in markdown
        assert "Cross-Scenario Robustness Analysis" in markdown

    def test_full_markdown_omits_empty_appendices(self) -> None:
        proposal = _make_proposal(forecasts=[_make_forecast(forecast_type="baseline")])
        markdown = proposal.get_full_markdown_with_footnotes()
        assert "Baseline Forecast Appendix" in markdown
        assert "Scenario Indicator Forecast Appendix" not in markdown
        assert "Scenario-Conditional Forecast Appendix" not in markdown


class TestCongressSession:
    def _make_session(self) -> CongressSession:
        from datetime import datetime, timezone

        return CongressSession(
            prompt="How should we regulate AI?",
            members_participating=[_make_member("Alice"), _make_member("Bob")],
            proposals=[_make_proposal("Alice"), _make_proposal("Bob")],
            aggregated_report_markdown="# Synthesis\n\nAgreed on X.",
            scenario_report="# Scenario Report\n\nScenario details.",
            timestamp=datetime.now(timezone.utc),
        )

    def test_get_all_forecasts(self) -> None:
        session = self._make_session()
        all_forecasts = session.get_all_forecasts()
        assert len(all_forecasts) == 8

    def test_get_all_baseline_forecasts(self) -> None:
        session = self._make_session()
        baseline = session.get_all_baseline_forecasts()
        assert len(baseline) == 2
        assert all(f.forecast_type == "baseline" for f in baseline)

    def test_get_all_scenario_indicator_forecasts(self) -> None:
        session = self._make_session()
        indicators = session.get_all_scenario_indicator_forecasts()
        assert len(indicators) == 2

    def test_get_all_scenario_conditional_forecasts(self) -> None:
        session = self._make_session()
        conditional = session.get_all_scenario_conditional_forecasts()
        assert len(conditional) == 2

    def test_get_all_proposal_conditional_forecasts(self) -> None:
        session = self._make_session()
        prop_cond = session.get_all_proposal_conditional_forecasts()
        assert len(prop_cond) == 2

    def test_get_forecasts_by_member(self) -> None:
        session = self._make_session()
        by_member = session.get_forecasts_by_member()
        assert "Alice" in by_member
        assert "Bob" in by_member
        assert len(by_member["Alice"]) == 4

    def test_get_all_scenarios_deduplicates(self) -> None:
        session = self._make_session()
        all_scenarios = session.get_all_scenarios()
        scenario_names = [s.name for s in all_scenarios]
        assert len(scenario_names) == len(set(scenario_names))

    def test_get_all_drivers_deduplicates(self) -> None:
        session = self._make_session()
        all_drivers = session.get_all_drivers()
        driver_names = [d.name for d in all_drivers]
        assert len(driver_names) == len(set(driver_names))

    def test_scenario_report_field(self) -> None:
        session = self._make_session()
        assert "Scenario Report" in session.scenario_report


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestSerialization:
    def test_scenario_round_trip(self) -> None:
        scenario = _make_scenario()
        json_data = scenario.to_json()
        restored = Scenario.from_json(json_data)
        assert restored.name == scenario.name
        assert restored.probability == scenario.probability
        assert len(restored.criteria) == len(scenario.criteria)
        assert len(restored.drivers) == len(scenario.drivers)

    def test_forecast_round_trip(self) -> None:
        forecast = _make_forecast(
            forecast_type="scenario_conditional",
            conditional_on_scenario="Tech Boom",
        )
        json_data = forecast.to_json()
        restored = ForecastDescription.from_json(json_data)
        assert restored.forecast_type == "scenario_conditional"
        assert restored.conditional_on_scenario == "Tech Boom"

    def test_proposal_round_trip(self) -> None:
        proposal = _make_proposal()
        json_data = proposal.to_json()
        restored = PolicyProposal.from_json(json_data)
        assert restored.selected_proposal_name == "Option A"
        assert len(restored.scenarios) == 2
        assert len(restored.forecasts) == 4
        assert len(restored.contingency_plans) == 1

    def test_session_round_trip(self) -> None:
        from datetime import datetime, timezone

        session = CongressSession(
            prompt="Test prompt",
            members_participating=[_make_member()],
            proposals=[_make_proposal()],
            aggregated_report_markdown="Report",
            scenario_report="Scenario Report",
            timestamp=datetime.now(timezone.utc),
        )
        json_data = session.to_json()
        restored = CongressSession.from_json(json_data)
        assert restored.prompt == "Test prompt"
        assert restored.scenario_report == "Scenario Report"
        assert len(restored.proposals) == 1


# =============================================================================
# MEMBER PROFILES TESTS
# =============================================================================


class TestMemberProfiles:
    def test_all_members_have_search_model(self) -> None:
        for member in AVAILABLE_MEMBERS:
            assert member.search_model is not None
            assert len(member.search_model) > 0

    def test_get_member_by_name_valid(self) -> None:
        member = get_member_by_name("Opus 4.5 (Anthropic)")
        assert member.name == "Opus 4.5 (Anthropic)"

    def test_get_member_by_name_invalid(self) -> None:
        with pytest.raises(ValueError):
            get_member_by_name("Nonexistent Member")

    def test_get_members_by_names(self) -> None:
        members = get_members_by_names(["Opus 4.5 (Anthropic)", "GPT 5.2 (OpenAI)"])
        assert len(members) == 2


# =============================================================================
# TOOLS TESTS
# =============================================================================


class TestBatchDiceRoll:
    def test_valid_json_input(self) -> None:
        forecasts = [
            {"id": "[^1]", "title": "GDP growth", "probability": 0.5},
            {"id": "[^2]", "title": "Rate cut", "probability": 0.8},
        ]
        result = roll_multiple_dice_raw(json.dumps(forecasts))
        assert "| ID |" in result
        assert "[^1]" in result
        assert "[^2]" in result
        assert "GDP growth" in result
        assert "Rate cut" in result

    def test_invalid_json_returns_error(self) -> None:
        result = roll_multiple_dice_raw("not valid json")
        assert "Error parsing JSON" in result

    def test_non_array_returns_error(self) -> None:
        result = roll_multiple_dice_raw('{"id": 1}')
        assert "Error" in result

    def test_out_of_range_probability(self) -> None:
        forecasts = [{"id": "1", "title": "Bad", "probability": 1.5}]
        result = roll_multiple_dice_raw(json.dumps(forecasts))
        assert "ERROR" in result

    def test_all_outcomes_are_occurred_or_not(self) -> None:
        forecasts = [
            {"id": str(i), "title": f"Q{i}", "probability": 0.5} for i in range(10)
        ]
        result = roll_multiple_dice_raw(json.dumps(forecasts))
        for line in result.split("\n")[2:]:
            if line.strip():
                assert "OCCURRED" in line or "DID NOT OCCUR" in line


# =============================================================================
# AGENT INSTRUCTION TESTS
# =============================================================================


class TestAgentInstructions:
    def test_instructions_include_all_phases(self) -> None:
        member = _make_member()
        agent = CongressMemberAgent(member)
        instructions = agent._build_agent_instructions("Should we regulate AI?")
        for phase_num in range(1, 16):
            assert f"PHASE {phase_num}:" in instructions

    def test_instructions_include_member_identity(self) -> None:
        member = _make_member(name="TestSenator")
        agent = CongressMemberAgent(member)
        instructions = agent._build_agent_instructions("Test prompt")
        assert "TestSenator" in instructions
        assert "Test Role" in instructions
        assert "Center" in instructions

    def test_instructions_include_scenario_mece_guidance(self) -> None:
        member = _make_member()
        agent = CongressMemberAgent(member)
        instructions = agent._build_agent_instructions("Test prompt")
        assert "MUTUALLY EXCLUSIVE" in instructions
        assert "COLLECTIVELY EXHAUSTIVE" in instructions
        assert "MECE" in instructions

    def test_instructions_include_contingency_plan_guidance(self) -> None:
        member = _make_member()
        agent = CongressMemberAgent(member)
        instructions = agent._build_agent_instructions("Test prompt")
        assert "Contingency Plans" in instructions
        assert "If X happens" in instructions

    def test_instructions_include_robustness_analysis(self) -> None:
        member = _make_member()
        agent = CongressMemberAgent(member)
        instructions = agent._build_agent_instructions("Test prompt")
        assert "Cross-Scenario Robustness" in instructions
        assert "no-regret" in instructions

    def test_instructions_request_20_plus_forecasts(self) -> None:
        member = _make_member()
        agent = CongressMemberAgent(member)
        instructions = agent._build_agent_instructions("Test prompt")
        assert "20+" in instructions

    def test_delphi_continuation_message_includes_other_reports(self) -> None:
        member = _make_member()
        agent = CongressMemberAgent(member)
        message = agent._build_delphi_continuation_message(
            "Test prompt",
            "My final report content",
            [("Alice", "Alice's report"), ("Bob", "Bob's report")],
            delphi_round=2,
        )
        assert "Alice" in message
        assert "Bob" in message
        assert "Alice's report" in message
        assert "Bob's report" in message
        assert "My final report content" in message
        assert "Delphi Round 2" in message

    def test_expertise_guidance_maps_to_known_areas(self) -> None:
        member = _make_member()
        member.expertise_areas = ["economics", "climate science"]
        agent = CongressMemberAgent(member)
        guidance = agent._get_expertise_specific_research_guidance()
        assert "Economic data" in guidance
        assert "Climate projections" in guidance

    def test_question_guidance_maps_to_known_traits(self) -> None:
        member = _make_member()
        member.personality_traits = ["data-driven", "pragmatic"]
        agent = CongressMemberAgent(member)
        guidance = agent._get_question_generation_guidance()
        assert "quantitative" in guidance
        assert "implementation" in guidance
