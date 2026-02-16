from __future__ import annotations

import pytest

from forecasting_tools.agents_and_tools.decision_hub.data_models import (
    AnalysisSession,
    ForecastQuestion,
    PolicyProposal,
    RedTeamResult,
    ResearchReport,
    Scenario,
    ScenarioDriver,
    ScenarioSet,
)
from forecasting_tools.agents_and_tools.decision_hub.database import SQLiteBackend

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test_decision_hub.db")
    return SQLiteBackend(db_path=db_path)


@pytest.fixture
def sample_session() -> AnalysisSession:
    return AnalysisSession(
        username="test_user",
        policy_question="Should the EU impose tariffs on Chinese EVs?",
    )


@pytest.fixture
def sample_research(sample_session) -> ResearchReport:
    return ResearchReport(
        session_id=sample_session.session_id,
        username="test_user",
        query="EU EV tariffs economic impact",
        report_markdown="# Research Report\n\nThis is test research.",
        sources=["https://example.com/1", "https://example.com/2"],
    )


@pytest.fixture
def sample_scenario_set(sample_session) -> ScenarioSet:
    return ScenarioSet(
        session_id=sample_session.session_id,
        username="test_user",
        drivers=[
            ScenarioDriver(
                name="EV Adoption Speed",
                description="How fast EVs are adopted globally",
                high_state="Rapid adoption (>50% by 2030)",
                low_state="Slow adoption (<25% by 2030)",
            ),
            ScenarioDriver(
                name="China Trade Relations",
                description="State of EU-China trade relations",
                high_state="Cooperative, reduced tensions",
                low_state="Retaliatory, trade war",
            ),
        ],
        scenarios=[
            Scenario(
                name="Green Boom",
                narrative="Fast EV adoption with cooperative trade relations",
                driver_states={
                    "EV Adoption Speed": "high",
                    "China Trade Relations": "high",
                },
                probability=0.25,
            ),
            Scenario(
                name="Trade War",
                narrative="Fast EV adoption but China retaliates",
                driver_states={
                    "EV Adoption Speed": "high",
                    "China Trade Relations": "low",
                },
                probability=0.30,
            ),
        ],
        rationale="These are the two most impactful external uncertainties.",
    )


@pytest.fixture
def sample_forecast_question(sample_session) -> ForecastQuestion:
    return ForecastQuestion(
        session_id=sample_session.session_id,
        username="test_user",
        question_text="Will EU auto market share exceed 20% by 2030?",
        resolution_criteria="Based on official EU market data",
        question_type="baseline",
    )


# =============================================================================
# DATA MODEL TESTS
# =============================================================================


class TestDataModelSerialization:
    def test_analysis_session_roundtrip(self, sample_session: AnalysisSession):
        json_data = sample_session.to_json()
        restored = AnalysisSession.from_json(json_data)
        assert restored.session_id == sample_session.session_id
        assert restored.username == sample_session.username
        assert restored.policy_question == sample_session.policy_question
        assert restored.status == "in_progress"

    def test_research_report_roundtrip(self, sample_research: ResearchReport):
        json_data = sample_research.to_json()
        restored = ResearchReport.from_json(json_data)
        assert restored.query == sample_research.query
        assert restored.report_markdown == sample_research.report_markdown
        assert len(restored.sources) == 2

    def test_scenario_set_roundtrip(self, sample_scenario_set: ScenarioSet):
        json_data = sample_scenario_set.to_json()
        restored = ScenarioSet.from_json(json_data)
        assert len(restored.drivers) == 2
        assert len(restored.scenarios) == 2
        assert restored.scenarios[0].name == "Green Boom"
        assert abs(restored.scenarios[0].probability - 0.25) < 1e-9
        assert restored.scenarios[0].driver_states["EV Adoption Speed"] == "high"

    def test_forecast_question_roundtrip(
        self, sample_forecast_question: ForecastQuestion
    ):
        json_data = sample_forecast_question.to_json()
        restored = ForecastQuestion.from_json(json_data)
        assert restored.question_text == sample_forecast_question.question_text
        assert restored.question_type == "baseline"
        assert restored.conditional_on_scenario is None

    def test_forecast_with_scenario_conditional(self, sample_session):
        question = ForecastQuestion(
            session_id=sample_session.session_id,
            username="test_user",
            question_text="Under Green Boom, will EU share > 20%?",
            resolution_criteria="Based on EU data",
            question_type="scenario_conditional",
            conditional_on_scenario="Green Boom",
        )
        json_data = question.to_json()
        restored = ForecastQuestion.from_json(json_data)
        assert restored.question_type == "scenario_conditional"
        assert restored.conditional_on_scenario == "Green Boom"

    def test_policy_proposal_roundtrip(self, sample_session):
        proposal = PolicyProposal(
            session_id=sample_session.session_id,
            username="test_user",
            proposal_markdown="# Proposal\n\nImplement graduated tariffs.",
            key_recommendations=["Phase in tariffs over 3 years", "Invest in R&D"],
            contingency_plans=["If trade war, reduce tariffs"],
        )
        json_data = proposal.to_json()
        restored = PolicyProposal.from_json(json_data)
        assert len(restored.key_recommendations) == 2
        assert len(restored.contingency_plans) == 1

    def test_red_team_result_without_session(self):
        result = RedTeamResult(
            username="test_user",
            input_text="Some proposal text",
            mode="devils_advocate",
            critique_markdown="## Critique\n\nThis is flawed because...",
        )
        json_data = result.to_json()
        restored = RedTeamResult.from_json(json_data)
        assert restored.session_id is None
        assert restored.mode == "devils_advocate"


# =============================================================================
# DATABASE TESTS
# =============================================================================


class TestSQLiteBackend:
    def test_create_and_get_user(self, db: SQLiteBackend):
        db.create_user("alice")
        user = db.get_user("alice")
        assert user is not None
        assert user["username"] == "alice"

    def test_get_nonexistent_user_returns_none(self, db: SQLiteBackend):
        assert db.get_user("nonexistent") is None

    def test_create_user_idempotent(self, db: SQLiteBackend):
        db.create_user("alice")
        db.create_user("alice")
        user = db.get_user("alice")
        assert user is not None

    def test_create_and_get_session(
        self, db: SQLiteBackend, sample_session: AnalysisSession
    ):
        db.create_user(sample_session.username)
        session_id = db.create_session(sample_session)
        assert session_id == sample_session.session_id

        loaded = db.get_session(session_id)
        assert loaded is not None
        assert loaded.policy_question == sample_session.policy_question
        assert loaded.status == "in_progress"

    def test_list_sessions_returns_only_user_sessions(self, db: SQLiteBackend):
        db.create_user("alice")
        db.create_user("bob")
        session_a = AnalysisSession(username="alice", policy_question="Question A")
        session_b = AnalysisSession(username="bob", policy_question="Question B")
        db.create_session(session_a)
        db.create_session(session_b)

        alice_sessions = db.list_sessions("alice")
        assert len(alice_sessions) == 1
        assert alice_sessions[0].policy_question == "Question A"

    def test_update_session(self, db: SQLiteBackend, sample_session: AnalysisSession):
        db.create_user(sample_session.username)
        db.create_session(sample_session)

        sample_session.status = "complete"
        db.update_session(sample_session)

        loaded = db.get_session(sample_session.session_id)
        assert loaded is not None
        assert loaded.status == "complete"

    def test_save_and_load_artifacts(
        self,
        db: SQLiteBackend,
        sample_session: AnalysisSession,
        sample_research: ResearchReport,
    ):
        db.create_user(sample_session.username)
        db.create_session(sample_session)

        artifact_id = db.save_artifact(
            sample_session.session_id,
            "research",
            sample_research.to_json(),
            sample_session.username,
        )
        assert artifact_id is not None

        artifacts = db.load_artifacts(sample_session.session_id, "research")
        assert len(artifacts) == 1
        restored = ResearchReport(**artifacts[0])
        assert restored.query == sample_research.query

    def test_load_artifacts_empty(
        self, db: SQLiteBackend, sample_session: AnalysisSession
    ):
        db.create_user(sample_session.username)
        db.create_session(sample_session)
        artifacts = db.load_artifacts(sample_session.session_id, "research")
        assert artifacts == []

    def test_save_multiple_artifacts_same_step(
        self, db: SQLiteBackend, sample_session: AnalysisSession
    ):
        db.create_user(sample_session.username)
        db.create_session(sample_session)

        for i in range(3):
            report = ResearchReport(
                session_id=sample_session.session_id,
                username="test_user",
                query=f"Query {i}",
                report_markdown=f"Report {i}",
            )
            db.save_artifact(
                sample_session.session_id,
                "research",
                report.to_json(),
                sample_session.username,
            )

        artifacts = db.load_artifacts(sample_session.session_id, "research")
        assert len(artifacts) == 3

    def test_delete_artifact(
        self,
        db: SQLiteBackend,
        sample_session: AnalysisSession,
        sample_research: ResearchReport,
    ):
        db.create_user(sample_session.username)
        db.create_session(sample_session)
        saved_id = db.save_artifact(
            sample_session.session_id,
            "research",
            sample_research.to_json(),
            sample_session.username,
        )

        db.delete_artifact(saved_id)
        artifacts = db.load_artifacts(sample_session.session_id, "research")
        assert len(artifacts) == 0

    def test_update_artifact(
        self,
        db: SQLiteBackend,
        sample_session: AnalysisSession,
        sample_research: ResearchReport,
    ):
        db.create_user(sample_session.username)
        db.create_session(sample_session)
        artifact_id = db.save_artifact(
            sample_session.session_id,
            "research",
            sample_research.to_json(),
            sample_session.username,
        )

        updated_data = sample_research.to_json()
        updated_data["report_markdown"] = "# Updated Report"
        db.update_artifact(artifact_id, updated_data)

        artifacts = db.load_artifacts(sample_session.session_id, "research")
        assert len(artifacts) == 1
        assert artifacts[0]["report_markdown"] == "# Updated Report"

    def test_artifacts_isolated_by_step_name(
        self, db: SQLiteBackend, sample_session: AnalysisSession
    ):
        db.create_user(sample_session.username)
        db.create_session(sample_session)

        research = ResearchReport(
            session_id=sample_session.session_id,
            username="test_user",
            query="test",
            report_markdown="research data",
        )
        db.save_artifact(
            sample_session.session_id,
            "research",
            research.to_json(),
            sample_session.username,
        )

        scenario_set = ScenarioSet(
            session_id=sample_session.session_id,
            username="test_user",
        )
        db.save_artifact(
            sample_session.session_id,
            "scenarios",
            scenario_set.to_json(),
            sample_session.username,
        )

        research_artifacts = db.load_artifacts(sample_session.session_id, "research")
        scenario_artifacts = db.load_artifacts(sample_session.session_id, "scenarios")
        assert len(research_artifacts) == 1
        assert len(scenario_artifacts) == 1

    def test_artifacts_isolated_by_session(self, db: SQLiteBackend):
        db.create_user("test_user")
        session_a = AnalysisSession(username="test_user", policy_question="Q A")
        session_b = AnalysisSession(username="test_user", policy_question="Q B")
        db.create_session(session_a)
        db.create_session(session_b)

        report_a = ResearchReport(
            session_id=session_a.session_id,
            username="test_user",
            query="test A",
            report_markdown="Data A",
        )
        report_b = ResearchReport(
            session_id=session_b.session_id,
            username="test_user",
            query="test B",
            report_markdown="Data B",
        )
        db.save_artifact(
            session_a.session_id, "research", report_a.to_json(), "test_user"
        )
        db.save_artifact(
            session_b.session_id, "research", report_b.to_json(), "test_user"
        )

        artifacts_a = db.load_artifacts(session_a.session_id, "research")
        artifacts_b = db.load_artifacts(session_b.session_id, "research")
        assert len(artifacts_a) == 1
        assert len(artifacts_b) == 1
        assert artifacts_a[0]["report_markdown"] == "Data A"
        assert artifacts_b[0]["report_markdown"] == "Data B"

    def test_save_artifact_with_upsert(
        self,
        db: SQLiteBackend,
        sample_session: AnalysisSession,
    ):
        db.create_user(sample_session.username)
        db.create_session(sample_session)

        scenario_set = ScenarioSet(
            session_id=sample_session.session_id,
            username="test_user",
            rationale="Initial rationale",
        )

        db.save_artifact(
            sample_session.session_id,
            "scenarios",
            scenario_set.to_json(),
            sample_session.username,
        )

        scenario_set.rationale = "Updated rationale"
        db.save_artifact(
            sample_session.session_id,
            "scenarios",
            scenario_set.to_json(),
            sample_session.username,
        )

        artifacts = db.load_artifacts(sample_session.session_id, "scenarios")
        assert len(artifacts) == 1
        assert artifacts[0]["rationale"] == "Updated rationale"

    def test_get_nonexistent_session_returns_none(self, db: SQLiteBackend):
        assert db.get_session("nonexistent-id") is None
