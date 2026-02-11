from __future__ import annotations

import logging

from forecasting_tools.agents_and_tools.situation_simulator.data_models import (
    AgentDefinition,
    Channel,
    CommunicationConfig,
    Environment,
    ItemDefinition,
    Message,
    MetadataItem,
    SimulationState,
    Situation,
)
from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.data_models import (
    ForecastCategory,
    HardMetricCriteria,
    InterventionForecast,
    InterventionRun,
    PolicyAgentResult,
)
from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.forecast_resolver import (
    calculate_brier_score,
    resolve_hard_metric_forecast,
)
from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.intervention_runner import (
    _create_intervention_situation,
    _inject_intervention_message,
)

logger = logging.getLogger(__name__)


def _make_test_situation() -> Situation:
    return Situation(
        name="Test Scenario",
        description="A test scenario",
        rules_text="Standard rules.",
        items=[
            ItemDefinition(name="gold", description="Currency", tradable=True),
            ItemDefinition(name="wood", description="Resource", tradable=True),
            ItemDefinition(name="stone", description="Resource", tradable=True),
        ],
        agents=[
            AgentDefinition(
                name="Alice",
                persona=[
                    MetadataItem(key="role", value="trader", hidden=False),
                    MetadataItem(key="secret", value="has gold stash", hidden=True),
                ],
                starting_inventory={"gold": 50, "wood": 10},
            ),
            AgentDefinition(
                name="Bob",
                persona=[
                    MetadataItem(key="role", value="builder", hidden=False),
                ],
                starting_inventory={"gold": 20, "stone": 15},
            ),
        ],
        environment=Environment(
            description="Test environment",
            inventory={"wood": 100, "stone": 100},
        ),
        communication=CommunicationConfig(
            channels=[Channel(name="general", members="everyone")],
        ),
        max_steps=20,
    )


def _make_test_state() -> SimulationState:
    return SimulationState(
        step_number=5,
        inventories={
            "Alice": {"gold": 45, "wood": 8},
            "Bob": {"gold": 25, "stone": 12},
        },
        environment_inventory={"wood": 92, "stone": 95},
        message_history=[
            Message(
                step=3,
                sender="Alice",
                channel="general",
                recipients=["Alice", "Bob"],
                content="Looking to trade wood for stone.",
            ),
        ],
    )


def _make_hard_metric_forecast(
    agent_name: str = "Alice",
    item_name: str = "gold",
    operator: str = ">=",
    threshold: int = 40,
    prediction: float = 0.7,
    is_conditional: bool = False,
) -> InterventionForecast:
    return InterventionForecast(
        question_title=f"{agent_name} {item_name} {operator} {threshold}",
        question_text=f"Will {agent_name} have {operator} {threshold} {item_name}?",
        resolution_criteria=f"Resolves YES if {agent_name} {item_name} {operator} {threshold}",
        prediction=prediction,
        reasoning="Based on current trajectory.",
        is_conditional=is_conditional,
        category=ForecastCategory.HARD_METRIC,
        hard_metric_criteria=HardMetricCriteria(
            agent_name=agent_name,
            item_name=item_name,
            operator=operator,
            threshold=threshold,
        ),
    )


# --- Brier Score Calculation ---


class TestBrierScoreCalculation:
    def test_perfect_prediction_true(self) -> None:
        score = calculate_brier_score(1.0, True)
        assert abs(score - 0.0) < 1e-9

    def test_perfect_prediction_false(self) -> None:
        score = calculate_brier_score(0.0, False)
        assert abs(score - 0.0) < 1e-9

    def test_worst_prediction_true(self) -> None:
        score = calculate_brier_score(0.0, True)
        assert abs(score - 1.0) < 1e-9

    def test_worst_prediction_false(self) -> None:
        score = calculate_brier_score(1.0, False)
        assert abs(score - 1.0) < 1e-9

    def test_moderate_prediction(self) -> None:
        score = calculate_brier_score(0.7, True)
        assert abs(score - 0.09) < 0.001

    def test_fifty_fifty(self) -> None:
        score = calculate_brier_score(0.5, True)
        assert abs(score - 0.25) < 1e-9


# --- Hard Metric Forecast Resolution ---


class TestHardMetricResolution:
    def test_resolves_true_when_condition_met(self) -> None:
        state = _make_test_state()
        forecast = _make_hard_metric_forecast(
            agent_name="Alice",
            item_name="gold",
            operator=">=",
            threshold=40,
        )
        result = resolve_hard_metric_forecast(forecast, state)
        assert result.resolved is True
        assert result.resolution is True
        assert result.brier_score is not None

    def test_resolves_false_when_condition_not_met(self) -> None:
        state = _make_test_state()
        forecast = _make_hard_metric_forecast(
            agent_name="Alice",
            item_name="gold",
            operator=">=",
            threshold=100,
        )
        result = resolve_hard_metric_forecast(forecast, state)
        assert result.resolved is True
        assert result.resolution is False

    def test_greater_than_operator(self) -> None:
        state = _make_test_state()
        forecast_exact = _make_hard_metric_forecast(
            agent_name="Alice",
            item_name="gold",
            operator=">",
            threshold=45,
        )
        result = resolve_hard_metric_forecast(forecast_exact, state)
        assert result.resolution is False

        forecast_less = _make_hard_metric_forecast(
            agent_name="Alice",
            item_name="gold",
            operator=">",
            threshold=44,
        )
        result = resolve_hard_metric_forecast(forecast_less, state)
        assert result.resolution is True

    def test_less_than_operator(self) -> None:
        state = _make_test_state()
        forecast = _make_hard_metric_forecast(
            agent_name="Bob",
            item_name="gold",
            operator="<",
            threshold=30,
        )
        result = resolve_hard_metric_forecast(forecast, state)
        assert result.resolution is True

    def test_equality_operator(self) -> None:
        state = _make_test_state()
        forecast = _make_hard_metric_forecast(
            agent_name="Alice",
            item_name="gold",
            operator="==",
            threshold=45,
        )
        result = resolve_hard_metric_forecast(forecast, state)
        assert result.resolution is True

    def test_not_equal_operator(self) -> None:
        state = _make_test_state()
        forecast = _make_hard_metric_forecast(
            agent_name="Alice",
            item_name="gold",
            operator="!=",
            threshold=45,
        )
        result = resolve_hard_metric_forecast(forecast, state)
        assert result.resolution is False

    def test_missing_item_defaults_to_zero(self) -> None:
        state = _make_test_state()
        forecast = _make_hard_metric_forecast(
            agent_name="Alice",
            item_name="stone",
            operator=">=",
            threshold=1,
        )
        result = resolve_hard_metric_forecast(forecast, state)
        assert result.resolution is False

    def test_missing_agent_defaults_to_empty_inventory(self) -> None:
        state = _make_test_state()
        forecast = _make_hard_metric_forecast(
            agent_name="Charlie",
            item_name="gold",
            operator=">=",
            threshold=1,
        )
        result = resolve_hard_metric_forecast(forecast, state)
        assert result.resolution is False

    def test_no_criteria_returns_unresolved(self) -> None:
        forecast = InterventionForecast(
            question_title="Missing criteria",
            question_text="Will something happen?",
            resolution_criteria="Unclear",
            prediction=0.5,
            reasoning="No criteria.",
            is_conditional=False,
            category=ForecastCategory.HARD_METRIC,
            hard_metric_criteria=None,
        )
        state = _make_test_state()
        result = resolve_hard_metric_forecast(forecast, state)
        assert result.resolved is False
        assert result.resolution is None

    def test_brier_score_calculated_correctly_on_resolution(self) -> None:
        state = _make_test_state()
        forecast = _make_hard_metric_forecast(
            agent_name="Alice",
            item_name="gold",
            operator=">=",
            threshold=40,
            prediction=0.8,
        )
        result = resolve_hard_metric_forecast(forecast, state)
        assert result.resolution is True
        expected_brier = (0.8 - 1.0) ** 2
        assert abs(result.brier_score - expected_brier) < 0.0001


# --- Intervention Injection ---


class TestInterventionInjection:
    def test_inject_message_adds_dm(self) -> None:
        state = _make_test_state()
        situation = _make_test_situation()
        target = situation.agents[0]
        initial_message_count = len(state.message_history)

        _inject_intervention_message(
            state,
            target,
            "Focus on gathering wood.",
            current_step=5,
        )

        assert len(state.message_history) == initial_message_count + 1
        new_msg = state.message_history[-1]
        assert new_msg.sender == "Intervention Advisor"
        assert new_msg.channel is None
        assert target.name in new_msg.recipients
        assert "Focus on gathering wood" in new_msg.content
        assert "MANDATORY" in new_msg.content

    def test_create_intervention_situation_modifies_rules(self) -> None:
        situation = _make_test_situation()
        target = situation.agents[0]

        modified = _create_intervention_situation(
            situation,
            target,
            "Trade all stone for gold.",
        )

        assert "MANDATORY INTERVENTION NOTICE" in modified.rules_text
        assert target.name in modified.rules_text
        assert situation.rules_text in modified.rules_text

    def test_original_situation_unchanged(self) -> None:
        situation = _make_test_situation()
        original_rules = situation.rules_text
        target = situation.agents[0]

        _create_intervention_situation(
            situation,
            target,
            "Some intervention.",
        )

        assert situation.rules_text == original_rules

    def test_intervention_message_step_matches(self) -> None:
        state = _make_test_state()
        situation = _make_test_situation()
        target = situation.agents[0]

        _inject_intervention_message(state, target, "Do something.", current_step=7)

        new_msg = state.message_history[-1]
        assert new_msg.step == 7


# --- Data Model Properties ---


class TestInterventionRunProperties:
    def _make_test_run(self) -> InterventionRun:
        forecasts = [
            _make_hard_metric_forecast(
                prediction=0.8,
                is_conditional=False,
            ),
            _make_hard_metric_forecast(
                prediction=0.6,
                is_conditional=True,
            ),
            InterventionForecast(
                question_title="Qualitative baseline",
                question_text="Will alliance form?",
                resolution_criteria="Evidence of alliance in messages",
                prediction=0.5,
                reasoning="Uncertain.",
                is_conditional=False,
                category=ForecastCategory.QUALITATIVE,
                resolved=True,
                resolution=True,
                brier_score=0.25,
            ),
            InterventionForecast(
                question_title="Qualitative conditional",
                question_text="Will conflict arise?",
                resolution_criteria="Evidence of conflict",
                prediction=0.3,
                reasoning="Low probability.",
                is_conditional=True,
                category=ForecastCategory.QUALITATIVE,
                resolved=True,
                resolution=False,
                brier_score=0.09,
            ),
        ]
        forecasts[0] = forecasts[0].model_copy(
            update={"resolved": True, "resolution": True, "brier_score": 0.04}
        )
        forecasts[1] = forecasts[1].model_copy(
            update={"resolved": True, "resolution": False, "brier_score": 0.36}
        )

        return InterventionRun(
            run_id="test123",
            model_name="test/model",
            situation_name="Test Scenario",
            target_agent_name="Alice",
            intervention_description="Test intervention",
            policy_proposal_markdown="# Proposal",
            evaluation_criteria=["Criterion 1", "Criterion 2"],
            warmup_steps=5,
            total_steps=20,
            forecasts=forecasts,
            total_cost=1.23,
        )

    def test_hard_metric_forecasts_filter(self) -> None:
        run = self._make_test_run()
        assert len(run.hard_metric_forecasts) == 2

    def test_qualitative_forecasts_filter(self) -> None:
        run = self._make_test_run()
        assert len(run.qualitative_forecasts) == 2

    def test_baseline_forecasts_filter(self) -> None:
        run = self._make_test_run()
        assert len(run.baseline_forecasts) == 2

    def test_conditional_forecasts_filter(self) -> None:
        run = self._make_test_run()
        assert len(run.conditional_forecasts) == 2

    def test_average_brier_score(self) -> None:
        run = self._make_test_run()
        avg = run.average_brier_score
        expected = (0.04 + 0.36 + 0.25 + 0.09) / 4
        assert avg is not None
        assert abs(avg - expected) < 0.0001

    def test_average_hard_metric_brier_score(self) -> None:
        run = self._make_test_run()
        avg = run.average_hard_metric_brier_score
        expected = (0.04 + 0.36) / 2
        assert avg is not None
        assert abs(avg - expected) < 0.0001

    def test_average_qualitative_brier_score(self) -> None:
        run = self._make_test_run()
        avg = run.average_qualitative_brier_score
        expected = (0.25 + 0.09) / 2
        assert avg is not None
        assert abs(avg - expected) < 0.0001


# --- Data Model Serialization ---


class TestInterventionDataSerialization:
    def test_intervention_forecast_round_trip(self) -> None:
        forecast = _make_hard_metric_forecast(
            prediction=0.75,
            is_conditional=True,
        )
        forecast = forecast.model_copy(
            update={"resolved": True, "resolution": True, "brier_score": 0.0625}
        )
        json_data = forecast.to_json()
        restored = InterventionForecast.from_json(json_data)

        assert abs(restored.prediction - 0.75) < 1e-9
        assert restored.is_conditional is True
        assert restored.category == ForecastCategory.HARD_METRIC
        assert restored.hard_metric_criteria is not None
        assert restored.hard_metric_criteria.agent_name == "Alice"
        assert abs(restored.brier_score - 0.0625) < 1e-9

    def test_intervention_run_round_trip(self) -> None:
        run = InterventionRun(
            run_id="abc123",
            model_name="test/model",
            situation_name="Test Scenario",
            target_agent_name="Alice",
            intervention_description="Trade more",
            policy_proposal_markdown="# Policy",
            evaluation_criteria=["Maximize gold"],
            warmup_steps=5,
            total_steps=20,
            forecasts=[
                _make_hard_metric_forecast(prediction=0.6),
            ],
            total_cost=0.50,
        )
        json_data = run.to_json()
        restored = InterventionRun.from_json(json_data)

        assert restored.run_id == "abc123"
        assert restored.model_name == "test/model"
        assert len(restored.forecasts) == 1
        assert abs(restored.forecasts[0].prediction - 0.6) < 1e-9

    def test_policy_agent_result_properties(self) -> None:
        result = PolicyAgentResult(
            agent_goals_analysis="Agent wants gold.",
            evaluation_criteria=["Maximize gold", "Minimize risk"],
            intervention_description="Focus on trades.",
            policy_proposal_markdown="# Proposal",
            forecasts=[
                _make_hard_metric_forecast(is_conditional=False),
                _make_hard_metric_forecast(is_conditional=True),
                InterventionForecast(
                    question_title="Qualitative",
                    question_text="Will trade happen?",
                    resolution_criteria="Evidence of trade",
                    prediction=0.5,
                    reasoning="Maybe.",
                    is_conditional=False,
                    category=ForecastCategory.QUALITATIVE,
                ),
            ],
        )
        assert len(result.baseline_forecasts) == 2
        assert len(result.conditional_forecasts) == 1
        assert len(result.hard_metric_forecasts) == 2
        assert len(result.qualitative_forecasts) == 1
