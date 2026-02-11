from __future__ import annotations

import logging

from forecasting_tools.agents_and_tools.situation_simulator.agent_runner import (
    SimulationAgentRunner,
)
from forecasting_tools.agents_and_tools.situation_simulator.data_models import (
    AgentDefinition,
    Channel,
    CommunicationConfig,
    Effect,
    Environment,
    InventoryCondition,
    InventoryRule,
    ItemDefinition,
    Message,
    MetadataItem,
    RandomOutcome,
    SimulationState,
    Situation,
    TradeProposal,
)
from forecasting_tools.agents_and_tools.situation_simulator.effect_engine import (
    EffectEngine,
)

logger = logging.getLogger(__name__)


def _make_simple_situation() -> Situation:
    return Situation(
        name="Test Situation",
        description="A test",
        rules_text="Test rules",
        items=[
            ItemDefinition(name="gold", description="Currency", tradable=True),
            ItemDefinition(name="sword", description="Weapon", tradable=True),
            ItemDefinition(
                name="badge", description="Non-tradable badge", tradable=False
            ),
        ],
        agents=[
            AgentDefinition(
                name="Alice",
                persona=[
                    MetadataItem(key="role", value="trader", hidden=False),
                    MetadataItem(key="secret", value="has hidden gold", hidden=True),
                ],
                starting_inventory={"gold": 10, "sword": 1},
            ),
            AgentDefinition(
                name="Bob",
                persona=[
                    MetadataItem(key="role", value="merchant", hidden=False),
                ],
                starting_inventory={"gold": 20},
            ),
        ],
        environment=Environment(
            description="Test environment",
            inventory={"gold": 100},
        ),
        communication=CommunicationConfig(
            channels=[
                Channel(name="general", members="everyone"),
                Channel(name="secret", members=["Alice"]),
            ],
            dm_blacklist=[("Alice", "Bob")],
        ),
    )


def _make_state_for_situation(situation: Situation) -> SimulationState:
    inventories = {
        agent.name: dict(agent.starting_inventory) for agent in situation.agents
    }
    return SimulationState(
        step_number=1,
        inventories=inventories,
        environment_inventory=dict(situation.environment.inventory),
    )


# --- EffectEngine: apply_effects ---


class TestEffectEngineAddItem:
    def test_add_item_to_actor(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [
            Effect(type="add_item", target="actor", item_name="gold", quantity=5)
        ]
        log = engine.apply_effects(effects, "Alice", {})

        assert state.inventories["Alice"]["gold"] == 15
        assert len(log) == 1
        assert "Added 5 gold to Alice" in log[0]

    def test_add_item_to_environment(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [
            Effect(type="add_item", target="environment", item_name="gold", quantity=50)
        ]
        engine.apply_effects(effects, "Alice", {})

        assert state.environment_inventory["gold"] == 150

    def test_add_item_to_named_agent(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [Effect(type="add_item", target="Bob", item_name="sword", quantity=1)]
        engine.apply_effects(effects, "Alice", {})

        assert state.inventories["Bob"]["sword"] == 1

    def test_add_new_item_creates_entry(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [
            Effect(type="add_item", target="actor", item_name="badge", quantity=1)
        ]
        engine.apply_effects(effects, "Alice", {})

        assert state.inventories["Alice"]["badge"] == 1


class TestEffectEngineRemoveItem:
    def test_remove_item_basic(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [
            Effect(type="remove_item", target="actor", item_name="gold", quantity=3)
        ]
        log = engine.apply_effects(effects, "Alice", {})

        assert state.inventories["Alice"]["gold"] == 7
        assert "Removed 3 gold" in log[0]

    def test_remove_capped_at_current_quantity(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [
            Effect(type="remove_item", target="actor", item_name="gold", quantity=999)
        ]
        log = engine.apply_effects(effects, "Alice", {})

        assert "gold" not in state.inventories["Alice"]
        assert "Removed 10 gold" in log[0]

    def test_remove_nonexistent_item_removes_zero(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [
            Effect(type="remove_item", target="actor", item_name="badge", quantity=5)
        ]
        log = engine.apply_effects(effects, "Alice", {})

        assert "Removed 0 badge" in log[0]


class TestEffectEngineTransferItem:
    def test_transfer_between_agents(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [
            Effect(
                type="transfer_item",
                source="actor",
                target="Bob",
                item_name="gold",
                quantity=5,
            )
        ]
        engine.apply_effects(effects, "Alice", {})

        assert state.inventories["Alice"]["gold"] == 5
        assert state.inventories["Bob"]["gold"] == 25

    def test_transfer_from_environment(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [
            Effect(
                type="transfer_item",
                source="environment",
                target="actor",
                item_name="gold",
                quantity=10,
            )
        ]
        engine.apply_effects(effects, "Alice", {})

        assert state.inventories["Alice"]["gold"] == 20
        assert state.environment_inventory["gold"] == 90

    def test_transfer_capped_at_source_quantity(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [
            Effect(
                type="transfer_item",
                source="actor",
                target="Bob",
                item_name="sword",
                quantity=5,
            )
        ]
        engine.apply_effects(effects, "Alice", {})

        assert "sword" not in state.inventories["Alice"]
        assert state.inventories["Bob"]["sword"] == 1


class TestEffectEngineParameterResolution:
    def test_quantity_from_parameter(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [
            Effect(
                type="add_item", target="actor", item_name="gold", quantity="{amount}"
            )
        ]
        engine.apply_effects(effects, "Alice", {"amount": "7"})

        assert state.inventories["Alice"]["gold"] == 17

    def test_item_name_from_parameter(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [
            Effect(
                type="add_item",
                target="actor",
                item_name="vote_for_{target}",
                quantity=1,
            )
        ]
        engine.apply_effects(effects, "Alice", {"target": "Bob"})

        assert state.inventories["Alice"]["vote_for_Bob"] == 1

    def test_target_from_parameter(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [
            Effect(type="remove_item", target="{target}", item_name="gold", quantity=5)
        ]
        engine.apply_effects(effects, "Alice", {"target": "Bob"})

        assert state.inventories["Bob"]["gold"] == 15


class TestEffectEngineRandomOutcome:
    def test_random_outcome_selects_one_branch(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        effects = [
            Effect(
                type="random_outcome",
                outcomes=[
                    RandomOutcome(
                        probability=0.5,
                        effects=[
                            Effect(
                                type="add_item",
                                target="actor",
                                item_name="gold",
                                quantity=10,
                            )
                        ],
                        description="Win",
                    ),
                    RandomOutcome(
                        probability=0.5,
                        effects=[
                            Effect(
                                type="remove_item",
                                target="actor",
                                item_name="gold",
                                quantity=5,
                            )
                        ],
                        description="Lose",
                    ),
                ],
            )
        ]
        engine.apply_effects(effects, "Alice", {})

        gold = state.inventories["Alice"].get("gold", 0)
        assert gold == 20 or gold == 5


# --- EffectEngine: Trade resolution ---


class TestTradeResolution:
    def _setup_trade(self) -> tuple[Situation, SimulationState, EffectEngine]:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        trade = TradeProposal(
            id="trade-1",
            proposer="Alice",
            eligible_acceptors=["Bob"],
            offering={"sword": 1},
            requesting={"gold": 15},
            proposed_at_step=1,
            expires_at_step=5,
        )
        state.pending_trades.append(trade)
        engine = EffectEngine(state, situation)
        return situation, state, engine

    def test_successful_trade(self) -> None:
        _, state, engine = self._setup_trade()

        success, _ = engine.resolve_trade("trade-1", "Bob")

        assert success is True
        assert state.inventories["Alice"]["gold"] == 25
        assert "sword" not in state.inventories["Alice"]
        assert state.inventories["Bob"]["sword"] == 1
        assert state.inventories["Bob"]["gold"] == 5
        assert len(state.trade_history) == 2

    def test_trade_by_ineligible_acceptor(self) -> None:
        _, _, engine = self._setup_trade()

        success, msg = engine.resolve_trade("trade-1", "Alice")

        assert success is False
        assert "not eligible" in msg

    def test_trade_nonexistent_id(self) -> None:
        _, _, engine = self._setup_trade()

        success, msg = engine.resolve_trade("nonexistent", "Bob")

        assert success is False
        assert "not found" in msg

    def test_trade_proposer_insufficient_items(self) -> None:
        _, state, engine = self._setup_trade()
        state.inventories["Alice"]["sword"] = 0

        success, _ = engine.resolve_trade("trade-1", "Bob")

        assert success is False

    def test_trade_acceptor_insufficient_items(self) -> None:
        _, state, engine = self._setup_trade()
        state.inventories["Bob"]["gold"] = 5

        success, _ = engine.resolve_trade("trade-1", "Bob")

        assert success is False

    def test_reject_trade(self) -> None:
        _, state, engine = self._setup_trade()

        success, _ = engine.reject_trade("trade-1")

        assert success is True
        assert state.pending_trades[0].status == "rejected"

    def test_expire_trades(self) -> None:
        _, state, engine = self._setup_trade()
        state.step_number = 6

        log = engine.expire_trades()

        assert len(log) == 1
        assert state.pending_trades[0].status == "expired"


# --- EffectEngine: Step-end inventory rules ---


class TestStepEndRules:
    def test_rule_triggers_when_conditions_met(self) -> None:
        situation = _make_simple_situation()
        situation.agents[0].inventory_rules = [
            InventoryRule(
                name="auto_convert",
                description="Convert 5 gold to 1 sword",
                conditions=[
                    InventoryCondition(item_name="gold", operator=">=", threshold=5),
                ],
                effects=[
                    Effect(
                        type="remove_item", target="actor", item_name="gold", quantity=5
                    ),
                    Effect(
                        type="add_item", target="actor", item_name="sword", quantity=1
                    ),
                ],
            )
        ]
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        log = engine.process_step_end_rules()

        assert len(log) >= 1
        assert state.inventories["Alice"]["gold"] == 5
        assert state.inventories["Alice"]["sword"] == 2

    def test_rule_does_not_trigger_when_conditions_not_met(self) -> None:
        situation = _make_simple_situation()
        situation.agents[0].inventory_rules = [
            InventoryRule(
                name="need_lots_of_gold",
                description="Needs 999 gold",
                conditions=[
                    InventoryCondition(item_name="gold", operator=">=", threshold=999),
                ],
                effects=[
                    Effect(
                        type="add_item", target="actor", item_name="sword", quantity=100
                    ),
                ],
            )
        ]
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        log = engine.process_step_end_rules()

        assert len(log) == 0
        assert state.inventories["Alice"]["sword"] == 1

    def test_multiple_conditions_all_must_pass(self) -> None:
        situation = _make_simple_situation()
        situation.agents[0].inventory_rules = [
            InventoryRule(
                name="need_gold_and_sword",
                description="Need both gold and sword",
                conditions=[
                    InventoryCondition(item_name="gold", operator=">=", threshold=5),
                    InventoryCondition(item_name="sword", operator=">=", threshold=1),
                ],
                effects=[
                    Effect(
                        type="add_item", target="actor", item_name="badge", quantity=1
                    ),
                ],
            )
        ]
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        engine.process_step_end_rules()

        assert state.inventories["Alice"]["badge"] == 1

    def test_environment_inventory_rules(self) -> None:
        situation = _make_simple_situation()
        situation.environment.inventory_rules = [
            InventoryRule(
                name="env_rule",
                description="Add gold if environment has enough",
                conditions=[
                    InventoryCondition(item_name="gold", operator=">=", threshold=50),
                ],
                effects=[
                    Effect(
                        type="add_item",
                        target="environment",
                        item_name="gold",
                        quantity=10,
                    ),
                ],
            )
        ]
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)

        engine.process_step_end_rules()

        assert state.environment_inventory["gold"] == 110

    def test_all_comparison_operators(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        engine = EffectEngine(state, situation)
        inventory = {"gold": 10}

        ops_and_expected = [
            (">=", 10, True),
            (">=", 11, False),
            ("<=", 10, True),
            ("<=", 9, False),
            ("==", 10, True),
            ("==", 9, False),
            (">", 9, True),
            (">", 10, False),
            ("<", 11, True),
            ("<", 10, False),
            ("!=", 9, True),
            ("!=", 10, False),
        ]
        for op, threshold, expected in ops_and_expected:
            conditions = [
                InventoryCondition(item_name="gold", operator=op, threshold=threshold)
            ]
            result = engine._evaluate_conditions(conditions, inventory)
            assert (
                result == expected
            ), f"Failed for {op} {threshold}: expected {expected}"


# --- AgentRunner: Visibility filtering ---


class TestAgentRunnerVisibility:
    def test_visible_messages_channel_filtering(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        state.message_history = [
            Message(
                step=1,
                sender="Alice",
                channel="general",
                recipients=["Alice", "Bob"],
                content="Hi all",
            ),
            Message(
                step=1,
                sender="Alice",
                channel="secret",
                recipients=["Alice"],
                content="Secret msg",
            ),
            Message(
                step=1,
                sender="Bob",
                channel="general",
                recipients=["Alice", "Bob"],
                content="Hello",
            ),
        ]
        runner = SimulationAgentRunner()

        alice_visible = runner.get_visible_messages("Alice", state, situation)
        bob_visible = runner.get_visible_messages("Bob", state, situation)

        assert len(alice_visible) == 3
        assert len(bob_visible) == 2
        assert all(m.channel != "secret" for m in bob_visible)

    def test_visible_messages_dm_filtering(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        state.message_history = [
            Message(
                step=1,
                sender="Alice",
                channel=None,
                recipients=["Alice", "Bob"],
                content="DM to Bob",
            ),
        ]
        runner = SimulationAgentRunner()

        alice_visible = runner.get_visible_messages("Alice", state, situation)
        bob_visible = runner.get_visible_messages("Bob", state, situation)

        assert len(alice_visible) == 1
        assert len(bob_visible) == 1

    def test_visible_metadata_hides_hidden_items(self) -> None:
        situation = _make_simple_situation()
        alice_def = situation.agents[0]
        runner = SimulationAgentRunner()

        alice_sees_alice = runner.get_visible_metadata("Alice", alice_def)
        bob_sees_alice = runner.get_visible_metadata("Bob", alice_def)

        assert len(alice_sees_alice) == 2
        assert len(bob_sees_alice) == 1
        assert bob_sees_alice[0].key == "role"

    def test_prompt_contains_key_sections(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        runner = SimulationAgentRunner()

        prompt = runner.build_agent_prompt(situation.agents[0], state, situation)

        assert "Test Situation" in prompt
        assert "Test rules" in prompt
        assert "Alice" in prompt
        assert "Your Inventory" in prompt
        assert "Available Actions" in prompt
        assert "no_action" in prompt
        assert "trade_propose" in prompt

    def test_prompt_includes_hidden_persona_for_self(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        runner = SimulationAgentRunner()

        prompt = runner.build_agent_prompt(situation.agents[0], state, situation)

        assert "has hidden gold" in prompt
        assert "HIDDEN" in prompt

    def test_prompt_excludes_hidden_persona_for_others(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        runner = SimulationAgentRunner()

        prompt = runner.build_agent_prompt(situation.agents[1], state, situation)

        assert "has hidden gold" not in prompt


# --- Data model: deep_copy ---


class TestSimulationStateDeepCopy:
    def test_deep_copy_is_independent(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        copy = state.deep_copy()

        state.inventories["Alice"]["gold"] = 999
        assert copy.inventories["Alice"]["gold"] == 10

    def test_deep_copy_preserves_data(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        state.message_history.append(
            Message(
                step=1,
                sender="Alice",
                channel="general",
                recipients=["Alice", "Bob"],
                content="Test",
            )
        )

        copy = state.deep_copy()

        assert len(copy.message_history) == 1
        assert copy.message_history[0].content == "Test"


# --- Data model: JSON round-trip ---


class TestJsonRoundTrip:
    def test_situation_round_trip(self) -> None:
        situation = _make_simple_situation()
        json_data = situation.to_json()
        restored = Situation.from_json(json_data)

        assert restored.name == situation.name
        assert len(restored.agents) == len(situation.agents)
        assert restored.agents[0].persona[1].hidden is True

    def test_simulation_state_round_trip(self) -> None:
        situation = _make_simple_situation()
        state = _make_state_for_situation(situation)
        state.pending_trades.append(
            TradeProposal(
                proposer="Alice",
                eligible_acceptors=["Bob"],
                offering={"gold": 5},
                requesting={"sword": 1},
                proposed_at_step=1,
                expires_at_step=3,
            )
        )

        json_data = state.to_json()
        restored = SimulationState.from_json(json_data)

        assert restored.inventories["Alice"]["gold"] == 10
        assert len(restored.pending_trades) == 1
        assert restored.pending_trades[0].proposer == "Alice"
