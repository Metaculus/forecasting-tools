from __future__ import annotations

import logging
import operator
import random

from forecasting_tools.agents_and_tools.situation_simulator.data_models import (
    Effect,
    InventoryCondition,
    SimulationState,
    Situation,
    TradeProposal,
    TradeRecord,
)

logger = logging.getLogger(__name__)

COMPARISON_OPERATORS: dict[str, callable] = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
    "!=": operator.ne,
}


class EffectEngine:
    def __init__(
        self,
        state: SimulationState,
        situation: Situation,
    ) -> None:
        self.state = state
        self.situation = situation

    def apply_effects(
        self,
        effects: list[Effect],
        actor: str,
        params: dict[str, str],
    ) -> list[str]:
        log: list[str] = []
        for effect in effects:
            log.extend(self._apply_single_effect(effect, actor, params))
        return log

    def resolve_trade(
        self,
        proposal_id: str,
        acceptor: str,
    ) -> tuple[bool, str]:
        proposal = self._find_pending_trade(proposal_id)
        if proposal is None:
            return False, f"Trade {proposal_id} not found or not pending"

        if acceptor not in proposal.eligible_acceptors:
            return False, f"{acceptor} is not eligible to accept trade {proposal_id}"

        if not self._agent_has_items(proposal.proposer, proposal.offering):
            proposal.status = "expired"
            return False, f"{proposal.proposer} no longer has the offered items"

        if not self._agent_has_items(acceptor, proposal.requesting):
            return False, f"{acceptor} does not have the requested items"

        self._transfer_items(proposal.proposer, acceptor, proposal.offering)
        self._transfer_items(acceptor, proposal.proposer, proposal.requesting)

        records = self._create_trade_records(proposal, acceptor)
        self.state.trade_history.extend(records)

        proposal.status = "accepted"
        return (
            True,
            f"Trade {proposal_id} completed between {proposal.proposer} and {acceptor}",
        )

    def reject_trade(self, proposal_id: str) -> tuple[bool, str]:
        proposal = self._find_pending_trade(proposal_id)
        if proposal is None:
            return False, f"Trade {proposal_id} not found or not pending"
        proposal.status = "rejected"
        return True, f"Trade {proposal_id} rejected"

    def process_step_end_rules(self) -> list[str]:
        log: list[str] = []
        for agent_def in self.situation.agents:
            inventory = self.state.inventories.get(agent_def.name, {})
            for rule in agent_def.inventory_rules:
                if self._evaluate_conditions(rule.conditions, inventory):
                    log.append(f"Rule '{rule.name}' triggered for {agent_def.name}")
                    log.extend(self.apply_effects(rule.effects, agent_def.name, {}))

        env_inventory = self.state.environment_inventory
        for rule in self.situation.environment.inventory_rules:
            if self._evaluate_conditions(rule.conditions, env_inventory):
                log.append(f"Environment rule '{rule.name}' triggered")
                log.extend(self.apply_effects(rule.effects, "environment", {}))

        return log

    def expire_trades(self) -> list[str]:
        log: list[str] = []
        for trade in self.state.pending_trades:
            if (
                trade.status == "pending"
                and self.state.step_number > trade.expires_at_step
            ):
                trade.status = "expired"
                log.append(f"Trade {trade.id} from {trade.proposer} expired")
        return log

    # --- Private helpers ---

    def _apply_single_effect(
        self,
        effect: Effect,
        actor: str,
        params: dict[str, str],
    ) -> list[str]:
        resolved_target = self._resolve_target(
            self._resolve_param_refs(effect.target, params), actor
        )
        resolved_quantity = self._resolve_quantity(effect.quantity, params)
        resolved_item = self._resolve_param_refs(effect.item_name, params)

        if effect.type == "add_item":
            self._modify_inventory(resolved_target, resolved_item, resolved_quantity)
            return [f"Added {resolved_quantity} {resolved_item} to {resolved_target}"]

        elif effect.type == "remove_item":
            current = self._get_inventory(resolved_target).get(resolved_item, 0)
            actual_removed = min(current, resolved_quantity)
            self._modify_inventory(resolved_target, resolved_item, -actual_removed)
            return [f"Removed {actual_removed} {resolved_item} from {resolved_target}"]

        elif effect.type == "transfer_item":
            resolved_source = self._resolve_target(
                self._resolve_param_refs(effect.source, params), actor
            )
            source_inventory = self._get_inventory(resolved_source)
            current = source_inventory.get(resolved_item, 0)
            actual_transferred = min(current, resolved_quantity)
            self._modify_inventory(resolved_source, resolved_item, -actual_transferred)
            self._modify_inventory(resolved_target, resolved_item, actual_transferred)
            return [
                f"Transferred {actual_transferred} {resolved_item} from {resolved_source} to {resolved_target}"
            ]

        elif effect.type == "random_outcome":
            return self._resolve_random_outcome(effect, actor, params)

        elif effect.type == "message":
            resolved_message = self._resolve_param_refs(effect.message_text, params)
            return [f"System message for {resolved_target}: {resolved_message}"]

        return []

    def _resolve_random_outcome(
        self,
        effect: Effect,
        actor: str,
        params: dict[str, str],
    ) -> list[str]:
        if not effect.outcomes:
            return ["Random outcome had no outcomes defined"]

        roll = random.random()
        cumulative = 0.0
        for outcome in effect.outcomes:
            cumulative += outcome.probability
            if roll <= cumulative:
                log = [f"Random outcome: {outcome.description}"]
                log.extend(self.apply_effects(outcome.effects, actor, params))
                return log

        last_outcome = effect.outcomes[-1]
        log = [f"Random outcome (fallback): {last_outcome.description}"]
        log.extend(self.apply_effects(last_outcome.effects, actor, params))
        return log

    def _evaluate_conditions(
        self,
        conditions: list[InventoryCondition],
        inventory: dict[str, int],
    ) -> bool:
        if not conditions:
            return True
        for condition in conditions:
            current_value = inventory.get(condition.item_name, 0)
            op_func = COMPARISON_OPERATORS.get(condition.operator, operator.ge)
            if not op_func(current_value, condition.threshold):
                return False
        return True

    def _find_pending_trade(self, proposal_id: str) -> TradeProposal | None:
        for trade in self.state.pending_trades:
            if trade.id == proposal_id and trade.status == "pending":
                return trade
        return None

    def _agent_has_items(
        self,
        agent_name: str,
        items: dict[str, int],
    ) -> bool:
        inventory = self._get_inventory(agent_name)
        for item_name, quantity in items.items():
            if inventory.get(item_name, 0) < quantity:
                return False
        return True

    def _transfer_items(
        self,
        from_agent: str,
        to_agent: str,
        items: dict[str, int],
    ) -> None:
        for item_name, quantity in items.items():
            self._modify_inventory(from_agent, item_name, -quantity)
            self._modify_inventory(to_agent, item_name, quantity)

    def _create_trade_records(
        self,
        proposal: TradeProposal,
        acceptor: str,
    ) -> list[TradeRecord]:
        records: list[TradeRecord] = []
        for item_name, quantity in proposal.offering.items():
            records.append(
                TradeRecord(
                    item_name=item_name,
                    quantity=quantity,
                    from_agent=proposal.proposer,
                    to_agent=acceptor,
                    step=self.state.step_number,
                    trade_id=proposal.id,
                )
            )
        for item_name, quantity in proposal.requesting.items():
            records.append(
                TradeRecord(
                    item_name=item_name,
                    quantity=quantity,
                    from_agent=acceptor,
                    to_agent=proposal.proposer,
                    step=self.state.step_number,
                    trade_id=proposal.id,
                )
            )
        return records

    def _resolve_target(self, target: str, actor: str) -> str:
        if target == "actor":
            return actor
        return target

    def _resolve_quantity(self, quantity: int | str, params: dict[str, str]) -> int:
        if isinstance(quantity, int):
            return quantity
        resolved = self._resolve_param_refs(str(quantity), params)
        try:
            return int(resolved)
        except ValueError:
            logger.warning(
                f"Could not resolve quantity '{quantity}' to int, defaulting to 0"
            )
            return 0

    def _resolve_param_refs(self, text: str, params: dict[str, str]) -> str:
        for key, value in params.items():
            text = text.replace(f"{{{key}}}", value)
        return text

    def _get_inventory(self, target: str) -> dict[str, int]:
        if target == "environment":
            return self.state.environment_inventory
        return self.state.inventories.get(target, {})

    def _modify_inventory(
        self,
        target: str,
        item_name: str,
        delta: int,
    ) -> None:
        if target == "environment":
            inventory = self.state.environment_inventory
        else:
            if target not in self.state.inventories:
                self.state.inventories[target] = {}
            inventory = self.state.inventories[target]

        current = inventory.get(item_name, 0)
        new_value = max(0, current + delta)
        if new_value == 0 and item_name in inventory:
            del inventory[item_name]
        elif new_value > 0:
            inventory[item_name] = new_value
