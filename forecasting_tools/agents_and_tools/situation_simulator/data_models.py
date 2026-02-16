from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from forecasting_tools.util.jsonable import Jsonable


class ItemDefinition(BaseModel, Jsonable):
    name: str
    description: str
    tradable: bool = True


class MetadataItem(BaseModel, Jsonable):
    key: str
    value: str
    hidden: bool = False


class ActionParameter(BaseModel, Jsonable):
    name: str
    description: str
    type: Literal["str", "int", "float", "agent_name", "item_name"]


class RandomOutcome(BaseModel, Jsonable):
    probability: float = Field(ge=0.0, le=1.0)
    effects: list[Effect] = Field(default_factory=list)
    description: str = ""


class Effect(BaseModel, Jsonable):
    type: Literal[
        "add_item",
        "remove_item",
        "transfer_item",
        "random_outcome",
        "message",
    ]
    target: str = "actor"
    item_name: str = ""
    quantity: int | str = 0
    source: str = ""
    outcomes: list[RandomOutcome] = Field(default_factory=list)
    message_text: str = ""


RandomOutcome.model_rebuild()


class ActionDefinition(BaseModel, Jsonable):
    name: str
    description: str
    parameters: list[ActionParameter] = Field(default_factory=list)
    effects: list[Effect] = Field(default_factory=list)
    available_to: list[str] | Literal["everyone"] = "everyone"


class InventoryCondition(BaseModel, Jsonable):
    item_name: str
    operator: Literal[">=", "<=", "==", ">", "<", "!="] = ">="
    threshold: int = 1


class InventoryRule(BaseModel, Jsonable):
    name: str
    description: str
    conditions: list[InventoryCondition] = Field(default_factory=list)
    effects: list[Effect] = Field(default_factory=list)


class AgentDefinition(BaseModel, Jsonable):
    name: str
    persona: list[MetadataItem] = Field(default_factory=list)
    starting_inventory: dict[str, int] = Field(default_factory=dict)
    special_actions: list[ActionDefinition] = Field(default_factory=list)
    inventory_rules: list[InventoryRule] = Field(default_factory=list)
    ai_model: str = "openrouter/anthropic/claude-sonnet-4.5"


class Channel(BaseModel, Jsonable):
    name: str
    members: list[str] | Literal["everyone"] = "everyone"
    description: str = ""


class CommunicationConfig(BaseModel, Jsonable):
    channels: list[Channel] = Field(default_factory=list)
    dm_blacklist: list[tuple[str, str]] = Field(default_factory=list)
    max_messages_per_turn: int = 7


class Environment(BaseModel, Jsonable):
    description: str = ""
    inventory: dict[str, int] = Field(default_factory=dict)
    global_actions: list[ActionDefinition] = Field(default_factory=list)
    inventory_rules: list[InventoryRule] = Field(default_factory=list)


class Situation(BaseModel, Jsonable):
    name: str
    description: str
    rules_text: str
    items: list[ItemDefinition] = Field(default_factory=list)
    agents: list[AgentDefinition] = Field(default_factory=list)
    environment: Environment = Field(default_factory=Environment)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    max_steps: int = 50


# --- Runtime State Models ---


class Message(BaseModel, Jsonable):
    step: int
    sender: str
    channel: str | None = None
    recipients: list[str] = Field(default_factory=list)
    content: str = ""


class TradeProposal(BaseModel, Jsonable):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    proposer: str = ""
    eligible_acceptors: list[str] = Field(default_factory=list)
    offering: dict[str, int] = Field(default_factory=dict)
    requesting: dict[str, int] = Field(default_factory=dict)
    message: str = ""
    proposed_at_step: int = 0
    expires_at_step: int = 0
    status: Literal["pending", "accepted", "rejected", "expired"] = "pending"


class TradeRecord(BaseModel, Jsonable):
    item_name: str
    quantity: int
    from_agent: str
    to_agent: str
    step: int
    trade_id: str


class AgentAction(BaseModel, Jsonable):
    agent_name: str = ""
    action_name: str = "no_action"
    parameters: dict[str, str] = Field(default_factory=dict)
    messages_to_send: list[Message] = Field(default_factory=list)
    trade_proposal: TradeProposal | None = None
    trade_acceptance_id: str | None = None


class SimulationState(BaseModel, Jsonable):
    step_number: int = 0
    inventories: dict[str, dict[str, int]] = Field(default_factory=dict)
    environment_inventory: dict[str, int] = Field(default_factory=dict)
    message_history: list[Message] = Field(default_factory=list)
    pending_trades: list[TradeProposal] = Field(default_factory=list)
    trade_history: list[TradeRecord] = Field(default_factory=list)
    action_log: list[AgentAction] = Field(default_factory=list)

    def deep_copy(self) -> SimulationState:
        return SimulationState.model_validate(self.model_dump())


class SimulationStep(BaseModel, Jsonable):
    step_number: int
    agent_actions: list[AgentAction] = Field(default_factory=list)
    triggered_effects_log: list[str] = Field(default_factory=list)
    state_before: SimulationState = Field(default_factory=SimulationState)
    state_after: SimulationState = Field(default_factory=SimulationState)


class SimulationResult(BaseModel, Jsonable):
    situation: Situation
    steps: list[SimulationStep] = Field(default_factory=list)
    final_state: SimulationState = Field(default_factory=SimulationState)
