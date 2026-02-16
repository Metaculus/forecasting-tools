from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from forecasting_tools.agents_and_tools.situation_simulator.data_models import (
    ActionDefinition,
    AgentAction,
    AgentDefinition,
    Channel,
    Message,
    MetadataItem,
    SimulationState,
    Situation,
    TradeProposal,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)

AGENT_TURN_TIMEOUT = 120


class LlmActionResponse(BaseModel, Jsonable):
    action_name: str = Field(
        default="no_action",
        description="The action to take. Use 'no_action' to skip, 'trade_propose' to propose a trade, 'trade_accept' or 'trade_reject' to respond to a trade, or any available action name.",
    )
    action_parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Parameters for the chosen action, as key-value string pairs.",
    )
    trade_proposal: LlmTradeProposal | None = Field(
        default=None,
        description="If action_name is 'trade_propose', fill this in.",
    )
    trade_response_id: str | None = Field(
        default=None,
        description="If action_name is 'trade_accept' or 'trade_reject', the ID of the trade proposal.",
    )
    messages: list[LlmMessage] = Field(
        default_factory=list,
        description="Messages to send this turn. Each message has a channel (or recipient for DMs) and content.",
    )
    reasoning: str = Field(
        default="",
        description="Brief internal reasoning for this action (not shown to other agents).",
    )


class LlmTradeProposal(BaseModel, Jsonable):
    eligible_acceptors: list[str] = Field(default_factory=list)
    offering: dict[str, int] = Field(default_factory=dict)
    requesting: dict[str, int] = Field(default_factory=dict)
    message: str = ""
    expires_in_steps: int = Field(default=3, ge=1)


class LlmMessage(BaseModel, Jsonable):
    channel: str | None = Field(
        default=None,
        description="Channel name to post in, or null/omitted for a DM.",
    )
    recipient: str | None = Field(
        default=None,
        description="For DMs only: the agent name to send to.",
    )
    content: str = ""


class SimulationAgentRunner:
    def __init__(
        self,
        default_model: str = "openrouter/anthropic/claude-sonnet-4.5",
        timeout: int = AGENT_TURN_TIMEOUT,
    ) -> None:
        self.default_model = default_model
        self.timeout = timeout

    async def get_agent_action(
        self,
        agent_def: AgentDefinition,
        state: SimulationState,
        situation: Situation,
    ) -> AgentAction:
        prompt = self.build_agent_prompt(agent_def, state, situation)
        model_name = agent_def.ai_model or self.default_model
        llm = GeneralLlm(model_name, temperature=0.7, timeout=self.timeout)

        raw_response = await llm.invoke(prompt)
        logger.info(f"Got response from {agent_def.name}, parsing action...")

        parsed = await structure_output(
            raw_response,
            LlmActionResponse,
            additional_instructions="Extract the agent's chosen action, parameters, messages, and trade info from their response.",
        )

        return self._convert_to_agent_action(parsed, agent_def.name, state, situation)

    def build_agent_prompt(
        self,
        agent_def: AgentDefinition,
        state: SimulationState,
        situation: Situation,
    ) -> str:
        sections: list[str] = []

        sections.append(
            self._build_system_section(agent_def, situation, state.step_number)
        )
        sections.append(self._build_other_agents_section(agent_def.name, situation))
        sections.append(self._build_inventory_section(agent_def.name, state))
        sections.append(self._build_messages_section(agent_def.name, state, situation))
        sections.append(self._build_trades_section(agent_def.name, state))
        sections.append(self._build_actions_section(agent_def, situation))
        sections.append(self._build_response_instructions(situation))

        return "\n\n---\n\n".join(sections)

    def get_visible_messages(
        self,
        agent_name: str,
        state: SimulationState,
        situation: Situation,
    ) -> list[Message]:
        accessible_channels = self._get_accessible_channels(agent_name, situation)
        channel_names = {ch.name for ch in accessible_channels}

        visible: list[Message] = []
        for msg in state.message_history:
            if msg.channel is not None:
                if msg.channel in channel_names:
                    visible.append(msg)
            else:
                if agent_name in msg.recipients or msg.sender == agent_name:
                    visible.append(msg)
        return visible

    def get_visible_metadata(
        self,
        viewer: str,
        target: AgentDefinition,
    ) -> list[MetadataItem]:
        if viewer == target.name:
            return target.persona
        return [item for item in target.persona if not item.hidden]

    # --- Prompt building sections ---

    def _build_system_section(
        self,
        agent_def: AgentDefinition,
        situation: Situation,
        step_number: int,
    ) -> str:
        persona_lines = []
        for item in agent_def.persona:
            visibility = " [HIDDEN - only you know this]" if item.hidden else ""
            persona_lines.append(f"- {item.key}: {item.value}{visibility}")
        persona_text = (
            "\n".join(persona_lines) if persona_lines else "No persona defined."
        )

        return clean_indents(
            f"""
            # Simulation: {situation.name}

            {situation.description}

            ## Rules

            {situation.rules_text}

            ## Your Identity: {agent_def.name}

            {persona_text}

            ## Current Step: {step_number}
            """
        ).strip()

    def _build_other_agents_section(
        self,
        viewer_name: str,
        situation: Situation,
    ) -> str:
        lines = ["## Other Agents"]
        for agent_def in situation.agents:
            if agent_def.name == viewer_name:
                continue
            public_metadata = self.get_visible_metadata(viewer_name, agent_def)
            if public_metadata:
                metadata_text = ", ".join(
                    f"{m.key}: {m.value}" for m in public_metadata
                )
                lines.append(f"- **{agent_def.name}**: {metadata_text}")
            else:
                lines.append(f"- **{agent_def.name}**")
        return "\n".join(lines)

    def _build_inventory_section(
        self,
        agent_name: str,
        state: SimulationState,
    ) -> str:
        inventory = state.inventories.get(agent_name, {})
        if inventory:
            items_text = "\n".join(
                f"- {name}: {qty}" for name, qty in inventory.items()
            )
        else:
            items_text = "Empty"
        return f"## Your Inventory\n\n{items_text}"

    def _build_messages_section(
        self,
        agent_name: str,
        state: SimulationState,
        situation: Situation,
    ) -> str:
        visible = self.get_visible_messages(agent_name, state, situation)
        if not visible:
            return "## Recent Messages\n\nNo messages yet."

        lines = ["## Recent Messages"]
        for msg in visible[-50:]:
            if msg.channel:
                lines.append(
                    f"[Step {msg.step}] #{msg.channel} | {msg.sender}: {msg.content}"
                )
            else:
                other = [r for r in msg.recipients if r != agent_name]
                dm_with = other[0] if other else msg.sender
                lines.append(
                    f"[Step {msg.step}] DM with {dm_with} | {msg.sender}: {msg.content}"
                )
        return "\n".join(lines)

    def _build_trades_section(
        self,
        agent_name: str,
        state: SimulationState,
    ) -> str:
        pending_for_agent = [
            t
            for t in state.pending_trades
            if t.status == "pending" and agent_name in t.eligible_acceptors
        ]
        own_pending = [
            t
            for t in state.pending_trades
            if t.status == "pending" and t.proposer == agent_name
        ]

        lines = ["## Trade Proposals"]
        if not pending_for_agent and not own_pending:
            lines.append("\nNo pending trades.")
            return "\n".join(lines)

        if pending_for_agent:
            lines.append("\n### Incoming (you can accept/reject):")
            for t in pending_for_agent:
                offering_str = ", ".join(f"{v} {k}" for k, v in t.offering.items())
                requesting_str = ", ".join(f"{v} {k}" for k, v in t.requesting.items())
                lines.append(
                    f"- Trade ID: {t.id}\n"
                    f"  From: {t.proposer}\n"
                    f"  Offering: {offering_str}\n"
                    f"  Requesting: {requesting_str}\n"
                    f"  Expires at step: {t.expires_at_step}\n"
                    f"  Message: {t.message}"
                )

        if own_pending:
            lines.append("\n### Your outgoing trades (pending):")
            for t in own_pending:
                offering_str = ", ".join(f"{v} {k}" for k, v in t.offering.items())
                requesting_str = ", ".join(f"{v} {k}" for k, v in t.requesting.items())
                lines.append(
                    f"- Trade ID: {t.id} | Offering: {offering_str} | Requesting: {requesting_str}"
                )

        return "\n".join(lines)

    def _build_actions_section(
        self,
        agent_def: AgentDefinition,
        situation: Situation,
    ) -> str:
        available = self._get_available_actions(agent_def, situation)
        lines = ["## Available Actions"]

        lines.append(
            "- **no_action**: Do nothing this turn (you can still send messages)."
        )
        lines.append("- **trade_propose**: Propose a trade to one or more agents.")
        lines.append(
            "- **trade_accept**: Accept a pending trade proposal (provide the trade ID)."
        )
        lines.append(
            "- **trade_reject**: Reject a pending trade proposal (provide the trade ID)."
        )

        for action in available:
            param_text = ""
            if action.parameters:
                params = ", ".join(
                    f"{p.name} ({p.type}): {p.description}" for p in action.parameters
                )
                param_text = f" Parameters: {params}"
            lines.append(f"- **{action.name}**: {action.description}{param_text}")

        return "\n".join(lines)

    def _build_response_instructions(self, situation: Situation) -> str:
        agent_names = [a.name for a in situation.agents]
        channel_names = [c.name for c in situation.communication.channels]
        max_messages = situation.communication.max_messages_per_turn

        return clean_indents(
            f"""
            ## Your Response

            Respond with your chosen action and any messages you want to send.
            You MUST include all of the following fields in your response:

            1. **action_name**: The name of the action you choose (e.g. "no_action", "trade_propose", or any available action name).
            2. **action_parameters**: A dict of parameter values if the action requires them.
            3. **trade_proposal**: If action_name is "trade_propose", include eligible_acceptors, offering, requesting, message, and expires_in_steps.
            4. **trade_response_id**: If action_name is "trade_accept" or "trade_reject", include the trade ID.
            5. **messages**: A list of messages to send. Each message needs either a "channel" (one of: {channel_names}) or a "recipient" (one of: {agent_names}) for DMs, and "content".
            6. **reasoning**: Brief internal reasoning (not shown to others).

            You may choose exactly ONE action per turn. You can send up to {max_messages} messages per turn (channel posts and DMs combined). Any messages beyond this limit will be dropped, so prioritize your most important communications.
            Think carefully about your goals and the current state before acting.
            """
        ).strip()

    # --- Helpers ---

    def _get_accessible_channels(
        self,
        agent_name: str,
        situation: Situation,
    ) -> list[Channel]:
        accessible: list[Channel] = []
        for channel in situation.communication.channels:
            if channel.members == "everyone" or agent_name in channel.members:
                accessible.append(channel)
        return accessible

    def _get_available_actions(
        self,
        agent_def: AgentDefinition,
        situation: Situation,
    ) -> list[ActionDefinition]:
        actions: list[ActionDefinition] = []
        for action in situation.environment.global_actions:
            if (
                action.available_to == "everyone"
                or agent_def.name in action.available_to
            ):
                actions.append(action)
        for action in agent_def.special_actions:
            actions.append(action)
        return actions

    def _can_dm(
        self,
        sender: str,
        recipient: str,
        situation: Situation,
    ) -> bool:
        for a, b in situation.communication.dm_blacklist:
            if (a == sender and b == recipient) or (a == recipient and b == sender):
                return False
        return True

    def _convert_to_agent_action(
        self,
        parsed: LlmActionResponse,
        agent_name: str,
        state: SimulationState,
        situation: Situation,
    ) -> AgentAction:
        messages: list[Message] = []
        for llm_msg in parsed.messages:
            if llm_msg.channel:
                accessible_channels = {
                    ch.name
                    for ch in self._get_accessible_channels(agent_name, situation)
                }
                if llm_msg.channel not in accessible_channels:
                    logger.warning(
                        f"{agent_name} tried to post in inaccessible channel #{llm_msg.channel}"
                    )
                    continue
                channel_def = next(
                    (
                        c
                        for c in situation.communication.channels
                        if c.name == llm_msg.channel
                    ),
                    None,
                )
                if channel_def is None:
                    continue
                if channel_def.members == "everyone":
                    recipients = [a.name for a in situation.agents]
                else:
                    recipients = list(channel_def.members)
                messages.append(
                    Message(
                        step=state.step_number,
                        sender=agent_name,
                        channel=llm_msg.channel,
                        recipients=recipients,
                        content=llm_msg.content,
                    )
                )
            elif llm_msg.recipient:
                if not self._can_dm(agent_name, llm_msg.recipient, situation):
                    logger.warning(
                        f"{agent_name} tried to DM {llm_msg.recipient} but it's blacklisted"
                    )
                    continue
                messages.append(
                    Message(
                        step=state.step_number,
                        sender=agent_name,
                        channel=None,
                        recipients=[llm_msg.recipient, agent_name],
                        content=llm_msg.content,
                    )
                )

        max_messages = situation.communication.max_messages_per_turn
        if len(messages) > max_messages:
            logger.warning(
                f"{agent_name} sent {len(messages)} messages but cap is "
                f"{max_messages}. Truncating to first {max_messages}."
            )
            messages = messages[:max_messages]

        trade_proposal = None
        if parsed.action_name == "trade_propose" and parsed.trade_proposal:
            tp = parsed.trade_proposal
            trade_proposal = TradeProposal(
                proposer=agent_name,
                eligible_acceptors=tp.eligible_acceptors,
                offering=tp.offering,
                requesting=tp.requesting,
                message=tp.message,
                proposed_at_step=state.step_number,
                expires_at_step=state.step_number + tp.expires_in_steps,
            )

        return AgentAction(
            agent_name=agent_name,
            action_name=parsed.action_name,
            parameters=parsed.action_parameters,
            messages_to_send=messages,
            trade_proposal=trade_proposal,
            trade_acceptance_id=(
                parsed.trade_response_id
                if parsed.action_name in ("trade_accept", "trade_reject")
                else None
            ),
        )
