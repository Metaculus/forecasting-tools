from __future__ import annotations

import logging

from forecasting_tools.agents_and_tools.situation_simulator.data_models import Situation
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)

GENERATION_TIMEOUT = 300

SITUATION_SCHEMA_GUIDE = clean_indents(
    """
    A Situation JSON defines a multi-agent simulation. Here is the schema:

    {
      "name": "string - Name of the simulation",
      "description": "string - Brief description",
      "rules_text": "string - Natural language rules all agents see. Use this for complex logic that agents reason about themselves. Only use structured effects/actions for things that need mathematical enforcement (dice, hidden computations, resource transfers).",
      "items": [
        {"name": "string", "description": "string", "tradable": true/false}
      ],
      "agents": [
        {
          "name": "string",
          "persona": [
            {"key": "string", "value": "string", "hidden": true/false}
          ],
          "starting_inventory": {"item_name": quantity},
          "special_actions": [
            {
              "name": "string",
              "description": "string",
              "parameters": [{"name": "string", "description": "string", "type": "str|int|float|agent_name|item_name"}],
              "effects": [<Effect objects>],
              "available_to": ["agent_name1"] or "everyone"
            }
          ],
          "inventory_rules": [
            {
              "name": "string",
              "description": "string",
              "conditions": [{"item_name": "string", "operator": ">=|<=|==|>|<|!=", "threshold": int}],
              "effects": [<Effect objects>]
            }
          ],
          "ai_model": "openrouter/anthropic/claude-sonnet-4.5"
        }
      ],
      "environment": {
        "description": "string",
        "inventory": {"item_name": quantity},
        "global_actions": [<ActionDefinition objects>],
        "inventory_rules": [<InventoryRule objects>]
      },
      "communication": {
        "channels": [
          {"name": "string", "members": ["agent1", "agent2"] or "everyone", "description": "string"}
        ],
        "dm_blacklist": [["agent1", "agent2"]],
        "max_messages_per_turn": 7
      },
      "max_steps": 50
    }

    Effect types:
    - {"type": "add_item", "target": "actor|environment|agent_name", "item_name": "string", "quantity": int_or_param_ref}
    - {"type": "remove_item", "target": "actor|environment|agent_name", "item_name": "string", "quantity": int_or_param_ref}
    - {"type": "transfer_item", "source": "actor|environment|agent_name", "target": "actor|environment|agent_name", "item_name": "string", "quantity": int_or_param_ref}
    - {"type": "random_outcome", "outcomes": [{"probability": 0.0-1.0, "effects": [<Effect>], "description": "string"}]}
    - {"type": "message", "target": "actor", "message_text": "string"}

    Parameter references: Use "{param_name}" in quantity or item_name to reference action parameters.
    """
).strip()

SITUATION_DESIGN_GUIDE = clean_indents(
    """
    ## Schema Design Principles

    - Use rules_text for complex game logic that agents can reason about (social deduction, negotiation strategy, win conditions). Only use structured actions/effects for things that need mathematical enforcement (randomness, hidden calculations, resource transfers).
    - Items are versatile: use them for currency ("gold_coin"), votes ("ballot"), health ("health_point"), status ("alive_status"), location ("location_castle"), military strength ("soldier"), unique possessions ("the_crown", "greenland"), etc.
    - Inventory rules fire at end of each step. Use conditions to gate them (e.g. auto-convert resources when threshold is met, trigger cascading consequences when a threshold is crossed).
    - Hidden metadata is powerful: use it for secret roles, hidden goals, private information, secret weaknesses.
    - DM blacklists prevent agents from privately coordinating when that would break the simulation.

    ## Core Design Requirements

    ### Scale and Agent Diversity
    - Always create at least 6 agents. More agents enable richer social dynamics.
    - Every agent must have a distinct personality, background, goals, and capabilities.
    - Give agents different specialties and resource advantages — not everyone can do everything equally.
    - Not every agent should be cooperative. Include self-interested, deceptive, unreliable, or antagonistic agents to create realistic social friction.
    - Agents should want different things. Avoid designs where everyone has the same objective.

    ### Dynamism and Replayability
    - Use random_outcome effects liberally so every playthrough produces different results.
    - Include enough hidden information and agent variety that outcomes are never predetermined.
    - Avoid designs where there is one dominant strategy. Create situations with many viable approaches and many "correct answers". The simulation should not be gameable.
    - Include trade-offs: every strategic choice should have both costs and benefits.

    ### News Feeds and Public Information Flow
    - Create dedicated news/announcement channels where designated agents post public updates about world events.
    - Consider a narrator or moderator agent (e.g., "Fate", "News Anchor", "Game Master") whose role is to broadcast events, outcomes, and external changes. Give only this agent (and possibly a select few) permission to post in the news channel. Other agents are told in rules_text to treat these announcements as ground truth.
    - Use the "message" effect type to deliver private notifications about action outcomes.
    - Also consider channels for specific subgroups (faction chat, committee channels, regional forums).

    ### Communication Bandwidth
    - Agents have a cap on how many messages (channel posts + DMs combined) they can send per turn, controlled by max_messages_per_turn (default: 7). Tune this for the situation.
    - A lower cap forces agents to prioritize who they talk to and what they say — use low values (2-3) for situations where communication is costly or limited (e.g., wartime, isolation scenarios, high-pressure negotiations).
    - A higher cap (6-15) suits situations with rich public discourse (e.g., parliamentary debates, open marketplaces).
    - This creates meaningful strategic choices about communication: who do you update, who do you lobby, who do you ignore?

    ### Hidden Information and Asymmetric Knowledge
    - Use hidden persona metadata extensively for secret roles, private goals, sensitive knowledge, and secret weaknesses/vulnerabilities.
    - Create private channels for factions/alliances to coordinate secretly.
    - Some agents should have information access others don't (e.g., an intelligence broker, a spy, an insider).
    - Design information that is valuable to trade or strategically reveal.

    ### Resource Constraints and Economics
    - Include limited resources that agents must compete for, trade, or manage.
    - Use environment inventory for shared/global resource pools that deplete over time.
    - Create resource interdependencies where different agents produce different things, encouraging trade and cooperation.
    - Include costs for actions to force meaningful resource allocation decisions.
    - Resources should be scarce enough that agents can't do everything they want — they must prioritize.

    ### Objectives — Both Hard and Soft
    - Every agent must have measurable objectives trackable through items/inventory (e.g., "accumulate the most gold", "survive until the end", "win the vote").
    - Also give agents soft qualitative objectives in their persona that aren't mechanically enforced (e.g., "maintain your reputation as trustworthy", "protect the vulnerable", "be seen as a leader", "avoid being seen as ruthless", "preserve your friendship with Agent X"). These should feel like real human motivations, not just game-win conditions.
    - Mix competitive and cooperative objectives. Some agents should benefit from helping each other.

    ### Situation Authenticity and Complexity
    - Model the essential dynamics of the real-world situation as accurately and closely as possible. Think deeply about what makes this type of situation tick.
    - If the situation is inherently zero-sum (elections, wars, single promotions), make it zero-sum. If it has positive-sum potential (business ecosystems, community building), enable genuine cooperation.
    - Include realistic external pressures and random events (market crashes, natural disasters, political shifts, supply chain disruptions). Use a master/narrator agent with random_outcome actions to generate these.
    - Make the rules complex, detailed, and rich. Real situations have many interacting factors. Don't oversimplify.
    - Include realistic constraints: deadlines, regulations, physical limitations, social norms.

    ## Advanced Design Strategies

    ### Master/Narrator Agent for External Events
    - Create a special agent (e.g., "Game Master", "Fate", "World Events") that uses random_outcome actions to determine external events like natural disasters, market shifts, discoveries, epidemics, or political changes.
    - This agent should narrate outcomes in a public news channel. Other agents are told in rules_text to accept this agent's announcements as ground truth.
    - Give this agent special actions: deal_damage, trigger_disaster, announce_market_change, grant_discovery, etc. — each using random_outcome to determine what happens.
    - This agent should NOT compete with other agents. Its role is purely to create a dynamic, unpredictable world.

    ### Items for Status, Position, and Power
    - Track agent status via items: alive_status, health_point, influence_level, morale, reputation_score, wounded_status.
    - Represent physical position via items (e.g., "location_castle", "location_market_square"). Create actions to move between locations. Use rules_text to explain that certain actions are only meaningful at certain locations.
    - Use single-instance non-tradable items for unique possessions, territories, or titles (e.g., "greenland", "the_crown", "master_key", "abigails_house", "CEO_title"). These create uniqueness and strategic value.
    - Represent military/political power as countable items: soldier, warship, spy_agent, political_capital, vote_bloc.

    ### Simulating Conflict and Competition
    - Represent military units, political capital, or competitive resources as items (soldier, militia, campaign_fund, market_share).
    - Create actions that resolve conflicts using random_outcome, where probabilities can be weighted by relative strength (e.g., more soldiers = higher chance of winning).
    - Use transfer_item to represent conquest, resource seizure, or competitive displacement.
    - Track casualties and losses via remove_item effects.

    ### Phases, Escalation, and Temporal Pressure
    - Use rules_text to define distinct phases (e.g., "Steps 1-5: Negotiation Phase. Steps 6-8: Action Phase. Steps 9-10: Resolution Phase").
    - Make different actions meaningful in different phases. Create urgency by having key decisions happen on specific steps.
    - Increase pressure over time: shrinking global resources, approaching deadlines, rising stakes, external threats escalating.
    - Include tipping points where the nature of the situation changes (e.g., "If war is declared, all trade between factions stops").

    ### Irreversible Decisions and Consequences
    - Design some actions as one-way doors: declaring war, making public accusations, spending unique items, revealing hidden information. These raise the stakes of decision-making.
    - Use inventory rules to create cascading consequences (e.g., "If health reaches 0, alive_status is removed" or "If treasury reaches 0, soldiers start deserting").
    - Actions should sometimes affect uninvolved third parties (externalities), preventing the simulation from decomposing into isolated pairwise interactions.

    ### Power Asymmetry Beyond Resources
    - Give some agents structural power: veto rights, agenda-setting ability, ability to call votes, ability to grant or revoke privileges.
    - Create information brokers who start with knowledge others need.
    - Design roles with unique abilities that cannot be replicated by other agents.
    - Balance structural power with vulnerabilities — a powerful agent should also have weaknesses or dependencies on others.

    ### Balancing Mechanics to Prevent Degenerate Outcomes
    - Include catch-up mechanics or diminishing returns to prevent one agent from running away with the game early.
    - Create natural checks and balances: powerful agents should have vulnerabilities, dominant strategies should have counters.
    - Ensure no single agent can achieve their objectives without interacting meaningfully with others.
    - Design situations where ignoring other agents is a losing strategy — interdependence drives engagement.
    """
).strip()


class SituationGenerator:
    def __init__(
        self,
        model: str = "openrouter/anthropic/claude-sonnet-4.5",
    ) -> None:
        self.model = model

    async def generate(self, prompt: str) -> Situation:
        logger.info(f"Generating situation from prompt: {prompt[:100]}...")

        system_prompt = clean_indents(
            f"""
            You are an expert simulation designer. Given a user's description, create a
            rich, complex, and realistic Situation JSON for a multi-agent simulation.

            Your goal is to create a simulation that is deeply engaging, unpredictable,
            and faithful to the real-world dynamics of the described scenario. Follow
            the design guide below carefully — every principle matters.

            {SITUATION_DESIGN_GUIDE}

            Use the following schema reference to build valid JSON:

            {SITUATION_SCHEMA_GUIDE}

            Return ONLY valid JSON. No markdown fences, no explanation.
            """
        ).strip()

        llm = GeneralLlm(self.model, temperature=0.8, timeout=GENERATION_TIMEOUT)
        situation = await llm.invoke_and_return_verified_type(
            f"{system_prompt}\n\n---\n\nTask:\n{prompt}",
            Situation,
            allowed_invoke_tries_for_failed_output=3,
        )

        logger.info(
            f"Generated situation '{situation.name}' with "
            f"{len(situation.agents)} agents and {len(situation.items)} items"
        )
        return situation

    def _clean_json_response(self, response: str) -> str:
        response = response.strip()
        if response.startswith("```json"):
            response = response[len("```json") :]
        if response.startswith("```"):
            response = response[len("```") :]
        if response.endswith("```"):
            response = response[: -len("```")]
        return response.strip()
