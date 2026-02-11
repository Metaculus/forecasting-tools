from __future__ import annotations

import logging

from forecasting_tools.agents_and_tools.situation_simulator.data_models import Situation
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.structure_output import structure_output
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
          "ai_model": "openrouter/anthropic/claude-sonnet-4"
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
        "dm_blacklist": [["agent1", "agent2"]]
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

    Design principles:
    - Use rules_text for complex game logic that agents can reason about (social deduction, negotiation strategy, win conditions)
    - Only use structured actions/effects for mathematical enforcement (randomness, hidden calculations, resource transfers)
    - Items are versatile: use them for currency ("gold_coin"), votes ("ballot"), health ("health_point"), etc.
    - Inventory rules fire at end of each step. Use conditions to gate them (e.g. auto-convert resources when threshold is met)
    - Hidden metadata is powerful: use it for secret roles, hidden goals, private information
    - DM blacklists prevent agents from privately coordinating when that would break the game
    """
).strip()


class SituationGenerator:
    def __init__(
        self,
        model: str = "openrouter/anthropic/claude-sonnet-4",
    ) -> None:
        self.model = model

    async def generate(self, prompt: str) -> Situation:
        logger.info(f"Generating situation from prompt: {prompt[:100]}...")

        system_prompt = clean_indents(
            f"""
            You are a simulation designer. Given a user's description, create a detailed
            Situation JSON for a multi-agent simulation.

            Make the simulation as realistic and engaging as possible. Include:
            - Interesting agent personas with distinct goals and personalities
            - Meaningful hidden information that creates asymmetric gameplay
            - Channels that reflect natural communication groupings
            - Items and actions that create interesting strategic choices
            - Inventory rules for automated game mechanics

            {SITUATION_SCHEMA_GUIDE}

            Return ONLY valid JSON. No markdown fences, no explanation.
            """
        ).strip()

        llm = GeneralLlm(self.model, temperature=0.8, timeout=GENERATION_TIMEOUT)
        raw_response = await llm.invoke(prompt, system_prompt=system_prompt)

        cleaned_response = self._clean_json_response(raw_response)

        situation = await structure_output(
            cleaned_response,
            Situation,
            additional_instructions="Parse this JSON into a Situation object. Preserve all fields exactly.",
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
