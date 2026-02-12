from __future__ import annotations

import logging

from forecasting_tools.agents_and_tools.situation_simulator.data_models import (
    AgentDefinition,
    SimulationState,
    Situation,
)
from forecasting_tools.agents_and_tools.situation_simulator.intervention_testing.data_models import (
    PolicyAgentResult,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)

POLICY_AGENT_TIMEOUT = 300


class InterventionPolicyAgent:
    def __init__(
        self,
        model_name: str = "openrouter/anthropic/claude-sonnet-4",
        timeout: int = POLICY_AGENT_TIMEOUT,
        structure_output_model: GeneralLlm | None = None,
    ) -> None:
        self.model_name = model_name
        self.timeout = timeout
        self.structure_output_model = structure_output_model or GeneralLlm(
            "openrouter/openai/gpt-4.1-mini",
            temperature=0.2,
            timeout=self.timeout,
        )

    async def analyze_and_propose(
        self,
        situation: Situation,
        state: SimulationState,
        target_agent: AgentDefinition,
    ) -> PolicyAgentResult:
        logger.info(
            f"Policy agent analyzing situation '{situation.name}' "
            f"for agent '{target_agent.name}' at step {state.step_number}"
        )
        prompt = self._build_prompt(situation, state, target_agent)
        llm = GeneralLlm(
            self.model_name,
            temperature=0.7,
            timeout=self.timeout,
        )
        raw_output = await llm.invoke(prompt)
        logger.info("Extracting policy agent result from LLM output")
        result = await self._extract_result(raw_output)
        logger.info(
            f"Policy agent produced {len(result.forecasts)} forecasts "
            f"and intervention: {result.intervention_description[:100]}..."
        )
        return result

    async def _extract_result(self, raw_output: str) -> PolicyAgentResult:
        extraction_instructions = clean_indents(
            """
            Extract the policy analysis from the agent's output.

            You must extract:
            1. agent_goals_analysis: The analysis of the target agent's goals and
               current situation (Phase 1 output)
            2. evaluation_criteria: The list of criteria as strings (Phase 2 output)
            3. intervention_description: The specific intervention/policy proposal
               that the agent should follow (Phase 4 output). This should be the
               actionable instructions, not the full proposal markdown.
            4. policy_proposal_markdown: The full policy proposal markdown including
               analysis, recommendations, and reasoning (Phase 4 output)
            5. forecasts: ALL forecasts from both baseline and conditional sections.
               For each forecast extract:
               - question_title: Short title
               - question_text: Full question
               - resolution_criteria: How it resolves
               - prediction: The probability as a float between 0.0 and 1.0
                 (e.g., "35%" becomes 0.35)
               - reasoning: The reasoning explanation
               - is_conditional: False for status quo (Phase 3) forecasts,
                 True for intervention (Phase 5) forecasts
               - category: "hard_metric" for inventory-based questions,
                 "qualitative" for qualitative event questions
               - hard_metric_criteria: For hard_metric forecasts ONLY, extract:
                 - agent_name: The agent whose inventory is being checked
                 - item_name: The inventory item name (must match item definitions)
                 - operator: The comparison operator (>=, >, <=, <, ==)
                 - threshold: The numeric threshold value

            Be thorough in extracting ALL forecasts. There should be at least 16
            total: 3 hard metric baseline + 5 qualitative baseline + 3 hard metric
            conditional + 5 qualitative conditional.
            """
        )
        return await structure_output(
            raw_output,
            PolicyAgentResult,
            model=self.structure_output_model,
            additional_instructions=extraction_instructions,
        )

    def _build_prompt(
        self,
        situation: Situation,
        state: SimulationState,
        target_agent: AgentDefinition,
    ) -> str:
        situation_context = self._build_situation_context(situation)
        agent_context = self._build_agent_context(situation, target_agent)
        state_context = self._build_state_context(situation, state, target_agent)
        item_definitions = self._build_item_definitions(situation)
        remaining_steps = situation.max_steps - state.step_number

        return clean_indents(
            f"""
            # Intervention Policy Agent

            You are a policy analyst tasked with analyzing a simulation in progress
            and proposing an intervention for a specific agent. You will create
            forecasts about outcomes both with and without your intervention.

            ---

            ## Simulation Context

            {situation_context}

            ---

            ## Target Agent: {target_agent.name}

            {agent_context}

            ---

            ## Current Simulation State (Step {state.step_number} of {situation.max_steps})

            {state_context}

            ---

            ## Item Definitions

            {item_definitions}

            ---

            ## Remaining Steps: {remaining_steps}

            The simulation will continue for {remaining_steps} more steps after
            the current state.

            ---

            You must complete ALL FIVE PHASES below in order.

            ## PHASE 1: Analyze Agent Goals

            Carefully analyze the target agent ({target_agent.name}):
            - What are their explicit and implicit goals based on their persona?
            - What is their current position (inventory, relationships, messages)?
            - What strategies have they been pursuing so far?
            - What challenges or opportunities do they face?
            - How are they positioned relative to other agents?

            Write a detailed "## Agent Goals Analysis" section.

            ---

            ## PHASE 2: Evaluation Criteria

            Define 4-6 criteria you will use to evaluate the effectiveness of any
            intervention. These should be measurable outcomes that matter for the
            target agent's success.

            For each criterion:
            - Name it clearly
            - Explain why it matters for this agent
            - Describe how you would measure success

            Write a "## Evaluation Criteria" section.

            ---

            ## PHASE 3: Status Quo Forecasts

            Generate forecasts about what will happen in the REMAINING {remaining_steps}
            steps if NO intervention is made. You must generate EXACTLY:

            ### Hard Metric Forecasts (exactly 3)

            These MUST be about specific inventory items for specific agents being
            above/below specific thresholds at the end of the simulation (step
            {situation.max_steps}).

            Format each as:
            - **Question Title**: e.g., "Marcus gold_coin above 50"
            - **Question**: "Will [agent_name] have [operator] [threshold]
              [item_name] at the end of step {situation.max_steps}?"
            - **Resolution Criteria**: "Resolves YES if [agent_name]'s inventory
              shows [item_name] [operator] [threshold] at step {situation.max_steps}"
            - **Agent Name**: The exact agent name from the simulation
            - **Item Name**: The exact item name from the item definitions
            - **Operator**: One of >=, >, <=, <, ==
            - **Threshold**: A specific integer value
            - **Prediction**: Your probability (e.g., "65%")
            - **Reasoning**: 3+ sentences explaining your forecast

            Choose items and thresholds that are interesting and non-trivial to predict.
            Consider current inventory levels, trends, and game mechanics.

            ### Qualitative Forecasts (exactly 5)

            These should be about qualitative events, behaviors, or outcomes in the
            simulation. Examples:
            - "Will [agent] and [agent] engage in a trade?"
            - "Will [agent] change their strategy to focus on [resource]?"
            - "Will there be a public conflict in the marketplace channel?"
            - "Will [agent] attempt to form an alliance?"
            - "Will [agent] accumulate more [resource] than [other agent]?"

            Format each as:
            - **Question Title**: Short descriptive title
            - **Question**: Full question text
            - **Resolution Criteria**: How this would be determined from the
              simulation transcript
            - **Prediction**: Your probability
            - **Reasoning**: 3+ sentences

            Write a "## Status Quo Forecasts" section with all 8 forecasts.

            ---

            ## PHASE 4: Propose Intervention

            Based on your analysis, propose a specific intervention for
            {target_agent.name}. This intervention will be delivered as a direct
            message from an "Intervention Advisor" and the agent MUST follow it.

            Your intervention should:
            - Be specific and actionable
            - Target a meaningful change in the agent's behavior
            - Have a plausible impact on measurable outcomes
            - Be interesting enough that the conditional forecasts will differ
              from baseline

            Write a "## Intervention Proposal" section with:
            - **Intervention Instructions**: The exact message that will be sent
              to the agent (this should be written as direct instructions)
            - **Expected Impact**: What you expect the intervention to change
            - **Rationale**: Why this intervention was chosen

            ---

            ## PHASE 5: Conditional Forecasts (With Intervention)

            Generate forecasts about what will happen in the remaining
            {remaining_steps} steps IF your intervention IS implemented.
            You must generate EXACTLY:

            ### Hard Metric Forecasts (exactly 3)

            Same format as Phase 3, but forecasting outcomes CONDITIONAL on the
            intervention being implemented. Use the SAME agents and items where
            possible to allow direct comparison with baseline forecasts, but adjust
            thresholds and probabilities to reflect the intervention's expected impact.

            ### Qualitative Forecasts (exactly 5)

            Same format as Phase 3, but conditional on the intervention. Some
            questions may be the same as baseline (with different probabilities),
            and some may be new questions specific to the intervention's effects.

            Write a "## Conditional Forecasts" section with all 8 forecasts.

            ---

            # Important Reminders

            - You MUST produce EXACTLY 16 forecasts total: 8 baseline + 8 conditional
            - Hard metric forecasts MUST use exact agent names and item names from
              the simulation
            - All predictions must be probabilities between 0% and 100%
            - Consider the game mechanics, inventory rules, and available actions
              when forecasting
            - Think about second-order effects and strategic responses from other agents
            - Your intervention should create a meaningful difference between the
              baseline and conditional forecasts

            Begin your analysis now. Start with Phase 1.
            """
        )

    def _build_situation_context(self, situation: Situation) -> str:
        return clean_indents(
            f"""
            **Name**: {situation.name}

            **Description**: {situation.description}

            **Rules**:
            {situation.rules_text}

            **Max Steps**: {situation.max_steps}
            """
        )

    def _build_agent_context(
        self,
        situation: Situation,
        target_agent: AgentDefinition,
    ) -> str:
        persona_text = self._format_persona(target_agent)
        other_agents_text = self._format_other_agents(situation, target_agent)
        special_actions_text = self._format_special_actions(target_agent)
        global_actions_text = self._format_global_actions(situation, target_agent)

        return clean_indents(
            f"""
            **Full Persona** (including hidden info):
            {persona_text}

            **Starting Inventory**: {target_agent.starting_inventory}

            **Special Actions**:
            {special_actions_text}

            **Available Global Actions**:
            {global_actions_text}

            **Other Agents**:
            {other_agents_text}
            """
        )

    @staticmethod
    def _format_persona(target_agent: AgentDefinition) -> str:
        if not target_agent.persona:
            return "No persona defined."
        lines = []
        for item in target_agent.persona:
            visibility = " [HIDDEN from others]" if item.hidden else ""
            lines.append(f"- {item.key}: {item.value}{visibility}")
        return "\n".join(lines)

    @staticmethod
    def _format_other_agents(
        situation: Situation, target_agent: AgentDefinition
    ) -> str:
        lines = []
        for agent_def in situation.agents:
            if agent_def.name == target_agent.name:
                continue
            public_metadata = [m for m in agent_def.persona if not m.hidden]
            if public_metadata:
                metadata_text = ", ".join(
                    f"{m.key}: {m.value}" for m in public_metadata
                )
                lines.append(f"- **{agent_def.name}**: {metadata_text}")
            else:
                lines.append(f"- **{agent_def.name}**")
        return "\n".join(lines) if lines else "No other agents."

    @staticmethod
    def _format_special_actions(target_agent: AgentDefinition) -> str:
        if not target_agent.special_actions:
            return "None"
        lines = []
        for action in target_agent.special_actions:
            params = ", ".join(f"{p.name} ({p.type})" for p in action.parameters)
            lines.append(f"- {action.name}: {action.description} [{params}]")
        return "\n".join(lines)

    @staticmethod
    def _format_global_actions(
        situation: Situation, target_agent: AgentDefinition
    ) -> str:
        lines = []
        for action in situation.environment.global_actions:
            if (
                action.available_to == "everyone"
                or target_agent.name in action.available_to
            ):
                params = ", ".join(f"{p.name} ({p.type})" for p in action.parameters)
                lines.append(f"- {action.name}: {action.description} [{params}]")
        return "\n".join(lines) if lines else "None"

    def _build_state_context(
        self,
        _situation: Situation,
        state: SimulationState,
        _target_agent: AgentDefinition,
    ) -> str:
        inventories_text = self._format_inventories(state)
        env_inventory_text = self._format_env_inventory(state)
        messages_text = self._format_recent_messages(state)
        actions_text = self._format_recent_actions(state)

        return clean_indents(
            f"""
            ### All Agent Inventories

            {inventories_text}

            ### Environment Inventory

            {env_inventory_text}

            ### Recent Messages (last 30)

            {messages_text}

            ### Recent Actions (last 20)

            {actions_text}
            """
        )

    @staticmethod
    def _format_inventories(state: SimulationState) -> str:
        lines = []
        for agent_name, inventory in state.inventories.items():
            items_text = ", ".join(f"{k}: {v}" for k, v in inventory.items())
            lines.append(f"- **{agent_name}**: {items_text}")
        return "\n".join(lines)

    @staticmethod
    def _format_env_inventory(state: SimulationState) -> str:
        text = ", ".join(f"{k}: {v}" for k, v in state.environment_inventory.items())
        return text if text else "Empty"

    @staticmethod
    def _format_recent_messages(state: SimulationState) -> str:
        recent = state.message_history[-30:]
        if not recent:
            return "No messages yet."
        lines = []
        for msg in recent:
            if msg.channel:
                lines.append(
                    f"[Step {msg.step}] #{msg.channel} | {msg.sender}: {msg.content}"
                )
            else:
                recipients = [r for r in msg.recipients if r != msg.sender]
                dm_target = recipients[0] if recipients else "unknown"
                lines.append(
                    f"[Step {msg.step}] DM {msg.sender} -> {dm_target}: {msg.content}"
                )
        return "\n".join(lines)

    @staticmethod
    def _format_recent_actions(state: SimulationState) -> str:
        recent = state.action_log[-20:]
        if not recent:
            return "No actions yet."
        lines = []
        for action in recent:
            if action.action_name == "no_action":
                lines.append(f"- {action.agent_name}: no_action")
            else:
                params_text = ", ".join(
                    f"{k}={v}" for k, v in action.parameters.items()
                )
                lines.append(
                    f"- {action.agent_name}: {action.action_name}({params_text})"
                )
        return "\n".join(lines)

    def _build_item_definitions(self, situation: Situation) -> str:
        if not situation.items:
            return "No items defined."
        lines = []
        for item in situation.items:
            tradable = "tradable" if item.tradable else "not tradable"
            lines.append(f"- **{item.name}**: {item.description} ({tradable})")
        return "\n".join(lines)
