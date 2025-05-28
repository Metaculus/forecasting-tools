import asyncio
import logging

from forecasting_tools.agents_and_tools.misc_tools import perplexity_pro_search
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AiAgent,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.benchmarking.control_group_prompt import (
    ControlGroupPrompt,
)
from forecasting_tools.benchmarking.prompt_data_models import (
    OptimizationResult,
    PromptConfig,
    PromptIdea,
)
from forecasting_tools.benchmarking.prompt_evaluator import PromptEvaluator
from forecasting_tools.forecast_helpers.structure_output import (
    structure_output,
)

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """
    A class to optimize a prompt for a given set of evaluation questions.
    """

    def __init__(
        self,
        num_prompts_to_try: int,
        forecast_llm: GeneralLlm,
        ideation_llm_name: str,
        evaluator: PromptEvaluator,
    ) -> None:
        self.num_prompts_to_try = num_prompts_to_try
        self.forecast_llm = forecast_llm
        self.ideation_llm_name = ideation_llm_name
        self.evaluator = evaluator

    async def create_optimized_prompt(
        self,
    ) -> OptimizationResult:
        ideas = await self._get_prompt_ideas()
        tasks = [self._prompt_idea_to_prompt_string(idea) for idea in ideas]
        prompt_templates = await asyncio.gather(*tasks)

        control_group_config = PromptConfig(
            prompt_template=ControlGroupPrompt.get_prompt(),
            llm=self.forecast_llm,
            original_idea=PromptIdea(
                short_name=f"Control Group v{ControlGroupPrompt.version()}",
                idea="The control group is a group of questions that are not optimized for the prompt. It is used to evaluate the performance of the optimized prompt.",
            ),
        )

        configs = [control_group_config]
        for prompt, idea in zip(prompt_templates, ideas):
            configs.append(
                PromptConfig(
                    prompt_template=prompt,
                    llm=self.forecast_llm,
                    original_idea=idea,
                )
            )
        evaluation_result = await self.evaluator.evaluate_prompts(configs)
        evaluated_prompts = evaluation_result.evaluated_prompts
        sorted_evaluated_prompts = sorted(
            evaluated_prompts, key=lambda x: x.score, reverse=True
        )
        return OptimizationResult(evaluated_prompts=sorted_evaluated_prompts)

    async def _get_prompt_ideas(self) -> list[PromptIdea]:
        agent = AiAgent(
            name="Prompt Ideator",
            model=AgentSdkLlm(self.ideation_llm_name),
            instructions=clean_indents(
                f"""
                You are creating instructions for an AI forecaster to forecast binary questions about the future.

                Please come up with {self.num_prompts_to_try} prompt ideas that asks a bot to forecast binary questions.
                There must be a final binary float given at the end, make sure to request for this.

                Consider general forecasting principles that are used in superforecasting.
                Give your ideas in a list format.

                If you need to, run up to 10 searches finding unique ways to approach the prompt.

                For instance a prompt that focuses on base rates, fermi estimates, scope sensitivity, etc.
                You should probably include a number of these at once.

                For instance one prompt might include consider all of the following:
                - The time left until the outcome to the question is known.
                - The status quo outcome if nothing changed.
                - A brief description of a scenario that results in a No or Yes outcome.
                - base rates
                - other estimation techniques

                Consider different formats of asking the question (which order, what items should influence what other items)?

                Return a list of ideas in the format:
                **1-6 word title**
                Idea Process 1

                **1-6 word title**
                Idea Process 2

                **1-6 word title**
                Idea Process 3
                """
            ),
            tools=[perplexity_pro_search],
        )
        output = await AgentRunner.run(
            agent, f"Please generate {self.num_prompts_to_try} prompt ideas"
        )
        ideas = await structure_output(output.final_output, list[PromptIdea])
        logger.info(f"Generated {len(ideas)} prompt ideas: {ideas}")
        return ideas

    async def _prompt_idea_to_prompt_string(
        self, prompt_idea: PromptIdea
    ) -> str:
        agent = AiAgent(
            name="Prompt Implementor",
            model=AgentSdkLlm(self.ideation_llm_name),
            instructions=clean_indents(
                f"""
                You need to implement a prompt that asks a bot to forecast binary questions.
                There must be a final binary float given at the end, make sure to request for this.

                The prompt should implement the below idea:
                Name: {prompt_idea.short_name}
                Idea: {prompt_idea.idea}

                This is a template prompt, and so you should add the following variables to the prompt:
                {{question_text}}
                {{background_info}}
                {{resolution_criteria}}
                {{fine_print}}
                {{today}}
                {{research}}

                """
            ),
        )
        output = await AgentRunner.run(
            agent,
            "Please implement a prompt. Return the prompt and nothing but the prompt. The prompt will be run as is",
        )
        prompt = output.final_output
        logger.info(f"Generated prompt:\n{prompt}")
        return prompt
