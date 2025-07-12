from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

from pydantic import BaseModel, Field

from forecasting_tools.agents_and_tools.minor_tools import (
    perplexity_reasoning_pro_search,
)
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AiAgent,
    general_trace_or_span,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import (
    clean_indents,
    retry_async_function,
)
from forecasting_tools.auto_optimizers.prompt_data_models import PromptIdea
from forecasting_tools.helpers.structure_output import structure_output

logger = logging.getLogger(__name__)


class ImplementedPrompt(BaseModel):
    text: str
    idea: PromptIdea
    originating_ideas: list[PromptIdea]
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptScore(BaseModel):
    value: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredPrompt(BaseModel):
    score: PromptScore
    prompt: ImplementedPrompt


class OptimizationRun(BaseModel):
    scored_prompts: list[ScoredPrompt]

    @property
    def best_prompt(self) -> ScoredPrompt:
        return max(self.scored_prompts, key=lambda x: x.score.value)


class PromptOptimizer:
    """A genetic algorithm-inspired prompt optimizer that evolves prompts through mutation, breeding, and selection.

    Args:
        initial_prompt: Starting prompt string to optimize from
        iterations: Number of optimization iterations to run
        ideation_llm_name: Name of the LLM model to use for generating prompt variations
        prompts_to_scores_func: async function that takes a list of prompts and returns a list of scores
        prompt_purpose_explanation: e.g. "You are making a prompt for an AI to forecast binary questions about future events. You want to optimize ..."
        prompt_requirements_explanation: e.g. "The prompt should 1) ask an AI to forecast a binary question, 2) ... \\n You have access to tools x,y,z with limits a, b, c "
        template_variables_explanation: e.g. "The prompt should include the following variables: {{question_text}}, {{background_info}}, {{resolution_criteria}}, {{fine_print}}, {{today}}, {{research}}. Always include at least one of each of these and only include research once"
        mutation_considerations: e.g. "As you develop the prompt consider: - Research formats and length (consider all varieties) - Research sources (consider all varieties) - Reasoning formats and length (consider all varieties)"
        format_scores_func: Async function that takes a list of scored prompts and returns a string of the scores and metadata. This is inserted into the prompt during prompt ideation.
        validate_prompt_func: Async function that takes a prompt and raises an exception if the prompt is invalid.
        initial_prompt_population_size: Target size of the initial prompt population
        survivors_per_iteration: Number of best prompts to keep after each iteration
        mutated_prompts_per_survivor: Number of mutated prompts to generate from each survivor
        breeded_prompts_per_iteration: Number of bred prompts to generate per iteration (all best survivors are considered when creating crossover ideas)
    """

    def __init__(
        self,  # NOSONAR
        initial_prompt: str,
        iterations: int,
        ideation_llm_name: str,
        prompt_purpose_explanation: str,
        prompt_requirements_explanation: str,
        template_variables_explanation: str,
        mutation_considerations: str,
        prompts_to_scores_func: Callable[
            [list[ImplementedPrompt]], Coroutine[Any, Any, list[PromptScore]]
        ],
        format_scores_func: Callable[[ScoredPrompt], Coroutine[Any, Any, str]] | None,
        validate_prompt_func: (
            Callable[[ImplementedPrompt], Coroutine[Any, Any, None]] | None
        ),
        initial_prompt_population_size: int = 25,
        survivors_per_iteration: int = 5,
        mutated_prompts_per_survivor: int = 4,
        breeded_prompts_per_iteration: int = 5,
    ) -> None:
        self.initial_prompt = initial_prompt
        self.iterations = iterations
        self.ideation_llm_name = ideation_llm_name
        self.prompt_to_scores_func = prompts_to_scores_func
        self.format_scores_func = format_scores_func
        self.validate_prompt_func = validate_prompt_func
        self.prompt_purpose_explanation = prompt_purpose_explanation
        self.prompt_requirements_explanation = prompt_requirements_explanation
        self.template_variables_explanation = template_variables_explanation
        self.mutation_considerations = mutation_considerations
        self.initial_prompt_population_size = initial_prompt_population_size
        self.survivors_per_iteration = survivors_per_iteration
        self.mutated_prompts_per_survivor = mutated_prompts_per_survivor
        self.breeded_prompts_per_iteration = breeded_prompts_per_iteration

        if (
            self.mutated_prompts_per_survivor == 0
            and self.breeded_prompts_per_iteration == 0
        ):
            raise ValueError(
                "At least one of mutated_prompts_per_surviving_prompt or breeded_prompts_per_iteration must be greater than 0"
            )

    async def create_optimized_prompt(self) -> OptimizationRun:
        with general_trace_or_span("Prompt Optimizer"):
            return await self._create_optimized_prompt()

    async def _create_optimized_prompt(self) -> OptimizationRun:
        iteration_num = 0
        with general_trace_or_span("Initial Population Generation (Iteration 1)"):
            iteration_num += 1
            logger.info(
                f"Generating initial prompt population of size {self.initial_prompt_population_size}"
            )
            seed_prompt = ImplementedPrompt(
                text=self.initial_prompt,
                idea=PromptIdea(
                    short_name="Initial Seed",
                    full_text="The user-provided initial prompt",
                ),
                originating_ideas=[],
            )
            starting_prompts: list[ImplementedPrompt] = [seed_prompt]
            prompts_still_needed = self.initial_prompt_population_size - len(
                starting_prompts
            )
            if prompts_still_needed > 0:
                additional_initial_prompts = await self._mutate_prompt(
                    starting_prompts[0],
                    prompts_still_needed,
                )
                starting_prompts.extend(additional_initial_prompts)

            offspring_prompts: list[ImplementedPrompt] = starting_prompts
            assert (
                seed_prompt in offspring_prompts
            ), "Seed prompt not found in offspring prompts"
            all_evaluated_prompts: list[ScoredPrompt] = (
                await self._evaluate_new_members(offspring_prompts)
            )
            survivors = await self._kill_the_weak(all_evaluated_prompts)

        while iteration_num < self.iterations:
            iteration_num += 1
            with general_trace_or_span(
                f"Prompt Optimizer Iteration {iteration_num + 1}",
                data={"survivors": [s.model_dump() for s in survivors]},
            ):
                logger.info(
                    f"Starting iteration {iteration_num + 1}/{self.iterations} - Current population size: {len(offspring_prompts)}"
                )

                offspring_prompts = await self._generate_new_prompts(survivors)

                evaluated_prompts = await self._evaluate_new_members(offspring_prompts)
                all_evaluated_prompts.extend(evaluated_prompts)
                updated_population = survivors + evaluated_prompts

                survivors = await self._kill_the_weak(updated_population)

                self._log_duplicate_prompts(all_evaluated_prompts)

        return OptimizationRun(scored_prompts=all_evaluated_prompts)

    async def _kill_the_weak(
        self, current_population: list[ScoredPrompt]
    ) -> list[ScoredPrompt]:
        current_population.sort(
            key=lambda x: x.score.value,
            reverse=True,
        )
        logger.debug(f"Current survivors: {current_population}")
        best_survivor = current_population[0]
        logger.info(
            f"Best survivor: {best_survivor.prompt.idea.short_name} with score {best_survivor.score.value:.4f}"
        )
        return current_population[: self.survivors_per_iteration]

    async def _generate_new_prompts(
        self, surviving_population: list[ScoredPrompt]
    ) -> list[ImplementedPrompt]:
        mutated_prompts: list[ImplementedPrompt] = []
        mutation_tasks = [
            self._mutate_prompt(ep.prompt, self.mutated_prompts_per_survivor)
            for ep in surviving_population
        ]
        mutation_results = await asyncio.gather(*mutation_tasks)
        for mutation_list in mutation_results:
            mutated_prompts.extend(mutation_list)
        logger.info(f"Generated {len(mutated_prompts)} mutated prompts.")

        bred_prompts: list[ImplementedPrompt] = []
        bred_prompts = await self._breed_prompts(surviving_population)

        logger.info(f"Generated {len(bred_prompts)} bred prompts.")

        new_prompts = mutated_prompts + bred_prompts
        return new_prompts

    async def _evaluate_new_members(
        self, prompts: list[ImplementedPrompt]
    ) -> list[ScoredPrompt]:
        scores = await self.prompt_to_scores_func(prompts)
        return [ScoredPrompt(prompt=p, score=s) for p, s in zip(prompts, scores)]

    @retry_async_function(tries=3)
    async def _mutate_prompt(
        self,
        input_prompt: ScoredPrompt | ImplementedPrompt,
        num_mutations_to_generate: int,
    ) -> list[ImplementedPrompt]:
        if isinstance(input_prompt, ImplementedPrompt):
            scores_str = ""
            prompt = input_prompt
        else:
            if self.format_scores_func is not None:
                scores_str = await self.format_scores_func(input_prompt)
            else:
                scores_str = ""
            prompt = input_prompt.prompt

        agent_mutate_ideas = AiAgent(
            name="Prompt Mutator Ideator",
            model=AgentSdkLlm(self.ideation_llm_name),
            instructions=clean_indents(
                f"""
                You are an expert prompt engineer. Your task is to generate {num_mutations_to_generate} new PROMPT IDEAS by mutating an existing prompt.
                Your ideas are being used to optimize this prompt using a Genetic Algorithm inspired approach.
                We are highlighting exploration over exploitation, but do want to strike a balance.
                Do not number the ideas (the idea title will be extracted and shared directly)

                # Purpose of Prompt
                {self.prompt_purpose_explanation}

                # Prompt Requirements
                The final prompt (i.e. not the ideas you will generate) will have the below requirements. Another agent will implement these requirements.

                {self.prompt_requirements_explanation}

                # Instructions
                1. Please analyze the scores from the previous prompt and identify what went wrong. You can run 3-5 searches to run this analysis.
                2. Run 3-10 searches on the web to find inspiration for novel prompt structures, techniques, and ideas that will solve the goal.
                3. Generate {num_mutations_to_generate} new, distinct PROMPT IDEAS based on the original.

                Please generate exactly {num_mutations_to_generate} new, distinct PROMPT IDEAS based on the original.
                Each mutation idea must be a concept for a new, complete prompt.

                For each idea please sequentially follow these policies to determine how much you try to mutate the original prompt:
                1st idea: "Normal variation, which should take a generally different approach and be a general rewrite while staying in general theme of the original",
                2nd idea: "Highly Diverse variation, that explores a substantially different structure or set of principles, focus on a completely different idea than in the original. Search until you find something novel.",
                nth idea: ... continue alternating between normal variation and highly diverse variation...

                # Original Prompt Idea Details
                Name: {prompt.idea.short_name}
                Core Idea: {prompt.idea.full_text}

                Original Prompt Template (for context only, do not reproduce it in your output):
                ```
                {prompt.text}
                ```

                {"# Scores from Original Prompt:" if scores_str else ""}
                {scores_str}

                # Ideation Considerations
                {self.mutation_considerations}

                # Format
                Below is the format you should use for your output. Fill in each part with your own words (e.g. don't literally say "Mutated Idea Title 1" in your output)

                ```
                **Mutated Idea Title 1**
                New idea for prompt mutation 1, specifying in detail how to implement the prompt reflecting the target variation. The implementor will not be shown the original prompt, just this idea and the prompt requirements (so don't repeat the requirements in your idea, but otherwise give a detailed overview).

                **Mutated Idea Title 2**
                New idea for prompt mutation 2, specifying in detail how to implement the prompt reflecting the target variation. The implementor will not be shown the original prompt, just this idea and the prompt requirements (so don't repeat the requirements in your idea, but otherwise give a detailed overview).
                ...
                (up to {num_mutations_to_generate} ideas)
                ```

                """
            ),
            tools=[perplexity_reasoning_pro_search],
        )

        mutation_agent_task = (
            f"Generate {num_mutations_to_generate} mutated prompt ideas for the prompt named '{prompt.idea.short_name}'. "
            f"Ensure each mutation aligns with the requested degree of variation."
        )
        output = await AgentRunner.run(agent_mutate_ideas, mutation_agent_task)
        mutated_ideas = await structure_output(output.final_output, list[PromptIdea])
        logger.info(
            f"Successfully structured {len(mutated_ideas)} mutation ideas for prompt '{prompt.idea.short_name}'. Requested {num_mutations_to_generate}."
        )

        if len(mutated_ideas) != num_mutations_to_generate:
            raise ValueError(
                f"Requested {num_mutations_to_generate} mutation ideas, but got {len(mutated_ideas)}"
            )

        implemented_prompts = await self._implement_prompt_ideas(
            mutated_ideas,
            [prompt.originating_ideas] * num_mutations_to_generate,
        )
        logger.info(
            f"Successfully created {len(implemented_prompts)} implemented prompts from {len(mutated_ideas)} mutation ideas."
        )
        return implemented_prompts

    @retry_async_function(tries=3)
    async def _breed_prompts(
        self, parent_scored_prompts: list[ScoredPrompt]
    ) -> list[ImplementedPrompt]:
        num_to_breed = self.breeded_prompts_per_iteration
        if num_to_breed == 0:
            return []
        if len(parent_scored_prompts) < 2:
            raise ValueError(
                f"Need at least 2 parent prompts, got {len(parent_scored_prompts)}."
            )

        parent_details_list = []
        for i, scored_prompt in enumerate(parent_scored_prompts):
            idea = scored_prompt.prompt.idea
            parent_details_list.append(
                clean_indents(
                    f"""
                    Parent Prompt {i + 1} (Original Name: '{idea.short_name}'):
                    Core Idea: {idea.full_text}
                    Full Template (for context):
                    ```
                    {scored_prompt.prompt.text}
                    ```
                    """
                )
            )
        parent_details_str = "\n".join(parent_details_list)

        agent_breed_ideas = AiAgent(
            name="Prompt Breeder Ideator",
            model=AgentSdkLlm(self.ideation_llm_name),
            instructions=clean_indents(
                f"""
                # Instructions
                You are an expert prompt engineer. Your task is to create {num_to_breed} new, high-quality PROMPT IDEAS
                by breeding (intelligently combining) ideas from several successful parent prompts.
                - Please generate exactly {num_to_breed} new, distinct prompt IDEAS.
                - Each new idea should represent a synergistic combination of the best elements from TWO OR MORE parent prompts.
                - Do not simply copy one parent or make only trivial combinations (e.g., just taking one sentence from one and one from another).
                - Aim for novel, potent combinations that are conceptually new prompt approaches derived from the parents.
                - Identify strengths in different parents and try to combine them. If parents have weaknesses, try to avoid them in the bred versions.
                - Each new prompt idea must be a concept for a new, complete prompt.

                # Purpose of Prompt
                {self.prompt_purpose_explanation}

                # Prompt Requirements
                The final prompt (i.e. not the ideas you will generate) will have the below requirements. Another agent will implement these requirements.

                {self.prompt_requirements_explanation}

                # Parent Prompts
                {parent_details_str}

                # Format
                Below is the format you should use for your output. Fill in each part with your own words (e.g. don't literally say "Mutated Idea Title 1" in your output)

                ```
                **Bred Idea Title 1**
                New idea for bred prompt 1, explaining how it combines elements from parents in detail. The implementor will not be shown the original prompts, just this idea and the prompt requirements (so don't repeat the requirements in your idea, but otherwise give a detailed overview).

                **Bred Idea Title 2**
                New idea for bred prompt 2, explaining how it combines elements from parents in detail. The implementor will not be shown the original prompts, just this idea and the prompt requirements (so don't repeat the requirements in your idea, but otherwise give a detailed overview).
                ...
                (up to {num_to_breed} ideas)
                ```
                """
            ),
            tools=[
                perplexity_reasoning_pro_search
            ],  # Allow research for inspiration if needed
        )

        breeding_agent_task = (
            f"Generate {num_to_breed} new prompt ideas by breeding from the provided {len(parent_scored_prompts)} parent prompts. "
            f"Focus on synergistic combinations."
        )
        output = await AgentRunner.run(agent_breed_ideas, breeding_agent_task)

        bred_ideas = await structure_output(output.final_output, list[PromptIdea])

        if len(bred_ideas) != num_to_breed:
            raise ValueError(
                f"Requested {num_to_breed} bred ideas, but got {len(bred_ideas)}"
            )
        new_prompts = await self._implement_prompt_ideas(
            bred_ideas,
            [[sp.prompt.idea for sp in parent_scored_prompts] * num_to_breed],
        )
        logger.info(
            f"Successfully created {len(new_prompts)} implemented prompts from bred ideas."
        )
        return new_prompts

    async def _implement_prompt_ideas(
        self,
        prompt_ideas: list[PromptIdea],
        originating_ideas: list[list[PromptIdea]],
    ) -> list[ImplementedPrompt]:
        if len(prompt_ideas) != len(originating_ideas):
            logger.warning(
                f"Number of prompt ideas ({len(prompt_ideas)}) does not match number of originating ideas ({len(originating_ideas)}). Cannot map these together."
            )
            originating_ideas = [
                [
                    PromptIdea(
                        short_name="Error",
                        full_text="Failed to match idea to originating ideas",
                    )
                ]
                for _ in range(len(prompt_ideas))
            ]
        tasks = [
            self._implement_prompt_idea(idea, originating_ideas)
            for idea, originating_ideas in zip(prompt_ideas, originating_ideas)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        implemented_prompts = [
            result for result in results if not isinstance(result, BaseException)
        ]
        errors = [result for result in results if isinstance(result, BaseException)]
        for error in errors:
            logger.error(f"Error implementing prompt idea: {error}")

        return implemented_prompts

    @retry_async_function(tries=3)
    async def _implement_prompt_idea(
        self, prompt_idea: PromptIdea, originating_ideas: list[PromptIdea]
    ) -> ImplementedPrompt:
        agent = AiAgent(
            name="Prompt Implementor",
            model=AgentSdkLlm(self.ideation_llm_name),
            instructions=clean_indents(
                f"""
                # Instructions
                You are an expert prompt engineer. Your task is to implement a prompt based on the below idea:
                Name: {prompt_idea.short_name}
                Idea: {prompt_idea.full_text}

                # Purpose of Prompt
                {self.prompt_purpose_explanation}

                # Prompt Requirements
                {self.prompt_requirements_explanation}

                # Template Variables
                This is a template prompt. Below are information about the variables you have access to:
                {self.template_variables_explanation}

                # Additional Notes
                - Make sure to put braces around the variables in the prompt (like you would a normal fstring or template string), and don't add any additional variables not listed.
                - Return the prompt and nothing but the prompt. The prompt will be run as is.
                - Ensure the prompt is complete, well-structured, and ready to use based on the idea provided.
                - Do not add any explanatory text before or after the prompt itself. Do not add any other text.
                """
            ),
        )
        output = await AgentRunner.run(
            agent,
            f"Please implement a prompt for the idea: '{prompt_idea.short_name}'. Return only the prompt text itself.",
        )
        prompt: str = output.final_output

        logger.info(
            f"Generated prompt string for idea '{prompt_idea.short_name}': {prompt}"
        )
        implemented_prompt = ImplementedPrompt(
            text=prompt,
            idea=prompt_idea,
            originating_ideas=originating_ideas,
        )
        if self.validate_prompt_func is not None:
            await self.validate_prompt_func(implemented_prompt)
        return implemented_prompt

    def _log_duplicate_prompts(self, prompts: list[ScoredPrompt]) -> None:
        for prompt in prompts:
            count = 0
            if prompt.prompt.text in [ep.prompt.text for ep in prompts]:
                count += 1
            if count > 1:  # expect to find that the prompt matches itself
                logger.warning(
                    f"Duplicate prompt template found: {prompt.prompt.text} {count - 1} times"
                )
