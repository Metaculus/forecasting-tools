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
    EvaluatedPrompt,
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

    def __init__(
        self,
        iterations: int,
        forecast_llm: GeneralLlm,
        ideation_llm_name: str,
        evaluator: PromptEvaluator,
        initial_prompt: str | None = None,
        initial_prompt_population_size: int = 25,
        survivors_per_iteration: int = 5,
        mutated_prompts_per_survivor: int = 4,
        breeded_prompts_per_iteration: int = 5,
    ) -> None:
        self.initial_prompt = initial_prompt or ControlGroupPrompt.get_prompt()
        self.iterations = iterations
        self.initial_prompt_population_size = initial_prompt_population_size
        self.survivors_per_iteration = survivors_per_iteration
        self.mutated_prompts_per_survivor = mutated_prompts_per_survivor
        self.breeded_prompts_per_iteration = breeded_prompts_per_iteration

        self.forecast_llm = forecast_llm
        self.ideation_llm_name = ideation_llm_name
        self.evaluator = evaluator

        if (
            self.mutated_prompts_per_survivor == 0
            and self.breeded_prompts_per_iteration == 0
        ):
            raise ValueError(
                "At least one of mutated_prompts_per_surviving_prompt or breeded_prompts_per_iteration must be greater than 0"
            )

    async def create_optimized_prompt(self) -> OptimizationResult:
        initial_seed_prompt_config = PromptConfig(
            prompt_template=self.initial_prompt,
            llm=self.forecast_llm,
            original_idea=PromptIdea(
                short_name="Initial Seed",
                idea="The user-provided initial prompt.",
            ),
        )

        new_prompt_configs: list[PromptConfig] = [initial_seed_prompt_config]
        if self.initial_prompt_population_size > 0:
            additional_initial_prompts = await self._mutate_prompt(
                initial_seed_prompt_config, self.initial_prompt_population_size
            )
            new_prompt_configs.extend(additional_initial_prompts)

        all_evaluated_prompts: list[EvaluatedPrompt] = []
        current_survivors: list[EvaluatedPrompt] = []

        for i in range(self.iterations):
            logger.info(
                f"Starting iteration {i + 1}/{self.iterations} - Current population size: {len(new_prompt_configs)}"
            )

            evaluation = await self.evaluator.evaluate_prompts(
                new_prompt_configs
            )
            new_evaluated_prompts = evaluation.evaluated_prompts
            all_evaluated_prompts.extend(new_evaluated_prompts)

            updated_population = current_survivors + new_evaluated_prompts
            updated_population.sort(
                key=lambda x: x.score,
                reverse=True,
            )
            current_survivors = updated_population[
                : self.survivors_per_iteration
            ]
            logger.debug(f"Current survivors: {current_survivors}")
            best_survivor = current_survivors[0]
            logger.info(
                f"Best survivor: {best_survivor.prompt_config.original_idea.short_name} with score {best_survivor.score:.4f}. Prompt:\n {best_survivor.prompt_config.prompt_template}"
            )

            mutated_configs: list[PromptConfig] = []
            mutation_tasks = [
                self._mutate_prompt(
                    ep.prompt_config, self.mutated_prompts_per_survivor
                )
                for ep in current_survivors
            ]
            initial_mutation_results = await asyncio.gather(
                *mutation_tasks, return_exceptions=True
            )
            mutation_results: list[list[PromptConfig]] = [
                result
                for result in initial_mutation_results
                if not isinstance(result, BaseException)
            ]
            for mutation_list in mutation_results:
                mutated_configs.extend(mutation_list)
            logger.info(f"Generated {len(mutated_configs)} mutated prompts.")

            bred_configs: list[PromptConfig] = []
            try:
                bred_configs = await self._breed_prompts(current_survivors)
            except Exception as e:
                logger.error(f"Failed to breed prompts: {e}")
                bred_configs = []
            logger.info(f"Generated {len(bred_configs)} bred prompts.")

            new_prompt_configs = mutated_configs + bred_configs
            for pc in new_prompt_configs:
                assert pc.prompt_template not in [
                    ep.prompt_config.prompt_template
                    for ep in all_evaluated_prompts
                ], f"Duplicate prompt template found: {pc.prompt_template}"

        return OptimizationResult(evaluated_prompts=all_evaluated_prompts)

    async def _mutate_prompt(
        self,
        prompt: EvaluatedPrompt | PromptConfig,
        num_mutations_to_generate: int,
    ) -> list[PromptConfig]:
        num_worst_reports = 3
        if isinstance(prompt, EvaluatedPrompt):
            parent_prompt_config = prompt.prompt_config
            worst_reports = prompt.benchmark.get_bottom_n_forecast_reports(
                num_worst_reports
            )
        else:
            parent_prompt_config = prompt
            worst_reports = []

        if worst_reports:
            report_str = f"Below are the worst {num_worst_reports} scores from the previous prompt. These are baseline scores (100pts is perfect forecast, -897pts is worst possible forecast, and 0pt is forecasting 50%):\n"
            report_str += "<><><><><><><><><><><><><><> START OF REPORTS <><><><><><><><><><><><><><>\n"
            for report in worst_reports:
                report_str += clean_indents(
                    f"""
                    ##  Question: {report.question.question_text} **(Score: {report.expected_baseline_score:.4f})**
                    **Summary**
                    ```{report.summary}```
                    **Research**
                    ```{report.research}```
                    **First rationale**
                    ```{report.first_rationale}```
                    """
                )
            report_str += "<><><><><><><><><><><><><><> END OF REPORTS <><><><><><><><><><><><><><>\n"
        else:
            report_str = ""

        agent_mutate_ideas = AiAgent(
            name="Prompt Mutator Ideator",
            model=AgentSdkLlm(self.ideation_llm_name),
            instructions=clean_indents(
                f"""
                You are an expert prompt engineer. Your task is to generate {num_mutations_to_generate} new PROMPT IDEAS by mutating an existing forecasting prompt.
                The original prompt is used by an AI to forecast binary questions.
                Your ideas are being used to optimize a forecasting bot using a Genetic Algorithm inspired approach.
                We are highlighting exploration over exploitation, but do want to strike a balance.

                # Instructions
                1. Please analyze the worst {num_worst_reports} scores from the previous prompt and identify what went wrong.
                2. Run 3-10 searches on the web to find inspiration for novel forecasting techniques or prompt structures.
                3. Generate {num_mutations_to_generate} new, distinct prompt IDEAS based on the original.

                Please generate exactly {num_mutations_to_generate} new, distinct prompt IDEAS based on the original.
                Each mutation idea must be a concept for a new, complete prompt. The implemented prompt will:
                1. Ask an AI to forecast a binary question.
                2. Require a final binary float as the output.

                For each idea please sequentially follow these policies to determine how much you try to mutate the original prompt:
                1st idea: "slight modification, like changing wording, adding/removing a sentences or a small paragraph, reording steps, adding emphasis, etc",
                2nd idea: "significant variation, which should take a generally different approach and be a general rewrite while staying in general theme of the original",
                3rd idea: "highly diverse mutation/experiment that explores a substantially different structure or set of principles, focus on a completely different idea than in the original. Search until you find something novel.",
                nth idea: ... continue alternating between significant variation and highly diverse (not slight)...

                # Original Prompt Idea Details
                Name: {parent_prompt_config.original_idea.short_name}
                Core Idea: {parent_prompt_config.original_idea.idea}

                Original Prompt Template (for context only, do not reproduce it in your output):
                ```
                {parent_prompt_config.prompt_template}
                ```

                {report_str}

                # Format
                **Mutated Idea Title 1**
                New idea for prompt mutation 1, specifying in detail how to implement the prompt reflecting the target variation.

                **Mutated Idea Title 2**
                New idea for prompt mutation 2, specifying in detail how to implement the prompt reflecting the target variation.
                ...
                (up to {num_mutations_to_generate} ideas)
                """
            ),
            tools=[perplexity_pro_search],
        )

        mutation_agent_task = (
            f"Generate {num_mutations_to_generate} mutated prompt ideas for the prompt named '{parent_prompt_config.original_idea.short_name}'. "
            f"Ensure each mutation aligns with the requested degree of variation."
        )
        output = await AgentRunner.run(agent_mutate_ideas, mutation_agent_task)
        mutated_ideas = await structure_output(
            output.final_output, list[PromptIdea]
        )
        logger.info(
            f"Successfully structured {len(mutated_ideas)} mutation ideas for prompt '{parent_prompt_config.original_idea.short_name}'. Requested {num_mutations_to_generate}."
        )

        initial_new_prompt_configs = await asyncio.gather(
            *[
                self._prompt_idea_to_prompt_config(idea)
                for idea in mutated_ideas
            ],
            return_exceptions=True,
        )
        new_prompt_configs: list[PromptConfig] = [
            result
            for result in initial_new_prompt_configs
            if not isinstance(result, BaseException)
        ]
        if len(mutated_ideas) != num_mutations_to_generate:
            logger.warning(
                f"Requested {num_mutations_to_generate} mutation ideas, but got {len(mutated_ideas)}. Returning {mutated_ideas[:num_mutations_to_generate]}"
            )
            mutated_ideas = mutated_ideas[:num_mutations_to_generate]

        logger.info(
            f"Successfully created {len(new_prompt_configs)} PromptConfig objects from mutation ideas."
        )
        return new_prompt_configs

    async def _breed_prompts(
        self, parent_evaluated_prompts: list[EvaluatedPrompt]
    ) -> list[PromptConfig]:
        num_to_breed = self.breeded_prompts_per_iteration
        if num_to_breed == 0:
            return []
        if len(parent_evaluated_prompts) < 2:
            raise ValueError(
                f"Need at least 2 parent prompts, got {len(parent_evaluated_prompts)}."
            )

        parent_details_list = []
        for i, ep in enumerate(parent_evaluated_prompts):
            pc = ep.prompt_config
            parent_details_list.append(
                clean_indents(
                    f"""
                    Parent Prompt {i + 1} (Original Name: '{pc.original_idea.short_name}'):
                    Core Idea: {pc.original_idea.idea}
                    Full Template (for context):
                    ```
                    {pc.prompt_template}
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
                You are an expert prompt engineer. Your task is to create {num_to_breed} new, high-quality forecasting PROMPT IDEAS
                by breeding (intelligently combining) ideas from several successful parent prompts.
                These prompts are used by an AI to forecast binary questions.

                Please generate exactly {num_to_breed} new, distinct prompt IDEAS.
                Each new idea should represent a synergistic combination of the best elements from TWO OR MORE parent prompts.
                Do not simply copy one parent or make only trivial combinations (e.g., just taking one sentence from one and one from another).
                Aim for novel, potent combinations that are conceptually new prompt approaches derived from the parents.
                Identify strengths in different parents and try to combine them. If parents have weaknesses, try to avoid them in the bred versions.

                Each new prompt idea must be a concept for a new, complete prompt. The implemented prompt will:
                1. Ask an AI to forecast a binary question.
                2. Require a final binary float as the output.

                # Parent Prompts
                {parent_details_str}

                # Format
                **Bred Idea Title 1**
                New idea for bred prompt 1, explaining how it combines elements from parents in detail.

                **Bred Idea Title 2**
                New idea for bred prompt 2, explaining how it combines elements from parents in detail.
                ...
                (up to {num_to_breed} ideas)
                """
            ),
            tools=[
                perplexity_pro_search
            ],  # Allow research for inspiration if needed
        )

        breeding_agent_task = (
            f"Generate {num_to_breed} new prompt ideas by breeding from the provided {len(parent_evaluated_prompts)} parent prompts. "
            f"Focus on synergistic combinations."
        )
        output = await AgentRunner.run(agent_breed_ideas, breeding_agent_task)

        bred_ideas = await structure_output(
            output.final_output, list[PromptIdea]
        )

        if len(bred_ideas) != num_to_breed:
            logger.warning(
                f"Requested {num_to_breed} bred ideas, but got {len(bred_ideas)}. Returning {bred_ideas[:num_to_breed]}"
            )
            bred_ideas = bred_ideas[:num_to_breed]
        initial_new_prompt_configs = await asyncio.gather(
            *[self._prompt_idea_to_prompt_config(idea) for idea in bred_ideas],
            return_exceptions=True,
        )
        new_prompt_configs: list[PromptConfig] = [
            result
            for result in initial_new_prompt_configs
            if not isinstance(result, BaseException)
        ]
        logger.info(
            f"Successfully created {len(new_prompt_configs)} PromptConfig objects from bred ideas."
        )
        return new_prompt_configs

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
                {{question_text}} - One sentence question text
                {{background_info}} - 1-2 paragraphs of background information
                {{resolution_criteria}} - 1-2 paragraphs of resolution criteria
                {{fine_print}} - 1-2 paragraphs of fine print
                {{today}} - The current date in the format YYYY-MM-DD
                {{research}} - Will contain 4-20 paragraphs of research

                Make sure to include the above variables in the prompt (e.g. in braces), and don't add any additional variables.
                Return the prompt and nothing but the prompt. The prompt will be run as is.
                Ensure the prompt is complete, well-structured, and ready to use based on the idea provided.
                Do not add any explanatory text before or after the prompt itself.
                """
            ),
        )
        output = await AgentRunner.run(
            agent,
            f"Please implement a prompt for the idea: '{prompt_idea.short_name}'. Return only the prompt text itself.",
        )
        prompt = output.final_output

        missing_vars = [
            var
            for var in [
                "{question_text}",
                "{background_info}",
                "{resolution_criteria}",
                "{fine_print}",
                "{today}",
                "{research}",
            ]
            if var not in prompt
        ]
        if missing_vars:
            logger.warning(
                f"Generated prompt for '{prompt_idea.short_name}' is missing template variables: {missing_vars}. Prompt: {prompt}"
            )

        try:
            prompt.format(
                question_text="test",
                background_info="test",
                resolution_criteria="test",
                fine_print="test",
                today="test",
                research="test",
            )
        except Exception as e:
            logger.error(f"Failed to fill in prompt: {e}")
            raise ValueError(f"Failed to fill-in-prompt test: {e}")

        logger.info(
            f"Generated prompt string for idea '{prompt_idea.short_name}': {prompt}"
        )
        return prompt

    async def _prompt_idea_to_prompt_config(
        self, prompt_idea: PromptIdea
    ) -> PromptConfig:
        prompt_template = await self._prompt_idea_to_prompt_string(prompt_idea)
        if not prompt_template:
            raise ValueError(
                f"No prompt template generated for idea: '{prompt_idea.short_name}'"
            )
        return PromptConfig(
            prompt_template=prompt_template,
            llm=self.forecast_llm,
            original_idea=prompt_idea,
        )
