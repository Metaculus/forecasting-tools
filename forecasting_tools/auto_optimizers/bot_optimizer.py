import logging

from pydantic import BaseModel

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.auto_optimizers.bot_evaluator import BotEvaluator
from forecasting_tools.auto_optimizers.control_prompt import ControlPrompt
from forecasting_tools.auto_optimizers.customizable_bot import CustomizableBot
from forecasting_tools.auto_optimizers.prompt_data_models import BotConfig, ResearchTool
from forecasting_tools.auto_optimizers.prompt_optimizer import (
    ImplementedPrompt,
    OptimizationRun,
    PromptOptimizer,
    PromptScore,
    ScoredPrompt,
)
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.questions import MetaculusQuestion

logger = logging.getLogger(__name__)


class BotOptimizer:

    @classmethod
    async def optimize_a_combined_research_and_reasoning_prompt(  # NOSONAR
        cls,
        research_agent_llm_name: str,  # e.g. "openai/gpt-4.1"
        reasoning_llm: GeneralLlm,
        ideation_llm_name: str,
        research_tools_bot_can_use: list[ResearchTool],
        evaluation_questions: list[MetaculusQuestion],
        remove_background_info_from_questions: bool,
        batch_size_for_question_evaluation: int,
        folder_to_save_benchmarks: str | None,
        num_iterations_per_run: int,
        initial_prompt_population_size: int = 20,
        survivors_per_iteration: int = 5,
        mutated_prompts_per_survivor: int = 3,
        breeded_prompts_per_iteration: int = 5,
    ) -> OptimizationRun:
        logger.info(f"Loaded {len(evaluation_questions)} questions")
        evaluation_questions = [
            question.model_copy(deep=True) for question in evaluation_questions
        ]
        if remove_background_info_from_questions:
            for question in evaluation_questions:
                question.background_info = None
        prompt_purpose_explanation = clean_indents(
            """
            You are making a prompt for an AI to forecast binary questions about future events on prediction markets/aggregators.
            This needs to optimize forecasting accuracy as measured by log/brier scores.
            """
        )
        research_tool_limits = "\n###".join(
            [
                f"{tool.tool.name}\nMax calls allowed: {tool.max_calls}\nDescription: {tool.tool.description}"
                for tool in research_tools_bot_can_use
            ]
        )
        prompt_requirements_explanation = clean_indents(
            f"""
            The prompt should be split into 2 main parts:
            - The research part of the prompt
            - The reasoning part of the prompt

            ## Prompt Format:
            The research part
            - should be used to generate a research report.
            - This research report will be passed to the reasoning prompt.
            - Should explicitly state limits on how much tools should be used (e.g. max tool calls overall or per step)
            - Should always include a mention of something like "Never include Metaculus predictions in your final research report and never search up Metaculus questions". Using Metaculus predictions as a source is disallowed.
            The reasoning part
            - should be used to generate a forecast for a binary question.
            - The forecast must be a probability between 0 and 1.
            Deliminator:
            - The research and reasoning part of the prompt should be separated by the following string: {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING}
            - There should be no other deliminators between the prompts, only this string.

            ## Research tools and their limits:
            {research_tool_limits}
            """
        )
        mutation_considerations = clean_indents(
            """
            As you develop the prompt consider:
            - Research formats and length (consider all varieties)
            - Research sources (consider all varieties)
            - Research strategies (i.e. which steps in which order, with what criteria for what steps)
            - Which tools to use and how much
            - Whether you want to call tools in parallel at each step or not
            - Reasoning formats and length (consider all varieties)
            - Reasoning strategies (i.e. which steps in which order, with what criteria for what steps)
            - Whether to cite sources/links in the research report or not
            - Whether to keep it simple or try something more complex
            """
        )
        template_variables_explanation = clean_indents(
            f"""
            ## Research part of the prompt
            REQUIRED: The research part of the prompt should include all of the following variables:
            {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES}

            Optionally the research part of the prompt can include the following variables:
            {CustomizableBot.OPTIONAL_RESEARCH_PROMPT_VARIABLES}

            ## Reasoning part of the prompt
            REQUIRED: The reasoning part of the prompt should include the all of the following variables:
            {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES}

            Remember to only include the {{research}} variable once in the reasoning part of the prompt (this will have a lot of text in it, so we don't want to repeat it).
            """
        )

        evaluator = BotEvaluator(
            input_questions=evaluation_questions,
            research_type=None,
            concurrent_evaluation_batch_size=batch_size_for_question_evaluation,
            file_or_folder_to_save_benchmarks=folder_to_save_benchmarks,
        )

        async def evaluate_combined_research_and_reasoning_prompts(
            prompts_to_evaluate: list[ImplementedPrompt],
        ) -> list[PromptScore]:
            configs = []
            seed_prompt_found = False
            for prompt in prompts_to_evaluate:
                research_prompt, reasoning_prompt = (
                    CustomizableBot.split_combined_research_reasoning_prompt(
                        prompt.text
                    )
                )
                if (
                    research_prompt.strip()
                    == ControlPrompt.get_seed_research_prompt().strip()
                ):
                    if not seed_prompt_found:
                        seed_prompt_found = True
                        logger.info(
                            "Found seed prompt and replacing it with control prompt"
                        )
                    else:
                        raise RuntimeError("Found multiple seed prompts")

                    research_prompt = ControlPrompt.get_control_research_prompt()
                    prompt.text = CustomizableBot.combine_research_reasoning_prompt(
                        research_prompt, reasoning_prompt
                    )  # This is bad practice, but I'm tired, and just need it to work lol. Fix this later.

                configs.append(
                    BotConfig(
                        reasoning_prompt_template=reasoning_prompt,
                        research_prompt_template=research_prompt,
                        research_tools=research_tools_bot_can_use,
                        research_llm=research_agent_llm_name,
                        reasoning_llm=reasoning_llm,
                        originating_idea=prompt.idea,
                        research_reports_per_question=1,
                        predictions_per_research_report=1,
                    )
                )
            evaluation_result = await evaluator.evaluate_bot_configs(configs)
            evaluated_bots = evaluation_result.evaluated_bots
            assert len(evaluated_bots) == len(
                prompts_to_evaluate
            ), f"Number of evaluated bots ({len(evaluated_bots)}) does not match number of prompts to evaluate ({len(prompts_to_evaluate)})"

            prompt_scores = []
            for evaluated_bot, prompt in zip(evaluated_bots, prompts_to_evaluate):
                benchmark = evaluated_bot.benchmark
                prompt_scores.append(
                    PromptScore(
                        value=evaluated_bot.score,
                        metadata={
                            "benchmark": evaluated_bot.benchmark,
                        },
                    )
                )
                assert (
                    benchmark.bot_prompt == prompt.text
                ), f"Prompt does not match prompt in benchmark.\nPrompt: {prompt.text}\n\n\n\nBenchmark Prompt: {benchmark.bot_prompt}"
            return prompt_scores

        async def validate_prompt(prompt: ImplementedPrompt) -> None:
            CustomizableBot.validate_combined_research_reasoning_prompt(prompt.text)
            check_metaculus_mention_prompt = clean_indents(
                f"""
                # Instructions
                Please check if the following prompt includes instructions to not share Metaculus predictions in the final research report.

                Please return a json as your answer with the following format:
                {{
                    "metaculus_disallowed": bool,
                    "reasoning": str
                }}

                # Examples
                A prompt that contains:
                > Never share Metaculus predictions in the final research report.
                would return:
                {{
                    "requires_no_metaculus_mention": True,
                    "reasoning": "The prompt contains the instruction 'Never share Metaculus predictions in the final research report.'"
                }}

                A prompt that contains no mention of Metaculus predictions would return:
                {{
                    "requires_no_metaculus_mention": False,
                    "reasoning": "The prompt does not contain any instructions to not share Metaculus predictions in the final research report."
                }}

                A prompt that contains:
                > Never search up any questions from Metaculus.
                would return:
                {{
                    "requires_no_metaculus_mention": True,
                    "reasoning": "The prompt contains the instruction 'Never search up any questions from Metaculus.'"
                }}

                # Your Turn

                The prompt is between the following deliminators <<< >>> deliminators below.

                <<<<<<<<<<<<<<<<<< START OF PROMPT >>>>>>>>>>>>>>>>
                ```
                {prompt.text}
                ```
                <<<<<<<<<<<<<<<<<< END OF PROMPT >>>>>>>>>>>>>>>>
                """
            )
            llm = GeneralLlm(model=ideation_llm_name)

            class CheckMetaculusMention(BaseModel):
                metaculus_disallowed: bool
                reasoning: str

            reasoning_result = await llm.invoke_and_return_verified_type(
                check_metaculus_mention_prompt,
                CheckMetaculusMention,
            )
            if not reasoning_result.metaculus_disallowed:
                raise RuntimeError(
                    f"Prompt does not require not mentioning metaculus predictions in the final research report: {reasoning_result.reasoning}"
                )

        optimizer = PromptOptimizer(
            initial_prompt=ControlPrompt.get_seed_combined_prompt(),
            iterations=num_iterations_per_run,
            ideation_llm_name=ideation_llm_name,
            prompts_to_scores_func=evaluate_combined_research_and_reasoning_prompts,
            prompt_purpose_explanation=prompt_purpose_explanation,
            prompt_requirements_explanation=prompt_requirements_explanation,
            template_variables_explanation=template_variables_explanation,
            mutation_considerations=mutation_considerations,
            format_scores_func=cls._format_worst_scores_and_context,
            validate_prompt_func=validate_prompt,
            initial_prompt_population_size=initial_prompt_population_size,
            survivors_per_iteration=survivors_per_iteration,
            mutated_prompts_per_survivor=mutated_prompts_per_survivor,
            breeded_prompts_per_iteration=breeded_prompts_per_iteration,
        )

        logger.info("Starting optimization run")
        optimization_run = await optimizer.create_optimized_prompt()
        logger.info("Optimization run complete")

        cls._log_best_prompts(optimization_run)
        return optimization_run

    @staticmethod
    def _log_best_prompts(optimization_run: OptimizationRun) -> None:
        best_prompts = optimization_run.scored_prompts
        best_prompts.sort(key=lambda x: x.score.value, reverse=True)
        best_prompts = best_prompts[:5]
        message = "Best prompts:\n"
        for sp in best_prompts:
            message += "\n\n\n\n################################\n\n\n\n"
            message += f"\nScore: {sp.score.value}"
            message += f"\nCost: {sp.score.metadata['benchmark'].total_cost}"
            message += f"\nIdea Name: {sp.prompt.idea.short_name}"
            message += f"\nIdea Description: {sp.prompt.idea.full_text}"
            message += f"\nPrompt: {sp.prompt.text}"
            message += f"\nOriginating Ideas: {[idea.short_name for idea in sp.prompt.originating_ideas]}"
        logger.info(message)

    @staticmethod
    async def _format_worst_scores_and_context(
        scored_prompt: ScoredPrompt,
    ) -> str:
        benchmark: BenchmarkForBot = scored_prompt.score.metadata["benchmark"]
        num_worst_reports = (
            3
            if len(benchmark.forecast_reports) > 3
            else len(benchmark.forecast_reports)
        )
        worst_reports = benchmark.get_bottom_n_forecast_reports(num_worst_reports)

        report_str = f"Below are the worst {num_worst_reports} scores from the previous prompt. These are baseline scores (100pts is perfect forecast, -897pts is worst possible forecast, and 0pt is forecasting 50%):\n"
        report_str += f"<><><><><><><><><><><><><><> TOP {num_worst_reports} WORST REPORTS <><><><><><><><><><><><><><>\n"
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
        report_str += (
            "<><><><><><><><><><><><><><> END OF REPORTS <><><><><><><><><><><><><><>\n"
        )
        return report_str
