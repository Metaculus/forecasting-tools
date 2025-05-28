import logging

import typeguard

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.benchmarking.benchmarker import Benchmarker
from forecasting_tools.benchmarking.control_group_prompt import (
    ControlGroupPrompt,
)
from forecasting_tools.benchmarking.customizable_bot import CustomizableBot
from forecasting_tools.benchmarking.prompt_data_models import (
    EvaluatedPrompt,
    OptimizationResult,
    PromptConfig,
    PromptIdea,
)
from forecasting_tools.benchmarking.question_research_snapshot import (
    QuestionResearchSnapshot,
    ResearchType,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot

logger = logging.getLogger(__name__)


class PromptEvaluator:
    def __init__(
        self,
        evaluation_questions: list[QuestionResearchSnapshot],
        research_type: ResearchType,
        concurrent_evaluation_batch_size: int,
        file_or_folder_to_save_benchmarks: str | None,
    ) -> None:
        self.evaluation_questions = evaluation_questions
        self.research_type = research_type
        self.concurrent_evaluation_batch_size = (
            concurrent_evaluation_batch_size
        )
        self.file_or_folder_to_save_benchmarks = (
            file_or_folder_to_save_benchmarks
        )

    async def evaluate_prompts(
        self, configurations: list[PromptConfig]
    ) -> OptimizationResult:
        bots = self._configs_to_bots(configurations)
        bots = typeguard.check_type(bots, list[ForecastBot])
        benchmarker = Benchmarker(
            forecast_bots=bots,
            questions_to_use=[
                snapshot.question for snapshot in self.evaluation_questions
            ],
            concurrent_question_batch_size=self.concurrent_evaluation_batch_size,
            file_path_to_save_reports=self.file_or_folder_to_save_benchmarks,
        )
        benchmarks = await benchmarker.run_benchmark()
        evaluated_prompts: list[EvaluatedPrompt] = []
        for config, benchmark in zip(configurations, benchmarks):
            benchmark.forecast_bot_class_name = (
                config.original_idea.short_name.replace(" ", "_")
            )
            evaluated_prompts.append(
                EvaluatedPrompt(prompt_config=config, benchmark=benchmark)
            )
        benchmarker.save_benchmarks_to_file_if_configured(benchmarks)
        return OptimizationResult(evaluated_prompts=evaluated_prompts)

    def _configs_to_bots(
        self, configs: list[PromptConfig]
    ) -> list[CustomizableBot]:
        bots = []
        for config in configs:
            if config.research_reports_per_question != 1:
                raise NotImplementedError(
                    "Currently only supports one research report per question"
                )
            bot = self._create_customizable_bot(
                config=config,
                research_snapshots=self.evaluation_questions,
                research_type=self.research_type,
            )
            bots.append(bot)
        return bots

    @classmethod
    def _create_customizable_bot(
        cls,
        config: PromptConfig,
        research_snapshots: list[QuestionResearchSnapshot],
        research_type: ResearchType,
    ) -> CustomizableBot:
        bot = CustomizableBot(
            originating_idea=config.original_idea,
            prompt=config.prompt_template,
            research_snapshots=research_snapshots,
            research_type=research_type,
            research_reports_per_question=config.research_reports_per_question,
            predictions_per_research_report=config.predictions_per_research_report,
            llms={
                "default": config.llm,
            },
            publish_reports_to_metaculus=False,
            enable_summarize_research=False,
        )
        return bot

    async def evaluate_best_benchmarked_prompts(
        self,
        benchmark_files: list[str],
        forecast_llm: GeneralLlm,
        top_n_prompts: int = 1,
        include_control_group_prompt: bool = True,
        include_worst_prompt: bool = False,
    ) -> OptimizationResult:
        best_benchmarks = self._get_best_benchmark_prompt(
            benchmark_files, top_n_prompts, include_worst_prompt
        )

        logger.info(
            f"Evaluating {len(best_benchmarks)} prompts with {forecast_llm.model}. Prompts are the best scoring from files {benchmark_files}"
        )
        configs = []
        if include_control_group_prompt:
            control_group_config = PromptConfig(
                prompt_template=ControlGroupPrompt.get_prompt(),
                llm=forecast_llm,
                original_idea=PromptIdea(
                    short_name=f"Control Group v{ControlGroupPrompt.version()}",
                    idea="The control group is a group of questions that are not optimized for the prompt. It is used to evaluate the performance of the optimized prompt.",
                ),
            )
            configs.append(control_group_config)
        for benchmark in best_benchmarks:
            prompt = benchmark.forecast_bot_config["prompt"]
            logger.info(
                f"{benchmark.forecast_bot_class_name} - {benchmark.average_expected_baseline_score}:\n{prompt}"
            )
            best_prompt_config = PromptConfig(
                prompt_template=prompt,
                llm=forecast_llm,
                original_idea=PromptIdea(
                    short_name=f"{benchmark.forecast_bot_class_name}_variation",
                    idea=f"Evaluate the prompt from {benchmark.forecast_bot_class_name} with model {forecast_llm.model} and {len(self.evaluation_questions)} questions",
                ),
            )
            configs.append(best_prompt_config)
        evaluation_result = await self.evaluate_prompts(configs)
        return evaluation_result

    @classmethod
    def _get_best_benchmark_prompt(
        cls,
        file_paths: list[str],
        top_n_prompts: int = 1,
        include_worst_prompt: bool = False,
    ) -> list[BenchmarkForBot]:
        logger.info(
            f"Attempting to get the best {top_n_prompts} prompts from {file_paths}"
        )
        all_benchmarks = []
        for file_path in file_paths:
            benchmarks = BenchmarkForBot.load_json_from_file_path(file_path)
            logger.info(
                f"Loaded {len(benchmarks)} benchmarks from {file_path}"
            )
            all_benchmarks.extend(benchmarks)
        sorted_benchmarks = sorted(
            all_benchmarks,
            key=lambda x: x.average_expected_baseline_score,
            reverse=True,
        )
        best_benchmarks = sorted_benchmarks[:top_n_prompts]
        if include_worst_prompt:
            worst_benchmark = sorted_benchmarks[-1]
            return best_benchmarks + [worst_benchmark]
        return best_benchmarks
