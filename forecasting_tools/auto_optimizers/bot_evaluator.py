import logging
from dataclasses import dataclass

import typeguard

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.auto_optimizers.control_prompt import ControlPrompt
from forecasting_tools.auto_optimizers.customizable_bot import CustomizableBot
from forecasting_tools.auto_optimizers.prompt_data_models import (
    BotConfig,
    PromptIdea,
    ResearchTool,
)
from forecasting_tools.auto_optimizers.question_plus_research import (
    QuestionPlusResearch,
    ResearchType,
)
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.cp_benchmarking.benchmarker import Benchmarker
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot

logger = logging.getLogger(__name__)


@dataclass
class EvaluatedBot:
    bot_config: BotConfig
    benchmark: BenchmarkForBot

    @property
    def score(self) -> float:
        if len(self.benchmark.forecast_reports) == 0:
            return -100.0
        return self.benchmark.average_expected_baseline_score


@dataclass
class BotEvaluation:
    evaluated_bots: list[EvaluatedBot]

    @property
    def best_bot(self) -> EvaluatedBot:
        if not self.evaluated_bots:
            raise ValueError("No evaluated bots available to determine the best bot")
        sorted_evaluated_bots = sorted(
            self.evaluated_bots, key=lambda x: x.score, reverse=True
        )
        return sorted_evaluated_bots[0]


class BotEvaluator:
    def __init__(
        self,
        input_questions: list[QuestionPlusResearch] | list[MetaculusQuestion],
        research_type: ResearchType | None,
        concurrent_evaluation_batch_size: int,
        file_or_folder_to_save_benchmarks: str | None,
    ) -> None:
        if research_type is None:
            input_questions = typeguard.check_type(
                input_questions, list[MetaculusQuestion]
            )
            self.evaluation_questions = input_questions
            self.research_snapshots = []
        else:
            input_questions = typeguard.check_type(
                input_questions, list[QuestionPlusResearch]
            )
            self.evaluation_questions = [
                snapshot.question for snapshot in input_questions
            ]
            self.research_snapshots = input_questions

        self.research_type = research_type
        self.concurrent_evaluation_batch_size = concurrent_evaluation_batch_size
        self.file_or_folder_to_save_benchmarks = file_or_folder_to_save_benchmarks

    async def evaluate_bot_configs(
        self, configurations: list[BotConfig]
    ) -> BotEvaluation:
        bots = self._configs_to_bots(configurations)
        bots = typeguard.check_type(bots, list[ForecastBot])
        benchmarker = Benchmarker(
            forecast_bots=bots,
            questions_to_use=self.evaluation_questions,
            concurrent_question_batch_size=self.concurrent_evaluation_batch_size,
            file_path_to_save_reports=self.file_or_folder_to_save_benchmarks,
        )
        benchmarks = await benchmarker.run_benchmark()
        if all(len(benchmark.forecast_reports) == 0 for benchmark in benchmarks):
            raise ValueError("All benchmarks have no forecast reports")
        evaluated_bots: list[EvaluatedBot] = []
        for config, benchmark in zip(configurations, benchmarks):
            benchmark.forecast_bot_class_name = (
                config.originating_idea.short_name.replace(" ", "_")
            )
            evaluated_bots.append(EvaluatedBot(bot_config=config, benchmark=benchmark))
            if len(benchmark.forecast_reports) == 0:
                logger.warning(
                    f"No forecast reports found for bot {config.originating_idea.short_name}"
                )
        return BotEvaluation(evaluated_bots=evaluated_bots)

    def _configs_to_bots(self, configs: list[BotConfig]) -> list[CustomizableBot]:
        bots = []
        for config in configs:
            if config.research_reports_per_question != 1:
                raise NotImplementedError(
                    "Currently only supports one research report per question"
                )
            custom_class_name = config.originating_idea.short_name.replace(" ", "_")
            CustomBotClass = type(custom_class_name, (CustomizableBot,), {})
            bot = CustomBotClass(
                originating_idea=config.originating_idea,
                research_prompt=config.research_prompt_template,
                reasoning_prompt=config.reasoning_prompt_template,
                research_tools=config.research_tools,
                cached_research=self.research_snapshots or None,
                cached_research_type=self.research_type,
                research_reports_per_question=config.research_reports_per_question,
                predictions_per_research_report=config.predictions_per_research_report,
                llms={
                    "default": config.reasoning_llm,
                    "researcher": config.research_llm,
                    "summarizer": None,
                },
                publish_reports_to_metaculus=False,
                enable_summarize_research=False,
            )
            bots.append(bot)
        return bots

    async def evaluate_best_benchmarked_prompts(
        self,
        benchmark_files: list[str],
        forecast_llm: GeneralLlm,
        research_llm_name: str,
        research_tools: list[ResearchTool],
        top_n_prompts: int = 1,
        include_control_group_prompt: bool = True,
        include_worst_prompt: bool = False,
        research_reports_per_question: int = 1,
        num_predictions_per_research_report: int = 1,
    ) -> BotEvaluation:
        best_benchmarks = self._get_best_benchmark_prompts(
            benchmark_files, top_n_prompts, include_worst_prompt
        )

        logger.info(
            f"Evaluating {len(best_benchmarks)} prompts with {forecast_llm.model}. Prompts are the best scoring from files {benchmark_files}"
        )
        configs = []
        if include_control_group_prompt:
            control_group_config = BotConfig(
                reasoning_prompt_template=ControlPrompt.get_reasoning_prompt(),
                research_prompt_template=ControlPrompt.get_control_research_prompt(),
                research_tools=ControlPrompt.get_control_research_tools(),
                reasoning_llm=forecast_llm,
                research_llm=research_llm_name,
                originating_idea=PromptIdea(
                    short_name=f"Control v{ControlPrompt.version()}",
                    full_text="The control is a prompt that is not optimized but used as a baseline comparison.",
                ),
                predictions_per_research_report=num_predictions_per_research_report,
                research_reports_per_question=research_reports_per_question,
            )
            configs.append(control_group_config)
        for benchmark in best_benchmarks:
            combined_research_reasoning_prompt = benchmark.bot_prompt
            research_prompt, reasoning_prompt = (
                CustomizableBot.split_combined_research_reasoning_prompt(
                    combined_research_reasoning_prompt
                )
            )
            logger.info(
                f"{benchmark.forecast_bot_class_name} - {benchmark.average_expected_baseline_score}"
            )
            best_prompt_config = BotConfig(
                reasoning_prompt_template=reasoning_prompt,
                research_prompt_template=research_prompt,
                research_tools=research_tools,  # Enable choosing to use bot.research_tools if benchmarks allow for different available tools
                reasoning_llm=forecast_llm,
                research_llm=research_llm_name,
                originating_idea=PromptIdea(
                    short_name=f"{benchmark.forecast_bot_class_name}",
                    full_text=f"Evaluate the prompt from {benchmark.forecast_bot_class_name} (originally found from a different dataset/origin) with model {forecast_llm.model} and {len(self.evaluation_questions)} questions",
                ),
                predictions_per_research_report=num_predictions_per_research_report,
                research_reports_per_question=research_reports_per_question,
            )
            configs.append(best_prompt_config)
        evaluation_result = await self.evaluate_bot_configs(configs)
        return evaluation_result

    @classmethod
    def _get_best_benchmark_prompts(
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
            logger.info(f"Loaded {len(benchmarks)} benchmarks from {file_path}")
            for benchmark in benchmarks:
                if len(benchmark.forecast_reports) > 0:
                    all_benchmarks.append(benchmark)
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
