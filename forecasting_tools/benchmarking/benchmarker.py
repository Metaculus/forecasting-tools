import inspect
import logging
import subprocess
import time
from datetime import datetime
from typing import Sequence

import typeguard

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.data_models.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.data_organizer import ReportTypes
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi

logger = logging.getLogger(__name__)


class QuestionBatch:
    def __init__(
        self,
        bot: ForecastBot,
        benchmark: BenchmarkForBot,
        questions: list[MetaculusQuestion],
    ):
        self.bot = bot
        self.benchmark = benchmark
        self.questions = questions


class Benchmarker:
    """
    This class is used to benchmark a list of forecast bots
    by comparing their predictions to the community prediction on a set of questions.

    For an idea of how many questions are 'enough' to test with read:
    https://forum.effectivealtruism.org/posts/DzqSh7akX28JEHf9H/comparing-two-forecasters-in-an-ideal-world

    TLDR: 100-200 questions is a decent starting point, but 500+ would be ideal.
    Lower than 100 can differentiate between bots of large skill differences,
    but not between bots of small skill differences. But even with 100 there is
    ~30% of the 'worse bot' winning if there are not large skill differences.
    """

    def __init__(
        self,
        forecast_bots: list[ForecastBot],
        number_of_questions_to_use: int | None = None,
        questions_to_use: Sequence[MetaculusQuestion] | None = None,
        file_path_to_save_reports: str | None = None,
        concurrent_question_batch_size: int = 10,
        additional_code_to_snapshot: list[type] | None = None,
    ) -> None:
        if (
            number_of_questions_to_use is not None
            and questions_to_use is not None
        ):
            raise ValueError(
                "Either number_of_questions_to_use or questions_to_use must be provided, not both"
            )
        if number_of_questions_to_use is None and questions_to_use is None:
            raise ValueError(
                "Either number_of_questions_to_use or questions_to_use must be provided"
            )

        self.forecast_bots = forecast_bots
        self.number_of_questions_to_use = number_of_questions_to_use
        self.questions_to_use = questions_to_use
        if (
            file_path_to_save_reports is not None
            and not file_path_to_save_reports.endswith("/")
        ):
            file_path_to_save_reports += "/"
        self.file_path_to_save_reports = file_path_to_save_reports
        self.initialization_timestamp = datetime.now()
        self.concurrent_question_batch_size = concurrent_question_batch_size
        self.code_to_snapshot = additional_code_to_snapshot

    async def run_benchmark(self) -> list[BenchmarkForBot]:
        if self.questions_to_use is None:
            assert (
                self.number_of_questions_to_use is not None
            ), "number_of_questions_to_use must be provided if questions_to_use is not provided"
            chosen_questions = MetaculusApi.get_benchmark_questions(
                self.number_of_questions_to_use,
            )
        else:
            chosen_questions = self.questions_to_use

        chosen_questions = typeguard.check_type(
            chosen_questions, list[MetaculusQuestion]
        )

        if self.number_of_questions_to_use is not None:
            assert len(chosen_questions) == self.number_of_questions_to_use

        benchmarks: list[BenchmarkForBot] = self._initialize_benchmarks(
            self.forecast_bots, chosen_questions
        )

        batches = self._batch_questions(
            self.forecast_bots,
            benchmarks,
            chosen_questions,
            self.concurrent_question_batch_size,
        )
        for batch in batches:
            await self._run_a_batch(batch)
            self._save_benchmarks_to_file_if_configured(benchmarks)
        return benchmarks

    async def _run_a_batch(self, batch: QuestionBatch) -> None:
        bot = batch.bot
        benchmark = batch.benchmark
        questions = batch.questions
        with MonetaryCostManager() as cost_manager:
            start_time = time.time()
            reports = await bot.forecast_questions(
                questions, return_exceptions=True
            )
            bot.log_report_summary(reports, raise_errors=False)
            valid_reports = [
                report
                for report in reports
                if not isinstance(report, BaseException)
            ]
            valid_reports = typeguard.check_type(
                valid_reports,
                list[ReportTypes],
            )
            failed_reports: list[BaseException] = [
                report
                for report in reports
                if isinstance(report, BaseException)
            ]
            benchmark.failed_report_errors.extend(
                [str(report) for report in failed_reports]
            )
            new_report_sequence = (
                list(benchmark.forecast_reports) + valid_reports
            )
            benchmark.forecast_reports = new_report_sequence
            end_time = time.time()

            if benchmark.time_taken_in_minutes is None:
                benchmark.time_taken_in_minutes = 0
            if benchmark.total_cost is None:
                benchmark.total_cost = 0
            benchmark.total_cost += cost_manager.current_usage
            benchmark.time_taken_in_minutes += (end_time - start_time) / 60

    @classmethod
    def _batch_questions(
        cls,
        bots: list[ForecastBot],
        benchmarks: list[BenchmarkForBot],
        questions: list[MetaculusQuestion],
        batch_size: int,
    ) -> list[QuestionBatch]:
        batches: list[QuestionBatch] = []
        question_batches = [
            questions[i : i + batch_size]
            for i in range(0, len(questions), batch_size)
        ]
        for question_batch in question_batches:
            for bot, benchmark in zip(bots, benchmarks):
                assert (
                    benchmark.forecast_bot_class_name == bot.__class__.__name__
                ), f"Benchmark {benchmark.forecast_bot_class_name} does not match bot {bot.__class__.__name__}"
                batches.append(
                    QuestionBatch(
                        bot=bot, benchmark=benchmark, questions=question_batch
                    )
                )
        assert len(batches) == len(bots) * len(question_batches)
        assert all(
            1 <= len(batch.questions) <= batch_size for batch in batches
        ), "All batches must have questions and be below batch size"
        return batches

    def _initialize_benchmarks(
        self,
        bots: list[ForecastBot],
        chosen_questions: list[MetaculusQuestion],
    ) -> list[BenchmarkForBot]:
        benchmarks: list[BenchmarkForBot] = []
        for bot in bots:
            try:
                source_code = inspect.getsource(bot.__class__)
                if self.code_to_snapshot:
                    for item in self.code_to_snapshot:
                        source_code += f"\n\n#------------{item.__name__}-------------\n\n{inspect.getsource(item)}"
            except Exception:
                logger.warning(
                    f"Could not get source code for {bot.__class__.__name__}"
                )
                source_code = None
            benchmark = BenchmarkForBot(
                forecast_bot_class_name=bot.__class__.__name__,
                forecast_reports=[],
                forecast_bot_config=bot.get_config(),
                time_taken_in_minutes=None,
                total_cost=None,
                git_commit_hash=self._get_git_commit_hash(),
                code=source_code,
                num_input_questions=len(chosen_questions),
            )
            benchmarks.append(benchmark)
        return benchmarks

    def _save_benchmarks_to_file_if_configured(
        self, benchmarks: list[BenchmarkForBot]
    ) -> None:
        if self.file_path_to_save_reports is None:
            return
        file_path_to_save_reports = (
            f"{self.file_path_to_save_reports}"
            f"benchmarks_"
            f"{self.initialization_timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"
            f".json"
        )
        BenchmarkForBot.save_object_list_to_file_path(
            benchmarks, file_path_to_save_reports
        )

    @classmethod
    def _get_git_commit_hash(cls) -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"]
                )
                .decode("ascii")
                .strip()
            )
        except Exception:
            return "no_git_hash"
