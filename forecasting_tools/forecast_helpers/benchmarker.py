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


class Benchmarker:
    """
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

        benchmarks: list[BenchmarkForBot] = []
        for bot in self.forecast_bots:
            try:
                source_code = inspect.getsource(bot.__class__)
            except Exception:
                logger.warning(
                    f"Could not get source code for {bot.__class__.__name__}"
                )
                source_code = None
            benchmark = BenchmarkForBot(
                forecast_reports=[],
                forecast_bot_config=bot.get_config(),
                description=f"This benchmark ran the {bot.__class__.__name__} bot on {self.number_of_questions_to_use} questions.",
                name=f"{bot.__class__.__name__}",
                time_taken_in_minutes=None,
                total_cost=None,
                git_commit_hash=self._get_git_commit_hash(),
                code=source_code,
            )
            benchmarks.append(benchmark)

        for bot, benchmark in zip(self.forecast_bots, benchmarks):
            with MonetaryCostManager() as cost_manager:
                start_time = time.time()
                for batch in self._batch_questions(
                    chosen_questions, self.concurrent_question_batch_size
                ):
                    reports = await bot.forecast_questions(
                        batch, return_exceptions=True
                    )
                    exceptions = [
                        report
                        for report in reports
                        if isinstance(report, Exception)
                    ]
                    if exceptions:
                        logger.error(
                            f"{len(exceptions)} reports failed. Exceptions: {exceptions}"
                        )
                    valid_reports = [
                        report
                        for report in reports
                        if not isinstance(report, Exception)
                    ]
                    valid_reports = typeguard.check_type(
                        valid_reports,
                        list[ReportTypes],
                    )
                    benchmark.forecast_reports.extend(valid_reports)
                    self._save_benchmarks_to_file_if_configured(benchmarks)
                end_time = time.time()
                benchmark.time_taken_in_minutes = (end_time - start_time) / 60
                benchmark.total_cost = cost_manager.current_usage
        self._save_benchmarks_to_file_if_configured(benchmarks)
        return benchmarks

    @classmethod
    def _batch_questions(
        cls, questions: list[MetaculusQuestion], batch_size: int
    ) -> list[list[MetaculusQuestion]]:
        return [
            questions[i : i + batch_size]
            for i in range(0, len(questions), batch_size)
        ]

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
