import inspect
import logging
import subprocess
import time
from datetime import datetime

import typeguard

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastBot,
)
from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecasting.questions_and_reports.benchmark_for_bot import (
    BenchmarkForBot,
)
from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    BinaryReport,
)
from forecasting_tools.forecasting.questions_and_reports.multiple_choice_report import (
    MultipleChoiceReport,
)
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    NumericReport,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    MetaculusQuestion,
)

logger = logging.getLogger(__name__)


class Benchmarker:
    """
    For an idea of how many questions are 'enough' to test with read:
    https://forum.effectivealtruism.org/posts/DzqSh7akX28JEHf9H/comparing-two-forecasters-in-an-ideal-world
    Also see the results of a rough (and probably flawed) simulation here:
    https://chatgpt.com/share/3fbc8106-829d-4fb3-a9e6-af0badf266df

    TLDR: 100-200 questions is a decent starting point, but 500+ would be ideal.
    Lower than 100 can differentiate between bots of large skill differences,
    but not between bots of small skill differences. But even with 100 there is
    ~30% of the 'worse bot' winning if there are not large skill differences.
    """

    def __init__(
        self,
        forecast_bots: list[ForecastBot],
        number_of_questions_to_use: int,
        file_path_to_save_reports: str | None = None,
        concurrent_question_batch_size: int = 10,
    ) -> None:
        self.forecast_bots = forecast_bots
        self.number_of_questions_to_use = number_of_questions_to_use
        if (
            file_path_to_save_reports is not None
            and not file_path_to_save_reports.endswith("/")
        ):
            file_path_to_save_reports += "/"
        self.file_path_to_save_reports = file_path_to_save_reports
        self.initialization_timestamp = datetime.now()
        self.concurrent_question_batch_size = concurrent_question_batch_size

    async def run_benchmark(self) -> list[BenchmarkForBot]:
        questions = MetaculusApi.get_benchmark_questions(
            self.number_of_questions_to_use,
        )

        questions = typeguard.check_type(questions, list[MetaculusQuestion])
        assert len(questions) == self.number_of_questions_to_use

        benchmarks = []
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
                name=f"Benchmark for {bot.__class__.__name__}",
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
                    questions, self.concurrent_question_batch_size
                ):
                    reports = await bot.forecast_questions(batch)
                    reports = typeguard.check_type(
                        reports,
                        list[
                            BinaryReport | MultipleChoiceReport | NumericReport
                        ],
                    )
                    benchmark.forecast_reports.extend(reports)
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
