from __future__ import annotations

import inspect
import logging
import subprocess
from datetime import datetime
from typing import Any

import typeguard
from pydantic import AliasChoices, BaseModel, Field

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.multiple_choice_report import (
    MultipleChoiceReport,
)
from forecasting_tools.data_models.numeric_report import NumericReport
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class BenchmarkForBot(BaseModel, Jsonable):
    explicit_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("name", "explicit_name"),
    )
    explicit_description: str | None = Field(
        default=None,
        validation_alias=AliasChoices("description", "explicit_description"),
    )
    forecast_bot_class_name: str | None = None
    num_input_questions: int | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    time_taken_in_minutes: float | None
    total_cost: float | None
    git_commit_hash: str | None = None
    forecast_bot_config: dict[str, Any]
    code: str | None = None
    failed_report_errors: list[str] = Field(default_factory=list)
    forecast_reports: list[BinaryReport | NumericReport | MultipleChoiceReport]

    @property
    def average_expected_baseline_score(self) -> float:
        if len(self.forecast_reports) == 0:
            raise ValueError("No forecast reports in benchmark")
        reports = typeguard.check_type(
            self.forecast_reports,
            list[ForecastReport],
        )
        return ForecastReport.calculate_average_expected_baseline_score(
            reports
        )

    def get_top_n_forecast_reports(self, n: int) -> list[ForecastReport]:
        reports = self._get_sorted_forecast_reports()
        return reports[:n]

    def get_bottom_n_forecast_reports(self, n: int) -> list[ForecastReport]:
        reports = self._get_sorted_forecast_reports()
        return reports[-n:]

    def _get_sorted_forecast_reports(self) -> list[ForecastReport]:
        if len(self.forecast_reports) == 0:
            raise ValueError("No forecast reports in benchmark")
        shallow_copied_reports = self.forecast_reports.copy()
        reports = typeguard.check_type(
            shallow_copied_reports,
            list[ForecastReport],
        )
        for report in reports:
            if report.expected_baseline_score is None:
                raise ValueError(
                    "No expected baseline score in forecast report"
                )
        reports.sort(key=lambda x: x.expected_baseline_score, reverse=True)  # type: ignore
        assert reports[0].expected_baseline_score >= reports[-1].expected_baseline_score, "Expected baseline scores are not sorted"  # type: ignore
        return reports

    @property
    def name(self) -> str:
        if self.explicit_name is not None:
            return self.explicit_name

        if self.forecast_bot_class_name is not None:
            class_name = f"{self.forecast_bot_class_name}"
        else:
            class_name = "n/a"

        try:
            research_reports = self.forecast_bot_config[
                "research_reports_per_question"
            ]
            predictions = self.forecast_bot_config[
                "predictions_per_research_report"
            ]
            num_runs_name = f"{research_reports} x {predictions}"
        except Exception:
            num_runs_name = "n/a"

        try:
            llms = self.forecast_bot_config["llms"]
            llms = typeguard.check_type(llms, dict[str, Any])
            try:
                default_llm = llms["default"]["original_model"]
            except Exception:
                default_llm = llms["default"]
            default_llm_display = default_llm[:50]
        except Exception:
            default_llm = "n/a"

        name = f"{class_name} | {num_runs_name} | {default_llm_display}"
        return name[:75]

    @property
    def description(self) -> str:
        if self.explicit_description is not None:
            return self.explicit_description
        return f"This benchmark ran the {self.forecast_bot_class_name} bot on {self.num_input_questions} questions."

    @property
    def num_failed_forecasts(self) -> int:
        return len(self.failed_report_errors)

    @classmethod
    def initialize_benchmark_for_bot(
        cls,
        bot: ForecastBot,
        num_input_questions: int,
        additional_code: list[type] | None = None,
    ) -> BenchmarkForBot:
        try:
            source_code = inspect.getsource(bot.__class__)
            if additional_code:
                for item in additional_code:
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
            git_commit_hash=cls._get_git_commit_hash(),
            code=source_code,
            num_input_questions=num_input_questions,
        )
        return benchmark

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
