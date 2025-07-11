from __future__ import annotations

import inspect
import logging
import subprocess
from datetime import datetime
from typing import Any

import typeguard
from pydantic import AliasChoices, BaseModel, Field

from forecasting_tools.auto_optimizers.customizable_bot import CustomizableBot
from forecasting_tools.auto_optimizers.prompt_data_models import ResearchTool
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
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
            raise ValueError(
                "No forecast reports in benchmark. Cannot calculate average"
            )
        reports = typeguard.check_type(
            self.forecast_reports,
            list[ForecastReport],
        )
        return ForecastReport.calculate_average_expected_baseline_score(reports)

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
                raise ValueError("No expected baseline score in forecast report")
        reports.sort(key=lambda x: x.expected_baseline_score, reverse=True)  # type: ignore
        assert reports[0].expected_baseline_score >= reports[-1].expected_baseline_score, "Expected baseline scores are not sorted"  # type: ignore
        return reports

    @property
    def name(self) -> str:
        if self.explicit_name is not None:
            return self.explicit_name

        if self.forecast_bot_class_name is not None:
            class_name = f"{self.forecast_bot_class_name}"
            max_length = 25
            if len(class_name) > max_length:
                class_name = class_name[:max_length] + "..."
        else:
            class_name = "n/a"

        try:
            research_reports = self.forecast_bot_config["research_reports_per_question"]
            predictions = self.forecast_bot_config["predictions_per_research_report"]
            num_runs_name = f"{research_reports} x {predictions}"
        except Exception:
            num_runs_name = ""

        try:
            llms = self.forecast_bot_config["llms"]
            llms = typeguard.check_type(llms, dict[str, Any])
            try:
                default_llm = llms["default"]["original_model"]
            except Exception:
                default_llm = llms["default"]
            final_path_part = default_llm.split("/")[-1]
            default_llm_display = final_path_part[:50]
        except Exception:
            default_llm_display = "n/a"

        name = ""
        if class_name:
            name += class_name
        if num_runs_name:
            name += f" | {num_runs_name}"
        if default_llm_display:
            name += f" | {default_llm_display}"
        return name[:75]

    @property
    def description(self) -> str:
        if self.explicit_description is not None:
            return self.explicit_description
        return f"This benchmark ran the {self.forecast_bot_class_name} bot on {self.num_input_questions} questions."

    @property
    def num_failed_forecasts(self) -> int:
        return len(self.failed_report_errors)

    @property
    def bot_prompt(self) -> str:
        try:
            return self.forecast_bot_config["prompt"]
        except Exception:
            logger.debug(
                f"No 'prompt' key found in forecast bot config for {self.forecast_bot_class_name}"
            )
            pass

        try:
            research_prompt = self.forecast_bot_config["research_prompt"]
            reasoning_prompt = self.forecast_bot_config["reasoning_prompt"]
            return CustomizableBot.combine_research_reasoning_prompt(
                research_prompt, reasoning_prompt
            )
        except Exception:
            logger.debug(
                f"No 'research_prompt' or 'reasoning_prompt' key found in forecast bot config for {self.forecast_bot_class_name}"
            )
            pass

        raise ValueError(
            f"Prompt not included in forecast bot config for {self.forecast_bot_class_name}"
        )

    @property
    def research_tools_used(self) -> list[ResearchTool]:
        try:
            research_tools = self.forecast_bot_config["research_tools"]
            research_tools = [ResearchTool(**tool) for tool in research_tools]
            research_tools = typeguard.check_type(research_tools, list[ResearchTool])
            return research_tools
        except Exception:
            raise ValueError(
                f"Research tools not included in forecast bot config for {self.forecast_bot_class_name}"
            )

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
            logger.warning(f"Could not get source code for {bot.__class__.__name__}")
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
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode("ascii")
                .strip()
            )
        except Exception:
            return "no_git_hash"
