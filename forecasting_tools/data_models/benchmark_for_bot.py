from datetime import datetime
from typing import Any, Sequence

import typeguard
from pydantic import AliasChoices, BaseModel, Field

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.multiple_choice_report import (
    MultipleChoiceReport,
)
from forecasting_tools.data_models.numeric_report import NumericReport
from forecasting_tools.util.jsonable import Jsonable


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
    timestamp: datetime = Field(default_factory=datetime.now)
    time_taken_in_minutes: float | None
    total_cost: float | None
    git_commit_hash: str
    forecast_bot_config: dict[str, Any]
    code: str | None = None
    forecast_reports: Sequence[
        BinaryReport | NumericReport | MultipleChoiceReport
    ]
    num_input_questions: int | None = None

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
            default_llm = f"default: {llms['default']}"
        except Exception:
            default_llm = "n/a"

        name = f"{class_name} | {num_runs_name} | {default_llm}"
        return name

    @property
    def description(self) -> str:
        if self.explicit_description is not None:
            return self.explicit_description
        return f"This benchmark ran the {self.forecast_bot_class_name} bot on {self.num_input_questions} questions."
