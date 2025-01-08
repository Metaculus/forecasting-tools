from datetime import datetime

import typeguard
from pydantic import BaseModel, Field

from forecasting_tools.forecasting.questions_and_reports.binary_report import (
    BinaryReport,
)
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ForecastReport,
)
from forecasting_tools.forecasting.questions_and_reports.multiple_choice_report import (
    MultipleChoiceReport,
)
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    NumericReport,
)
from forecasting_tools.util.jsonable import Jsonable


class BenchmarkForBot(BaseModel, Jsonable):
    name: str
    description: str
    timestamp: datetime = Field(default_factory=datetime.now)
    time_taken_in_minutes: float | None
    total_cost: float | None
    git_commit_hash: str
    forecast_bot_config: dict[str, str]
    code: str | None = None
    forecast_reports: list[BinaryReport | NumericReport | MultipleChoiceReport]

    @property
    def average_inverse_expected_log_score(self) -> float:
        reports = typeguard.check_type(
            self.forecast_reports,
            list[ForecastReport],
        )
        return ForecastReport.calculate_average_inverse_expected_log_score(
            reports
        )
