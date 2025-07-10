from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.data_organizer import ReportTypes
from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
from forecasting_tools.data_models.numeric_report import NumericReport


class GeneratedQuestion(SimpleQuestion):
    forecast_report: ReportTypes | None = None
    error_message: str | None = None

    @property
    def is_uncertain(self) -> bool:
        """Determines if a forecast shows sufficient uncertainty."""
        report = self.forecast_report
        if report is None or isinstance(report, Exception):
            return False

        # Use heuristics to determine uncertainty of question
        if isinstance(report, BinaryReport):
            probability = report.prediction
            is_uncertain = 0.1 <= probability <= 0.9
        elif isinstance(report, NumericReport):
            is_uncertain = True
        elif isinstance(report, MultipleChoiceReport):
            for option in report.prediction.predicted_options:
                if option.probability > 0.8 or option.probability < 0.05:
                    is_uncertain = False
                    break
            else:
                is_uncertain = True
        else:
            is_uncertain = False
        return is_uncertain
