import logging

from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    Percentile,
)
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot

logger = logging.getLogger(__name__)


class UniformProbabilityBot(ForecastBot):
    """
    This bot predicts a uniform probability for all options.
    """

    async def run_research(self, question: MetaculusQuestion) -> str:
        return "No research needed for coin flip bot"

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        return ReasonedPrediction(
            prediction_value=0.5, reasoning="Always predict 50%"
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        options = question.options
        prediction_per_option = 1 / len(options)
        prediction: PredictedOptionList = PredictedOptionList(
            predicted_options=[
                PredictedOption(
                    option_name=option,
                    probability=prediction_per_option,
                )
                for option in options
            ]
        )
        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning="Predicted equal probability for all options",
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        lower_bound = question.lower_bound
        upper_bound = question.upper_bound
        distribution_range = upper_bound - lower_bound

        percentiles = [
            Percentile(
                value=lower_bound + 0.1 * distribution_range,
                percentile=0.1,
            ),
            Percentile(
                value=lower_bound + 0.3 * distribution_range,
                percentile=0.3,
            ),
            Percentile(
                value=lower_bound + 0.5 * distribution_range,
                percentile=0.5,
            ),
            Percentile(
                value=lower_bound + 0.7 * distribution_range,
                percentile=0.7,
            ),
            Percentile(
                value=lower_bound + 0.9 * distribution_range,
                percentile=0.9,
            ),
        ]

        distribution = NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )

        return ReasonedPrediction(
            prediction_value=distribution,
            reasoning=(
                "Created a uniform distribution between the lower and upper bounds. "
                "NOTE: The cdf will have sloping probability at the edges if the bounds are open"
            ),
        )
