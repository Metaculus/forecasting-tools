from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, Field, field_validator

from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import DiscreteQuestion, NumericQuestion

logger = logging.getLogger(__name__)


class Percentile(BaseModel):
    percentile: float = Field(
        ge=0,
        le=1,
        description="A number between 0 and 1 (e.g. '90% of people are age 60 or younger' translates to '0.9')",
    )
    value: float = Field(
        description="The number matching the percentile (e.g. '90% of people are age 60 or younger' translates to '60')",
    )


class NumericDistribution(BaseModel):
    declared_percentiles: list[Percentile]
    open_upper_bound: bool
    open_lower_bound: bool
    upper_bound: float
    lower_bound: float
    zero_point: float | None
    cdf_size: int | None = (
        None  # Normal numeric questions have 201 points, but discrete questions have fewer
    )

    @classmethod
    def from_question(
        cls, percentiles: list[Percentile], question: NumericQuestion
    ) -> NumericDistribution:
        return NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
            cdf_size=question.cdf_size,
        )

    @property
    def inversed_expected_log_score(self) -> float | None:
        raise NotImplementedError("Not implemented")

    @property
    def community_prediction(self) -> NumericDistribution | None:
        raise NotImplementedError("Not implemented")

    @field_validator("declared_percentiles")
    def validate_percentiles_order(
        cls: NumericDistribution, percentiles: list[Percentile]
    ) -> list[Percentile]:
        for i in range(len(percentiles) - 1):
            if percentiles[i].percentile >= percentiles[i + 1].percentile:
                raise ValueError("Percentiles must be in strictly increasing order")
            if percentiles[i].value > percentiles[i + 1].value:
                raise ValueError("Values must be in strictly increasing order")
        if len(percentiles) < 2:
            raise ValueError("NumericDistribution must have at least 2 percentiles")
        return percentiles

    @property
    def cdf(self) -> list[Percentile]:
        """
        Turns a list of percentiles into a full distribution (201 points, if numeric, otherwise based on discrete values)
        between upper and lower bound (taking into account probability assigned above and below the bounds)
        that is compatible with Metaculus questions.

        cdf stands for 'continuous distribution function'
        """
        # TODO: This function needs to be cleaned up and made more readable

        percentiles = self.declared_percentiles
        open_upper_bound = self.open_upper_bound
        open_lower_bound = self.open_lower_bound
        upper_bound = self.upper_bound
        lower_bound = self.lower_bound
        zero_point = self.zero_point
        cdf_size = self.cdf_size or 201

        # Convert to dict so we don't have to rewrite this function
        percentile_values: dict[float, float] = {
            percentile.percentile * 100: percentile.value for percentile in percentiles
        }

        percentile_max = max(float(key) for key in percentile_values.keys())
        percentile_min = min(float(key) for key in percentile_values.keys())
        range_min = lower_bound
        range_max = upper_bound
        range_size = abs(range_max - range_min)
        buffer = 1 if range_size > 100 else 0.01 * range_size

        # Adjust any values that are exactly at the bounds
        for percentile, value in list(percentile_values.items()):
            if not open_lower_bound and value <= range_min + buffer:
                percentile_values[percentile] = range_min + buffer
            if not open_upper_bound and value >= range_max - buffer:
                percentile_values[percentile] = range_max - buffer

        # Set cdf values outside range
        if open_upper_bound:
            if range_max > percentile_values[percentile_max]:
                percentile_values[int(100 - (0.5 * (100 - percentile_max)))] = range_max
        else:
            percentile_values[100] = range_max

        # Set cdf values outside range
        if open_lower_bound:
            if range_min < percentile_values[percentile_min]:
                percentile_values[int(0.5 * percentile_min)] = range_min
        else:
            percentile_values[0] = range_min

        sorted_percentile_values = dict(sorted(percentile_values.items()))

        # Normalize percentile keys
        normalized_percentile_values = {}
        for key, value in sorted_percentile_values.items():
            percentile = float(key) / 100
            normalized_percentile_values[percentile] = value

        value_percentiles = {
            value: key for key, value in normalized_percentile_values.items()
        }

        # function for log scaled questions
        def generate_cdf_locations(
            range_min: float, range_max: float, zero_point: float | None
        ) -> list[float]:
            if zero_point is None:
                scale = lambda x: range_min + (range_max - range_min) * x
            else:
                deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
                scale = lambda x: range_min + (range_max - range_min) * (
                    deriv_ratio**x - 1
                ) / (deriv_ratio - 1)
            return [scale(x) for x in np.linspace(0, 1, cdf_size)]

        cdf_xaxis = generate_cdf_locations(range_min, range_max, zero_point)

        def linear_interpolation(
            x_values: list[float], xy_pairs: dict[float, float]
        ) -> list[float]:
            # Sort the xy_pairs by x-values
            sorted_pairs = sorted(xy_pairs.items())

            # Extract sorted x and y values
            known_x = [pair[0] for pair in sorted_pairs]
            known_y = [pair[1] for pair in sorted_pairs]

            # Initialize the result list
            y_values = []

            for x in x_values:
                # Check if x is exactly in the known x values
                if x in known_x:
                    y_values.append(known_y[known_x.index(x)])
                else:
                    # Find the indices of the two nearest known x-values
                    i = 0
                    while i < len(known_x) and known_x[i] < x:
                        i += 1
                    # If x is outside the range of known x-values, use the nearest endpoint
                    if i == 0:
                        y_values.append(known_y[0])
                    elif i == len(known_x):
                        y_values.append(known_y[-1])
                    else:
                        # Perform linear interpolation
                        x0, x1 = known_x[i - 1], known_x[i]
                        y0, y1 = known_y[i - 1], known_y[i]

                        # Linear interpolation formula
                        y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                        y_values.append(y)

            return y_values

        continuous_cdf = linear_interpolation(cdf_xaxis, value_percentiles)

        percentiles = [
            Percentile(value=value, percentile=percentile)
            for value, percentile in zip(cdf_xaxis, continuous_cdf)
        ]
        assert len(percentiles) == cdf_size

        # Validate minimum spacing between consecutive values
        for i in range(len(percentiles) - 1):
            assert (
                abs(percentiles[i + 1].percentile - percentiles[i].percentile) >= 5e-05
            ), (
                f"Percentiles at indices {i} and {i+1} are too close: "
                f"{percentiles[i].percentile} and {percentiles[i+1].percentile} "
                f"at values {percentiles[i].value} and {percentiles[i+1].value}. "
                "It is possible that your prediction is mostly or completely out of the upper/lower bound range "
                "Thus making this cdf mostly meaningless."
            )

        return percentiles

    def get_representative_percentiles(
        self, num_percentiles: int = 5
    ) -> list[Percentile]:
        if num_percentiles <= 1:
            raise ValueError("Number of percentiles must be at least 2")

        starting_percentiles = self.declared_percentiles
        if num_percentiles > len(starting_percentiles):
            logger.warning(
                f"Number of percentiles requested ({num_percentiles}) is greater than the number of declared percentiles in the distribution ({len(starting_percentiles)}). Using all percentiles."
            )
            num_percentiles = len(starting_percentiles)

        desired_percentile_points = np.linspace(
            0, len(starting_percentiles) - 1, num_percentiles
        )
        desired_indices = [int(round(point)) for point in desired_percentile_points]

        representative_percentiles = [
            starting_percentiles[idx] for idx in desired_indices
        ]
        return representative_percentiles


class NumericReport(ForecastReport):
    question: NumericQuestion
    prediction: NumericDistribution

    @classmethod
    async def aggregate_predictions(
        cls, predictions: list[NumericDistribution], question: NumericQuestion
    ) -> NumericDistribution:
        assert predictions, "No predictions to aggregate"
        cdfs = [prediction.cdf for prediction in predictions]
        all_percentiles_of_cdf: list[list[float]] = []
        all_values_of_cdf: list[list[float]] = []
        x_axis: list[float] = [percentile.value for percentile in cdfs[0]]
        for cdf in cdfs:
            all_percentiles_of_cdf.append([percentile.percentile for percentile in cdf])
            all_values_of_cdf.append([percentile.value for percentile in cdf])

        for cdf in cdfs:
            for i in range(len(cdf)):
                if cdf[i].value != x_axis[i]:
                    raise ValueError("X axis between cdfs is not the same")

        median_percentile_list: list[float] = np.median(
            np.array(all_percentiles_of_cdf), axis=0
        ).tolist()
        median_cdf = [
            Percentile(value=value, percentile=percentile)
            for value, percentile in zip(x_axis, median_percentile_list)
        ]

        if not predictions:
            raise ValueError("No predictions to aggregate")

        return NumericDistribution.from_question(median_cdf, question)

    @classmethod
    def make_readable_prediction(cls, prediction: NumericDistribution) -> str:
        num_percentiles = len(prediction.declared_percentiles)
        if num_percentiles > 10:
            num_display_percentiles = 5
        else:
            num_display_percentiles = num_percentiles
        representative_percentiles = prediction.get_representative_percentiles(
            num_display_percentiles
        )
        readable = "Probability distribution:\n"
        for percentile in representative_percentiles:
            readable += f"- {percentile.percentile:.2%} chance of value below {round(percentile.value,4)}\n"
        return readable

    async def publish_report_to_metaculus(self) -> None:
        from forecasting_tools.helpers.metaculus_api import MetaculusApi

        if self.question.id_of_question is None:
            raise ValueError("Publishing to Metaculus requires a question ID")

        if self.question.id_of_post is None:
            raise ValueError(
                "Publishing to Metaculus requires a post ID for the question"
            )

        prediction = self.prediction
        if prediction.cdf_size is None:
            prediction = NumericDistribution.from_question(
                prediction.declared_percentiles, self.question
            )

        cdf_probabilities = [percentile.percentile for percentile in prediction.cdf]

        MetaculusApi.post_numeric_question_prediction(
            self.question.id_of_question, cdf_probabilities
        )
        MetaculusApi.post_question_comment(self.question.id_of_post, self.explanation)


class DiscreteReport(NumericReport):
    question: DiscreteQuestion
    prediction: NumericDistribution
