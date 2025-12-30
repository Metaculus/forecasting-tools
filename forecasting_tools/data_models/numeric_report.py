from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import typing_extensions
from pydantic import BaseModel, Field, model_validator

from forecasting_tools.data_models.forecast_report import ForecastReport

if TYPE_CHECKING:
    from forecasting_tools.data_models.questions import (
        DateQuestion,
        DiscreteQuestion,
        NumericQuestion,
    )
    from forecasting_tools.helpers.metaculus_client import MetaculusClient
logger = logging.getLogger(__name__)


class NumericDefaults:
    DEFAULT_CDF_SIZE = (
        201  # Discrete questions have fewer points, Numeric will have 201 points
    )
    DEFAULT_INBOUND_OUTCOME_COUNT = DEFAULT_CDF_SIZE - 1
    MAX_NUMERIC_PMF_VALUE = 0.2

    @classmethod
    def get_max_pmf_value(
        cls, cdf_size: int, include_wiggle_room: bool = True
    ) -> float:
        # cap depends on inboundOutcomeCount (0.2 if it is the default 200)
        inbound_outcome_count = cdf_size - 1
        normal_cap = cls.MAX_NUMERIC_PMF_VALUE * (
            cls.DEFAULT_INBOUND_OUTCOME_COUNT / inbound_outcome_count
        )

        if include_wiggle_room:
            return normal_cap * 0.95
        else:
            return normal_cap


class Percentile(BaseModel):
    percentile: float = Field(
        description="A number between 0 and 1 (e.g. '90% of people are age 60 or younger' translates to '0.9')",
    )
    value: float = Field(
        description="The number matching the percentile (e.g. '90% of people are age 60 or younger' translates to '60')",
    )

    @model_validator(mode="after")
    def validate_percentile(self: Percentile) -> Percentile:
        if self.percentile < 0 or self.percentile > 1:
            raise ValueError(
                f"Percentile must be between 0 and 1, but was {self.percentile}"
            )
        if np.isnan(self.percentile):
            raise ValueError(f"Percentile must be a number, but was {self.percentile}")
        return self


class DatePercentile(BaseModel):
    percentile: float = Field(
        description="A number between 0 and 1 (e.g. '90% likelihood of AGI by 2040-01-01' translates to '0.9')",
    )
    value: datetime = Field(
        description="The date matching the percentile (e.g. '90% likelihood of AGI by 2040-01-01' translates to '2040-01-01')",
    )

    @model_validator(mode="after")
    def validate_percentile(self: DatePercentile) -> DatePercentile:
        if self.percentile < 0 or self.percentile > 1:
            raise ValueError(
                f"Percentile must be between 0 and 1, but was {self.percentile}"
            )
        if np.isnan(self.percentile):
            raise ValueError(f"Percentile must be a number, but was {self.percentile}")
        return self


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
    standardize_cdf: bool = True
    strict_validation: bool = True
    is_date: bool = False

    @model_validator(mode="after")
    def validate_percentiles(self: NumericDistribution) -> NumericDistribution:
        percentiles = self.declared_percentiles
        self._check_percentiles_increasing()
        self._check_log_scaled_fields()

        if not self.strict_validation:
            return self

        self._check_percentile_spacing()

        if self.standardize_cdf:
            self._check_too_far_from_bounds(percentiles)
        if self.standardize_cdf and len(percentiles) == self.cdf_size:
            self._check_distribution_too_tall(percentiles)

        self.declared_percentiles = self._check_and_update_repeating_values(percentiles)
        return self

    def _check_percentiles_increasing(self) -> None:
        percentiles = self.declared_percentiles
        for i in range(len(percentiles) - 1):
            if percentiles[i].percentile >= percentiles[i + 1].percentile:
                raise ValueError("Percentiles must be in strictly increasing order")
            if percentiles[i].value > percentiles[i + 1].value:
                raise ValueError("Values must be in strictly increasing order")
        if len(percentiles) < 2:
            raise ValueError("NumericDistribution must have at least 2 percentiles")

    def _check_percentile_spacing(self) -> None:
        percentiles = self.declared_percentiles
        for i in range(len(percentiles) - 1):
            if abs(percentiles[i + 1].percentile - percentiles[i].percentile) < 5e-05:
                raise ValueError(
                    f"Percentiles at indices {i} and {i+1} are too close. CDF must be increasing by at least 5e-05 at every step. "
                    f"{percentiles[i].percentile} and {percentiles[i+1].percentile} "
                    f"at values {percentiles[i].value} and {percentiles[i+1].value}. "
                    "One possible reason is that your prediction is mostly or completely out of the upper/lower "
                    "bound range thus assigning very little probability to any one x-axis value."
                )

    def _check_log_scaled_fields(self) -> None:
        if self.zero_point is not None and self.lower_bound <= self.zero_point:
            raise ValueError(
                f"Lower bound {self.lower_bound} is less than or equal to the zero point {self.zero_point}. "
                "Lower bound must be greater than the zero point."
            )

        for percentile in self.declared_percentiles:
            if self.zero_point is not None and percentile.value < self.zero_point:
                raise ValueError(
                    f"Percentile value {percentile.value} is less than the zero point {self.zero_point}. "
                    "Determining probability less than zero point is currently not supported."
                )

    def _check_and_update_repeating_values(
        self, percentiles: list[Percentile]
    ) -> list[Percentile]:
        unique_value_count = Counter(percentile.value for percentile in percentiles)
        final_percentiles = []
        for percentile in percentiles:
            value = percentile.value
            count = unique_value_count[value]
            repeated_value = count > 1
            value_in_bounds = self.lower_bound < value < self.upper_bound
            value_above_bound = value >= self.upper_bound
            value_below_bound = value <= self.lower_bound
            epsilon = 1e-10
            if not repeated_value:
                final_percentiles.append(percentile)
            elif value_in_bounds:
                greater_epsilon = 1e-6  # TODO: Figure out why normal epsilon doesn't work. Could cause brittle behavior.
                modification = (1 - percentile.percentile) * greater_epsilon
                final_percentiles.append(
                    Percentile(
                        value=value - modification,
                        percentile=percentile.percentile,
                    )
                )
            elif value_above_bound:
                modification = epsilon * percentile.percentile
                final_percentiles.append(
                    Percentile(
                        value=self.upper_bound + modification,
                        percentile=percentile.percentile,
                    )
                )
            elif value_below_bound:
                modification = epsilon * (1 - percentile.percentile)
                final_percentiles.append(
                    Percentile(
                        value=self.lower_bound - modification,
                        percentile=percentile.percentile,
                    )
                )
            else:
                raise ValueError(
                    f"Unexpected state: value {value} is repeated {count} times. Bound is {self.lower_bound} and {self.upper_bound}"
                )
        return final_percentiles

    def _check_too_far_from_bounds(self, percentiles: list[Percentile]) -> None:
        max_to_min_range = self.upper_bound - self.lower_bound

        # TODO: Better handle log scaled questions (a fixed wiggle room percentage doesn't work well for them)
        wiggle_percent = 0.25
        wiggle_room = max_to_min_range * wiggle_percent
        upper_bound_plus_wiggle_room = self.upper_bound + wiggle_room
        lower_bound_minus_wiggle_room = self.lower_bound - wiggle_room
        percentiles_within_bounds_plus_wiggle_room = [
            percentile
            for percentile in percentiles
            if lower_bound_minus_wiggle_room
            <= percentile.value
            <= upper_bound_plus_wiggle_room
        ]
        if len(percentiles_within_bounds_plus_wiggle_room) == 0:
            raise ValueError(
                f"No declared percentiles are within the range of the question +/- {wiggle_percent * 100}%. "
                f"Lower bound: {self.lower_bound}, upper bound: {self.upper_bound}. "
                f"Percentiles: {percentiles}"
            )

        max_to_min_range_buffer = max_to_min_range * 2
        percentiles_far_exceeding_bounds = [
            percentile
            for percentile in percentiles
            if percentile.value < self.lower_bound - max_to_min_range_buffer
            or percentile.value > self.upper_bound + max_to_min_range_buffer
        ]
        if len(percentiles_far_exceeding_bounds) > 0:
            raise ValueError(
                "Some declared percentiles are far exceeding the bounds of the question. "
                f"Lower bound: {self.lower_bound}, upper bound: {self.upper_bound}. "
                f"Percentiles: {percentiles_far_exceeding_bounds}"
            )

    def _check_distribution_too_tall(self, cdf: list[Percentile]) -> None:
        if len(cdf) != self.cdf_size:
            raise ValueError(
                f"CDF size is not the same as the declared percentiles. CDF size: {len(cdf)}, declared percentiles: {self.cdf_size}"
            )
        cap = NumericDefaults.get_max_pmf_value(len(cdf), include_wiggle_room=False)

        for i in range(len(cdf) - 1):
            pmf_value = cdf[i + 1].percentile - cdf[i].percentile
            if pmf_value > cap:
                raise ValueError(
                    f"Distribution is too concentrated. The probability mass between "
                    f"values {cdf[i].value} and {cdf[i + 1].value} is {pmf_value:.4f}, "
                    f"which exceeds the maximum allowed of {cap:.4f}."
                )

    @classmethod
    def from_question(
        cls,
        percentiles: list[Percentile],
        question: NumericQuestion | DateQuestion,
        standardize_cdf: bool | None = None,
    ) -> NumericDistribution:
        from forecasting_tools.data_models.questions import DateQuestion

        is_date = isinstance(question, DateQuestion)

        if is_date:
            upper_bound_float: float = question.upper_bound.timestamp()
            lower_bound_float: float = question.lower_bound.timestamp()
        else:
            upper_bound_float = question.upper_bound
            lower_bound_float = question.lower_bound

        if standardize_cdf is None:
            return NumericDistribution(
                declared_percentiles=percentiles,
                open_upper_bound=question.open_upper_bound,
                open_lower_bound=question.open_lower_bound,
                upper_bound=upper_bound_float,
                lower_bound=lower_bound_float,
                zero_point=question.zero_point,
                cdf_size=question.cdf_size,
                is_date=is_date,
            )
        else:
            return NumericDistribution(
                declared_percentiles=percentiles,
                open_upper_bound=question.open_upper_bound,
                open_lower_bound=question.open_lower_bound,
                upper_bound=upper_bound_float,
                lower_bound=lower_bound_float,
                zero_point=question.zero_point,
                cdf_size=question.cdf_size,
                standardize_cdf=standardize_cdf,
                is_date=is_date,
            )

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

    @property
    @typing_extensions.deprecated(
        "NumericDistribution.cdf (property) will be replaced with NumericDistribution.get_cdf (method). Please switch.",
        category=None,
    )
    def cdf(self) -> list[Percentile]:
        return self.get_cdf()

    def get_cdf(self) -> list[Percentile]:
        """
        Turns a list of percentiles into a full distribution (201 points, if numeric, otherwise based on discrete values)
        between upper and lower bound (taking into account probability assigned above and below the bounds)
        that is compatible with Metaculus questions.

        cdf stands for 'continuous distribution function'

        At Metaculus CDFs are often represented with 201 points. Each point has:
        - percentile ("X% of values are below this point". This is the y axis of the cdf graph)
        - 'value' or 'nominal location' (The real world number that answers the question)
        - cdf location (a number between 0 and 1 representing where the point is on the cdf x axis, where 0 is range min, and 1 is range max)
        """

        cdf_size = self.cdf_size or NumericDefaults.DEFAULT_CDF_SIZE
        continuous_cdf = []
        cdf_xaxis = []
        cdf_eval_locations = [i / (cdf_size - 1) for i in range(cdf_size)]
        for l in cdf_eval_locations:
            continuous_cdf.append(self._get_cdf_at(l))
            cdf_xaxis.append(self._cdf_location_to_nominal_location(l))

        if self.standardize_cdf:
            continuous_cdf = self._standardize_cdf(continuous_cdf)

        percentiles = [
            Percentile(value=value, percentile=percentile)
            for value, percentile in zip(cdf_xaxis, continuous_cdf)
        ]
        assert len(percentiles) == cdf_size

        validation_distribution = NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=self.open_upper_bound,
            open_lower_bound=self.open_lower_bound,
            upper_bound=self.upper_bound,
            lower_bound=self.lower_bound,
            zero_point=self.zero_point,
            standardize_cdf=self.standardize_cdf,
        )
        NumericDistribution.model_validate(validation_distribution)
        return percentiles

    @classmethod
    def _percentile_list_to_dict(
        cls, percentiles: list[Percentile], multiply_by_100: bool
    ) -> dict[float, float]:
        return {
            (
                percentile.percentile * 100
                if multiply_by_100
                else percentile.percentile
            ): percentile.value
            for percentile in percentiles
        }

    @classmethod
    def _dict_to_percentile_list(
        cls, percentile_dict: dict[float, float], divide_by_100: bool
    ) -> list[Percentile]:
        return [
            Percentile(
                percentile=percentile / 100 if divide_by_100 else percentile,
                value=value,
            )
            for percentile, value in percentile_dict.items()
        ]

    def _add_explicit_upper_lower_bound_percentiles(
        self,
        input_percentiles: list[Percentile],
    ) -> list[Percentile]:
        open_upper_bound = self.open_upper_bound
        open_lower_bound = self.open_lower_bound
        range_max = self.upper_bound
        range_min = self.lower_bound

        return_percentiles = self._percentile_list_to_dict(
            input_percentiles, multiply_by_100=True
        )
        percentile_max = max(percentile for percentile in return_percentiles.keys())
        percentile_min = min(percentile for percentile in return_percentiles.keys())
        range_size = abs(range_max - range_min)
        buffer = 1 if range_size > 100 else 0.01 * range_size

        # Adjust any values that are exactly at the bounds
        for percentile, value in list(return_percentiles.items()):
            # TODO: Handle this more gracefully for log scaled questions
            #  (where buffer could be quite a bit on the lower bound side)
            if not open_lower_bound and value <= range_min + buffer:
                return_percentiles[percentile] = range_min + buffer
            if not open_upper_bound and value >= range_max - buffer:
                return_percentiles[percentile] = range_max - buffer

        # Set cdf values outside range
        if open_upper_bound:
            if range_max > return_percentiles[percentile_max]:
                halfway_between_max_and_100th_percentile = 100 - (
                    0.5 * (100 - percentile_max)
                )
                return_percentiles[halfway_between_max_and_100th_percentile] = range_max
        else:
            return_percentiles[100] = range_max

        # Set cdf values outside range
        if open_lower_bound:
            if range_min < return_percentiles[percentile_min]:
                halfway_between_min_and_0th_percentile = 0.5 * percentile_min
                return_percentiles[halfway_between_min_and_0th_percentile] = range_min
        else:
            return_percentiles[0] = range_min

        sorted_return_percentiles = dict(sorted(return_percentiles.items()))

        return_list = self._dict_to_percentile_list(
            sorted_return_percentiles, divide_by_100=True
        )
        return return_list

    def _nominal_location_to_cdf_location(self, nominal_value: float) -> float:
        """
        Takes the real world value (like $17k - that would answer the forecasting question)
        and converts it to a cdf location between 0 and 1 depending on
        how far it is between the upper and lower bound
        (it can go over 1 or below 0 if beyond the bounds)
        """
        range_max = self.upper_bound
        range_min = self.lower_bound
        zero_point = self.zero_point

        if zero_point is not None:
            # logarithmically scaled question
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            if nominal_value == zero_point:
                # If nominal = zero point, then you would take the log of 0. Add a small epsilon to avoid this.
                nominal_value += 1e-10
            unscaled_location = (
                np.log(
                    (nominal_value - range_min) * (deriv_ratio - 1)
                    + (range_max - range_min)
                )
                - np.log(range_max - range_min)
            ) / np.log(deriv_ratio)
        else:
            # linearly scaled question
            unscaled_location = (nominal_value - range_min) / (range_max - range_min)
        return float(unscaled_location)

    def _get_cdf_at(self, cdf_location: float) -> float:
        """
        Helper function that takes a cdf location and returns
        the height (percentile) of the cdf at that location, linearly
        interpolating between values
        """
        bounded_percentiles = self._add_explicit_upper_lower_bound_percentiles(
            self.declared_percentiles
        )
        cdf_location_to_percentile_mapping: list[tuple[float, float]] = []
        for percentile in bounded_percentiles:
            height = percentile.percentile
            location = self._nominal_location_to_cdf_location(percentile.value)
            cdf_location_to_percentile_mapping.append((location, height))
        previous = cdf_location_to_percentile_mapping[0]
        for i in range(1, len(cdf_location_to_percentile_mapping)):
            current = cdf_location_to_percentile_mapping[i]
            epsilon = 1e-10
            if previous[0] - epsilon <= cdf_location <= current[0] + epsilon:
                result = previous[1] + (current[1] - previous[1]) * (
                    cdf_location - previous[0]
                ) / (current[0] - previous[0])
                if np.isnan(result):
                    raise ValueError(f"Result is NaN for cdf location {cdf_location}")
                return result
            previous = current
        raise ValueError(f"CDF location Input {cdf_location} cannot be found")

    def _standardize_cdf(self, cdf: list[float] | np.ndarray) -> list[float]:
        """
        See documentation: https://metaculus.com/api/#:~:text=CDF%20generation%20details in the
            "CDF generation details and examples" section

        Takes a cdf and returns a standardized version of it

        - assigns no mass outside of closed bounds (scales accordingly)
        - assigns at least a minimum amount of mass outside of open bounds
        - increasing by at least the minimum amount (0.01 / 200 = 0.0005)
        - caps the maximum growth to 0.2

        Note, thresholds change with different `inbound_outcome_count`s
        """

        lower_open = self.open_lower_bound
        upper_open = self.open_upper_bound

        # apply lower bound & enforce boundary values
        scale_lower_to = 0 if lower_open else cdf[0]
        scale_upper_to = 1.0 if upper_open else cdf[-1]
        rescaled_inbound_mass = scale_upper_to - scale_lower_to

        def apply_minimum(F: float, location: float) -> float:
            # `F` is the height of the cdf at `location` (in range [0, 1])
            # rescale
            rescaled_F = (F - scale_lower_to) / rescaled_inbound_mass
            # offset
            if lower_open and upper_open:
                return 0.988 * rescaled_F + 0.01 * location + 0.001
            elif lower_open:
                return 0.989 * rescaled_F + 0.01 * location + 0.001
            elif upper_open:
                return 0.989 * rescaled_F + 0.01 * location
            return 0.99 * rescaled_F + 0.01 * location

        for i, value in enumerate(cdf):
            cdf[i] = apply_minimum(value, i / (len(cdf) - 1))

        # apply upper bound
        # operate in PMF space
        pmf = np.diff(cdf, prepend=0, append=1)
        cap = NumericDefaults.get_max_pmf_value(len(cdf))

        def cap_pmf(scale: float) -> np.ndarray:
            return np.concatenate(
                [pmf[:1], np.minimum(cap, scale * pmf[1:-1]), pmf[-1:]]
            )

        def capped_sum(scale: float) -> float:
            return float(cap_pmf(scale).sum())

        # find the appropriate scale search space
        lo = hi = scale = 1.0
        while capped_sum(hi) < 1.0:
            hi *= 1.2
        # hone in on scale value that makes capped sum 1
        for _ in range(100):
            scale = 0.5 * (lo + hi)
            s = capped_sum(scale)
            if s < 1.0:
                lo = scale
            else:
                hi = scale
            if s == 1.0 or (hi - lo) < 2e-5:
                break
        # apply scale and renormalize
        pmf = cap_pmf(scale)
        pmf[1:-1] *= (cdf[-1] - cdf[0]) / pmf[1:-1].sum()
        # back to CDF space
        cdf = np.cumsum(pmf)[:-1]

        # round to minimize floating point errors
        cdf = np.round(cdf, 10)
        return cdf.tolist()

    def _cdf_location_to_nominal_location(self, cdf_location: float) -> float:
        range_max = self.upper_bound
        range_min = self.lower_bound
        zero_point = self.zero_point

        if zero_point is None:
            scaled_location = range_min + (range_max - range_min) * cdf_location
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scaled_location = range_min + (range_max - range_min) * (
                deriv_ratio**cdf_location - 1
            ) / (deriv_ratio - 1)
        if np.isnan(scaled_location):
            raise ValueError(f"Scaled location is NaN for cdf location {cdf_location}")
        return scaled_location


class NumericReport(ForecastReport):
    question: NumericQuestion
    prediction: NumericDistribution

    @classmethod
    async def aggregate_predictions(
        cls,
        predictions: list[NumericDistribution],
        question: NumericQuestion | DateQuestion,
    ) -> NumericDistribution:
        assert predictions, "No predictions to aggregate"
        cdfs = [prediction.get_cdf() for prediction in predictions]
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

        final_distribution = NumericDistribution.from_question(median_cdf, question)
        return final_distribution

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
            if prediction.is_date:
                formatted_value = datetime.fromtimestamp(
                    percentile.value, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                formatted_value = str(round(percentile.value, 6))
            readable += f"- {percentile.percentile:.2%} chance of value below {formatted_value}\n"
        return readable

    async def publish_report_to_metaculus(
        self, metaculus_client: MetaculusClient | None = None
    ) -> None:
        from forecasting_tools.helpers.metaculus_client import MetaculusClient

        metaculus_client = metaculus_client or MetaculusClient()

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

        cdf_probabilities = [
            percentile.percentile for percentile in prediction.get_cdf()
        ]

        metaculus_client.post_numeric_question_prediction(
            self.question.id_of_question, cdf_probabilities
        )
        metaculus_client.post_question_comment(
            self.question.id_of_post, self.explanation
        )


class DiscreteReport(NumericReport):
    question: DiscreteQuestion
    prediction: NumericDistribution


class DateReport(NumericReport):
    question: DateQuestion
    prediction: NumericDistribution
