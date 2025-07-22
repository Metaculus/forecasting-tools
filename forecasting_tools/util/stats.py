from math import sqrt

import numpy as np
from pydantic import BaseModel
from scipy.stats import norm, shapiro, t


class ProportionStatCalculator:
    def __init__(self, number_of_successes: int, number_of_trials: int):
        self.number_of_successes: int = number_of_successes
        self.number_of_trials: int = number_of_trials

    def determine_if_population_proportion_is_above_p0(
        self, p0: float, desired_confidence: float
    ) -> tuple[float, bool, str]:
        """
        Requirements
        - Simple random sample
        - Binomial distribution conditions are satisfied
        - np >= 5 and nq >= 5 where n is the number of trials, p is the population proportion, and q is 1 - p

        Input
        - p0: float - the population proportion that is being tested
        - desired_confidence: float - the desired confidence level

        Output
        - p_value: float - the p-value of the test
        - hypothesis_rejected: bool - whether the null hypothesis is rejected
        - written_conclusion: str - a written conclusion of the test

        Useful values
        - A sample success of 98/100 (98%) needed to show that proportion is above 95% with 90% confidence
        - A sample success of 194/200 (97%) needed to show that proportion is above 95% with 90% confidence
        - A sample success of 27/30 (90%) needed to show that proportion is above 80% with 90% confidence
        - A sample success of 86/100 (86%) needed to show that proportion is above 80% with 90% confidence

        For an online test Calculator see the below (it may or may not use t distribution at lower sample sizes)
        https://www.statssolver.com/hypothesis-testing.html
        """

        if self.number_of_trials * p0 < 5 or self.number_of_trials * (1 - p0) < 5:
            raise ValueError(
                "The normal distribution approximation conditions are not satisfied. Too few samples given the desired p0 to test"
            )

        sample_proportion = self.number_of_successes / self.number_of_trials

        if (
            self.number_of_trials * sample_proportion < 5
            or self.number_of_trials * (1 - sample_proportion) < 5
        ):
            raise ValueError(
                "The normal distribution approximation conditions are not satisfied. Sample proportion is too close to 0 or 1"
            )

        standard_error = sqrt(p0 * (1 - p0) / self.number_of_trials)
        z_score = (sample_proportion - p0) / standard_error

        # Since we're testing 'larger', find the area to the right of the z-score
        p_value: float = float(1 - norm.cdf(z_score))

        alpha = 1 - desired_confidence
        hypothesis_rejected = p_value < alpha

        if hypothesis_rejected:
            written_conclusion = f"If the null hypothesis is true (the proportion is equal to and not greater than {p0*100:.2f}%), then there is a {p_value*100:.2f}% probability that the sample (estimated) proportion is {sample_proportion*100:.2f}% or more (the success percentage found in the sample). Thus, we reject the null hypothesis with {desired_confidence*100:.2f}% confidence since at the {alpha*100:.2f}% level of significance, the sample data do give enough evidence to conclude that the proportion is greater than {p0*100:.2f}%. There were {self.number_of_successes} successes in {self.number_of_trials} trials."
        else:
            written_conclusion = f"If the null hypothesis is true (the proportion is equal to and not greater than {p0*100:.2f}%), then there is a {p_value*100:.2f}% probability that the sample (estimated) proportion is {sample_proportion*100:.2f}% or more (the success percentage found in the sample). Thus, we fail to reject the null hypothesis since at the {alpha*100:.2f}% level of significance, the sample data do not give enough evidence to conclude that the proportion is greater than {p0*100:.2f}%. There were {self.number_of_successes} successes in {self.number_of_trials} trials."
        return p_value, hypothesis_rejected, written_conclusion


class ConfidenceInterval(BaseModel):
    mean: float
    margin_of_error: float
    standard_deviation: float

    @property
    def lower_bound(self) -> float:
        return self.mean - self.margin_of_error

    @property
    def upper_bound(self) -> float:
        return self.mean + self.margin_of_error


class ConfidenceIntervalCalculator:

    @classmethod
    def confidence_interval_from_observations(
        cls, observations: list[float], confidence: float = 0.9
    ) -> ConfidenceInterval:
        """
        This solves the following stats problem:
        'estimating population mean with unknown population standard deviation'

        Requirements
        - Simple random sample
        - Either the sample is from a normally distributed population or n >30
        - Observations are independent
        """
        assert 0 <= confidence <= 1, "Confidence must be between 0 and 1"
        assert len(observations) > 0, "Observations must be non-empty"
        assert (
            len(observations) > 3
        ), "Must have at least 3 observations to check for normality"

        sample_size = len(observations)
        if sample_size < 2:
            raise ValueError("Not enough data for T-based confidence interval")

        test_normality_assumption(sample_size, observations)

        sample_mean = np.mean(observations)
        sample_std = np.std(observations, ddof=1)

        return cls.confidence_interval_from_mean_and_std(
            float(sample_mean), float(sample_std), sample_size, confidence
        )

    @classmethod
    def confidence_interval_from_mean_and_std(
        cls,
        sample_mean: float,
        sample_std: float,
        sample_size: int,
        confidence: float,
    ) -> ConfidenceInterval:
        standard_error = sample_std / np.sqrt(sample_size)
        alpha = 1 - confidence
        critical_value = t.ppf(1 - alpha / 2, sample_size - 1)
        margin_of_error = critical_value * standard_error

        return ConfidenceInterval(
            mean=float(sample_mean),
            margin_of_error=margin_of_error,
            standard_deviation=float(sample_std),
        )


class HypothesisTest(BaseModel):
    p_value: float
    hypothesis_rejected: bool
    written_conclusion: str | None = None


class ObservationStats(BaseModel):
    average: float
    standard_deviation: float
    count: int

    @property
    def standard_error(self) -> float:
        return self.standard_deviation / np.sqrt(self.count)


class MeanHypothesisCalculator:

    @classmethod
    def test_if_mean_is_greater_than_hypothesis_mean(
        cls,
        observations: list[float],
        hypothesis_mean: float,
        confidence: float = 0.95,
    ) -> HypothesisTest:
        """
        Stat test name:
        Hypothesis Testing: Mean with population standard deviation is not known

        Hypothesis mean is same is null hypothesis

        Assumptions
        1. The sample is a simple rnadom sample
        2. The value of htep opulation standard devaiaition is not known
        3. Either or both of these conditions is satisfied: The population is normally distributed or n > 30
        """
        assert len(observations) > 0, "Observations must be non-empty"
        assert 0 < confidence < 1, "Confidence must be between 0 and 1"
        test_normality_assumption(len(observations), observations)
        observation_stats = cls._get_observation_stats(observations)
        return cls._test_if_mean_is_greater_w_observation_stats(
            observation_stats, hypothesis_mean, confidence
        )

    @classmethod
    def test_if_mean_is_equal_to_than_hypothesis_mean(
        cls,
        observations: list[float],
        hypothesis_mean: float,
        confidence: float = 0.95,
    ) -> HypothesisTest:
        """
        Stat test name:
        Hypothesis Testing: Mean with population standard deviation is not known

        Hypothesis mean is same is null hypothesis

        Assumptions
        1. The sample is a simple rnadom sample
        2. The value of htep opulation standard devaiaition is not known
        3. Either or both of these conditions is satisfied: The population is normally distributed or n > 30
        """
        assert len(observations) > 0, "Observations must be non-empty"
        assert 0 < confidence < 1, "Confidence must be between 0 and 1"
        test_normality_assumption(len(observations), observations)
        observation_stats = cls._get_observation_stats(observations)
        return cls._test_if_mean_is_equal_to_than_hypothesis_mean_w_observation_stats(
            observation_stats, hypothesis_mean, confidence
        )

    @classmethod
    def _get_observation_stats(cls, observations: list[float]) -> ObservationStats:
        average = np.mean(observations)
        standard_deviation = np.std(observations, ddof=1)
        count = len(observations)

        return ObservationStats(
            average=float(average),
            standard_deviation=float(standard_deviation),
            count=int(count),
        )

    @classmethod
    def _test_if_mean_is_greater_w_observation_stats(
        cls,
        observation_stats: ObservationStats,
        hypothesis_mean: float,
        confidence: float = 0.95,
    ) -> HypothesisTest:
        test_normality_assumption(observation_stats.count)
        average = observation_stats.average
        std_error = observation_stats.standard_error
        count = observation_stats.count

        t_statistic = (average - hypothesis_mean) / std_error
        cdf = float(t.cdf(t_statistic, df=count - 1))
        p_value = (
            1 - cdf
        )  # One-tailed p-value for testing if mean is greater than hypothesis (would be just cdf if testing for "less than")
        alpha = 1 - confidence

        hypothesis_rejected = p_value < alpha
        if hypothesis_rejected:
            written_conclusion = f"We reject the null hypothesis (the population mean is less than or equal to {hypothesis_mean}) with {confidence*100:.2f}% confidence since at the {alpha*100:.2f}% level of significance, the sample data do, in fact, give enough evidence to conclude that the population mean is greater than {hypothesis_mean}. If the null hypothesis is true, then there is a {p_value*100:.2f}% probability that the sample (observed) mean would be observed at {average} or more. Since the mean value observed in the sample was {average} we can reject the null hypothesis. This also means there is sufficient evidence to support the alternative hypothesis that the population mean is greater than {hypothesis_mean}. The sample consisted of {count} observations."
        else:
            written_conclusion = f"We fail to reject the null hypothesis (the population mean is less than or equal to {hypothesis_mean}) since at the {alpha*100:.2f}% level of significance, the sample data do not give enough evidence to conclude that the population mean is greater than {hypothesis_mean}. If the null hypothesis is true, then there is a {p_value*100:.2f}% probability that the sample (observed) mean would be observed at {average} or more. Thus there is not enough evidence to suggest that the population mean is greater than {hypothesis_mean}. The sample consisted of {count} observations."
        return HypothesisTest(
            p_value=p_value,
            hypothesis_rejected=hypothesis_rejected,
            written_conclusion=written_conclusion,
        )

    @classmethod
    def _test_if_mean_is_equal_to_than_hypothesis_mean_w_observation_stats(
        cls,
        observation_stats: ObservationStats,
        hypothesis_mean: float,
        confidence: float = 0.95,
    ) -> HypothesisTest:
        test_normality_assumption(observation_stats.count)
        average = observation_stats.average
        std_error = observation_stats.standard_error
        count = observation_stats.count
        t_statistic = (average - hypothesis_mean) / std_error

        # Calculate CDF and p-value
        cdf = float(t.cdf(t_statistic, df=count - 1))
        p_value = 2 * min(cdf, 1 - cdf)  # Two-tailed p-value

        alpha = 1 - confidence
        hypothesis_rejected = p_value < alpha
        if hypothesis_rejected:
            written_conclusion = f"We reject the null hypothesis (i.e. the population mean is equal to {hypothesis_mean}) in favour of the alternative hypothesis. At the {alpha*100:.2f}% significance level there is enough evidence to suggest that population mean is not equal to {hypothesis_mean}. The sample consisted of {count} observations."
        else:
            written_conclusion = f"We fail to reject the null hypothesis (i.e. the population mean is equal to {hypothesis_mean}). At the {alpha*100:.2f}% level of significance, the sample data do not give enough evidence to conclude that the population mean is not equal to {hypothesis_mean}. The sample consisted of {count} observations."
        return HypothesisTest(
            p_value=p_value,
            hypothesis_rejected=hypothesis_rejected,
            written_conclusion=written_conclusion,
        )


def test_normality_assumption(
    sample_size: int, observations: list[float] | None = None
) -> None:
    if sample_size < 2:
        raise ValueError("Not enough data for T-based confidence interval")

    if sample_size < 30 and observations:
        _, normality_pvalue = shapiro(observations)
        if normality_pvalue < 0.05:
            raise ValueError(
                "Data fails normality assumption for T-based confidence interval"
            )
