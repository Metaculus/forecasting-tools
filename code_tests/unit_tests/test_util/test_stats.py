import logging

import pytest

from forecasting_tools.util.stats import (
    ConfidenceIntervalCalculator,
    MeanHypothesisCalculator,
    ObservationStats,
    ProportionStatCalculator,
)

logger = logging.getLogger(__name__)


class TestProportionStatCalculator:
    def run_test_on_binomial_calculator(
        self,
        successes: int,
        trials: int,
        desired_confidence: float,
        test_proportion_is_greater_than: float,
        correct_p_value: float,
        correct_hypothesis_rejected: bool,
        correct_conclusion: str,
    ) -> None:
        test_problem = ProportionStatCalculator(successes, trials)
        p_value, hypothesis_rejected, written_conclusion = (
            test_problem.determine_if_population_proportion_is_above_p0(
                test_proportion_is_greater_than, desired_confidence
            )
        )

        logger.info(
            f"p_value: {p_value}, hypothesis_rejected: {hypothesis_rejected}, written_conclusion: {written_conclusion}"
        )

        tolerance = 1e-3
        assert (
            abs(p_value - correct_p_value) < tolerance
        ), f"p_value was {p_value} but should have been {correct_p_value}"
        assert (
            hypothesis_rejected == correct_hypothesis_rejected
        ), f"hypothesis_rejected was {hypothesis_rejected} but should have been {correct_hypothesis_rejected}"

    def test_1_proportion_hypothesis_rejection(self) -> None:
        successes = 17
        trials = 42
        desired_confidence = 0.99
        test_proportion_is_greater_than = 0.25
        correct_p_value = 0.0103
        correct_hypothesis_rejected = False
        correct_conclusion = "If the null hypothesis is true (the proportion is 25.00%), then there is a 1.03% probability that the sample (estimated) proportion is 40.48% or more (the success percentage found in the sample). Thus, we fail to reject the null hypothesis since at the 1.00% level of significance, the sample data do now give enough evidence to conclude that the proportion is greater than 25.00%"

        self.run_test_on_binomial_calculator(
            successes,
            trials,
            desired_confidence,
            test_proportion_is_greater_than,
            correct_p_value,
            correct_hypothesis_rejected,
            correct_conclusion,
        )

    def test_2_proportion_hypothesis_rejection(self) -> None:
        # For this problem, see https://stats.libretexts.org/Courses/Las_Positas_College/Math_40%3A_Statistics_and_Probability/08%3A_Hypothesis_Testing_with_One_Sample/8.04%3A_Hypothesis_Test_Examples_for_Proportions#:~:text=In%20words%2C%20CLEARLY%20state
        successes = 13173
        trials = 25468
        desired_confidence = 0.95
        test_proportion_is_greater_than = 0.5
        correct_p_value = 0
        correct_hypothesis_rejected = True
        correct_conclusion = "If the null hypothesis is true (the proportion is 50.00%), then there is a 0.00% probability that the sample (estimated) proportion is 51.72% or more (the success percentage found in the sample). Thus, we reject the null hypothesis with 95.00% confidence since at the 5.00% level of significance, the sample data do give enough evidence to conclude that the proportion is greater than 50.00%"

        self.run_test_on_binomial_calculator(
            successes,
            trials,
            desired_confidence,
            test_proportion_is_greater_than,
            correct_p_value,
            correct_hypothesis_rejected,
            correct_conclusion,
        )

    def test_3_proportion_hypothesis_rejection(self) -> None:
        # For this problem see https://ecampusontario.pressbooks.pub/introstats/chapter/8-8-hypothesis-tests-for-a-population-proportion/#:~:text=households%20that%20have-,at%20least%20three%20cell%20phones%20is,-30%25.%C2%A0%20A%20cell
        successes = 54
        trials = 150
        desired_confidence = 0.99
        test_proportion_is_greater_than = 0.3
        correct_p_value = 0.0544
        correct_hypothesis_rejected = False
        correct_conclusion = "If the null hypothesis is true (the proportion is 30.00%), then there is a 5.44% probability that the sample (estimated) proportion is 36.00% or more. Thus, we fail to reject the null hypothesis since at the 1.00% level of significance, the sample data do now give enough evidence to conclude that the proportion is greater than 30.00%"

        self.run_test_on_binomial_calculator(
            successes,
            trials,
            desired_confidence,
            test_proportion_is_greater_than,
            correct_p_value,
            correct_hypothesis_rejected,
            correct_conclusion,
        )

    def test_4_proportion_hypothesis_rejection(self) -> None:
        # For this problem see https://courses.lumenlearning.com/wm-concepts-statistics/chapter/hypothesis-test-for-a-population-proportion-2-of-3/
        successes = 664
        trials = 800
        desired_confidence = 0.95
        test_proportion_is_greater_than = 0.8
        correct_p_value = 0.017
        correct_hypothesis_rejected = True
        correct_conclusion = "If the null hypothesis is true (the proportion is 80.00%), then there is a 1.70% probability that the sample (estimated) proportion is 83.00% or more (the success percentage found in the sample). Thus, we reject the null hypothesis with 95.00% confidence since at the 5.00% level of significance, the sample data do give enough evidence to conclude that the proportion is greater than 80.00%"

        self.run_test_on_binomial_calculator(
            successes,
            trials,
            desired_confidence,
            test_proportion_is_greater_than,
            correct_p_value,
            correct_hypothesis_rejected,
            correct_conclusion,
        )

    def test_error_thrown_if_normal_distribution_assumption_not_satisfied_high_p0(
        self,
    ) -> None:
        # Make n*(1-p) less than 5
        successes = 5
        trials = 10
        desired_confidence = 0.95
        test_proportion_is_greater_than = 0.95

        with pytest.raises(ValueError):
            test_problem = ProportionStatCalculator(successes, trials)
            test_problem.determine_if_population_proportion_is_above_p0(
                test_proportion_is_greater_than, desired_confidence
            )

    def test_error_thrown_if_normal_distribution_assumption_not_satisfied_low_p0(
        self,
    ) -> None:
        # Make n*p less than 5
        successes = 5
        trials = 10
        desired_confidence = 0.95
        test_proportion_is_greater_than = 0.05

        with pytest.raises(ValueError):
            test_problem = ProportionStatCalculator(successes, trials)
            test_problem.determine_if_population_proportion_is_above_p0(
                test_proportion_is_greater_than, desired_confidence
            )

    def test_error_thrown_if_100_percent_proportion_values_found(self) -> None:
        successes = 0
        trials = 10
        desired_confidence = 0.95
        test_proportion_is_greater_than = 0.5

        with pytest.raises(ValueError):
            test_problem = ProportionStatCalculator(successes, trials)
            test_problem.determine_if_population_proportion_is_above_p0(
                test_proportion_is_greater_than, desired_confidence
            )

    def test_error_thrown_if_0_percent_proportion_values_found(self) -> None:
        successes = 10
        trials = 10
        desired_confidence = 0.95
        test_proportion_is_greater_than = 0.5

        with pytest.raises(ValueError):
            test_problem = ProportionStatCalculator(successes, trials)
            test_problem.determine_if_population_proportion_is_above_p0(
                test_proportion_is_greater_than, desired_confidence
            )


class TestConfidenceInterval:

    def test_confidence_interval_from_mean_and_std_v1(self) -> None:
        # https://www.khanacademy.org/math/statistics-probability/confidence-intervals-one-sample/estimating-population-mean/v/calculating-a-one-sample-t-interval-for-a-mean
        num_observations = 14
        mean = 700
        std = 50
        confidence = 0.95

        confidence_interval = (
            ConfidenceIntervalCalculator.confidence_interval_from_mean_and_std(
                mean, std, num_observations, confidence
            )
        )

        assert confidence_interval.mean == mean
        assert confidence_interval.margin_of_error == pytest.approx(28.9, 0.1)
        assert confidence_interval.lower_bound == pytest.approx(671.1, 0.1)
        assert confidence_interval.upper_bound == pytest.approx(728.9, 0.1)

    def test_confidence_interval_from_mean_andstd_v2(self) -> None:
        # https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Introductory_Statistics_1e_(OpenStax)/08%3A_Confidence_Intervals/8.E%3A_Confidence_Intervals_(Exercises)
        num_observations = 100
        mean = 23.6
        std = 7
        confidence = 0.95

        confidence_interval = (
            ConfidenceIntervalCalculator.confidence_interval_from_mean_and_std(
                mean, std, num_observations, confidence
            )
        )

        assert confidence_interval.margin_of_error == pytest.approx(1.372, 0.1)
        assert confidence_interval.lower_bound == pytest.approx(22.228, 0.1)
        assert confidence_interval.upper_bound == pytest.approx(24.972, 0.1)

    def test_confidence_interval_from_observations_v1(self) -> None:
        # https://www.statskingdom.com/confidence-interval-calculator.html
        data: list[float] = [
            2.016958256852196,
            -2.1545547595542955,
            0.17643058731468048,
            -0.17899745527693173,
            -1.4400636862377763,
            1.5761768611550813,
            -0.1980518250021597,
            -0.011601732681319138,
            -1.7437464244027827,
            -0.3944474061704416,
            0.43389005591630586,
            1.1077540943600064,
            -0.6687719492567181,
            -0.9757879464441855,
            -0.5618087528418959,
            -0.9865765689103303,
            -0.7048146469454047,
            2.612506442801254,
            0.808876533367124,
            -0.13846324336650978,
            -0.5766259715841434,
            -0.29767038112826727,
            0.06583008361659538,
            0.7460570036633725,
            -0.640818986028395,
        ]
        confidence = 0.9
        confidence_interval = (
            ConfidenceIntervalCalculator.confidence_interval_from_observations(
                data, confidence
            )
        )

        assert confidence_interval.mean == pytest.approx(-0.08513, 0.1)
        assert confidence_interval.margin_of_error == pytest.approx(0.3818, 0.1)
        assert confidence_interval.lower_bound == pytest.approx(-0.4669, 0.1)
        assert confidence_interval.upper_bound == pytest.approx(0.2967, 0.1)

    @pytest.mark.skip(
        reason="all 3 of my code, the online stat calculator, and gpt analysis calculate std as 829k which different from than the textbook 909k. The textbook also seems to be using z scores (t is better here?)"
    )
    def test_confidence_interval_from_observations_v2(self) -> None:
        # https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Introductory_Statistics_1e_(OpenStax)/08%3A_Confidence_Intervals/8.E%3A_Confidence_Intervals_(Exercises)
        data: list[float] = [
            3600,
            1243900,
            10900,
            385200,
            581500,
            7400,
            2900,
            400,
            3714500,
            632500,
            391000,
            467400,
            56800,
            5800,
            405200,
            733200,
            8000,
            468700,
            75200,
            41000,
            13300,
            9500,
            953800,
            1113500,
            1109300,
            353900,
            986100,
            88600,
            378200,
            13200,
            3800,
            745100,
            5800,
            3072100,
            1626700,
            512900,
            2309200,
            6600,
            202400,
            15800,
        ]
        confidence = 0.95
        confidence_interval = (
            ConfidenceIntervalCalculator.confidence_interval_from_observations(
                data, confidence
            )
        )

        assert confidence_interval.mean == pytest.approx(568_873, 0.1)
        assert confidence_interval.standard_deviation == pytest.approx(909_200)
        assert confidence_interval.margin_of_error == pytest.approx(281_764)
        assert confidence_interval.lower_bound == pytest.approx(287_109)
        assert confidence_interval.upper_bound == pytest.approx(850_637)

    def test_confidence_interval_from_observations_v3(self) -> None:
        # https://stats.libretexts.org/Workbench/PSYC_2200%3A_Elementary_Statistics_for_Behavioral_and_Social_Science_(Oja)_WITHOUT_UNITS/08%3A_One_Sample_t-test/8.05%3A_Confidence_Intervals/8.5.01%3A_Practice_with_Confidence_Interval_Calculations
        data: list[float] = [
            8.6,
            9.4,
            7.9,
            6.8,
            8.3,
            7.3,
            9.2,
            9.6,
            8.7,
            11.4,
            10.3,
            5.4,
            8.1,
            5.5,
            6.9,
        ]
        confidence = 0.95
        confidence_interval = (
            ConfidenceIntervalCalculator.confidence_interval_from_observations(
                data, confidence
            )
        )

        assert confidence_interval.mean == pytest.approx(8.2267, 0.1)
        assert confidence_interval.margin_of_error == pytest.approx(0.924, 0.1)
        assert confidence_interval.lower_bound == pytest.approx(7.3, 0.1)
        assert confidence_interval.upper_bound == pytest.approx(9.15, 0.1)

    def test_confidence_interval_from_observations_v4(self) -> None:
        # https://stats.libretexts.org/Workbench/PSYC_2200%3A_Elementary_Statistics_for_Behavioral_and_Social_Science_(Oja)_WITHOUT_UNITS/08%3A_One_Sample_t-test/8.05%3A_Confidence_Intervals/8.5.01%3A_Practice_with_Confidence_Interval_Calculations
        data: list[float] = [
            79,
            145,
            147,
            160,
            116,
            100,
            159,
            151,
            156,
            126,
            137,
            83,
            156,
            94,
            121,
            144,
            123,
            114,
            139,
            99,
        ]
        confidence = 0.9
        confidence_interval = (
            ConfidenceIntervalCalculator.confidence_interval_from_observations(
                data, confidence
            )
        )

        assert confidence_interval.mean == pytest.approx(127.45, 0.1)
        assert confidence_interval.standard_deviation == pytest.approx(25.965, 0.1)
        assert confidence_interval.margin_of_error == pytest.approx(10.038, 0.1)
        assert confidence_interval.lower_bound == pytest.approx(117.412, 0.1)
        assert confidence_interval.upper_bound == pytest.approx(137.488, 0.1)

    def test_non_normal_data_errors(self) -> None:
        with pytest.raises(ValueError):
            data: list[float] = [1, 5, 8, 7, 79, 3, 45, 67, 43, 65, 87, 12]
            confidence = 0.9
            ConfidenceIntervalCalculator.confidence_interval_from_observations(
                data, confidence
            )

    def test_insufficient_data_errors(self) -> None:
        with pytest.raises(Exception):
            data: list[float] = []
            confidence = 0.9
            ConfidenceIntervalCalculator.confidence_interval_from_observations(
                data, confidence
            )


class TestMeanStatCalculator:

    def test_mean_is_greater_than_hypothesis_mean(self) -> None:
        # https://ecampusontario.pressbooks.pub/introstats/chapter/8-7-hypothesis-tests-for-a-population-mean-with-unknown-population-standard-deviation/
        observations = [
            65.0,
            67.0,
            66.0,
            68.0,
            72.0,
            65.0,
            70.0,
            63.0,
            63.0,
            71.0,
        ]
        hypothesis_mean = 65.0
        confidence = 0.99

        hypothesis_test = (
            MeanHypothesisCalculator.test_if_mean_is_greater_than_hypothesis_mean(
                observations, hypothesis_mean, confidence
            )
        )

        assert hypothesis_test.p_value == pytest.approx(0.0396, 0.01)
        assert hypothesis_test.hypothesis_rejected == False

    def test_mean_is_equal_to_hypothesis_mean(self) -> None:
        # https://ecampusontario.pressbooks.pub/introstats/chapter/8-7-hypothesis-tests-for-a-population-mean-with-unknown-population-standard-deviation/
        hypothesis_mean = 3.78
        count = 100
        average = 3.62
        standard_deviation = 0.7
        confidence = 0.95

        observation_stats = ObservationStats(
            average=average,
            standard_deviation=standard_deviation,
            count=count,
        )

        hypothesis_test = MeanHypothesisCalculator._test_if_mean_is_equal_to_than_hypothesis_mean_w_observation_stats(
            observation_stats, hypothesis_mean, confidence
        )

        assert hypothesis_test.p_value == pytest.approx(0.0244, 0.01)
        assert hypothesis_test.hypothesis_rejected == True
