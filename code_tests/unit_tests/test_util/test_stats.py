import logging

import pytest

from forecasting_tools.util.stats import ProportionStatCalculator

logger = logging.getLogger(__name__)


def run_test_on_binomial_calculator(
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


def test_1_proportion_hypothesis_rejection():
    successes = 17
    trials = 42
    desired_confidence = 0.99
    test_proportion_is_greater_than = 0.25
    correct_p_value = 0.0103
    correct_hypothesis_rejected = False
    correct_conclusion = "If the null hypothesis is true (the proportion is 25.00%), then there is a 1.03% probability that the sample (estimated) proportion is 40.48% or more (the success percentage found in the sample). Thus, we fail to reject the null hypothesis since at the 1.00% level of significance, the sample data do now give enough evidence to conclude that the proportion is greater than 25.00%"

    run_test_on_binomial_calculator(
        successes,
        trials,
        desired_confidence,
        test_proportion_is_greater_than,
        correct_p_value,
        correct_hypothesis_rejected,
        correct_conclusion,
    )


def test_2_proportion_hypothesis_rejection():
    # For this problem, see https://stats.libretexts.org/Courses/Las_Positas_College/Math_40%3A_Statistics_and_Probability/08%3A_Hypothesis_Testing_with_One_Sample/8.04%3A_Hypothesis_Test_Examples_for_Proportions#:~:text=In%20words%2C%20CLEARLY%20state
    successes = 13173
    trials = 25468
    desired_confidence = 0.95
    test_proportion_is_greater_than = 0.5
    correct_p_value = 0
    correct_hypothesis_rejected = True
    correct_conclusion = "If the null hypothesis is true (the proportion is 50.00%), then there is a 0.00% probability that the sample (estimated) proportion is 51.72% or more (the success percentage found in the sample). Thus, we reject the null hypothesis with 95.00% confidence since at the 5.00% level of significance, the sample data do give enough evidence to conclude that the proportion is greater than 50.00%"

    run_test_on_binomial_calculator(
        successes,
        trials,
        desired_confidence,
        test_proportion_is_greater_than,
        correct_p_value,
        correct_hypothesis_rejected,
        correct_conclusion,
    )


def test_3_proportion_hypothesis_rejection():
    # For this problem see https://ecampusontario.pressbooks.pub/introstats/chapter/8-8-hypothesis-tests-for-a-population-proportion/#:~:text=households%20that%20have-,at%20least%20three%20cell%20phones%20is,-30%25.%C2%A0%20A%20cell
    successes = 54
    trials = 150
    desired_confidence = 0.99
    test_proportion_is_greater_than = 0.3
    correct_p_value = 0.0544
    correct_hypothesis_rejected = False
    correct_conclusion = "If the null hypothesis is true (the proportion is 30.00%), then there is a 5.44% probability that the sample (estimated) proportion is 36.00% or more. Thus, we fail to reject the null hypothesis since at the 1.00% level of significance, the sample data do now give enough evidence to conclude that the proportion is greater than 30.00%"

    run_test_on_binomial_calculator(
        successes,
        trials,
        desired_confidence,
        test_proportion_is_greater_than,
        correct_p_value,
        correct_hypothesis_rejected,
        correct_conclusion,
    )


def test_4_proportion_hypothesis_rejection():
    # For this problem see https://courses.lumenlearning.com/wm-concepts-statistics/chapter/hypothesis-test-for-a-population-proportion-2-of-3/
    successes = 664
    trials = 800
    desired_confidence = 0.95
    test_proportion_is_greater_than = 0.8
    correct_p_value = 0.017
    correct_hypothesis_rejected = True
    correct_conclusion = "If the null hypothesis is true (the proportion is 80.00%), then there is a 1.70% probability that the sample (estimated) proportion is 83.00% or more (the success percentage found in the sample). Thus, we reject the null hypothesis with 95.00% confidence since at the 5.00% level of significance, the sample data do give enough evidence to conclude that the proportion is greater than 80.00%"

    run_test_on_binomial_calculator(
        successes,
        trials,
        desired_confidence,
        test_proportion_is_greater_than,
        correct_p_value,
        correct_hypothesis_rejected,
        correct_conclusion,
    )


def test_error_thrown_if_normal_distribution_assumption_not_satisfied_high_p0():
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


def test_error_thrown_if_normal_distribution_assumption_not_satisfied_low_p0():
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


def test_error_thrown_if_100_percent_proportion_values_found():
    successes = 0
    trials = 10
    desired_confidence = 0.95
    test_proportion_is_greater_than = 0.5

    with pytest.raises(ValueError):
        test_problem = ProportionStatCalculator(successes, trials)
        test_problem.determine_if_population_proportion_is_above_p0(
            test_proportion_is_greater_than, desired_confidence
        )


def test_error_thrown_if_0_percent_proportion_values_found():
    successes = 10
    trials = 10
    desired_confidence = 0.95
    test_proportion_is_greater_than = 0.5

    with pytest.raises(ValueError):
        test_problem = ProportionStatCalculator(successes, trials)
        test_problem.determine_if_population_proportion_is_above_p0(
            test_proportion_is_greater_than, desired_confidence
        )
