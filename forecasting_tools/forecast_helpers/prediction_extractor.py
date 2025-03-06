import re

from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    Percentile,
)
from forecasting_tools.data_models.questions import NumericQuestion


class PredictionExtractor:

    @staticmethod
    def extract_last_percentage_value(
        text: str, max_prediction: float, min_prediction: float
    ) -> float:
        if not text or text.strip() == "":
            raise ValueError(
                "While trying to extract last percentage value found that the text is None or an empty string"
            )
        assert (
            0 <= max_prediction <= 1
        ), f"Max prediction {max_prediction} is not between 0 and 1"
        assert (
            0 <= min_prediction <= 1
        ), f"Min prediction {min_prediction} is not between 0 and 1"
        assert (
            max_prediction >= min_prediction
        ), f"Max prediction {max_prediction} is not greater than or equal to min prediction {min_prediction}"
        matches = re.findall(r"(\d+)%", text)
        if matches:
            # Return the last number found before a '%'
            original_number = int(matches[-1]) / 100
            clamped_number = min(
                max_prediction, max(min_prediction, original_number)
            )
            assert (
                min_prediction <= clamped_number <= max_prediction
            ), f"Clamped number {clamped_number} is not between {min_prediction} and {max_prediction}"
            return clamped_number
        else:
            raise ValueError(
                f"Could not extract prediction from response. The text was: {text}"
            )

    @staticmethod
    def extract_option_list_with_percentage_afterwards(
        text: str, options: list[str]
    ) -> PredictedOptionList:
        if not text or text.strip() == "":
            raise ValueError(
                "While trying to extract option list found that the text is None or an empty string"
            )

        option_probabilities = []

        # Iterate through each line in the text
        for expected_option in options:
            expected_option = expected_option.strip()
            probability_found = False
            matching_lines = []
            for line in text.split("\n"):
                expected_option_with_underscores = expected_option.replace(
                    " ", "_"
                )
                if (
                    expected_option.lower() in line.lower()
                    or expected_option_with_underscores.lower() in line.lower()
                ):
                    matching_lines.append(line)

            if matching_lines:
                last_matching_line = matching_lines[-1]
                # Extract all numbers from the line
                numbers_as_string = re.findall(
                    r"-?\d+(?:,\d{3})*(?:\.\d+)?", last_matching_line
                )
                numbers_as_float = [
                    float(num.replace(",", "")) for num in numbers_as_string
                ]
                if len(numbers_as_float) >= 1:
                    last_number = numbers_as_float[-1]
                    assert (
                        0 <= last_number <= 100
                    ), f"Probability {last_number} is not between 0 and 100 for option: {expected_option}"
                    option_probabilities.append(last_number)
                    probability_found = True

            if not probability_found:
                raise ValueError(
                    f"No probability found for option: {expected_option}"
                )

        assert len(option_probabilities) == len(
            options
        ), f"Number of option probabilities {len(option_probabilities)} does not match number of options {len(options)}"

        total_sum = sum(option_probabilities)
        assert (
            total_sum > 1.5
        ), f"Total sum of option probabilities {total_sum} is not greater than 1.5 indicating rationale was working in decimal probabilities"
        decimal_list = [x / total_sum for x in option_probabilities]

        # Step 1: Clamp values
        clamped_list = [max(min(x, 0.999), 0.001) for x in decimal_list]

        # Step 2: Calculate the sum of all elements
        total_sum_decimal = sum(clamped_list)

        # Step 3: Normalize the list so that all elements add up to 1
        normalized_list = [x / total_sum_decimal for x in clamped_list]

        # Step 4: Adjust for any small floating-point errors
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment
        normalized_option_probabilities = normalized_list

        predicted_options: list[PredictedOption] = []
        for i in range(len(options)):
            predicted_options.append(
                PredictedOption(
                    option_name=options[i],
                    probability=normalized_option_probabilities[i],
                )
            )

        return PredictedOptionList(predicted_options=predicted_options)

    @staticmethod
    def extract_numeric_distribution_from_list_of_percentile_number_and_probability(
        text: str, question: NumericQuestion
    ) -> NumericDistribution:

        if not text or text.strip() == "":
            raise ValueError(
                "While trying to extract numeric distribution from response found that the reasoning is None or an empty string"
            )

        pattern = r"^.*[Pp]ercentile.*$"
        number_pattern = r"-\s*(?:[^\d\-]*\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?)"
        results = []

        for line in text.split("\n"):
            if re.match(pattern, line):
                numbers = re.findall(number_pattern, line)
                numbers_no_commas = [
                    next(num for num in match if num).replace(",", "")
                    for match in numbers
                ]
                numbers = [
                    float(num) if "." in num else int(num)
                    for num in numbers_no_commas
                ]
                if len(numbers) > 1:
                    first_number = numbers[0]
                    last_number = numbers[-1]
                    # Check if the original line had a negative sign before the last number
                    if "-" in line.split(":")[-1]:
                        last_number = -abs(last_number)
                    results.append((first_number, last_number))

        percentiles = [
            Percentile(
                value=value,
                percentile=percentile / 100,
            )
            for percentile, value in results
        ]

        if not percentiles:
            raise ValueError(
                f"Couldn't extract numeric distribution from response. The text was: {text}"
            )

        return NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )
