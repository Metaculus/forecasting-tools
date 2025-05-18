import logging
import re
import string

from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    Percentile,
)
from forecasting_tools.data_models.questions import NumericQuestion

logger = logging.getLogger(__name__)


class PredictionExtractor:

    @staticmethod
    def extract_last_percentage_value(
        text: str, max_prediction: float = 1.0, min_prediction: float = 0.0
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
            return float(clamped_number)
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

        alphabet_abc_option_letters = list(
            string.ascii_uppercase[: len(options)]
        )
        cleaned_options = [option.strip() for option in options]
        underscored_options = [
            option.strip().replace(" ", "_") for option in options
        ]
        option_lists_to_try = [
            options,
            cleaned_options,
            underscored_options,
            [f"Option {option}" for option in options],
            [f"Option {cleaned_option}" for cleaned_option in cleaned_options],
            [
                f"Option {underscored_option}"
                for underscored_option in underscored_options
            ],
            [
                f"Option_{underscored_option}"
                for underscored_option in underscored_options
            ],
            [f"Option {i}" for i in range(1, len(options) + 1)],
            [f"Option_{i}" for i in range(1, len(options) + 1)],
            [f"Option {letter}" for letter in alphabet_abc_option_letters],
            [f"Option_{letter}" for letter in alphabet_abc_option_letters],
            [f"{letter}" for letter in alphabet_abc_option_letters],
        ]

        exceptions = []
        raise_exceptions = True
        for option_list in option_lists_to_try:
            try:
                option_probabilities = PredictionExtractor._extract_option_probabilities_through_name_matching(
                    text, option_list
                )
            except Exception as e:
                exceptions.append(e)
                continue
            else:
                raise_exceptions = False
                break

        if raise_exceptions:
            raise ValueError(
                f"No option list variations worked. First exception: {exceptions[0]}"
            )

        assert len(option_probabilities) == len(
            options
        ), f"Number of option probabilities {len(option_probabilities)} does not match number of options {len(options)}"

        normalized_option_probabilities = (
            PredictionExtractor._normalize_option_probabilities(
                option_probabilities
            )
        )

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
    def _extract_option_probabilities_through_name_matching(
        text: str, options: list[str]
    ) -> list[float]:
        option_probabilities = []
        for expected_option in options:
            expected_option = expected_option.strip()
            probability_found = False
            matching_lines = []
            for line in text.split("\n"):
                pattern = rf".*(?:^|[^0-9.])(?:{re.escape(expected_option.lower())})(?:\s*|\s*['\"]?\s*)[^.,0-9]+(-?\d*\.\d+|-?\d+).*"
                match = re.match(pattern, line.strip().lower())
                if match:
                    matching_lines.append(line)

            if matching_lines:
                last_matching_line = matching_lines[-1]
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
        return option_probabilities

    @staticmethod
    def _normalize_option_probabilities(
        option_probabilities: list[float],
    ) -> list[float]:
        total_sum = sum(option_probabilities)
        threshold_for_decimal_probability_presence = 1.9
        if total_sum < threshold_for_decimal_probability_presence:
            logger.debug(
                (
                    f"Total sum of option probabilities {total_sum} is less than ",
                    f"{threshold_for_decimal_probability_presence} ",
                    "indicating rationale was working in decimal probabilities. ",
                    "Now these are being treated as decimal probabilities.",
                )
            )
            decimal_list = option_probabilities
        else:
            decimal_list = [x / 100 for x in option_probabilities]

        sum_of_probabilities = sum(decimal_list)
        if sum_of_probabilities > 1.11 or sum_of_probabilities < 0.89:
            raise ValueError(
                f"Sum of option probabilities {sum_of_probabilities} is "
                "too far from 1 to be confident that normalization will deliver an "
                "intended prediction."
            )

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

        return normalized_option_probabilities

    @staticmethod
    def extract_numeric_distribution_from_list_of_percentile_number_and_probability(
        text: str, question: NumericQuestion
    ) -> NumericDistribution:

        if not text or text.strip() == "":
            raise ValueError(
                "While trying to extract numeric distribution from response found that the reasoning is None or an empty string"
            )

        # More specific regex to identify lines likely containing percentile data.
        # It looks for patterns like "Percentile X: Y" or "Xth Percentile - Y".
        line_filter_pattern = r"^\s*(?:(?:[Pp]ercentile\s+\d+)|\d+\s*(?:th|st|nd|rd)?\s*[Pp]ercentile)\s*[:\-\s]\s*\S+.*"

        # Regex to extract numbers, allowing for an optional sign, optional leading symbols (like currency),
        # and commas within the number.
        # Group 1: Optional sign (-).
        # Group 2: The number string (digits, possibly with commas/decimal).
        number_extraction_pattern = (
            r"(-)?\s*(?:[^\w\s,.-]*\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)"
        )

        final_percentiles: list[Percentile] = []

        for line in text.split("\n"):
            stripped_line = line.strip()
            if re.match(line_filter_pattern, stripped_line):
                raw_number_matches = re.findall(
                    number_extraction_pattern, stripped_line
                )

                parsed_numbers_in_line: list[float | int] = []
                for sign_group, number_val_str in raw_number_matches:
                    num_str_no_commas = number_val_str.replace(",", "")

                    if "." in num_str_no_commas:
                        parsed_num = float(num_str_no_commas)
                    else:
                        parsed_num = int(num_str_no_commas)

                    if sign_group == "-":
                        parsed_num = -parsed_num

                    parsed_numbers_in_line.append(parsed_num)

                if len(parsed_numbers_in_line) > 1:
                    # Assuming the first number is the percentile level
                    # the second is the main value, and everything else should be
                    # in sets of 3 indicating that there was a space used for the thousands/millions
                    percentile_level = parsed_numbers_in_line[0]
                    value_at_percentile = parsed_numbers_in_line[1]
                    value_is_negative = value_at_percentile < 0

                    if len(parsed_numbers_in_line) > 2:
                        for other_number in parsed_numbers_in_line[2:]:
                            if other_number >= 1000:
                                raise ValueError(
                                    "There should not be more than 3 digits in number sections separated by spaces"
                                )
                            if other_number < 0:
                                raise ValueError(
                                    "There should not be negative numbers in number sections separated by spaces"
                                )
                            value_at_percentile = (
                                value_at_percentile * 1000
                                + other_number
                                * (-1 if value_is_negative else 1)
                            )

                    final_percentiles.append(
                        Percentile(
                            value=value_at_percentile,
                            percentile=percentile_level / 100.0,
                        )
                    )

        if not final_percentiles:
            raise ValueError(
                f"Couldn't extract numeric distribution from response. The text was: {text}"
            )

        return NumericDistribution(
            declared_percentiles=final_percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )
