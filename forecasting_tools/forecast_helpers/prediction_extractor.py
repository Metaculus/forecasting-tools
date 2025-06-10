import logging
import re
import string
from dataclasses import dataclass

from unidecode import unidecode_expect_ascii

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
        if not 0 <= max_prediction <= 1:
            raise ValueError(
                f"Max prediction {max_prediction} is not between 0 and 1"
            )
        if not 0 <= min_prediction <= 1:
            raise ValueError(
                f"Min prediction {min_prediction} is not between 0 and 1"
            )
        if max_prediction < min_prediction:
            raise ValueError(
                f"Max prediction {max_prediction} is not greater than or equal to min prediction {min_prediction}"
            )
        if not 0 <= min_prediction <= 1:
            raise ValueError(
                f"Min prediction {min_prediction} is not between 0 and 1"
            )
        if max_prediction < min_prediction:
            raise ValueError(
                f"Max prediction {max_prediction} is not greater than or equal to min prediction {min_prediction}"
            )
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
        if matches:
            # Return the last number found before a '%'
            original_number = float(matches[-1]) / 100
            if not 0 <= original_number <= 1:
                raise ValueError(
                    f"Probability {original_number} is not between 0 and 1, so won't attempt to clamp the value"
                )
            clamped_number = min(
                max_prediction, max(min_prediction, original_number)
            )
            return float(clamped_number)
        else:
            raise ValueError(
                f"Could not extract prediction from response. The text was: {text}"
            )

    @classmethod
    def extract_option_list_with_percentage_afterwards(
        cls, text: str, options: list[str]
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
            [f'Option "{option}"' for option in options],
            [f"Option '{option}'" for option in options],
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
                option_probabilities = (
                    cls._extract_option_probabilities_through_name_matching(
                        text, option_list
                    )
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

        normalized_option_probabilities = cls._normalize_option_probabilities(
            option_probabilities
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

    @classmethod
    def _extract_option_probabilities_through_name_matching(
        cls, text: str, options: list[str]
    ) -> list[float]:
        option_probabilities = []
        for expected_option in options:
            escaped = re.escape(cls._clean_text(expected_option))
            pattern = rf"""^\s*\W*                 # any leading non-word stuff
                {escaped}               # the key phrase itself
                (?![\w.,-])             # must NOT be followed by word/comma/period/hyphen
                \s*\W*                  # optional quotes / spaces after the phrase
                [|:]                    # delimiter can be ‘:’ or table ‘|’
                \s*                     # optional space
                (-?                     # ⎫ capture group 1 = number
                    \d[\d,]*            # | digits with optional thousands commas
                    (?:\.\d+)?          # | optional decimal part
                )                       # ⎭
                """
            compiled_pattern = re.compile(pattern, re.IGNORECASE | re.VERBOSE)

            @dataclass
            class MatchingLine:
                line: str
                probability: float  # decimal or percentage (convert later)

            matching_lines: list[MatchingLine] = []
            for line in text.split("\n"):
                cleaned_line = cls._clean_text(line)
                match = compiled_pattern.search(cleaned_line)
                if match:
                    probability = float(match.group(1).replace(",", ""))
                    matching_lines.append(
                        MatchingLine(line=line, probability=probability)
                    )

            if len(matching_lines) != 1:
                if len(text) > 1000:
                    display_text = f"{text[:200]} ... {text[-200:]}"
                else:
                    display_text = text
                raise ValueError(
                    f"Expected exactly one match for pattern '{expected_option}', found {len(matching_lines)}. Matching lines: {matching_lines}. Text: \"{display_text}\""
                )
            matching_line = matching_lines[0]
            option_probabilities.append(matching_line.probability)

        return option_probabilities

    @classmethod
    def _clean_text(cls, text: str) -> str:
        return unidecode_expect_ascii(text.strip().lower())

    @classmethod
    def _normalize_option_probabilities(
        cls,
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
        if sum_of_probabilities > 1.02 or sum_of_probabilities < 0.99:
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
        percentile_lines = PredictionExtractor._get_percentile_lines(text)
        final_percentiles: list[Percentile] = []
        for line in percentile_lines:
            numbers = PredictionExtractor._parse_numbers_from_line(line)
            percentile = PredictionExtractor._percentile_from_numbers(numbers)
            if percentile:
                final_percentiles.append(percentile)
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

    @staticmethod
    def _get_percentile_lines(text: str) -> list[str]:
        # Regex looks for patterns like "Percentile X: Y" or "Xth Percentile - Y".
        line_filter_pattern = r"^\s*(?:(?:[Pp]ercentile\s+\d+)|\d+\s*(?:th|st|nd|rd)?\s*[Pp]ercentile)\s*[:\-\s]\s*\S+.*"
        return [
            line.strip()
            for line in text.split("\n")
            if re.match(line_filter_pattern, line.strip())
        ]

    @staticmethod
    def _parse_numbers_from_line(line: str) -> list[float | int]:
        # Regex to extract numbers, allowing for an optional sign,
        # optional leading symbols (like currency),
        # and commas within the number.
        # Group 1: Optional sign (-).
        # Group 2: The number string (digits, possibly with commas/decimal).
        number_extraction_pattern = (
            r"(-)?\s*(?:[^\w\s,.-]*\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)"
        )
        raw_number_matches = re.findall(number_extraction_pattern, line)
        parsed_numbers: list[float | int] = []
        for sign_group, number_val_str in raw_number_matches:
            num_str_no_commas = number_val_str.replace(",", "")
            if "." in num_str_no_commas:
                parsed_num = float(num_str_no_commas)
            else:
                parsed_num = int(num_str_no_commas)
            if sign_group == "-":
                parsed_num = -parsed_num
            parsed_numbers.append(parsed_num)
        return parsed_numbers

    @staticmethod
    def _percentile_from_numbers(
        numbers: list[float | int],
    ) -> Percentile | None:
        """
        Example:
        Percentile 10: -12,222
        Would parse to [10, -12222]

        Example:
        Percentile 10: -12 222

        Would parse to [10, -12, 222]
        """
        if len(numbers) <= 1:
            raise ValueError(
                "There should be at least two numbers in lines parsed for percentiles"
            )
        percentile_level = numbers[0]
        original_value_at_percentile = numbers[1]
        value_is_negative = original_value_at_percentile < 0

        value_at_percentile = original_value_at_percentile
        if len(numbers) > 2:
            if value_at_percentile >= 1000:
                raise ValueError(
                    "There should not be more than 3 digits per space separated section for number"
                )
            for other_number in numbers[2:]:
                if other_number >= 1000:
                    raise ValueError(
                        "There should not be more than 3 digits per space separated section for number"
                    )
                if other_number < 0:
                    raise ValueError(
                        "There should not be negative numbers in number section separated by spaces"
                    )
                value_at_percentile = (
                    value_at_percentile * 1000
                    + other_number * (-1 if value_is_negative else 1)
                )
        return Percentile(
            value=value_at_percentile,
            percentile=percentile_level / 100.0,
        )
