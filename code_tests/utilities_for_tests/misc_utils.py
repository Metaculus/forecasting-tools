from typing import Any


def determine_percent_correct(actual: list[Any], expected: list[Any]) -> float:
    if len(actual) != len(expected):
        raise ValueError(
            f"Length of actual ({len(actual)}) does not match length of expected ({len(expected)})"
        )

    number_of_correct: int = 0
    for i in range(len(actual)):
        if actual[i] == expected[i]:
            number_of_correct += 1
    percent_correct: float = number_of_correct / len(actual)

    return percent_correct


def replace_tzinfo_in_string(string: str) -> str:
    updated_s = (
        string.replace("datetime.timezone.utc", "")
        .replace("TZInfo(UTC)", "")
        .replace("TzInfo(0)", "")
        .replace("pendulum.timezone('UTC')", "")
        .replace("Timezone('UTC')", "")
        .replace("TzInfo(UTC)", "")
        .replace("Timezone('Etc/UTC')", "")
        .replace("Timezone('America/New_York')", "")
        .replace("TzInfo(-05:00)", "")
        .replace("datetime.datetime", "")
        .replace("datetime", "")
        .replace("DateTime", "")
    )
    return updated_s
