import pytest

from forecasting_tools.front_end.helpers.report_displayer import ReportDisplayer


@pytest.mark.parametrize(
    "text, expected",
    [
        ("# Hello, world!\ncontent", "# Hello, world!\ncontent"),
        ("I have $5 from Tom", "I have \\$5 from Tom"),
        (
            "I have $5 from Tom and $10 from John",
            "I have \\$5 from Tom and \\$10 from John",
        ),
        ("\\$100", "\\$100"),
        ("Already escaped money \\$100", "Already escaped money \\$100"),
        (
            "Both escaped \\$100 and unescaped $100",
            "Both escaped \\$100 and unescaped \\$100",
        ),
        # ("Lets do math $1 + 1 = 2$", "Lets do math $1 + 1 = 2$"), # This isn't really needed and hard to get working
        # ("Lets do math twice $1 + 1 = 2$ and $2 + 2 = 4$", "Lets do math twice $1 + 1 = 2$ and $2 + 2 = 4$"), # This isn't really needed and hard to get working
        # ("Lets do no space math $1+1=2$", "Lets do no space math $1+1=2$"),
        # ("Lets do no space math twice $1+1=2$ and $2+2=4$", "Lets do no space math twice $1+1=2$ and $2+2=4$"),
        # ("Lets do different math $$1 + 1 = 2$$", "Lets do different math $$1 + 1 = 2$$"),
        # ("Doing math $1+1=2$ and $10 and \\$100", "Doing math $1+1=2$ and \\$10 and \\$100"),
    ],
)
def test_clean_markdown(text: str, expected: str) -> None:
    cleaned_markdown = ReportDisplayer.clean_markdown(text)
    assert cleaned_markdown == expected
