from __future__ import annotations

from forecasting_tools.bot_review.report_parsing import (
    final_prediction,
    forecaster_rationales,
    parse_forecasters,
    split_sections,
    was_truncated,
)

# Stock forecasting-tools output: unannotated bullets, two research reports.
TEMPLATE_EXPLANATION = """
# SUMMARY
*Question*: Will X happen?
*Final Prediction*: 35.0%
*Total Cost*: $0.35 (estimated)

## Report 1 Summary
### Forecasts
*Forecaster 1*: 30.0%
*Forecaster 2*: 40.0%

### Research Summary
A short summary of report 1.

## Report 2 Summary
### Forecasts
*Forecaster 1*: 50.0%
*Forecaster 2*: 60.0%

### Research Summary
A short summary of report 2.

# RESEARCH
## Report 1 Research
### Research heading 1
Body one.
## Report 2 Research
### Research heading 2
Body two.

# FORECASTS

## R1: Forecaster 1 Reasoning
### A heading demoted by the framework
R1F1 reasoning.

## R1: Forecaster 2 Reasoning
R1F2 reasoning.

## R2: Forecaster 1 Reasoning
R2F1 reasoning.

## R2: Forecaster 2 Reasoning
R2F2 reasoning.
""".strip()

# Some bots add a parenthesised suffix to the bullets; it must not break parsing.
ANNOTATED_EXPLANATION = """
# SUMMARY
*Question*: Will X happen?
*Final Prediction*: 3.0%

## Report 1 Summary
### Forecasts
*Forecaster 1 (a-model)*: 3.0%
*Forecaster 2 (another-model)*: 4.0%

### Research Summary
_Full research in the RESEARCH section below._

# RESEARCH
## Report 1 Research
Some news.

# FORECASTS
## R1: Forecaster 1 Reasoning
First rationale.

## R1: Forecaster 2 Reasoning
Second rationale.
""".strip()

MULTILINE_SUMMARY = """
*Final Prediction*:
- A: 60.00%
- B: 40.00%

*Total Cost*: disabled

## Report 1 Summary
### Forecasts
*Forecaster 1*: - A: 62.0%
- B: 38.0%

*Forecaster 2*: - A: 58.0%
- B: 42.0%

### Research Summary
_elsewhere_
""".strip()


class TestSplitSections:
    def test_finds_the_three_sections(self):
        assert set(split_sections(TEMPLATE_EXPLANATION)) == {
            "summary",
            "research",
            "forecasts",
        }

    def test_research_stops_where_forecasts_start(self):
        research = split_sections(TEMPLATE_EXPLANATION)["research"]
        assert "Body two." in research
        assert "Forecaster 1 Reasoning" not in research

    def test_missing_sections_are_omitted(self):
        # the heading line stays in, as it does on ForecastReport.summary
        assert split_sections("# SUMMARY\nonly this") == {
            "summary": "# SUMMARY\nonly this"
        }


class TestParseForecasters:
    def test_unannotated_bullets_across_two_reports(self):
        summary = split_sections(TEMPLATE_EXPLANATION)["summary"]
        assert parse_forecasters(summary) == [
            {"key": "R1:F1", "prediction": "30.0%"},
            {"key": "R1:F2", "prediction": "40.0%"},
            {"key": "R2:F1", "prediction": "50.0%"},
            {"key": "R2:F2", "prediction": "60.0%"},
        ]

    def test_annotated_bullets_still_parse(self):
        summary = split_sections(ANNOTATED_EXPLANATION)["summary"]
        assert parse_forecasters(summary) == [
            {"key": "R1:F1", "prediction": "3.0%"},
            {"key": "R1:F2", "prediction": "4.0%"},
        ]

    def test_research_summary_is_not_read_as_a_prediction(self):
        summary = split_sections(ANNOTATED_EXPLANATION)["summary"]
        assert "Full research" not in parse_forecasters(summary)[-1]["prediction"]

    def test_multiline_predictions_are_kept_whole(self):
        parsed = parse_forecasters(MULTILINE_SUMMARY)
        assert parsed[0]["prediction"] == "- A: 62.0%\n- B: 38.0%"
        assert parsed[1]["prediction"] == "- A: 58.0%\n- B: 42.0%"


class TestForecasterRationales:
    def test_second_research_report_does_not_overwrite_the_first(self):
        rationales = forecaster_rationales(TEMPLATE_EXPLANATION)
        assert sorted(rationales) == ["R1:F1", "R1:F2", "R2:F1", "R2:F2"]
        assert "R1F1 reasoning." in rationales["R1:F1"]
        assert "R2F1 reasoning." in rationales["R2:F1"]

    def test_subheadings_stay_with_their_forecaster(self):
        rationales = forecaster_rationales(TEMPLATE_EXPLANATION)
        assert "A heading demoted by the framework" in rationales["R1:F1"]
        assert "R1F2 reasoning" not in rationales["R1:F1"]

    def test_no_forecasts_section(self):
        assert forecaster_rationales("# SUMMARY\nnothing else") == {}


class TestFinalPrediction:
    def test_single_line(self):
        summary = split_sections(TEMPLATE_EXPLANATION)["summary"]
        assert final_prediction(summary) == "35.0%"

    def test_multiline(self):
        assert final_prediction(MULTILINE_SUMMARY) == "- A: 60.00%\n- B: 40.00%"

    def test_absent(self):
        assert final_prediction("# SUMMARY\nnothing here") == ""


class TestWasTruncated:
    def test_detects_the_frameworks_truncation_notice(self):
        assert was_truncated(
            "# SUMMARY\n---\n The comment size exceeded max size and has been truncated"
        )
        assert not was_truncated(TEMPLATE_EXPLANATION)
