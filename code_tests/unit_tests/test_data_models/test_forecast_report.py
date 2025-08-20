import os
import textwrap

import pytest

from code_tests.unit_tests.forecasting_test_manager import ForecastingTestManager
from code_tests.utilities_for_tests.misc_utils import replace_tzinfo_in_string
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.markdown_tree import MarkdownTree
from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
from forecasting_tools.data_models.numeric_report import NumericReport


def test_metaculus_report_is_jsonable() -> None:
    temp_writing_path = "temp/temp_metaculus_report.json"
    read_report_path = "code_tests/unit_tests/test_data_models/forecasting_test_data/metaculus_forecast_report_examples.json"
    reports = DataOrganizer.load_reports_from_file_path(read_report_path)
    assert any(isinstance(report, NumericReport) for report in reports)
    assert any(isinstance(report, BinaryReport) for report in reports)
    assert any(isinstance(report, MultipleChoiceReport) for report in reports)

    DataOrganizer.save_reports_to_file_path(reports, temp_writing_path)
    reports_2 = DataOrganizer.load_reports_from_file_path(temp_writing_path)
    assert len(reports) == len(reports_2)
    for report, report_2 in zip(reports, reports_2):
        assert report.question.question_text == report_2.question.question_text
        assert report.prediction == report_2.prediction
        assert report.question.id_of_post == report_2.question.id_of_post
        assert report.question.state == report_2.question.state
        assert replace_tzinfo_in_string(str(report)) == replace_tzinfo_in_string(
            str(report_2)
        )

    os.remove(temp_writing_path)


def test_report_sections_are_parsed_correctly() -> None:
    fake_report = ForecastingTestManager.get_fake_forecast_report()
    fake_explanation = textwrap.dedent(
        """
        # Intro
        This is a test section
        # Summary
        This is a test summary
        ## Summary 1
        Summary part 1

        # Explanation
        This is a test explanation

        ## Analysis
        ### Analysis 1
        This is a test analysis

        ### Analysis 2
        This is a test analysis
        #### Analysis 2.1
        This is a test analysis
        #### Analysis 2.2
        This is a test analysis
        - Conclusion 1
        - Conclusion 2

        # Conclusion
        This is a test conclusion
        - Conclusion 1
        - Conclusion 2
        """
    )
    fake_report.explanation = fake_explanation

    sections = fake_report.report_sections

    assert len(sections) == 4
    assert sections[0].title == "Intro"
    assert "This is a test section" in sections[0].section_content.strip()
    assert len(sections[0].sub_sections) == 0

    summary_section = sections[1]
    assert summary_section.title == "Summary"
    assert "This is a test summary" in summary_section.section_content
    assert len(summary_section.sub_sections) == 1

    summary_1 = summary_section.sub_sections[0]
    assert summary_1.title == "Summary 1"
    assert "Summary part 1" in summary_1.section_content

    explanation_section = sections[2]
    assert explanation_section.title == "Explanation"
    assert "This is a test explanation" in explanation_section.section_content
    assert len(explanation_section.sub_sections) == 1

    analysis_section = explanation_section.sub_sections[0]
    assert analysis_section.title == "Analysis"
    assert len(analysis_section.sub_sections) == 2

    analysis_1 = analysis_section.sub_sections[0]
    assert analysis_1.title == "Analysis 1"
    assert "This is a test analysis" in analysis_1.section_content

    analysis_2 = analysis_section.sub_sections[1]
    assert analysis_2.title == "Analysis 2"
    assert len(analysis_2.sub_sections) == 2

    analysis_2_1 = analysis_2.sub_sections[0]
    assert analysis_2_1.title == "Analysis 2.1"
    assert "This is a test analysis" in analysis_2_1.section_content

    analysis_2_2 = analysis_2.sub_sections[1]
    assert analysis_2_2.title == "Analysis 2.2"
    assert "This is a test analysis" in analysis_2_2.section_content
    assert "- Conclusion 1" in analysis_2_2.section_content
    assert "- Conclusion 2" in analysis_2_2.section_content

    conclusion_section = sections[3]
    assert conclusion_section.title == "Conclusion"
    assert "This is a test conclusion" in conclusion_section.section_content
    assert "- Conclusion 1" in conclusion_section.section_content
    assert "- Conclusion 2" in conclusion_section.section_content

    combined_content = combine_all_section_content(sections)
    assert combined_content.replace("\n", "") == fake_explanation.replace("\n", "")


def combine_all_section_content(sections: list[MarkdownTree]) -> str:
    # Only goes to level h4 in the report list
    combined_content = ""
    for section in sections:
        combined_content += section.section_content
        for sub_section in section.sub_sections:
            combined_content += sub_section.section_content
            for sub_sub_section in sub_section.sub_sections:
                combined_content += sub_sub_section.section_content
                for sub_sub_sub_section in sub_sub_section.sub_sections:
                    combined_content += sub_sub_sub_section.section_content
    return combined_content


@pytest.mark.skip("Not implemented")
def test_combine_forecast_reports_works() -> None:
    raise NotImplementedError


@pytest.mark.skip("Not implemented")
def test_summary_section_is_correct() -> None:
    raise NotImplementedError


@pytest.mark.skip("Not implemented")
def test_research_section_is_correct() -> None:
    raise NotImplementedError


@pytest.mark.skip("Not implemented")
def test_forecasts_rationale_section_is_correct() -> None:
    raise NotImplementedError


@pytest.mark.skip("Not implemented")
def test_each_report_type_is_jsonable() -> None:
    raise NotImplementedError
