import pytest

from code_tests.unit_tests.test_frontend.front_end_test_utils import (
    FrontEndTestUtils,
)
from front_end.app_pages.benchmark_page import BenchmarkPage
from front_end.helpers.report_displayer import ReportDisplayer


def test_all_markdown_is_clean_after_reports_displayed() -> None:
    app_test = FrontEndTestUtils.convert_page_to_app_tester(BenchmarkPage)
    app_test.run()
    all_markdown = app_test.markdown
    for markdown_element in all_markdown:
        assert ReportDisplayer.markdown_is_clean(markdown_element.value)


@pytest.mark.skip(
    reason="The app is working, but app_test system is not. Not worth fixing since this is a private page"
)
def test_reports_are_on_page() -> None:
    app_test = FrontEndTestUtils.convert_page_to_app_tester(BenchmarkPage)
    # NOTE: Add navigation through the custom auth
    app_test.run()
    select_box = app_test.selectbox(BenchmarkPage.BENCHMARK_FILE_SELECTBOX_KEY)
    for file_name in select_box.options:
        select_box.select(file_name)
        app_test.run()
        expected_reports = BenchmarkPage._load_benchmark_reports(file_name)
        FrontEndTestUtils.assert_x_valid_forecast_reports_are_on_the_page(
            app_test, expected_reports
        )


@pytest.mark.skip(reason="Don't need to test since this is a private page")
def test_there_is_a_score_for_each_benchmark_file() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="Don't need to test since this is a private page")
def test_questions_and_preditions_shown_for_each_benchmark_file() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="Don't need to test since this is a private page")
def test_question_details_shown_for_each_report() -> None:
    raise NotImplementedError
