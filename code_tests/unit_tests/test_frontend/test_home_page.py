import logging

from code_tests.unit_tests.test_frontend.front_end_test_utils import FrontEndTestUtils
from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.front_end.Home import HomePage

logger = logging.getLogger(__name__)
import pytest


@pytest.mark.parametrize("page", HomePage.NON_HOME_PAGES)
def test_home_page(page: type[AppPage]) -> None:
    at = FrontEndTestUtils.convert_page_to_app_tester(HomePage)
    at.run()
    assert not at.exception, f"Exception occurred: {at.exception}"
    button = at.button(key=page.PAGE_DISPLAY_NAME)
    button.click()
    at.run()
    assert not at.exception, f"Exception occurred: {at.exception}"


@pytest.mark.parametrize(
    "page_class",
    [page for page in HomePage.NON_HOME_PAGES],
)
def test_all_pages_compile(
    page_class: type[AppPage],
) -> None:
    app_test = FrontEndTestUtils.convert_page_to_app_tester(page_class)
    app_test.run()
    assert not app_test.exception
    assert not app_test.error
