import logging

from streamlit.testing.v1 import AppTest

from forecasting_tools.front_end.Home import AppPage

logger = logging.getLogger(__name__)


class FrontEndTestUtils:

    @staticmethod
    def convert_page_to_app_tester(app_page: type[AppPage]) -> AppTest:
        module_name = app_page.__module__
        project_path = module_name.replace(".", "/") + ".py"
        app_test = AppTest.from_file(project_path, default_timeout=600)
        return app_test
