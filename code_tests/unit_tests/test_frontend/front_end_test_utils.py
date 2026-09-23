import importlib
import logging
from pathlib import Path

from streamlit.testing.v1 import AppTest

from forecasting_tools.front_end.Home import AppPage

logger = logging.getLogger(__name__)


class FrontEndTestUtils:

    @staticmethod
    def convert_page_to_app_tester(app_page: type[AppPage]) -> AppTest:
        module = importlib.import_module(app_page.__module__)
        if module.__file__ is None:
            raise RuntimeError(f"Cannot find the file backing {app_page.__module__}")
        script_path = Path(module.__file__).resolve()
        return AppTest.from_file(script_path, default_timeout=600)
