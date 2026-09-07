import importlib
from pathlib import Path

from streamlit.testing.v1 import AppTest

from forecasting_tools.cp_benchmarking.benchmark_displayer import (
    run_benchmark_streamlit_page,
)


def test_benchmark_displayer() -> None:
    module = importlib.import_module(run_benchmark_streamlit_page.__module__)
    if module.__file__ is None:
        raise RuntimeError(
            f"Cannot find the file backing {run_benchmark_streamlit_page.__module__}"
        )
    app_test = AppTest.from_file(Path(module.__file__).resolve(), default_timeout=600)
    app_test.run()
    assert not app_test.exception, f"Exception occurred: {app_test.exception}"
    assert len(app_test.title) > 0
