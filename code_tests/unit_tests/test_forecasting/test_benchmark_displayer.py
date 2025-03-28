from streamlit.testing.v1 import AppTest

from forecasting_tools.forecast_helpers.benchmark_displayer import (
    run_benchmark_streamlit_page,
)


def test_benchmark_displayer() -> None:
    module_name = run_benchmark_streamlit_page.__module__
    project_path = module_name.replace(".", "/") + ".py"
    app_test = AppTest.from_file(project_path, default_timeout=600)
    app_test.run()
    assert not app_test.exception, f"Exception occurred: {app_test.exception}"
    assert len(app_test.title) > 0
