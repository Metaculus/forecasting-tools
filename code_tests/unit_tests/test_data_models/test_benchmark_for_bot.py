from code_tests.utilities_for_tests import jsonable_assertations
from forecasting_tools.benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.multiple_choice_report import (
    MultipleChoiceReport,
)
from forecasting_tools.data_models.numeric_report import NumericReport


def test_benchmark_for_bot() -> None:
    read_path = "code_tests/unit_tests/test_data_models/forecasting_test_data/benchmark_object_examples.json"
    temp_write_path = "temp/temp_benchmark_object_examples.json"
    jsonable_assertations.assert_reading_and_printing_from_file_works(
        BenchmarkForBot,
        read_path,
        temp_write_path,
    )

    benchmarks = BenchmarkForBot.load_json_from_file_path(read_path)
    all_reports = [
        report
        for benchmark in benchmarks
        for report in benchmark.forecast_reports
    ]
    assert any(isinstance(report, NumericReport) for report in all_reports)
    assert any(
        isinstance(report, MultipleChoiceReport) for report in all_reports
    )
    assert any(isinstance(report, BinaryReport) for report in all_reports)
