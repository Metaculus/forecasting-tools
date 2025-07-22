import logging

from code_tests.utilities_for_tests.coroutine_testing import (
    assert_coroutines_run_under_x_times_duration_of_benchmark,
)
from forecasting_tools.agents_and_tools.research.smart_searcher import SmartSearcher

logger = logging.getLogger(__name__)


async def test_ask_question_concurrent_requests() -> None:
    searcher = SmartSearcher()
    num_test_calls = 6
    num_benchmark_calls = 3
    question = "What is AI?"
    tasks = [searcher.invoke(question) for _ in range(num_test_calls)]
    benchmark = [searcher.invoke(question) for _ in range(num_benchmark_calls)]
    assert_coroutines_run_under_x_times_duration_of_benchmark(benchmark, tasks, 2)
