# This file is run before any tests are run in order to configure tests


import dotenv
import pytest

from forecasting_tools.util.custom_logger import CustomLogger

PROFILING_DIR = "logs/profiling"


@pytest.fixture(scope="session", autouse=True)
def setup_logging() -> None:
    dotenv.load_dotenv()
    CustomLogger.setup_logging()


def sanitize_filename(name: str) -> str:
    """Remove characters that are invalid for filenames."""


# Can be used to profile each test
# @pytest.fixture(autouse=True)
# def profile_each_test(request: pytest.FixtureRequest) -> None:
#     """Profiles each test function and saves the results sorted by time."""
#     profiler = cProfile.Profile()
#     profiler.enable()

#     yield  # Run the test

#     profiler.disable()

#     os.makedirs(PROFILING_DIR, exist_ok=True)

#     test_node_id = request.node.nodeid
#     sanitized_test_name = re.sub(r'[<>:"/\\|?*]', "_", test_node_id).replace("::", "_")  # Pytest uses :: to separate scopes
#     profile_file_path = os.path.join(
#         PROFILING_DIR, f"{sanitized_test_name}.prof"
#     )

#     # Save the profiling stats, sorted by cumulative time
#     with open(profile_file_path, "w") as f:
#         stats = pstats.Stats(profiler, stream=f)
#         stats.sort_stats("cumtime")
#         stats.print_stats()
