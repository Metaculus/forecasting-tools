import subprocess
import sys
from pathlib import Path

import forecasting_tools

REPO_ROOT = Path(__file__).resolve().parents[2]

# CI installs every extra, so a base install can only be exercised by hiding the
# optional packages from the import system in a fresh interpreter.
_HIDE_OPTIONAL_PACKAGES = """
import sys

class _Blocker:
    blocked = {"agents", "hyperbrowser", "streamlit"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.blocked:
            raise ImportError(f"simulated missing package: {fullname}")
        return None

sys.meta_path.insert(0, _Blocker())
"""


def _run_as_base_install(snippet: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _HIDE_OPTIONAL_PACKAGES + snippet],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_bot_infrastructure_works_without_optional_packages() -> None:
    _run_as_base_install(
        """
import sys

import forecasting_tools
import run_bots  # noqa: F401

still_loaded = [
    package
    for package in ("agents", "hyperbrowser", "streamlit")
    if package in sys.modules
]
assert not still_loaded, f"optional packages imported on a base install: {still_loaded}"

bot = forecasting_tools.TemplateBot(research_reports_per_question=1)
assert type(bot).__name__ == "TemplateBot"
"""
    )


def test_gated_names_say_which_extra_to_install() -> None:
    _run_as_base_install(
        """
import forecasting_tools

extra_for_name = {
    "Benchmarker": "agents",
    "BenchmarkForBot": "agents",
    "BotOptimizer": "agents",
    "DataAnalyzer": "agents",
    "QuestionDecomposer": "agents",
    "ComputerUse": "computer-use",
    "run_benchmark_streamlit_page": "front-end",
}
for name, extra in extra_for_name.items():
    try:
        getattr(forecasting_tools, name)
    except AttributeError as error:
        assert f"forecasting-tools[{extra}]" in str(error), (name, str(error))
    else:
        raise AssertionError(f"{name} should not be reachable without its extra")

# AttributeError rather than ImportError, so capability checks degrade quietly.
assert hasattr(forecasting_tools, "Benchmarker") is False
"""
    )


def test_subpackages_are_reachable_as_attributes() -> None:
    assert forecasting_tools.cp_benchmarking is not None
    assert forecasting_tools.auto_optimizers is not None
    assert forecasting_tools.data_models is not None


def test_unknown_attribute_still_raises_attribute_error() -> None:
    try:
        forecasting_tools.NotARealName
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError for an unknown attribute")
