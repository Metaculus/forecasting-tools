import logging

from forecasting_tools.benchmarking.benchmark_displayer import (
    run_benchmark_streamlit_page,
)
from forecasting_tools.front_end.helpers.app_page import AppPage

logger = logging.getLogger(__name__)


class ChatMessage:
    def __init__(self, role: str, content: str, reasoning: str = "") -> None:
        self.role = role
        self.content = content
        self.reasoning = reasoning

    def to_open_ai_message(self) -> dict:
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
            reasoning=data.get("reasoning", ""),
        )


class BenchmarkPage(AppPage):
    PAGE_DISPLAY_NAME: str = "🏆 Benchmarks"
    URL_PATH: str = "/benchmark"
    ENABLE_HEADER: bool = False
    ENABLE_FOOTER: bool = False

    @classmethod
    async def _async_main(cls) -> None:
        run_benchmark_streamlit_page("logs/forecasts/benchmarks/")


if __name__ == "__main__":
    BenchmarkPage.main()
