import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from forecasting_tools import BinaryQuestion, TemplateBot
from forecasting_tools.data_models.binary_report import BinaryPrediction
from forecasting_tools.forecast_bots.official_bots import template_bot_2026_fall
from forecasting_tools.helpers.parallel_searcher import ParallelSearcher


@pytest.fixture
async def mcp_server(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[dict]:
    state = {"requests": [], "error": False, "empty": False, "delay": 0}

    async def handle(request: web.Request) -> web.Response:
        if request.method != "POST":
            return web.Response(status=405)
        body = await request.json()
        state["requests"].append((body, dict(request.headers)))
        if "id" not in body:
            return web.Response(status=202)
        if body["method"] == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "search-fixture", "version": "1"},
            }
        elif body["method"] == "tools/list":
            result = {
                "tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]
            }
        else:
            assert body["method"] == "tools/call"
            await asyncio.sleep(state["delay"])
            result = {
                "isError": state["error"],
                "content": [],
                "structuredContent": {
                    "results": (
                        []
                        if state["empty"]
                        else [
                            {
                                "title": "Official hurricane outlook",
                                "url": "https://example.org/outlook",
                                "publish_date": "2026-05-21",
                                "excerpts": [
                                    "The outlook anticipates major hurricanes."
                                ],
                            }
                        ]
                    )
                },
            }
        return web.json_response({"jsonrpc": "2.0", "id": body["id"], "result": result})

    app = web.Application()
    app.router.add_route("*", "/mcp", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setattr(ParallelSearcher, "_endpoint", f"http://127.0.0.1:{port}/mcp")
    try:
        yield state
    finally:
        await runner.cleanup()


async def test_selected_researcher_reaches_native_forecast(
    mcp_server: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PARALLEL_API_KEY", "ignored-test-key")
    bot = TemplateBot(
        llms={"researcher": "parallel/search"},
        research_reports_per_question=1,
        predictions_per_research_report=1,
        enable_summarize_research=False,
    )
    model = bot.get_llm("default", "llm")
    invoke = AsyncMock(
        return_value="The outlook supports a likely event. Probability: 70%"
    )
    monkeypatch.setattr(model, "invoke", invoke)
    monkeypatch.setattr(
        template_bot_2026_fall,
        "structure_output",
        AsyncMock(return_value=BinaryPrediction(prediction_in_decimal=0.7)),
    )
    question = BinaryQuestion(question_text="Will a major hurricane make landfall?")
    report = await bot.forecast_question(question)
    assert report.prediction == 0.7
    assert "https://example.org/outlook" in report.explanation
    assert "The outlook anticipates major hurricanes." in invoke.call_args.args[0]
    assert bot.make_llm_dict()["researcher"] == "parallel/search"
    calls = mcp_server["requests"]
    for _, headers in calls:
        assert "Authorization" not in headers
        assert headers["User-Agent"].startswith("forecasting-tools/")
    initialize = next(body for body, _ in calls if body["method"] == "initialize")
    assert initialize["params"]["clientInfo"]["name"] == "forecasting-tools"
    call = next(body for body, _ in calls if body["method"] == "tools/call")
    assert call["params"]["name"] == "web_search"
    assert call["params"]["arguments"]["search_queries"] == [question.question_text]
    assert len(call["params"]["arguments"]["session_id"]) == 32


async def test_search_empty_results_and_tool_error(mcp_server: dict) -> None:
    mcp_server["empty"] = True
    result = await ParallelSearcher().invoke(
        "Research hurricanes", ["Atlantic hurricane outlook"]
    )
    assert result == "No relevant sources found by Parallel Search MCP."
    mcp_server["error"] = True
    with pytest.raises(RuntimeError, match="tool error"):
        await ParallelSearcher().invoke(
            "Research hurricanes", ["Atlantic hurricane outlook"]
        )


async def test_search_timeout_and_cancellation(mcp_server: dict) -> None:
    mcp_server["delay"] = 1
    with pytest.raises(TimeoutError):
        await ParallelSearcher(timeout=0.1).invoke(
            "Research", ["Atlantic hurricane outlook"]
        )
    task = asyncio.create_task(
        ParallelSearcher().invoke("Research", ["Atlantic hurricane outlook"])
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    "prompt,queries", [("", ["query"]), ("Research", []), ("Research", [""])]
)
async def test_invalid_search_inputs(prompt: str, queries: list[str]) -> None:
    with pytest.raises(ValueError):
        await ParallelSearcher().invoke(prompt, queries)


def test_default_research_selection_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in [
        "ASKNEWS_CLIENT_ID",
        "ASKNEWS_SECRET",
        "ASKNEWS_API_KEY",
        "PERPLEXITY_API_KEY",
        "OPENROUTER_API_KEY",
        "EXA_API_KEY",
        "OPENAI_API_KEY",
        "METACULUS_TOKEN",
    ]:
        monkeypatch.delenv(key, raising=False)
    assert TemplateBot().get_llm("researcher", "llm").model == "perplexity/sonar-pro"
    monkeypatch.setenv("ASKNEWS_API_KEY", "test-key")
    assert TemplateBot().get_llm("researcher") == "asknews/news-summaries"
