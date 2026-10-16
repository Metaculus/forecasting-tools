import asyncio
import json
from datetime import timedelta
from importlib.metadata import version
from uuid import uuid4

from forecasting_tools.util.optional_imports import missing_optional_package_error

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import CallToolResult, Implementation, TextContent
except ImportError as e:
    raise missing_optional_package_error("mcp", "parallel-search") from e


class ParallelSearcher:
    """Return cited search excerpts using the anonymous Parallel Search MCP."""

    _endpoint = "https://search.parallel.ai/mcp"

    def __init__(self, timeout: float = 60) -> None:
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        self.timeout = timeout

    async def invoke(self, prompt: str, search_queries: list[str]) -> str:
        if (
            not prompt.strip()
            or not search_queries
            or any(not query.strip() for query in search_queries)
        ):
            raise ValueError(
                "A research prompt and nonempty search queries are required"
            )

        task = asyncio.create_task(self._search(prompt, search_queries))
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=self.timeout)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        return self._format_result(result)

    async def _search(self, prompt: str, search_queries: list[str]) -> CallToolResult:
        package_version = version("forecasting-tools")
        async with streamablehttp_client(
            self._endpoint,
            headers={"User-Agent": f"forecasting-tools/{package_version}"},
            timeout=self.timeout,
            sse_read_timeout=self.timeout,
        ) as (read_stream, write_stream, _):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self.timeout),
                client_info=Implementation(
                    name="forecasting-tools", version=package_version
                ),
            ) as session:
                await session.initialize()
                return await session.call_tool(
                    "web_search",
                    arguments={
                        "objective": prompt,
                        "search_queries": search_queries,
                        "session_id": uuid4().hex,
                    },
                )

    @staticmethod
    def _format_result(result: CallToolResult) -> str:
        if result.isError:
            raise RuntimeError("Parallel Search MCP returned a tool error")
        payload = result.structuredContent
        if payload is None:
            text = "\n".join(
                block.text for block in result.content if isinstance(block, TextContent)
            )
            payload = json.loads(text)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("results"), list
        ):
            raise ValueError("Parallel Search MCP returned an invalid search response")

        sources = []
        for source in payload["results"]:
            title = source.get("title") or source["url"]
            date = source.get("publish_date")
            excerpts = source.get("excerpts", [])
            sources.append(
                f"### {title}\nSource: {source['url']}\n"
                + (f"Published: {date}\n" if date else "")
                + "\n"
                + "\n\n".join(excerpts)
            )
        return (
            "\n\n".join(sources) or "No relevant sources found by Parallel Search MCP."
        )
