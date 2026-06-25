"""A lightweight Exa search tool that returns directly quotable evidence.

Unlike ``SmartSearcher`` (which synthesises an answer from search results), this
tool returns the raw building blocks an agent needs to *quote and attribute*
sources itself: for each result it returns the title, URL, author, publish date,
a short article summary, and the most relevant highlight quotes. This is meant to
be handed to an agent as a single tool so the agent can read what sources say and
record the forecast each source implies, with verbatim quotes.
"""

import logging

from forecasting_tools.ai_models.exa_searcher import ExaSearcher, ExaSource, SearchInput

logger = logging.getLogger(__name__)


class ExaQuoteSearcher:
    """Searches the web with Exa and returns quotable highlights + summaries."""

    def __init__(
        self,
        num_results: int = 6,
        num_quotes_per_source: int = 4,
        max_summary_chars: int = 600,
    ) -> None:
        self.num_results = num_results
        self.num_quotes_per_source = num_quotes_per_source
        self.max_summary_chars = max_summary_chars
        self.exa_searcher = ExaSearcher(
            include_text=False,
            include_highlights=True,
            include_summary=True,
            num_results=num_results,
        )

    async def search_for_quotes(
        self,
        query: str,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> str:
        search_input = SearchInput(
            web_search_query=query,
            highlight_query=query,
            include_domains=include_domains or [],
            exclude_domains=exclude_domains or [],
            include_text=None,
            start_published_date=None,
            end_published_date=None,
        )
        try:
            sources = await self.exa_searcher.invoke(search_input)
        except Exception as error:
            logger.warning(f"Exa quote search failed for '{query}': {error}")
            return f"No Exa search results available for '{query}' (error: {error})."
        if not sources:
            return f"No Exa search results found for '{query}'."
        return self._format_sources(query, sources)

    def _format_sources(self, query: str, sources: list[ExaSource]) -> str:
        blocks: list[str] = [f'Exa results for "{query}":']
        for index, source in enumerate(sources, start=1):
            blocks.append(self._format_single_source(index, source))
        return "\n\n".join(blocks)

    def _format_single_source(self, index: int, source: ExaSource) -> str:
        title = source.title or "(untitled)"
        author = f" by {source.author}" if source.author else ""
        header = (
            f"[{index}] {title}{author} — {source.readable_publish_date}\n"
            f"URL: {source.url or 'unknown'}"
        )
        summary = self._summary_text(source)
        quotes = self._top_quotes(source)
        quote_block = (
            "\n".join(f'  - "{quote}"' for quote in quotes)
            if quotes
            else "  - (no highlight quotes returned)"
        )
        return f"{header}\nSummary: {summary}\nQuotes:\n{quote_block}"

    def _summary_text(self, source: ExaSource) -> str:
        if not source.summary:
            return "(no summary returned)"
        summary = source.summary.strip()
        if len(summary) > self.max_summary_chars:
            summary = summary[: self.max_summary_chars].rstrip() + "…"
        return summary

    def _top_quotes(self, source: ExaSource) -> list[str]:
        scores = source.highlight_scores or [1.0] * len(source.highlights)
        scored_quotes = sorted(
            zip(source.highlights, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        top_quotes = [
            quote.strip()
            for quote, _ in scored_quotes[: self.num_quotes_per_source]
            if quote.strip()
        ]
        return top_quotes
