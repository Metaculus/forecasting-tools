import urllib.parse

from forecasting_tools.forecast_helpers.smart_searcher import ExaHighlightQuote


class WorksCitedCreator:

    @classmethod
    def create_works_cited_list(
        cls,
        quotes_in_citation_order: list[ExaHighlightQuote],
        report_with_citations: str,
    ) -> str:
        works_cited_dict = cls.__build_works_cited_dict(
            quotes_in_citation_order, report_with_citations
        )
        return cls.__format_works_cited_list(works_cited_dict)

    @classmethod
    def __build_works_cited_dict(
        cls, citations: list[ExaHighlightQuote], report: str
    ) -> dict[str, list[tuple[int, str]]]:
        works_cited_dict: dict[str, list[tuple[int, str]]] = {}
        for i, citation in enumerate(citations):
            if f"[{i+1}]" not in report:
                continue
            url_domain = cls.__extract_url_domain_from_highlight(citation)
            source_key = f"{citation.source.title} ({url_domain})"
            works_cited_dict.setdefault(source_key, []).append(
                (i + 1, citation.highlight_text)
            )
        return works_cited_dict

    @classmethod
    def __extract_url_domain_from_highlight(
        cls,
        citation: ExaHighlightQuote,
    ) -> str | None:
        try:
            assert citation.source.url is not None, "Source URL is None"
            url_domain = urllib.parse.urlparse(citation.source.url).netloc
            return url_domain
        except Exception:
            return citation.source.url

    @classmethod
    def __format_works_cited_list(
        cls, works_cited_dict: dict[str, list[tuple[int, str]]]
    ) -> str:
        works_cited_list = ""
        for source_num, (source, highlights) in enumerate(
            works_cited_dict.items(), 1
        ):
            works_cited_list += f"Source {source_num}: {source}\n"
            for citation_num, highlight in highlights:
                works_cited_list += (
                    f'- [{citation_num}] Quote: "{highlight}"\n'
                )
            works_cited_list += "\n"
        return works_cited_list
