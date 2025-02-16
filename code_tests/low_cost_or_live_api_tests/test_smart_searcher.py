import logging
import re

import pytest

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecasting.helpers.smart_searcher import SmartSearcher

logger = logging.getLogger(__name__)


async def test_ask_question_basic() -> None:
    searcher = SmartSearcher(
        include_works_cited_list=True,
        num_searches_to_run=1,
        num_sites_per_search=3,
    )
    question = "What is the recent news on SpaceX?"
    report = await searcher.invoke(question)
    logger.info(f"Report:\n{report}")
    validate_search_report(report)


async def test_ask_question_with_different_llm() -> None:
    temperature = 0.7
    chosen_model = "gpt-3.5-turbo"
    searcher = SmartSearcher(
        model=chosen_model,
        include_works_cited_list=True,
        num_searches_to_run=1,
        num_sites_per_search=3,
        temperature=temperature,
    )

    assert isinstance(searcher.llm, GeneralLlm)
    assert searcher.llm.model == chosen_model
    assert searcher.llm.litellm_kwargs["temperature"] == pytest.approx(
        temperature
    )
    assert searcher.llm.litellm_kwargs["model"] == chosen_model

    question = "What is the recent news on SpaceX?"
    report = await searcher.invoke(question)
    logger.info(f"Report:\n{report}")
    validate_search_report(report)


def validate_search_report(report: str) -> None:
    assert report, "Result should not be empty"
    assert isinstance(report, str), "Result should be a string"

    citation_numbers: list[int] = [
        int(num) for num in re.findall(r"\[(\d+)\]", report)
    ]
    for citation_number in citation_numbers:
        citation_number_appears_at_least_twice = (
            report.count(f"[{citation_number}]") >= 2
        )
        assert (
            citation_number_appears_at_least_twice
        ), f"Citation [{citation_number}] should appear at least twice. Once in the report body and once in the citation list."

        hyperlinks = re.findall(
            r"\\\[\[{}\]\((.*?)\)\\\]".format(citation_number), report
        )
        assert (
            len(hyperlinks) >= 2
        ), f"Citation [{citation_number}] should be part of at least two markdown hyperlinks"

        assert (
            len(set(hyperlinks)) == 1
        ), f"All hyperlinks for citation [{citation_number}] should be identical"


@pytest.mark.skip(
    "Not implemented yet. Cost would not be worth increase in visibility"
)
async def test_ask_question_without_works_cited_list() -> None:
    raise NotImplementedError


async def test_ask_question_empty_prompt() -> None:
    searcher = SmartSearcher()
    with pytest.raises(ValueError):
        await searcher.invoke("")
