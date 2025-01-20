import logging
import re

import pytest

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecasting.helpers.smart_searcher import SmartSearcher

logger = logging.getLogger(__name__)


async def test_ask_question_basic() -> None:
    num_searches_to_run = 1
    num_sites_per_search = 3
    searcher = SmartSearcher(
        include_works_cited_list=True,
        num_searches_to_run=num_searches_to_run,
        num_sites_per_search=num_sites_per_search,
    )
    question = "What is the recent news on SpaceX?"
    report = await searcher.invoke(question)
    logger.info(f"Report:\n{report}")

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


@pytest.mark.skip("Run this when needed as it's purely a qualitative test")
async def test_screenshot_question_2() -> None:
    with MonetaryCostManager() as cost_manager:
        searcher = SmartSearcher(num_sites_to_deep_dive=2)
        question = "Please tell me about the recent trends in the Federal Funds Effective Rate."
        result = await searcher.invoke(question)
        logger.info(f"Result: {result}")
        logger.info(f"Cost: {cost_manager.current_usage}")
