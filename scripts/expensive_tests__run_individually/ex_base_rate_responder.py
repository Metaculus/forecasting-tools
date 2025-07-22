import asyncio
from datetime import datetime

from forecasting_tools.agents_and_tools.base_rates.base_rate_researcher import (
    BaseRateReport,
    BaseRateResearcher,
    DenominatorOption,
    ReferenceClassWithCount,
)

# TODO: Test these specifically if worth the cost to run each time
# [ ] Correctly solves the base rate for examples (when certain parts are mocked)
# [ ] Gives a reasonable rate when only resolve date is given but a known historical rate can be found
# [ ] Gives correct past rate but no specific future date if resolution criteria are not passed in
# [ ] Errors if not all correct information is given
# [ ] Metaculus question asking actually calls base rate questions with right information
# [ ] Costs are under a certain amount


def test_responder_gets_example_base_rate_right_for_per_day_question() -> None:
    question = "What are the chances per year that a US president will be assassinated?"
    responder = BaseRateResearcher(question)
    base_rate_report = asyncio.run(responder.make_base_rate_report())
    assert_base_rate_report_parts_exist(base_rate_report)


def test_responder_gets_example_base_rate_right_for_per_event_class_question() -> None:
    question = "How many US presidential elections have been won by Democrats?"
    responder = BaseRateResearcher(question)
    base_rate_report = asyncio.run(responder.make_base_rate_report())
    assert_base_rate_report_parts_exist(base_rate_report)


def assert_base_rate_report_parts_exist(
    base_rate_report: BaseRateReport,
) -> None:
    assert isinstance(base_rate_report.historical_rate, float)
    assert isinstance(base_rate_report, BaseRateReport)
    assert isinstance(base_rate_report.markdown_report, str)
    assert isinstance(base_rate_report.question, str)
    assert isinstance(base_rate_report.historical_rate, float)
    assert isinstance(base_rate_report.markdown_report, str)
    assert isinstance(
        base_rate_report.numerator_reference_class, ReferenceClassWithCount
    )
    assert isinstance(
        base_rate_report.denominator_reference_class, ReferenceClassWithCount
    )
    assert isinstance(base_rate_report.denominator_type, DenominatorOption)
    assert isinstance(base_rate_report.start_date, datetime)
    assert isinstance(base_rate_report.end_date, datetime)
