from typing import Any

import pytest
from pydantic import BaseModel

from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.helpers.structure_output import structure_output


class MathProblem(BaseModel):
    reasoning: str
    answer: int


class GroceryItem(BaseModel):
    item_name: str
    quantity: int


NUMERIC_FORECAST_1 = """
(a) Time left until the outcome is known
- Roughly 1–4 months. If a reconciliation bill passing both chambers and being signed by the president occurs, the official CBO score on education impacts would follow soon after. If no such bill is enacted by the end of FY2025 (Sept 30, 2025), the question would be annulled.

(b) The outcome if nothing changed
- Annulled. Without a 2025 reconciliation bill affecting education, there would be no new formal reduction in education outlays scored for 2025–2034.

(c) The outcome if the current trend continued
- Central forecast (10-year impact on education outlays): about a net reduction of 350 billion dollars (in billions, -$350). This reflects publicly discussed figures suggesting large-scale education cuts in the reconciliation package (e.g., roughly a few hundred billion in discretionary education spending over a decade, with additional programmatic reductions possible under various Title I/IDEA-type mechanisms).

(d) The expectations of experts and markets
- Experts expect the official CBO scoring to lean toward substantial education reductions if a GOP-dominated reconciliation bill advances, with CRFB-style tallies in the low-to-mid hundreds of billions of dollars over 10 years. Market commentary around the time suggested sensitivity to whether education cuts accompany broader tax and entitlement changes, and whether any offsets or protections for certain programs (e.g., Pell, IDEA) are preserved. In short: strong bets on sizable reductions are plausible if education is treated as a primary target in the package, but large uncertainties remain until a score is released.

(e) An unexpected scenario that results in a low outcome (larger cuts)
- A surprise late-stage agreement that drops additional education programs or tightens Byrd Rule constraints, leading to even larger reductions (e.g., exceeding -$500 billion over the 10-year window). This could come from a narrower reconciliation package that concentrates cuts in discretionary education and restricts potential offsets or protections.

(f) An unexpected scenario that results in a high outcome (smaller cuts or net increase)
- A bipartisan compromise that shields or restores key education programs (Title I, IDEA, Pell-like supports) and strips other areas of larger cuts, or a CBO re-score that shifts baseline assumptions, resulting in far smaller net reductions or even a net increase in education outlays over the decade. Administrative changes, unintended consequences, or political derailments could also tilt the score toward smaller cuts than currently anticipated.

Final answer
Percentile 10: -520,000,000,000
Percentile 20: -430,000,000,000
Percentile 40: -360,000,000,000
Percentile 60: -320,000,000,000
Percentile 80: -290,000,000,000
Percentile 90: -250,000,000,000
"""

PARSING_INSTRUCTIONS_1 = """
The text given to you is trying to give a forecast distribution for a numeric question.
- This text is trying to answer the numeric question: "How much will the education outlays be reduced by in the next 10 years?".
- When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
- The units for the forecast are: B $
- Your work will be shown publicly with these units stated verbatim after the numbers your parse.
- As an example, someone else guessed that the answer will be between -520 B $ and -250 B $.
- If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
- If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
- Turn any values that are in scientific notation into regular numbers.
"""


# https://www.metaculus.com/questions/10642/
NUMERIC_FORECAST_2 = """
Final answer
Percentile 10: 10
Percentile 20: 11
Percentile 40: 11
Percentile 60: 11
Percentile 80: 11
Percentile 90: 11
"""

PARSING_INSTRUCTIONS_2 = """
The text given to you is trying to give a forecast distribution for a numeric question.
- This text is trying to answer the numeric question: "How many UN member states will formally recognize Taiwan at the end of 2025?".
- When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
- The units for the forecast are: countries
- Your work will be shown publicly with these units stated verbatim after the numbers your parse.
- As an example, someone else guessed that the answer will be between 0 and 220 countries.
- If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
- If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
- Turn any values that are in scientific notation into regular numbers.
"""


@pytest.mark.parametrize(
    "output,parsing_instructions,output_type,expected",
    [
        (
            "My list of items are apple, banana, and orange",
            None,
            list[str],
            ["apple", "banana", "orange"],
        ),
        (
            "If Mary has 10 apples and she gives 2 to John, how many apples does she have left? Reasoning: Mary has 10 apples and she gives 2 to John, so she has 10 - 2 = 8 apples left.",
            None,
            MathProblem,
            MathProblem(
                reasoning="Mary has 10 apples and she gives 2 to John, so she has 10 - 2 = 8 apples left.",
                answer=8,
            ),
        ),
        (
            "I need 2 apples, 5 banana, and 3 oranges",
            None,
            list[GroceryItem],
            [
                GroceryItem(item_name="apples", quantity=2),
                GroceryItem(item_name="banana", quantity=5),
                GroceryItem(item_name="oranges", quantity=3),
            ],
        ),
        (
            "How many piano tuners are there in New York City? Let me break this down step by step:\n\n1. Population of NYC: ~8.5 million\n2. Average household size: ~2.5 people\n3. Number of households: 8.5M/2.5 = 3.4M households\n4. % of households with pianos: ~1%\n5. Number of pianos: 3.4M * 0.01 = 34,000 pianos\n6. Pianos tuned per year: ~1 per piano\n7. Tunings per tuner per year: ~200 (5 per day * 40 weeks)\n8. Number of tuners needed: 34,000/200 = 170 tuners\n\nFinal answer: 170 piano tuners",
            None,
            float,
            170.0,
        ),
        (
            "How many piano tuners are there in New York City? Let me break this down step by step:\n\n1. Population of NYC: ~8.5 million\n2. Average household size: ~2.5 people\n3. Number of households: 8.5M/2.5 = 3.4M households\n4. % of households with pianos: ~1%\n5. Number of pianos: 3.4M * 0.01 = 34,000 pianos\n6. Pianos tuned per year: ~1 per piano\n7. Tunings per tuner per year: ~200 (5 per day * 40 weeks)\n8. Number of tuners needed: 34,000/200 = 170 tuners\n\nThus my final from the previous question is 30%",
            "Give your answer as a percentage.",
            float,
            30.0,
        ),
        (
            NUMERIC_FORECAST_1,
            PARSING_INSTRUCTIONS_1,
            list[Percentile],
            [
                Percentile(value=-520, percentile=0.1),
                Percentile(value=-430, percentile=0.2),
                Percentile(value=-360, percentile=0.4),
                Percentile(value=-320, percentile=0.6),
                Percentile(value=-290, percentile=0.8),
                Percentile(value=-250, percentile=0.9),
            ],
        ),
        (
            NUMERIC_FORECAST_2,
            PARSING_INSTRUCTIONS_2,
            list[Percentile],
            [
                Percentile(value=10, percentile=0.1),
                Percentile(value=11, percentile=0.2),
                Percentile(value=11, percentile=0.4),
                Percentile(value=11, percentile=0.6),
                Percentile(value=11, percentile=0.8),
                Percentile(value=11, percentile=0.9),
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_structure_output_parametrized(
    output: str, parsing_instructions: str | None, output_type: type, expected: Any
) -> None:
    result = await structure_output(
        output,
        output_type,
        additional_instructions=parsing_instructions,
        num_validation_samples=2,
    )
    assert result == expected
