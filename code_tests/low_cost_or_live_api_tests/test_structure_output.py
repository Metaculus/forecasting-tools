from typing import Any

import pytest
from pydantic import BaseModel

from forecasting_tools.forecast_helpers.structure_output import (
    structure_output,
)


class MathProblem(BaseModel):
    reasoning: str
    answer: int


class GroceryItem(BaseModel):
    item_name: str
    quantity: int


@pytest.mark.parametrize(
    "output,output_type,expected",
    [
        (
            "My list of items are apple, banana, and orange",
            list[str],
            ["apple", "banana", "orange"],
        ),
        (
            "If Mary has 10 apples and she gives 2 to John, how many apples does she have left? Reasoning: Mary has 10 apples and she gives 2 to John, so she has 10 - 2 = 8 apples left.",
            MathProblem,
            MathProblem(
                reasoning="Mary has 10 apples and she gives 2 to John, so she has 10 - 2 = 8 apples left.",
                answer=8,
            ),
        ),
        (
            "I need 2 apples, 5 banana, and 3 oranges",
            list[GroceryItem],
            [
                GroceryItem(item_name="apples", quantity=2),
                GroceryItem(item_name="banana", quantity=5),
                GroceryItem(item_name="oranges", quantity=3),
            ],
        ),
        (
            "How many piano tuners are there in New York City? Let me break this down step by step:\n\n1. Population of NYC: ~8.5 million\n2. Average household size: ~2.5 people\n3. Number of households: 8.5M/2.5 = 3.4M households\n4. % of households with pianos: ~1%\n5. Number of pianos: 3.4M * 0.01 = 34,000 pianos\n6. Pianos tuned per year: ~1 per piano\n7. Tunings per tuner per year: ~200 (5 per day * 40 weeks)\n8. Number of tuners needed: 34,000/200 = 170 tuners\n\nFinal answer: 170 piano tuners",
            float,
            170.0,
        ),
        (
            "How many piano tuners are there in New York City? Let me break this down step by step:\n\n1. Population of NYC: ~8.5 million\n2. Average household size: ~2.5 people\n3. Number of households: 8.5M/2.5 = 3.4M households\n4. % of households with pianos: ~1%\n5. Number of pianos: 3.4M * 0.01 = 34,000 pianos\n6. Pianos tuned per year: ~1 per piano\n7. Tunings per tuner per year: ~200 (5 per day * 40 weeks)\n8. Number of tuners needed: 34,000/200 = 170 tuners\n\nThus my final from the previous question is 30%",
            float,
            30.0,
        ),
    ],
)
@pytest.mark.asyncio
async def test_structure_output_parametrized(
    output: str, output_type: type, expected: Any
) -> None:
    result = await structure_output(output, output_type)
    assert result == expected
