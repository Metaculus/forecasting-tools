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
    ],
)
@pytest.mark.asyncio
async def test_structure_output_parametrized(
    output: str, output_type: type, expected: Any
) -> None:
    result = await structure_output(output, output_type, "gpt-4o-mini")
    assert result == expected
