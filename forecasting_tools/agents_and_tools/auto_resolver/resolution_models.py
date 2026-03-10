"""Pydantic models for structured resolution output parsing.

This module defines the data models used to parse the output from the resolver
agent into typed resolution decisions.
"""
from forecasting_tools.data_models.questions import BinaryResolution, CanceledResolution

from pydantic import BaseModel, Field
from typing import Literal, Optional


class DeadlineCheckResult(BaseModel):
    """Structured output from the LLM-based implicit deadline analysis.

    Used to determine whether a question's effective deadline (as described
    in its text, resolution criteria, or fine print) has already passed.

    Attributes:
        has_deadline: Whether the question contains a discernible deadline.
        deadline_date: The effective deadline as an ISO-8601 date string
            (YYYY-MM-DD), or None if no deadline was found.
        reasoning: Brief explanation of how the deadline was determined.
    """

    has_deadline: bool = Field(
        description="Whether the question contains a discernible deadline or time-bound condition"
    )
    deadline_date: Optional[str] = Field(
        default=None,
        description="The effective deadline in YYYY-MM-DD format, or null if no deadline found",
    )
    reasoning: str = Field(
        description="1-2 sentence explanation of how the deadline was identified"
    )


class BinaryResolutionResult(BaseModel):
    """Structured output for binary question resolution.

    This model is used to parse the final output from the resolver agent
    into a typed resolution decision with supporting evidence and reasoning.

    Attributes:
        resolution_status: The final resolution determination (TRUE, FALSE,
            AMBIGUOUS, ANNULLED, or NOT_YET_RESOLVABLE)
        reasoning: A 2-4 sentence explanation of why this resolution was chosen
        key_evidence: A list of 3-5 key pieces of evidence supporting this resolution
    """

    resolution_status: Literal[
        "TRUE", "FALSE", "AMBIGUOUS", "ANNULLED", "NOT_YET_RESOLVABLE"
    ] = Field(description="The final resolution determination")

    reasoning: str = Field(
        description="2-4 sentence explanation of why this resolution was chosen"
    )

    key_evidence: list[str] = Field(
        description="3-5 key pieces of evidence supporting this resolution",
        min_length=3,
        max_length=5,
    )


    def convert_to_binary_resolution(self) -> Optional[BinaryResolution]:
        """Convert structured result to typed binary resolution.

        Args:
            result: Parsed resolution result

        Returns:
            Typed BinaryResolution or None

        Raises:
            ValueError: If resolution status is unexpected
        """
     
        match self.resolution_status:
            case "TRUE":
                return True
            case "FALSE":
                return False
            case "AMBIGUOUS":
                return CanceledResolution.AMBIGUOUS
            case "ANNULLED":
                return CanceledResolution.ANNULLED
            case "NOT_YET_RESOLVABLE":
                return None
            case _:
                raise ValueError(
                    f"Unexpected resolution status: {result.resolution_status}"
                )
    
