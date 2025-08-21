from typing import Literal, Self

from pydantic import BaseModel

from forecasting_tools.data_models.questions import BinaryQuestion, MetaculusQuestion

DirectionsType = Literal["positive", "negative"]

StrengthsType = Literal["low", "medium", "high"]

LinkTypesType = Literal["causal"]


class CoherenceLink(BaseModel):
    question1_id: int
    question1: MetaculusQuestion
    question2_id: int
    question2: MetaculusQuestion
    direction: DirectionsType
    strength: StrengthsType
    type: LinkTypesType
    id: int

    @classmethod
    def from_metaculus_api_json(cls, api_json: dict) -> Self:
        return cls(
            question1_id=api_json["question1_id"],
            question2_id=api_json["question2_id"],
            direction=api_json["direction"],
            strength=api_json["strength"],
            type=api_json["type"],
            id=api_json["id"],
            question1=BinaryQuestion.from_metaculus_api_json(
                api_json["question1"], is_question_json=True
            ),
            question2=BinaryQuestion.from_metaculus_api_json(
                api_json["question2"], is_question_json=True
            ),
        )
