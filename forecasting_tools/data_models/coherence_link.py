from typing import List, Literal, Self

from pydantic import BaseModel

from forecasting_tools.data_models.questions import MetaculusQuestion

LinkTypesType = Literal["causal"]


class CoherenceLink(BaseModel):
    question1_id: int
    question2_id: int
    direction: int
    strength: int
    type: LinkTypesType
    id: int


class DetailedCoherenceLink(CoherenceLink, BaseModel):
    question1: MetaculusQuestion
    question2: MetaculusQuestion

    @classmethod
    def from_metaculus_api_json(cls, api_json: dict) -> Self:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question1: dict = api_json["question1"]
        question2: dict = api_json["question2"]
        return cls(
            question1_id=api_json["question1_id"],
            question2_id=api_json["question2_id"],
            direction=api_json["direction"],
            strength=api_json["strength"],
            type=api_json["type"],
            id=api_json["id"],
            question1=DataOrganizer.get_question_from_question_json(question1),
            question2=DataOrganizer.get_question_from_question_json(question2),
        )


class NeedsUpdateResponse(BaseModel):
    questions: List[MetaculusQuestion]
    links: list[CoherenceLink]
