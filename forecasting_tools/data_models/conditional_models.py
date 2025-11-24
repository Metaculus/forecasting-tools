from pydantic import BaseModel

from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution


class PredictionAffirmed(BaseModel):
    pass


ConditionalPredictionTypesNonAffirmable = (
    NumericDistribution | PredictedOptionList | float
)
ConditionalPredictionTypes = (
    ConditionalPredictionTypesNonAffirmable | PredictionAffirmed
)


class ConditionalPrediction(BaseModel):
    parent: ConditionalPredictionTypes
    child: ConditionalPredictionTypes
    prediction_yes: ConditionalPredictionTypesNonAffirmable
    prediction_no: ConditionalPredictionTypesNonAffirmable
