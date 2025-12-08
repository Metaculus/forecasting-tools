from abc import ABC
from datetime import datetime

from pydantic import BaseModel

from forecasting_tools.data_models.binary_report import BinaryPrediction
from forecasting_tools.data_models.numeric_report import NumericDistribution


class TimeStampedPrediction(BaseModel, ABC):
    timestamp: datetime
    timestamp_end: datetime | None


class BinaryTimestampedPrediction(BinaryPrediction, TimeStampedPrediction):
    pass


class NumericTimestampedDistribution(NumericDistribution, TimeStampedPrediction):
    strict_validation: bool = False
