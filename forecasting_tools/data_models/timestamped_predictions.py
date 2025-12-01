from abc import ABC
from datetime import datetime

from forecasting_tools import BinaryPrediction, NumericDistribution


class TimeStampedPrediction(ABC):
    timestamp: datetime


class BinaryTimestampedPrediction(BinaryPrediction, TimeStampedPrediction):
    pass


class NumericTimestampedDistribution(NumericDistribution, TimeStampedPrediction):
    pass
