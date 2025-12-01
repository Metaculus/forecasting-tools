from datetime import datetime

from forecasting_tools import BinaryPrediction, NumericDistribution


class BinaryTimestampedPrediction(BinaryPrediction):
    timestamp: datetime


class NumericTimestampedDistribution(NumericDistribution):
    timestamp: datetime
