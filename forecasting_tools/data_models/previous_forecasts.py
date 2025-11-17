from datetime import datetime

from pydantic import BaseModel


class BinaryPreviousForecast(BaseModel):
    value: float
    timestamp: datetime
