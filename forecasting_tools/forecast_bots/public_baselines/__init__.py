"""Public-baseline forecasting bots.

These bots do NOT try to give the most accurate forecast. Instead, each one
estimates the forecast that a particular *group of people* would collectively
give if a randomized, representative sample of that group were asked the
question. They are meant to be cheap, agentic stand-ins for "what the public /
experts / news / left / center / right would predict", so the Metaculus
Community Prediction can be benchmarked against these public baselines.

All bots are built on PydanticAI agents and wrapped in the ``ForecastBot``
interface so they run identically to the other open-source bots.
"""

from forecasting_tools.forecast_bots.public_baselines.center_leaning_bot import (
    CenterLeaningBaselineBot,
)
from forecasting_tools.forecast_bots.public_baselines.credible_news_bot import (
    CredibleNewsBaselineBot,
)
from forecasting_tools.forecast_bots.public_baselines.expert_opinion_bot import (
    ExpertOpinionBaselineBot,
)
from forecasting_tools.forecast_bots.public_baselines.left_leaning_bot import (
    LeftLeaningBaselineBot,
)
from forecasting_tools.forecast_bots.public_baselines.population_baseline_bot import (
    PopulationBaselineBot,
)
from forecasting_tools.forecast_bots.public_baselines.public_sentiment_bot import (
    PublicSentimentBaselineBot,
)
from forecasting_tools.forecast_bots.public_baselines.right_leaning_bot import (
    RightLeaningBaselineBot,
)

__all__ = [
    "PopulationBaselineBot",
    "PublicSentimentBaselineBot",
    "ExpertOpinionBaselineBot",
    "CredibleNewsBaselineBot",
    "LeftLeaningBaselineBot",
    "CenterLeaningBaselineBot",
    "RightLeaningBaselineBot",
]
