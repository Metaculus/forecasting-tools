from forecasting_tools.ai_models.gpto1preview import GptO1Preview
from forecasting_tools.forecasting.forecast_bots.experiments.exa_q4_binary import (
    ExaQ4BinaryBot,
)


class ExaQ4BinaryO1PreviewBot(ExaQ4BinaryBot):
    FINAL_DECISION_LLM = GptO1Preview()
