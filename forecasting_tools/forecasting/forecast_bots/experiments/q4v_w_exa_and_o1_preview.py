from forecasting_tools.ai_models.gpto1preview import GptO1Preview
from forecasting_tools.forecasting.forecast_bots.experiments.q4v_w_exa import (
    Q4VeritasWithExa,
)


class Q4VeritasWithExaAndO1Preview(Q4VeritasWithExa):
    FINAL_DECISION_LLM = GptO1Preview()
