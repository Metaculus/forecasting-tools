from forecasting_tools.ai_models.deepseek_r1 import DeepSeekR1
from forecasting_tools.forecasting.forecast_bots.experiments.q4v_w_exa_and_o1_preview import (
    Q4VeritasWithExaAndO1Preview,
)


class Q4VeritasWithExaAndDeepSeekR1(Q4VeritasWithExaAndO1Preview):
    FINAL_DECISION_LLM = DeepSeekR1(temperature=0.1)
