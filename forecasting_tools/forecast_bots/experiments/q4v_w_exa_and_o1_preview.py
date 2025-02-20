from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_bots.experiments.q4v_w_exa import (
    Q4VeritasWithExa,
)


class Q4VeritasWithExaAndO1Preview(Q4VeritasWithExa):
    FINAL_DECISION_LLM = GeneralLlm(model="o1-preview", temperature=0.1)
