from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_bots.experiments.q4v_w_exa_and_o1_preview import (
    Q4VeritasWithExaAndO1Preview,
)


class Q4VeritasWithExaAndDeepSeekR1(Q4VeritasWithExaAndO1Preview):
    FINAL_DECISION_LLM = GeneralLlm(
        model="deepseek/deepseek-reasoner", temperature=0.1
    )
