from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot2025,
)


class MainBot(Q1TemplateBot2025):
    """
    The verified highest accuracy bot available.
    Main bot is the template bot for now till we can confirm other bots are better.
    """

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        return {
            "default": "openai/o1",
            "summarizer": "openai/gpt-4o-mini",
        }
