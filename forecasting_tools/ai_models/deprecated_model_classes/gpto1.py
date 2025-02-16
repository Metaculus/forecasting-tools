import typing_extensions

from forecasting_tools.ai_models.deprecated_model_classes.gpto1preview import (
    GptO1Preview,
)


@typing_extensions.deprecated(
    "LLM calls will slowly be moved to the GeneralLlm class", category=None
)
class GptO1(GptO1Preview):
    MODEL_NAME: str = "o1"
