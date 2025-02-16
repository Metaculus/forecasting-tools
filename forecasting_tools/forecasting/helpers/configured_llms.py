from forecasting_tools.ai_models.deprecated_model_classes.gpt4o import Gpt4o
from forecasting_tools.ai_models.deprecated_model_classes.gpt4ovision import (
    Gpt4oVision,
    Gpt4VisionInput,
)

default_llms = {
    "basic": "gpt-4o",
    "advanced": "gpt-4o",
}


class BasicLlm(Gpt4o):
    # NOTE: If need be, you can force an API key here through OpenAI Client class variable
    pass


class AdvancedLlm(Gpt4o):
    pass


class VisionLlm(Gpt4oVision):
    pass


class VisionData(Gpt4VisionInput):
    pass
