from forecasting_tools.ai_models.gemini2exp import Gemini2Exp
from forecasting_tools.ai_models.gemini2flash import Gemini2Flash
from forecasting_tools.ai_models.gpt4ovision import (
    Gpt4oVision,
    Gpt4VisionInput,
)


class BasicLlm(Gemini2Flash):
    # NOTE: If need be, you can force an API key here through OpenAI Client class variable
    pass


class AdvancedLlm(Gemini2Exp):
    pass


class VisionLlm(Gpt4oVision):
    pass


class VisionData(Gpt4VisionInput):
    pass
