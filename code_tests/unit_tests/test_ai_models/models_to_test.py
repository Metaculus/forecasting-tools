from litellm import model_cost

from forecasting_tools.ai_models.ai_utils.openai_utils import VisionMessageData
from forecasting_tools.ai_models.deprecated_model_classes.deepseek_r1 import DeepSeekR1
from forecasting_tools.ai_models.deprecated_model_classes.gpt4o import Gpt4o
from forecasting_tools.ai_models.deprecated_model_classes.gpt4ovision import Gpt4oVision
from forecasting_tools.ai_models.deprecated_model_classes.metaculus4o import (
    Gpt4oMetaculusProxy,
)
from forecasting_tools.ai_models.deprecated_model_classes.perplexity import Perplexity
from forecasting_tools.ai_models.exa_searcher import ExaSearcher
from forecasting_tools.ai_models.general_llm import GeneralLlm, ModelInputType
from forecasting_tools.ai_models.model_interfaces.ai_model import AiModel
from forecasting_tools.ai_models.model_interfaces.incurs_cost import IncursCost
from forecasting_tools.ai_models.model_interfaces.outputs_text import OutputsText
from forecasting_tools.ai_models.model_interfaces.request_limited_model import (
    RequestLimitedModel,
)
from forecasting_tools.ai_models.model_interfaces.retryable_model import RetryableModel
from forecasting_tools.ai_models.model_interfaces.time_limited_model import (
    TimeLimitedModel,
)
from forecasting_tools.ai_models.model_interfaces.token_limited_model import (
    TokenLimitedModel,
)
from forecasting_tools.ai_models.model_interfaces.tokens_incur_cost import (
    TokensIncurCost,
)


class ModelsToTest:

    def litellm_has_model_cost(self, model: str) -> bool:
        assert isinstance(model_cost, dict)
        available_models = model_cost.keys()
        return model in available_models

    ALL_MODELS = [
        Gpt4o,
        Gpt4oMetaculusProxy,
        Gpt4oVision,
        # GptO1Preview,
        # GptO1,
        # Claude35Sonnet,
        Perplexity,
        ExaSearcher,
        DeepSeekR1,
    ]
    BASIC_MODEL_LIST: list[type[AiModel]] = [
        model for model in ALL_MODELS if issubclass(model, AiModel)
    ]
    RETRYABLE_LIST: list[type[RetryableModel]] = [
        model for model in ALL_MODELS if issubclass(model, RetryableModel)
    ]
    TIME_LIMITED_LIST: list[type[TimeLimitedModel]] = [
        model for model in ALL_MODELS if issubclass(model, TimeLimitedModel)
    ]
    REQUEST_LIMITED_LIST: list[type[RequestLimitedModel]] = [
        model for model in ALL_MODELS if issubclass(model, RequestLimitedModel)
    ]
    TOKEN_LIMITED_LIST: list[type[TokenLimitedModel]] = [
        model for model in ALL_MODELS if issubclass(model, TokenLimitedModel)
    ]
    INCURS_COST_LIST: list[type[IncursCost]] = [
        model for model in ALL_MODELS if issubclass(model, IncursCost)
    ]
    OUTPUTS_TEXT: list[type[OutputsText]] = [
        model for model in ALL_MODELS if issubclass(model, OutputsText)
    ]
    TOKENS_INCUR_COST_LIST: list[type[TokensIncurCost]] = [
        model for model in ALL_MODELS if issubclass(model, TokensIncurCost)
    ]


class ModelTest:
    def __init__(self, llm: GeneralLlm, model_input: ModelInputType) -> None:
        self.llm = llm
        self.model_input = model_input


class LLMTestData:
    SMALL_BASE_64_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    CHEAP_VISION_MESSAGE_DATA = VisionMessageData(
        prompt="Hi", b64_image=SMALL_BASE_64_IMAGE, image_resolution="low"
    )

    def get_cheap_user_message(self) -> str:
        return "Hi"

    def get_cheap_vision_message_data(self) -> VisionMessageData:
        return self.CHEAP_VISION_MESSAGE_DATA
