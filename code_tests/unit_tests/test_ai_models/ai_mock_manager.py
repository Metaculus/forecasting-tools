import asyncio
import logging
from unittest.mock import Mock

from forecasting_tools.ai_models.model_interfaces.ai_model import AiModel
from forecasting_tools.ai_models.model_interfaces.request_limited_model import (
    RequestLimitedModel,
)
from forecasting_tools.ai_models.model_interfaces.time_limited_model import (
    TimeLimitedModel,
)
from forecasting_tools.ai_models.model_interfaces.token_limited_model import (
    TokenLimitedModel,
)
from forecasting_tools.ai_models.model_interfaces.tokens_are_calculatable import (
    TokensAreCalculatable,
)

logger = logging.getLogger(__name__)
from typing import Any


class AiModelMockManager:

    @staticmethod
    def get_direct_call_function_path_as_string(
        subclass: type[AiModel],
    ) -> str:
        full_function_path = f"{subclass._mockable_direct_call_to_model.__module__}.{subclass._mockable_direct_call_to_model.__qualname__}"
        return full_function_path

    @staticmethod
    def mock_ai_model_direct_call_with_2_errors_then_success(
        mocker: Mock, subclass: type[AiModel]
    ) -> Mock:
        mock_function = mocker.patch(
            AiModelMockManager.get_direct_call_function_path_as_string(subclass)
        )
        mock_function.side_effect = [
            Exception("Mock Exception 1"),
            Exception("Mock Exception 2"),
            subclass._get_mock_return_for_direct_call_to_model_using_cheap_input(),
        ]
        return mock_function

    @staticmethod
    def mock_ai_model_direct_call_with_long_wait(
        mocker: Mock, subclass: type[AiModel]
    ) -> Mock:
        if not issubclass(subclass, TimeLimitedModel):
            raise ValueError("Model must be TimeLimited")

        timeout_time = subclass().TIMEOUT_TIME
        time_longer_than_timeout: int = timeout_time + 1

        def wait_longer_than_timeout(*args, **kwargs) -> Any:
            asyncio.run(asyncio.sleep(time_longer_than_timeout))
            return (
                subclass._get_mock_return_for_direct_call_to_model_using_cheap_input()
            )

        mock_function = mocker.patch(
            AiModelMockManager.get_direct_call_function_path_as_string(subclass),
            side_effect=wait_longer_than_timeout,
        )
        return mock_function

    @staticmethod
    def mock_ai_model_direct_call_with_predefined_mock_value(
        mocker: Mock, subclass: type[AiModel]
    ) -> Mock:
        input_value = (
            subclass._get_mock_return_for_direct_call_to_model_using_cheap_input()
        )
        mock_function = AiModelMockManager.mock_ai_model_direct_call_with_value(
            mocker, subclass, input_value
        )
        return mock_function

    @staticmethod
    def mock_ai_model_direct_call_with_value(
        mocker: Mock, subclass: type[AiModel], value: Any
    ) -> Mock:
        direct_call_function = subclass._mockable_direct_call_to_model
        if isinstance(direct_call_function, Mock):
            return direct_call_function

        full_function_path = AiModelMockManager.get_direct_call_function_path_as_string(
            subclass
        )
        mock_function = mocker.patch(full_function_path)
        mock_function.return_value = value
        return mock_function

    @staticmethod
    def mock_ai_model_direct_call_with_only_errors(
        mocker: Mock, subclass: type[AiModel]
    ) -> Mock:
        def throw_error(*args, **kwargs) -> None:
            raise RuntimeError("Mock Exception")

        mock_function = mocker.patch(
            AiModelMockManager.get_direct_call_function_path_as_string(subclass),
            side_effect=throw_error,
        )
        return mock_function

    @staticmethod
    def mock_input_to_tokens_with_value(
        mocker: Mock,
        subclass: type[TokensAreCalculatable],
        mock_value: int = 1,
    ) -> Mock:
        input_to_tokens_function = subclass.input_to_tokens
        if isinstance(input_to_tokens_function, Mock):
            return input_to_tokens_function

        mock_function = mocker.patch(
            f"{subclass.input_to_tokens.__module__}.{subclass.input_to_tokens.__qualname__}"
        )
        mock_function.return_value = mock_value
        return mock_function

    @staticmethod
    def mock_function_that_throws_error_if_test_limit_reached(
        mocker: Mock,
    ) -> Mock:
        mock_function = mocker.patch(
            f"{AiModel._increment_calls_then_error_if_testing_call_limit_reached.__module__}.{AiModel._increment_calls_then_error_if_testing_call_limit_reached.__qualname__}"
        )
        return mock_function

    @staticmethod
    def reinitialize_limiters(
        subclass: type[RequestLimitedModel | TokenLimitedModel],
    ) -> None:
        if issubclass(subclass, RequestLimitedModel):
            subclass._reinitialize_request_rate_limiter()

        if issubclass(subclass, TokenLimitedModel):
            subclass._reinitialize_token_limiter()
