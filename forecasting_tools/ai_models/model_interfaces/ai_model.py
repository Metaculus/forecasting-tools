import os
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Any


class AiModel(ABC):

    _num_calls_to_dependent_model: ContextVar[int] = ContextVar(
        "_num_calls_to_dependent_model", default=0
    )

    @abstractmethod
    async def invoke(self, input: Any) -> Any:
        """
        This function is the main function that should be called to use the model.
        """
        pass

    @abstractmethod
    async def _mockable_direct_call_to_model(self, input: Any) -> Any:
        """
        This function calls the model directly.
        The function will probably be mocked in tests with a static return value.
        Do not put any logic that is tested into this function if the mocked return value would break the test.
        """

    @staticmethod
    @abstractmethod
    def _get_mock_return_for_direct_call_to_model_using_cheap_input() -> Any:
        """
        This is a value that tests can use to mock the
        return value of the function for the direct call to the model.
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_cheap_input_for_invoke() -> Any:
        """
        This is an input to the invoke function that is cheap to run
        for the tests that run real requests to the model
        """
        pass

    def _everything_special_to_call_before_direct_call(self) -> Any:
        """
        This function is called before the direct call to the model.
        It is a place to put any logic that should be run before the direct call.
        """
        self._increment_calls_then_error_if_testing_call_limit_reached()

    def _increment_calls_then_error_if_testing_call_limit_reached(
        self, max_calls: int = 30
    ) -> None:
        current_calls = self._num_calls_to_dependent_model.get()
        self._num_calls_to_dependent_model.set(current_calls + 1)
        if (
            "PYTEST_CURRENT_TEST" in os.environ
            and self._num_calls_to_dependent_model.get() > max_calls
        ):
            raise RuntimeError(
                f"model called more than {max_calls} times during testing"
            )
