import asyncio
import logging
import time

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)

logger = logging.getLogger(__name__)


async def stress_test_proxy() -> None:
    prompt = "Write me a long 10 paragraph story"
    with MonetaryCostManager() as cost_manager:
        model = GeneralLlm(
            model="metaculus/gpt-4o-mini",
            timeout=160,
            allowed_tries=2,
            temperature=None,
        )
        num_calls = 500
        coroutines = [model.invoke(prompt) for _ in range(num_calls)]

        start_time = time.time()
        responses = await asyncio.gather(*coroutines, return_exceptions=True)
        end_time = time.time()
        for i, response in enumerate(responses):
            print(f"------------- {i} -------------------")
            if isinstance(response, BaseException):
                logger.error(f"Error {i}: {response}")
            else:
                logger.info(f"Response {i}: {response[0:20]}")

        print("--------------------------------")
        logger.info(f"Time taken: {(end_time - start_time)/60} minutes")
        number_of_errors = len(
            [response for response in responses if isinstance(response, BaseException)]
        )
        logger.info(f"Number of errors: {number_of_errors}")
        number_of_responses = len(
            [
                response
                for response in responses
                if not isinstance(response, BaseException)
            ]
        )
        logger.info(f"Number of responses: {number_of_responses}")
        logger.info(f"Cost: {cost_manager.current_usage}")

        # Count errors by type
        error_types = {}
        for response in responses:
            if isinstance(response, BaseException):
                error_type = type(response).__name__
                error_types[error_type] = error_types.get(error_type, 0) + 1

        # Log each error type count
        for error_type, count in error_types.items():
            logger.info(f"Number of {error_type} errors: {count}")

        print("Finished")
        if number_of_errors > 0:
            raise RuntimeError("There were errors")

        if cost_manager.current_usage > 3:
            raise RuntimeError("Cost is too high")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(stress_test_proxy())
