# Overview
This module manages functionality of a bunch of AI models.
This readme was last updated on Aug 15, 2024

# Example Usage
Below is some example code illustrating the main functionality of this module
```python
from forecasting_tools.ai_models.gpt4o import Gpt4o
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import MonetaryCostManager

import asyncio
import logging
from pydantic import BaseModel
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    class President(BaseModel):
        name: str
        age_at_death: int
        biggest_accomplishment: str

    max_cost = 2
    with MonetaryCostManager(max_cost) as cost_manager:
        model = Gpt4o(temperature=0, system_prompt="You are a helpful assistant", allowed_tries=3)
        pydantic_model_explanation = model.get_schema_format_instructions_for_pydantic_type(President)
        prompt = clean_indents(f"""
            You are a historian helping with a presidential research project. Please answer the following question:
            Who is Abraham Lincoln?
            Please provide the information in the following format:
            {pydantic_model_explanation}
            """)
        type_required_coroutine = model.invoke_and_return_verified_type(prompt, President, allowed_invoke_tries_for_failed_output=2)
        regular_coroutine = model.invoke(prompt)

        verified_president = asyncio.run(type_required_coroutine)
        regular_output = asyncio.run(regular_coroutine)

        logger.debug(f"President is pydantic: {isinstance(verified_president, President)}")
        logger.info(f"President: {verified_president}")
        logger.info(f"Regular output: {regular_output}")
        logger.info(f"Cost: {cost_manager.current_usage}")
```

The output would be
```
    President is pydantic: True

    President: name='Abraham Lincoln' age_at_death=56 biggest_accomplishment='Leading the United States through the Civil War and abolishing slavery with the Emancipation Proclamation.'

    Regular output: ```json
    {
    "name": "Abraham Lincoln",
    "age_at_death": 56,
    "biggest_accomplishment": "Leading the United States through the Civil War and abolishing slavery with the Emancipation Proclamation and the passage of the 13th Amendment."
    }
    ```

    Cost: 0.004274999999999999
```

# Features
## Model Endpoints
There is access to various models from
- OpenAI
- Anthropic
- Perplexity
- Exa

Its easy to set up a new model with rate limits, etc by creating a new model file in the top level directory. Set the model name, which model archetype it inherits from, and its request/token rate limit, etc.

## Cost Tracking
The `MonetatryCostManager` allows for tracking costs between all model providers. Any call made within the context of the manager will be logged in the manager. CostManagers are async safe and can nest inside of each other. CostManagers always can take a "max_cost" value and all model calls will error if that max cost is exceeded. Please note that cost is only tracked after a call finishes, so if you concurrently batch 1000 calls, they will all go through even if the cost is exceeded during the middle of the execution of the batch.

Cost management (for the most part) is tracked through langchain, so make sure to keep the langchain packages updated so model pricing stays current. As of writing, Perplexity costs have to be updated manually.

## Type Validated Outputs
The `invoke_and_return_verified_type()` function will guarantee returning any primitive type (e.g. str, int, list[str], list[dict], list[tuple[int,str]]) or pydantic model type you specify. It will retry the prompt a number of times until it is able to parse the result in the desired format.

## Rate Limiting
Rate limiting values is setup inside the models as class variables. The model will run a lot of calls in a quick burst, and when the rate limit is about to be reached (for tokens or requests) it will slow down to maintain the usage per period defined in the class variables. All rates have to be updated manually. There are plans to make a configuration file for this.

## Mini Code Interpreter
You can use `invoke_and_unsafely_run_and_return_generated_code()` to generate and run code locally.

## Prompt indent cleaning
Prompts are notorious for breaking indentation levels. Usually you have to either not indent your prompt in the code, or you send a bunch of indents to the model. With `clean_indents` you can have the best of both worlds.

## Timeouts
Each TimeoutLimitedModel allows you to set a timeout for that model. If any requests takes longer to finish then the timeout then it stops early

## Normal Retries
All RetryableModels have a `allowed_tries` argument when initializing the model. The model will try that many times before giving up

## System Prompts
Most models also allow for a system prompt to be set when initializing the model object.


# Testing
There are a few tests in this module that make actual calls to models (e.g. to check that predicted token usage meets actual token usage, etc). There should only be around ~8 calls per model per full test run on a simple prompt like "Hi".

Whenever you add a new model, add the model to the "models to test" list in `tests/test_ai_models/models_to_test.py`
