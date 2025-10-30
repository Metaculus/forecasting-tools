import asyncio

from forecasting_tools.ai_models.general_llm import GeneralLlm

llms = [
    GeneralLlm(model="metaculus/gpt-4o"),
    GeneralLlm.thinking_budget_model(
        model="metaculus/claude-3-7-sonnet-latest",
    ),
    GeneralLlm(model="metaculus/o1"),
    GeneralLlm(
        model="metaculus/o4-mini",
    ),
    GeneralLlm(
        model="metaculus/gpt-4.1",
    ),
    GeneralLlm(
        model="metaculus/gpt-4o-mini",
    ),
    GeneralLlm(
        model="metaculus/o3-mini",
    ),
]

for i, llm in enumerate(llms):
    response = asyncio.run(llm.invoke("What is your name?"))
    print(f"Response {i+1}:")
    print(response)
    print("-" * 100)
