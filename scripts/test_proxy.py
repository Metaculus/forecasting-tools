import asyncio

from forecasting_tools.ai_models.general_llm import GeneralLlm

llm = GeneralLlm(
    model="metaculus/gpt-4o",
)

for i in range(10):
    response = asyncio.run(llm.invoke("What is your name?"))
    print("-" * 100)
    print(f"Response {i+1}:")
    print(response)
