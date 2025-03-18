import asyncio

from forecasting_tools.ai_models.general_llm import GeneralLlm

for i in range(10):
    llm1 = GeneralLlm(
        model="metaculus/gpt-4o", api_key="this-is-to-keep-it-from-erroring"
    )
    llm2 = GeneralLlm(
        model="metaculus/claude-3-5-sonnet-20241022",
        api_key="this-is-to-keep-it-from-erroring",
    )
    response1 = asyncio.run(llm1.invoke("What is your name?"))
    response2 = asyncio.run(llm2.invoke("What is your name?"))
    print("-" * 100)
    print(f"Response {i+1}:")
    print(response1)
    print(response2)
