import asyncio
import logging

import dotenv

from forecasting_tools.ai_models.gemini2exp import Gemini2Exp
from forecasting_tools.ai_models.gemini2flash import Gemini2Flash
from forecasting_tools.ai_models.gemini2flashthinking import (
    Gemini2FlashThinking,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def test_model(name: str, model):
    print(f"\nTesting {name}...")
    print(f"Model name: {model.MODEL_NAME}")
    try:
        response = await model.invoke("Hello, how are you?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {str(e)}")


async def main():
    models = {
        "Gemini Flash Thinking": Gemini2FlashThinking(temperature=0.7),
        "Gemini Flash": Gemini2Flash(temperature=0.7),
        "Gemini Exp": Gemini2Exp(temperature=0.7),
    }

    for name, model in models.items():
        await test_model(name, model)


if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(main())
