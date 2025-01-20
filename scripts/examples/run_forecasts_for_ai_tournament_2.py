from __future__ import annotations

import asyncio
import logging

from forecasting_tools import MetaculusApi
from forecasting_tools.forecasting.forecast_bots.experiments.gemini_bots import (
    GeminiExpBot,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)


async def main():
    exp_bot = GeminiExpBot(
        research_reports_per_question=2,
        predictions_per_research_report=3,
        publish_reports_to_metaculus=True,  # Set to True to publish results
        folder_to_save_reports_to="logs/forecasts/exp/",
    )

    tournament_id = (
        MetaculusApi.CURRENT_QUARTERLY_CUP_ID
    )  # Use the correct tournament ID
    reports = await exp_bot.forecast_on_tournament(tournament_id)

    # Print results
    for report in reports:
        print(f"\nQuestion: {report.question.question_text}")
        print(f"Prediction: {report.prediction}")
        print("\nReasoning:")
        print(report.explanation)


if __name__ == "__main__":
    asyncio.run(main())
if __name__ == "__main__":
    asyncio.run(main())
