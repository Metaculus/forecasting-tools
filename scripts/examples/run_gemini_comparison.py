import asyncio
import logging

from forecasting_tools.forecasting.forecast_bots.experiments.gemini_bots import (
    GeminiExpBot,
    GeminiFlash2Bot,
    GeminiFlashThinkingExpBot,
)
from forecasting_tools import MetaculusApi

# Configure logging - only show INFO and above
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",  # Simplified format
)


async def main():
    flash_thinking_bot = GeminiFlashThinkingExpBot(
        research_reports_per_question=2,
        predictions_per_research_report=3,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to="logs/forecasts/flash_thinking/",
    )

    flash2_bot = GeminiFlash2Bot(
        research_reports_per_question=3,
        predictions_per_research_report=5,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to="logs/forecasts/flash2/",
    )

    exp_bot = GeminiExpBot(
        research_reports_per_question=2,
        predictions_per_research_report=3,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to="logs/forecasts/exp/",
    )

    question = MetaculusApi.get_question_by_url(
        "https://www.metaculus.com/questions/26245/5y-after-agi-biological-human-population/"
    )

    print("\nRunning Flash Thinking Bot...")
    flash_thinking_reports = await flash_thinking_bot.forecast_questions(
        [question]
    )

    print("\nRunning Flash 2.0 Bot...")
    flash2_reports = await flash2_bot.forecast_questions([question])

    print("\nRunning Exp Bot...")
    exp_reports = await exp_bot.forecast_questions([question])

    # Print results in a cleaner format
    for title, reports in [
        ("Flash Thinking Bot", flash_thinking_reports),
        ("Flash 2.0 Bot", flash2_reports),
        ("Exp Bot", exp_reports),
    ]:
        print(f"\n=== {title} Results ===")
        for report in reports:
            print(f"\nQuestion: {report.question.question_text}")
            print(f"Prediction: {report.prediction}")
            print("\nReasoning:")
            print(report.explanation)


if __name__ == "__main__":
    asyncio.run(main())
