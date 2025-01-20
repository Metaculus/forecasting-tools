import asyncio

from forecasting_tools import MetaculusApi, TemplateBot


async def main():
    # Initialize the bot
    bot = TemplateBot(
        research_reports_per_question=2,
        predictions_per_research_report=3,
        publish_reports_to_metaculus=True,  # Set to True to post to Metaculus
        folder_to_save_reports_to="logs/forecasts/",
        skip_previously_forecasted_questions=True,
    )

    # Run forecasts on Q4 2024 AI Tournament
    TOURNAMENT_ID = MetaculusApi.AI_COMPETITION_ID_Q4
    reports = await bot.forecast_on_tournament(TOURNAMENT_ID)

    # Print results
    for report in reports:
        print(f"\nQuestion: {report.question.question_text}")
        print(f"Prediction: {report.prediction}")


if __name__ == "__main__":
    asyncio.run(main())
