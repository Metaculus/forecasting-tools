import asyncio
import logging
import os
import sys
from datetime import datetime

import dotenv

# Dynamically determine the absolute path to the top-level directory
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(top_level_dir)

from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastBot,
)
from forecasting_tools.forecasting.questions_and_reports.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    NumericDistribution,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    MultipleChoiceQuestion,
    NumericQuestion,
)
from johnathan_bot import JohnathanBot
from forecasting_tools import MetaculusApi
from forecasting_tools.forecasting.helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.util.custom_logger import CustomLogger

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

EXAMPLE_QUESTIONS = [  # (question_id, post_id)
    (31609, 31609),  # Human Extinction - Binary - https://www.metaculus.com/questions/578/human-extinction-by-2100/
]


async def run_johnathan_bot(
    tournament_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
    use_example_questions: bool,
    use_flash_thinking_for_research: bool,
) -> None:
    """
    Runs the JohnathanBot on a specified Metaculus tournament.
    """
    CustomLogger.setup_logging()

    johnathan_bot = JohnathanBot(
        research_reports_per_question=2,
        predictions_per_research_report= num_runs_per_question,
        publish_reports_to_metaculus=submit_prediction,
        folder_to_save_reports_to="logs/forecasts/johnathan_bot/",
        skip_previously_forecasted_questions=skip_previously_forecasted_questions,
        use_flash_thinking_for_research=use_flash_thinking_for_research,
    )

    if use_example_questions:
        question_post_id_pairs = EXAMPLE_QUESTIONS
        questions = [
            MetaculusApi.get_question_by_post_id(post_id)
            for _, post_id in question_post_id_pairs
        ]
    else:
        # Fetch questions from the tournament
        question_post_id_pairs = MetaculusApi.get_all_open_questions_from_tournament(tournament_id)
        questions = [
            MetaculusApi.get_question_by_post_id(post_id)
            for _, post_id in question_post_id_pairs
        ]


    if skip_previously_forecasted_questions:
        questions_to_forecast = []
        for question in questions:
            if not question.already_forecasted:
                questions_to_forecast.append(question)
            else:
                logger.info(f"Skipping question {question.id_of_post} as it has already been forecasted on.")
        questions = questions_to_forecast

    if len(questions) == 0:
        logger.info("No questions to forecast on. Exiting.")
        return

    # Forecast on each question
    reports = await johnathan_bot.forecast_questions(questions)

    # Print results (or save to a file, etc.)
    for report in reports:
        logger.info(f"\nQuestion: {report.question.question_text}")
        if report.prediction is not None:
          logger.info(f"Prediction: {report.prediction}")
        else:
          logger.info("Prediction: Not Available (Forecast Failed)")

        logger.info("\nReasoning:")
        logger.info(report.explanation)


if __name__ == "__main__":

    TOURNAMENT_ID = MetaculusApi.Q1_2025_QUARTERLY_CUP  # Or your desired tournament ID
    SUBMIT_PREDICTION = True
    SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = False
    USE_EXAMPLE_QUESTIONS = True
    USE_FLASH_THINKING = True
    NUM_RUNS_PER_QUESTION = 1

    asyncio.run(
        run_johnathan_bot(
            TOURNAMENT_ID,
            SUBMIT_PREDICTION,
            NUM_RUNS_PER_QUESTION,
            SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
            USE_EXAMPLE_QUESTIONS,
            USE_FLASH_THINKING,
        )
    )