import argparse
import asyncio
import logging
import os
from typing import Literal

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.official_bots.q2_template_bot import (
    Q2TemplateBot2025,
)
from forecasting_tools.forecast_helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi

logger = logging.getLogger(__name__)


async def run_bot(mode: str) -> None:
    ai_tournament = MetaculusApi.CURRENT_AI_COMPETITION_ID

    bot = create_bot_from_mode(mode)
    reports = await bot.forecast_on_tournament(
        ai_tournament, return_exceptions=True
    )
    bot.log_report_summary(reports)


def create_bot_from_mode(mode: str) -> ForecastBot:
    if os.getenv("METACULUS_TOKEN") is not None:
        raise ValueError(
            "Metaculus token will be overridden by the specific mode chosen"
        )

    default_temperature = 0.3
    default_bot = Q2TemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    if mode == "gpt-4o_asknews":
        _set_metaculus_token("METAC_GPT_4O_ASKNEWS_TOKEN")
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="gpt-4o",
                temperature=default_temperature,
            )
        )
    elif mode == "gpt-4o_exa":
        _set_metaculus_token("METAC_GPT_4O_EXA_TOKEN")
        _make_sure_search_keys_dont_conflict("exa-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="gpt-4o",
                temperature=default_temperature,
            )
        )
    elif mode == "gpt-4o_perplexity":
        _set_metaculus_token("METAC_GPT_4O_PERPLEXITY_TOKEN")
        _make_sure_search_keys_dont_conflict("perplexity-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="gpt-4o",
                temperature=default_temperature,
            )
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return bot


def _set_metaculus_token(variable_name: str) -> None:
    token_value = os.getenv(variable_name)
    if token_value is None:
        raise ValueError(f"Token {variable_name} is not set")
    os.environ["METACULUS_TOKEN"] = token_value


def _make_sure_search_keys_dont_conflict(
    mode: Literal["asknews-mode", "exa-mode", "perplexity-mode"],
) -> None:
    if mode == "asknews-mode":
        assert (
            os.getenv("PERPLEXITY_API_KEY") is None
        ), "Perplexity API key is set, but it should not be set for asknews-mode"
        assert (
            os.getenv("EXA_API_KEY") is None
        ), "Exa API key is set, but it should not be set for asknews-mode"
    elif mode == "exa-mode":
        assert (
            os.getenv("PERPLEXITY_API_KEY") is None
        ), "Perplexity API key is set, but it should not be set for exa-mode"
        assert (
            os.getenv("ASKNEWS_SECRET_KEY") is None
        ), "Asknews secret key is set, but it should not be set for exa-mode"
    elif mode == "perplexity-mode":
        assert (
            os.getenv("EXA_API_KEY") is None
        ), "Exa API key is set, but it should not be set for perplexity-mode"
        assert (
            os.getenv("ASKNEWS_SECRET_KEY") is None
        ), "Asknews secret key is set, but it should not be set for perplexity-mode"


async def _save_reports_to_database(reports: list[ForecastReport]) -> None:
    for report in reports:
        await asyncio.sleep(5)
        try:
            ForecastDatabaseManager.add_forecast_report_to_database(
                report, ForecastRunType.REGULAR_FORECAST
            )
        except Exception as e:
            logger.error(f"Error adding forecast report to database: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a forecasting bot with the specified mode"
    )
    parser.add_argument(
        "mode",
        type=str,
        help="Bot mode to run",
    )

    args = parser.parse_args()
    asyncio.run(run_bot(args.mode))
