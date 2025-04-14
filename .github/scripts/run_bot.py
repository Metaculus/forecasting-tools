import argparse
import asyncio
import logging
import os
from typing import Literal

import dotenv

dotenv.load_dotenv()

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.forecast_bots.community.uniform_probability_bot import (
    UniformProbabilityBot,
)
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

    if "metaculus-cup" in mode:
        chosen_tournament = MetaculusApi.CURRENT_METACULUS_CUP_ID
        skip_previously_forecasted_questions = False
        token = mode.split("+")[0]
    else:
        chosen_tournament = MetaculusApi.CURRENT_AI_COMPETITION_ID
        skip_previously_forecasted_questions = True
        token = mode

    default_temperature = 0.3
    default_bot = Q2TemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=skip_previously_forecasted_questions,
    )

    if token == "METAC_GPT_4O_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="gpt-4o",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_O1_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="o1",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_O1_HIGH_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="o1",
                temperature=default_temperature,
                reasoning_effort="high",
            )
        )
    elif token == "METAC_O3_MINI_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="o3-mini",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_O3_MINI_HIGH_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="o3-mini",
                temperature=default_temperature,
                reasoning_effort="high",
            )
        )
    elif token == "METAC_O1_MINI_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="o1-mini",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_GPT_4_5_PREVIEW_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="gpt-4.5-preview",
                temperature=default_temperature,
                timeout=90,
            )
        )
        bot.research_reports_per_question = 1
        bot.predictions_per_research_report = 3
    elif token == "METAC_GPT_3_5_TURBO_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="gpt-3.5-turbo",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_GPT_4O_MINI_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="gpt-4o-mini",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_CLAUDE_3_7_SONNET_LATEST_THINKING_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="claude-3-7-sonnet-latest",
                temperature=1,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 32000,
                },
                max_tokens=40000,
                timeout=160,
            )
        )
    elif token == "METAC_CLAUDE_3_7_SONNET_LATEST_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="claude-3-7-sonnet-latest",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_CLAUDE_3_5_SONNET_LATEST_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="claude-3-5-sonnet-latest",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_CLAUDE_3_5_SONNET_20240620_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="claude-3.5-Sonnet-20240620",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_GEMINI_2_5_PRO_PREVIEW_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="gemini/gemini-2.5-pro-preview-03-25",
                temperature=default_temperature,
                timeout=90,
            )
        )
    elif token == "METAC_GEMINI_2_0_FLASH_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="gemini/gemini-2.0-flash-001",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_LLAMA_4_MAVERICK_17B_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="openrouter/meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_LLAMA_3_3_NEMOTRON_49B_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="openrouter/meta-llama/llama-3.3-nemotron-super-49b-v1",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_QWEN_2_5_MAX_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="openrouter/qwen/qwen2.5-max",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_DEEPSEEK_R1_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="openrouter/deepseek/deepseek-r1",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_DEEPSEEK_V3_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="openrouter/deepseek/deepseek-chat",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_GROK_3_LATEST_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="openrouter/xai/grok-3-latest",
                temperature=default_temperature,
            )
        )
    elif token == "METAC_GROK_3_MINI_LATEST_HIGH_TOKEN":
        _make_sure_search_keys_dont_conflict("asknews-mode")
        bot = default_bot
        bot.set_llm(
            GeneralLlm(
                model="openrouter/xai/grok-3-mini-latest",
                temperature=default_temperature,
                reasoning_effort="high",
            )
        )
    elif token == "METAC_UNIFORM_PROBABILITY_BOT_TOKEN":
        bot = UniformProbabilityBot(
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=True,
            skip_previously_forecasted_questions=True,
        )
    else:
        raise ValueError(f"Invalid mode: {token}")

    reports = await bot.forecast_on_tournament(
        chosen_tournament, return_exceptions=True
    )
    bot.log_report_summary(reports)


def _set_metaculus_token(variable_name: str) -> None:
    token_value = os.getenv(variable_name)
    if token_value is None:
        raise ValueError(f"Token {variable_name} is not set")
    os.environ["METACULUS_TOKEN"] = token_value


def _make_sure_search_keys_dont_conflict(
    mode: Literal["asknews-mode", "exa-mode", "perplexity-mode"],
) -> None:
    if mode == "asknews-mode":
        assert not os.getenv(
            "PERPLEXITY_API_KEY"
        ), "Perplexity API key is set, but it should not be set for asknews-mode"
        assert not os.getenv(
            "EXA_API_KEY"
        ), "Exa API key is set, but it should not be set for asknews-mode"
    elif mode == "exa-mode":
        assert not os.getenv(
            "PERPLEXITY_API_KEY"
        ), "Perplexity API key is set, but it should not be set for exa-mode"
        assert not os.getenv(
            "ASKNEWS_SECRET_KEY"
        ), "Asknews secret key is set, but it should not be set for exa-mode"
    elif mode == "perplexity-mode":
        assert not os.getenv(
            "EXA_API_KEY"
        ), "Exa API key is set, but it should not be set for perplexity-mode"
        assert not os.getenv(
            "ASKNEWS_SECRET_KEY"
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run a forecasting bot with the specified mode"
    )
    parser.add_argument(
        "mode",
        type=str,
        help="Bot mode to run",
    )

    args = parser.parse_args()
    token = args.mode

    asyncio.run(run_bot(token))
