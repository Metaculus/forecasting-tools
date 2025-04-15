import argparse
import asyncio
import logging
import os
from typing import Literal

import dotenv

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.forecast_bots.community.uniform_probability_bot import (
    UniformProbabilityBot,
)
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
dotenv.load_dotenv()


def create_bot(
    llm: GeneralLlm,
    predictions_per_research_report: int = 5,
) -> ForecastBot:
    default_bot = Q2TemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=predictions_per_research_report,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
        llms={
            "default": llm,
            "summarizer": "gpt-4o-mini",
        },
    )
    return default_bot


default_temperature = 0.3
roughly_gpt_4o_cost = 0.05
roughly_gpt_4o_mini_cost = 0.005
roughly_sonnet_3_5_cost = 0.10
all_base_bots = {
    "METAC_O1_HIGH_TOKEN": {
        "estimated_cost_per_question": 1.18,
        "bot": create_bot(
            GeneralLlm(
                model="o1",
                temperature=default_temperature,
                reasoning_effort="high",
            ),
        ),
    },
    "METAC_O1_TOKEN": {
        "estimated_cost_per_question": 0.8,
        "bot": create_bot(
            GeneralLlm(
                model="o1",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_O1_MINI_TOKEN": {
        "estimated_cost_per_question": roughly_gpt_4o_cost,
        "bot": create_bot(
            GeneralLlm(
                model="o1-mini",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_GPT_4_5_PREVIEW_TOKEN": {
        "estimated_cost_per_question": 1.5,
        "bot": create_bot(
            GeneralLlm(
                model="gpt-4.5-preview",
                temperature=default_temperature,
            ),
            predictions_per_research_report=3,
        ),
    },
    "METAC_O3_MINI_HIGH_TOKEN": {
        "estimated_cost_per_question": roughly_gpt_4o_cost,
        "bot": create_bot(
            GeneralLlm(
                model="o3-mini",
                temperature=default_temperature,
                reasoning_effort="high",
            ),
        ),
    },
    "METAC_O3_MINI_TOKEN": {
        "estimated_cost_per_question": roughly_gpt_4o_cost,
        "bot": create_bot(
            GeneralLlm(
                model="o3-mini",
                temperature=default_temperature,
                reasoning_effort="medium",
            ),
        ),
    },
    "METAC_GPT_4O_TOKEN": {
        "estimated_cost_per_question": roughly_gpt_4o_cost,
        "bot": create_bot(
            GeneralLlm(
                model="gpt-4o",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_GPT_4O_MINI_TOKEN": {
        "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
        "bot": create_bot(
            GeneralLlm(
                model="gpt-4o-mini",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_GPT_3_5_TURBO_TOKEN": {
        "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
        "bot": create_bot(
            GeneralLlm(
                model="gpt-3.5-turbo",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_CLAUDE_3_7_SONNET_LATEST_THINKING_TOKEN": {
        "estimated_cost_per_question": 0.37,
        "bot": create_bot(
            GeneralLlm(
                model="anthropic/claude-3-7-sonnet-latest",  # NOSONAR
                temperature=1,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 32000,
                },
                max_tokens=40000,
                timeout=160,
            ),
        ),
    },
    "METAC_CLAUDE_3_7_SONNET_LATEST_TOKEN": {
        "estimated_cost_per_question": roughly_sonnet_3_5_cost,
        "bot": create_bot(
            GeneralLlm(
                model="anthropic/claude-3-7-sonnet-latest",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_CLAUDE_3_5_SONNET_LATEST_TOKEN": {
        "estimated_cost_per_question": roughly_sonnet_3_5_cost,
        "bot": create_bot(
            GeneralLlm(
                model="anthropic/claude-3-5-sonnet-latest",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_CLAUDE_3_5_SONNET_20240620_TOKEN": {
        "estimated_cost_per_question": roughly_sonnet_3_5_cost,
        "bot": create_bot(
            GeneralLlm(
                model="anthropic/claude-3-5-sonnet-20240620",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_GEMINI_2_5_PRO_PREVIEW_TOKEN": {
        "estimated_cost_per_question": 0.30,
        "bot": create_bot(
            GeneralLlm(
                model="gemini/gemini-2.5-pro-preview-03-25",
                temperature=default_temperature,
                timeout=90,
            ),
        ),
    },
    "METAC_GEMINI_2_0_FLASH_TOKEN": {
        "estimated_cost_per_question": 0.05,
        "bot": create_bot(
            GeneralLlm(
                model="gemini/gemini-2.0-flash-001",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_LLAMA_4_MAVERICK_17B_TOKEN": {
        "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
        "bot": create_bot(
            GeneralLlm(
                model="openrouter/meta-llama/llama-4-maverick",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_LLAMA_3_3_NEMOTRON_49B_TOKEN": {
        "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
        "bot": create_bot(
            GeneralLlm(
                model="openrouter/nvidia/llama-3.3-nemotron-super-49b-v1:free",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_QWEN_2_5_MAX_TOKEN": {
        "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
        "bot": create_bot(
            GeneralLlm(
                model="openrouter/qwen/qwen-max",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_DEEPSEEK_R1_TOKEN": {
        "estimated_cost_per_question": roughly_gpt_4o_cost,
        "bot": create_bot(
            GeneralLlm(
                model="openrouter/deepseek/deepseek-r1",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_DEEPSEEK_V3_TOKEN": {
        "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
        "bot": create_bot(
            GeneralLlm(
                model="openrouter/deepseek/deepseek-chat",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_GROK_3_LATEST_TOKEN": {
        "estimated_cost_per_question": 0.13,
        "bot": create_bot(
            GeneralLlm(
                model="xai/grok-3-latest",
                temperature=default_temperature,
            ),
        ),
    },
    "METAC_GROK_3_MINI_LATEST_HIGH_TOKEN": {
        "estimated_cost_per_question": 0.10,
        "bot": create_bot(
            GeneralLlm(
                model="xai/grok-3-mini-latest",
                temperature=default_temperature,
                reasoning_effort="high",
            ),
        ),
    },
    "METAC_UNIFORM_PROBABILITY_BOT_TOKEN": {
        "estimated_cost_per_question": 0.00,
        "bot": UniformProbabilityBot(
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=True,
            skip_previously_forecasted_questions=True,
        ),
    },
}


async def run_bot(
    mode: str, return_bot_dont_run: bool = False, check_env_vars: bool = True
) -> ForecastBot | list[ForecastReport]:

    if "metaculus-cup" in mode:
        chosen_tournament = MetaculusApi.CURRENT_METACULUS_CUP_ID
        skip_previously_forecasted_questions = False
        token = mode.split("+")[0]
    else:
        chosen_tournament = MetaculusApi.CURRENT_AI_COMPETITION_ID
        skip_previously_forecasted_questions = True
        token = mode

    bot = all_base_bots[token]["bot"]
    assert isinstance(bot, ForecastBot)
    bot.skip_previously_forecasted_questions = (
        skip_previously_forecasted_questions
    )

    if check_env_vars:
        _make_sure_search_keys_dont_conflict("asknews-mode")

    if return_bot_dont_run:
        return bot
    else:
        logger.info(f"LLMs for bot are: {bot.make_llm_dict()}")
        reports = await bot.forecast_on_tournament(
            chosen_tournament, return_exceptions=True
        )
        bot.log_report_summary(reports)
        return reports


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
        assert os.getenv(
            "ASKNEWS_SECRET"
        ), "Asknews secret key is not set for asknews-mode"
    elif mode == "exa-mode":
        assert not os.getenv(
            "PERPLEXITY_API_KEY"
        ), "Perplexity API key is set, but it should not be set for exa-mode"
        assert not os.getenv(
            "ASKNEWS_SECRET"
        ), "Asknews secret key is set, but it should not be set for exa-mode"
        assert os.getenv("EXA_API_KEY"), "Exa API key is not set for exa-mode"
    elif mode == "perplexity-mode":
        assert not os.getenv(
            "EXA_API_KEY"
        ), "Exa API key is set, but it should not be set for perplexity-mode"
        assert not os.getenv(
            "ASKNEWS_SECRET"
        ), "Asknews secret key is set, but it should not be set for perplexity-mode"
        assert os.getenv(
            "PERPLEXITY_API_KEY"
        ), "Perplexity API key is not set for perplexity-mode"


async def _save_reports_to_database(reports: list[ForecastReport]) -> None:
    for report in reports:
        await asyncio.sleep(5)
        try:
            ForecastDatabaseManager.add_forecast_report_to_database(
                report, ForecastRunType.REGULAR_FORECAST
            )
        except Exception as e:
            logger.error(f"Error adding forecast report to database: {e}")


async def get_all_bots(check_env_vars: bool = True) -> list[ForecastBot]:
    bots = []
    keys = list(all_base_bots.keys())
    for key in keys:
        bots.append(
            await run_bot(
                key, return_bot_dont_run=True, check_env_vars=check_env_vars
            )
        )
    return bots


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
