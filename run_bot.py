"""
This is the main file used to run the bots in the Metaulus AI Competition.

It is run by workflows in the .github/workflows directory.
"""

import argparse
import asyncio
import logging
from typing import Any

import dotenv

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.official_bots.q2_template_bot import (
    Q2TemplateBot2025,
)
from forecasting_tools.forecast_bots.other.uniform_probability_bot import (
    UniformProbabilityBot,
)
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


default_for_skipping_questions = True
default_for_publish_to_metaculus = True
default_for_using_summary = False


async def configure_and_run_bot(
    mode: str, return_bot_dont_run: bool = False
) -> ForecastBot | list[ForecastReport | BaseException]:

    if "metaculus-cup" in mode:
        chosen_tournament = MetaculusApi.CURRENT_METACULUS_CUP_ID
        skip_previously_forecasted_questions = False
        token = mode.split("+")[0]
    else:
        chosen_tournament = MetaculusApi.CURRENT_AI_COMPETITION_ID
        skip_previously_forecasted_questions = True
        token = mode

    bot = get_default_bot_dict()[token]["bot"]
    assert isinstance(bot, ForecastBot)
    bot.skip_previously_forecasted_questions = (
        skip_previously_forecasted_questions
    )

    if return_bot_dont_run:
        return bot
    else:
        logger.info(f"LLMs for bot are: {bot.make_llm_dict()}")
        reports = await bot.forecast_on_tournament(
            chosen_tournament, return_exceptions=True
        )
        bot.log_report_summary(reports)
        return reports


async def get_all_bots() -> list[ForecastBot]:
    bots = []
    keys = list(get_default_bot_dict().keys())
    for key in keys:
        bots.append(await configure_and_run_bot(key, return_bot_dont_run=True))
    return bots


def create_bot(
    llm: GeneralLlm,
    researcher: str | GeneralLlm = "asknews/news-summaries",
    predictions_per_research_report: int = 5,
) -> ForecastBot:
    default_bot = Q2TemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=predictions_per_research_report,
        use_research_summary_to_forecast=default_for_using_summary,
        publish_reports_to_metaculus=default_for_publish_to_metaculus,
        skip_previously_forecasted_questions=default_for_skipping_questions,
        llms={
            "default": llm,
            "summarizer": "gpt-4o-mini",
            "researcher": researcher,
        },
    )
    return default_bot


def get_default_bot_dict() -> dict[str, Any]:  # NOSONAR
    default_temperature = 0.3

    # NOTE: Anything that uses the "roughly" cost value (other than the original model the variable matches to)
    # is estimated value and was not measured directly. These estimates were derived from Litellm's pricing functionality.
    roughly_gpt_4o_cost = 0.05
    roughly_gpt_4o_mini_cost = 0.005
    roughly_sonnet_3_5_cost = 0.10
    roughly_gemini_2_5_pro_preview_cost = 0.30  # TODO: Double check this
    roughly_deepseek_r1_cost = 0.039
    guess_at_deepseek_plus_search = roughly_deepseek_r1_cost + 0.015

    gemini_2_5_pro_preview = "openrouter/google/gemini-2.5-pro-preview"  # "gemini/gemini-2.5-pro-preview-03-25"
    gemini_default_timeout = 120
    default_perplexity_settings = {
        "web_search_options": {"search_context_size": "high"},
        "reasoning_effort": "high",
    }
    gemini_grounding_llm = GeneralLlm(
        model=gemini_2_5_pro_preview,
        generationConfig={
            "thinkingConfig": {
                "thinkingBudget": 0,
            },
            "responseMimeType": "text/plain",
        },
        tools=[
            {"googleSearch": {}},
        ],
    )
    default_deepseek_research_bot_llm = GeneralLlm(
        model="openrouter/deepseek/deepseek-r1",
        temperature=default_temperature,
    )

    mode_base_bot_mapping = {
        "METAC_GEMINI_2_5_PRO_GEMINI_2_5_PRO_GROUNDING": {
            "estimated_cost_per_question": 0.16,
            "bot": create_bot(
                GeneralLlm(
                    model=gemini_2_5_pro_preview,
                    temperature=default_temperature,
                    timeout=gemini_default_timeout,
                ),
                researcher=gemini_grounding_llm,
            ),
        },
        "METAC_GEMINI_2_5_PRO_SONAR_REASONING_PRO": {
            "estimated_cost_per_question": roughly_gemini_2_5_pro_preview_cost,
            "bot": create_bot(
                GeneralLlm(
                    model=gemini_2_5_pro_preview,
                    temperature=default_temperature,
                    timeout=gemini_default_timeout,
                ),
                researcher=GeneralLlm(
                    model="perplexity/sonar-reasoning-pro",
                    **default_perplexity_settings,
                ),
            ),
        },
        "METAC_GEMINI_2_5_EXA_PRO": {
            "estimated_cost_per_question": roughly_gemini_2_5_pro_preview_cost,
            "bot": create_bot(
                GeneralLlm(
                    model=gemini_2_5_pro_preview,
                    temperature=default_temperature,
                    timeout=gemini_default_timeout,
                ),
                researcher=GeneralLlm(model="exa/exa-pro"),
            ),
        },
        "METAC_DEEPSEEK_R1_SONAR_PRO": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_deepseek_research_bot_llm,
                researcher=GeneralLlm(
                    model="perplexity/sonar-pro",
                    **default_perplexity_settings,
                ),
            ),
        },
        "METAC_DEEPSEEK_R1_SONAR": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_deepseek_research_bot_llm,
                researcher=GeneralLlm(
                    model="perplexity/sonar",
                    **default_perplexity_settings,
                ),
            ),
        },
        "METAC_DEEPSEEK_R1_SONAR_DEEP_RESEARCH": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_deepseek_research_bot_llm,
                researcher=GeneralLlm(
                    model="perplexity/sonar-deep-research",
                    **default_perplexity_settings,
                ),
            ),
        },
        "METAC_DEEPSEEK_R1_SONAR_REASONING_PRO": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_deepseek_research_bot_llm,
                researcher=GeneralLlm(
                    model="perplexity/sonar-reasoning-pro",
                    **default_perplexity_settings,
                ),
            ),
        },
        "METAC_DEEPSEEK_R1_SONAR_REASONING": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_deepseek_research_bot_llm,
                researcher=GeneralLlm(
                    model="perplexity/sonar-reasoning",
                    **default_perplexity_settings,
                ),
            ),
        },
        "METAC_ONLY_SONAR_REASONING_PRO": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                GeneralLlm(
                    model="perplexity/sonar-reasoning-pro",
                    **default_perplexity_settings,
                ),
                researcher="None",
            ),
        },
        "METAC_DEEPSEEK_R1_GPT_4O_SEARCH_PREVIEW": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_deepseek_research_bot_llm,
                researcher=GeneralLlm(
                    model="openai/gpt-4o-search-preview", temperature=None
                ),
            ),
        },
        "METAC_DEEPSEEK_R1_GEMINI_2_5_PRO_GROUNDING": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_deepseek_research_bot_llm,
                researcher=gemini_grounding_llm,
            ),
        },
        "METAC_DEEPSEEK_R1_EXA_SMART_SEARCHER": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_deepseek_research_bot_llm,
                researcher="smart-searcher/openrouter/deepseek/deepseek-r1",
            ),
        },
        "METAC_DEEPSEEK_R1_ASK_EXA_PRO": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_deepseek_research_bot_llm,
                researcher=GeneralLlm(model="exa/exa-pro"),
            ),
        },
        "METAC_DEEPSEEK_R1_DEEPNEWS": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_deepseek_research_bot_llm,
                researcher="asknews/deep-research/high-depth",
            ),
        },
        "METAC_O3_HIGH_TOKEN": {
            "estimated_cost_per_question": 0.7,
            "bot": create_bot(
                GeneralLlm(
                    model="o3",
                    temperature=1,
                    reasoning_effort="high",
                    timeout=300,
                ),
            ),
        },
        "METAC_O3_TOKEN": {
            "estimated_cost_per_question": 0.5,
            "bot": create_bot(
                GeneralLlm(
                    model="o3",
                    temperature=1,
                    reasoning_effort="medium",
                ),
            ),
        },
        "METAC_O4_MINI_HIGH_TOKEN": {
            "estimated_cost_per_question": 0.07,
            "bot": create_bot(
                GeneralLlm(
                    model="o4-mini",
                    temperature=1,
                    reasoning_effort="high",
                ),
            ),
        },
        "METAC_O4_MINI_TOKEN": {
            "estimated_cost_per_question": 0.043,
            "bot": create_bot(
                GeneralLlm(
                    model="o4-mini",
                    temperature=1,
                    reasoning_effort="medium",
                ),
            ),
        },
        "METAC_4_1_TOKEN": {
            "estimated_cost_per_question": 0.07,
            "bot": create_bot(
                GeneralLlm(model="gpt-4.1", temperature=default_temperature),
            ),
        },
        "METAC_4_1_MINI_TOKEN": {
            "estimated_cost_per_question": 0.015,
            "bot": create_bot(
                GeneralLlm(
                    model="gpt-4.1-mini",
                    temperature=default_temperature,
                ),
            ),
        },
        "METAC_4_1_NANO_TOKEN": {
            "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="gpt-4.1-nano",
                    temperature=default_temperature,
                ),
            ),
        },
        "METAC_GEMINI_2_5_FLASH_PREVIEW_TOKEN": {
            "estimated_cost_per_question": 0.03,
            "bot": create_bot(
                GeneralLlm(
                    model="gemini/gemini-2.5-flash-preview-04-17",
                    temperature=default_temperature,
                    timeout=gemini_default_timeout,
                ),
            ),
        },
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
            "estimated_cost_per_question": 1.15,
            "bot": create_bot(
                GeneralLlm(
                    model="o1",
                    temperature=default_temperature,
                    reasoning_effort="medium",
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
            "estimated_cost_per_question": roughly_gemini_2_5_pro_preview_cost,
            "bot": create_bot(
                GeneralLlm(
                    model=gemini_2_5_pro_preview,
                    temperature=default_temperature,
                    timeout=gemini_default_timeout,
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
            "estimated_cost_per_question": 0.039,
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
                use_research_summary_to_forecast=default_for_using_summary,
                publish_reports_to_metaculus=default_for_publish_to_metaculus,
                skip_previously_forecasted_questions=default_for_skipping_questions,
            ),
        },
    }

    modes = list(mode_base_bot_mapping.keys())
    bots: list[ForecastBot] = [
        mode_base_bot_mapping[key]["bot"] for key in modes
    ]
    for mode, bot in zip(modes, bots):
        if "sonar" in mode.lower():
            researcher = bot.get_llm("researcher", "llm")
            if "only" in mode.lower():
                researcher = bot.get_llm("default", "llm")

            assert researcher.model.startswith("perplexity/")
            assert (
                researcher.litellm_kwargs["web_search_options"][
                    "search_context_size"
                ]
                == "high"
            )
            assert researcher.litellm_kwargs["reasoning_effort"] == "high"
        elif "grounding" in mode.lower():
            researcher = bot.get_llm("researcher", "llm")
            assert researcher.model.startswith(
                "gemini/"
            ) or researcher.model.startswith("openrouter/google/")
            assert len(researcher.litellm_kwargs["tools"]) == 1
        elif "deepseek" in mode.lower():
            researcher = bot.get_llm("default", "llm")
            assert researcher.model.startswith("openrouter/deepseek/")

    return mode_base_bot_mapping


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

    asyncio.run(configure_and_run_bot(token))
