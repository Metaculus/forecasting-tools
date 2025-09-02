"""
This is the main file used to run the bots in the Metaulus AI Competition.

It is run by workflows in the .github/workflows directory.
"""

import argparse
import asyncio
import logging
from typing import Any, Literal

import dotenv

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.official_bots.fall_research_only_bot import (
    FallResearchOnlyBot2025,
)
from forecasting_tools.forecast_bots.official_bots.gpt_4_1_optimized_bot import (
    GPT41OptimizedBot,
)
from forecasting_tools.forecast_bots.official_bots.uniform_probability_bot import (
    UniformProbabilityBot,
)
from forecasting_tools.forecast_bots.template_bot import TemplateBot
from forecasting_tools.helpers.metaculus_api import MetaculusApi
from forecasting_tools.helpers.structure_output import DEFAULT_STRUCTURE_OUTPUT_MODEL

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


default_for_skipping_questions = True
default_for_publish_to_metaculus = True
default_for_using_summary = False
default_num_forecasts_for_research_only_bot = 3


async def configure_and_run_bot(
    mode: str, return_bot__dont_run: bool = False
) -> ForecastBot | None | list[ForecastReport | BaseException]:

    if "metaculus-cup" in mode:
        assert (
            "+" in mode
        ), "Metaculus cup mode must be in the format 'token+tournament_id'"
        token = mode.split("+")[0]
        chosen_tournaments = [MetaculusApi.CURRENT_METACULUS_CUP_ID]
        skip_previously_forecasted_questions = False
    elif "+" in mode:
        parts = mode.split("+")
        assert len(parts) == 2
        token = parts[0]
        chosen_tournaments = [parts[1]]
        skip_previously_forecasted_questions = False
    else:
        chosen_tournaments = [
            MetaculusApi.CURRENT_AI_COMPETITION_ID,
            MetaculusApi.CURRENT_MINIBENCH_ID,
        ]
        skip_previously_forecasted_questions = True
        token = mode

    is_discontinued = get_default_bot_dict()[token].get("discontinued", False)
    if is_discontinued:
        logger.warning(f"Bot {token} is discontinued, skipping")
        return None

    bot = get_default_bot_dict()[token]["bot"]
    if bot is not None:
        assert isinstance(bot, ForecastBot)
        bot.skip_previously_forecasted_questions = skip_previously_forecasted_questions

    if return_bot__dont_run:
        return bot
    else:
        assert isinstance(bot, ForecastBot)
        logger.info(f"LLMs for bot are: {bot.make_llm_dict()}")
        all_reports = []
        for tournament in chosen_tournaments:
            reports = await bot.forecast_on_tournament(
                tournament, return_exceptions=True
            )
            all_reports.extend(reports)
        bot.log_report_summary(all_reports)
        return all_reports


async def get_all_bots() -> list[ForecastBot]:
    bots = []
    keys = list(get_default_bot_dict().keys())
    for key in keys:
        bots.append(await configure_and_run_bot(key, return_bot__dont_run=True))
    return bots


def create_bot(
    llm: GeneralLlm,
    researcher: str | GeneralLlm = "asknews/news-summaries",
    predictions_per_research_report: int | None = None,
    bot_type: Literal["template", "gpt_4_1_optimized", "research_only"] = "template",
) -> ForecastBot:
    default_summarizer = "openrouter/openai/gpt-4.1-mini"

    if bot_type == "research_only":
        return FallResearchOnlyBot2025(
            research_reports_per_question=1,
            predictions_per_research_report=predictions_per_research_report
            or default_num_forecasts_for_research_only_bot,
            use_research_summary_to_forecast=default_for_using_summary,
            publish_reports_to_metaculus=default_for_publish_to_metaculus,
            skip_previously_forecasted_questions=default_for_skipping_questions,
            llms={
                "default": llm,
                "summarizer": None,
                "researcher": "no_research",
                "parser": DEFAULT_STRUCTURE_OUTPUT_MODEL,
            },
            enable_summarize_research=False,
            extra_metadata_in_explanation=True,
        )

    if bot_type == "template":
        bot_class = TemplateBot
    elif bot_type == "gpt_4_1_optimized":
        bot_class = GPT41OptimizedBot
    else:
        raise ValueError(f"Invalid bot type: {bot_type}")

    default_bot = bot_class(
        research_reports_per_question=1,
        predictions_per_research_report=predictions_per_research_report or 5,
        use_research_summary_to_forecast=default_for_using_summary,
        publish_reports_to_metaculus=default_for_publish_to_metaculus,
        skip_previously_forecasted_questions=default_for_skipping_questions,
        llms={
            "default": llm,
            "summarizer": default_summarizer,
            "researcher": researcher,
            "parser": DEFAULT_STRUCTURE_OUTPUT_MODEL,
        },
        extra_metadata_in_explanation=True,
    )
    return default_bot


def get_default_bot_dict() -> dict[str, Any]:  # NOSONAR
    """
    Each entry in the dict has a key which is the environment variable set in the project secrets, and also used in the Workflows that run the bots.

    Anything that uses the "roughly" cost value (other than the original model the variable matches to)
    is estimated value and was not measured directly. These estimates were derived from Litellm's pricing functionality.
    Also, a lot of pricing is probably outdated. Unless otherwise stated, costs are "cost per question".

    Useful Links:
    OpenRouter reasoning control settings: https://openrouter.ai/docs/use-cases/reasoning-tokens#controlling-reasoning-tokens
    """
    default_temperature = 0.3

    roughly_gpt_4o_cost = 0.05
    roughly_gpt_4o_mini_cost = 0.005
    roughly_sonnet_3_5_cost = 0.10
    roughly_gemini_2_5_pro_preview_cost = 0.30  # TODO: Double check this
    roughly_deepseek_r1_cost = 0.039
    guess_at_search_cost = 0.015
    guess_at__research_only_bot__search_costs = (
        guess_at_search_cost * default_num_forecasts_for_research_only_bot
    )
    guess_at_deepseek_plus_search = roughly_deepseek_r1_cost + guess_at_search_cost
    guess_at_deepseek_v3_1_cost = roughly_deepseek_r1_cost / 2
    roughly_one_call_to_grok_4_llm = 0.084
    roughly_sonar_deep_research_cost_per_call = 1.35399 / 3

    sonnet_4_name = "anthropic/claude-sonnet-4-20250514"
    gemini_2_5_pro = "openrouter/google/gemini-2.5-pro"  # Used to be gemini-2.5-pro-preview (though automatically switched to regular pro when preview was deprecated)
    gemini_default_timeout = 120
    deepnews_model = "asknews/deep-research/high-depth/o3"

    default_perplexity_settings: dict = {
        "web_search_options": {"search_context_size": "high"},
        "reasoning_effort": "high",
    }
    flex_price_settings: dict = {"service_tier": "flex"}
    claude_thinking_settings: dict = {
        "temperature": 1,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 16000,
        },
        "max_tokens": 32000,
        "timeout": 160,
    }

    gemini_grounding_llm = GeneralLlm(
        model=gemini_2_5_pro,
        # generationConfig={
        #     "thinkingConfig": {
        #         "thinkingBudget": 0,
        #     },
        #     "responseMimeType": "text/plain",
        # }, # In Q2 had thinking turned off
        tools=[
            {"googleSearch": {}},
        ],
    )  # https://ai.google.dev/gemini-api/docs/google-search#rest
    default_research_comparison_forecast_llm = GeneralLlm(
        model="openrouter/deepseek/deepseek-r1",
        temperature=default_temperature,
    )
    grok_4_search_llm = GeneralLlm(
        model="xai/grok-4-latest",
        search_parameters={"mode": "auto"},
    )  # https://docs.x.ai/docs/guides/live-search
    sonnet_4_search_llm = GeneralLlm(
        model=sonnet_4_name,
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 10,
            }
        ],
        **claude_thinking_settings,
    )  # https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool
    deepseek_r1_exa_online_llm = GeneralLlm(
        model="openrouter/deepseek/deepseek-r1:online",
    )
    o4_mini_deep_research_llm = GeneralLlm(
        model="openai/o4-mini-deep-research",
        responses_api=True,
        temperature=None,
        tools=[
            {"type": "web_search"},
            # {"type": "code_interpreter", "container": {"type": "auto", "file_ids": []}}, # TODO: Consider adding code interpreter
        ],
    )
    sonar_deep_research_llm = GeneralLlm(
        model="perplexity/sonar-deep-research",
        **default_perplexity_settings,
    )
    gpt_5_with_search = GeneralLlm(
        model="openai/gpt-5",
        responses_api=True,
        temperature=None,
        tools=[
            {"type": "web_search"},
            # {"type": "code_interpreter", "container": {"type": "auto", "file_ids": []}}, # TODO: Consider adding code interpreter
        ],
        **flex_price_settings,
    )

    kimi_k2_basic_bot = {
        "estimated_cost_per_question": roughly_deepseek_r1_cost,
        "bot": create_bot(
            GeneralLlm(
                model="openrouter/moonshotai/kimi-k2",
                temperature=default_temperature,
            ),
        ),
    }
    deepseek_r1_bot = {
        "estimated_cost_per_question": roughly_deepseek_r1_cost,
        "bot": create_bot(
            GeneralLlm(
                model="openrouter/deepseek/deepseek-r1",
                temperature=default_temperature,
            ),
        ),
    }
    deepseek_v3_1_bot = {
        "estimated_cost_per_question": guess_at_deepseek_v3_1_cost,
        "bot": create_bot(
            GeneralLlm(
                model="openrouter/deepseek/deepseek-chat-v3.1",
                temperature=default_temperature,
            ),
        ),
    }

    mode_base_bot_mapping = {
        ############################ Bots started in Fall 2025 ############################
        ### Regular Bots
        "METAC_GPT_5_HIGH": {
            "estimated_cost_per_question": 0.37868,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="openai/gpt-5",
                    reasoning_effort="high",
                    temperature=default_temperature,
                    timeout=15 * 60,
                    **flex_price_settings,
                ),
            ),
        },
        "METAC_GPT_5": {
            "estimated_cost_per_question": 0.19971,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="openai/gpt-5",
                    temperature=default_temperature,
                    timeout=15 * 60,
                    **flex_price_settings,
                ),
            ),
        },
        "METAC_GPT_5_MINI": {
            "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="openai/gpt-5-mini",
                    temperature=default_temperature,
                    **flex_price_settings,
                ),
            ),
        },
        "METAC_GPT_5_NANO": {
            "estimated_cost_per_question": roughly_deepseek_r1_cost,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="openai/gpt-5-nano",
                    temperature=default_temperature,
                ),
            ),
        },
        "METAC_CLAUDE_4_SONNET_HIGH_16K": {
            "estimated_cost_per_question": 0.33980,
            "bot": create_bot(
                llm=GeneralLlm(
                    model=sonnet_4_name,
                    **claude_thinking_settings,
                ),
            ),
        },
        "METAC_CLAUDE_4_SONNET": {
            "estimated_cost_per_question": 0.25190,
            "bot": create_bot(
                llm=GeneralLlm(
                    model=sonnet_4_name,
                    temperature=default_temperature,
                ),
            ),
        },
        "METAC_CLAUDE_4_1_OPUS_HIGH_16K": {
            "estimated_cost_per_question": 1.56,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="anthropic/claude-opus-4-1",
                    **claude_thinking_settings,
                ),
            ),
        },
        "METAC_GROK_4": {
            "estimated_cost_per_question": 5 * roughly_one_call_to_grok_4_llm,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="xai/grok-4-latest",
                    temperature=default_temperature,
                ),
            ),
        },
        "METAC_KIMI_K2": kimi_k2_basic_bot,
        "METAC_KIMI_K2_VARIANCE_TEST": kimi_k2_basic_bot,
        "METAC_DEEPSEEK_R1_VARIANCE_TEST": deepseek_r1_bot,  # See METAC_DEEPSEEK_R1_TOKEN below
        "METAC_GPT_OSS_120B": {
            "estimated_cost_per_question": roughly_deepseek_r1_cost,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="openrouter/openai/gpt-oss-120b",
                    temperature=default_temperature,
                ),
            ),
        },
        "METAC_ZAI_GLM_4_5": {
            "estimated_cost_per_question": roughly_deepseek_r1_cost,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="openrouter/z-ai/glm-4.5",
                    temperature=default_temperature,
                    reasoning={
                        "enabled": True,
                    },
                ),
            ),
        },
        "METAC_DEEPSEEK_V3_1_REASONING": {
            "estimated_cost_per_question": guess_at_deepseek_v3_1_cost * 1.2,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="openrouter/deepseek/deepseek-chat-v3.1",
                    temperature=default_temperature,
                    reasoning={
                        "enabled": True,
                    },
                ),
            ),
        },
        "METAC_DEEPSEEK_V3_1": deepseek_v3_1_bot,
        "METAC_DEEPSEEK_V3_1_VARIANCE_TEST_1": deepseek_v3_1_bot,
        "METAC_DEEPSEEK_V3_1_VARIANCE_TEST_2": deepseek_v3_1_bot,
        ### Research-Only Bots
        "METAC_O4_MINI_DEEP_RESEARCH": {
            "estimated_cost_per_question": 1.5,  # Top down calculation was 1.5, while bottom up was 0.64674
            "bot": create_bot(
                llm=o4_mini_deep_research_llm,
                bot_type="research_only",
            ),
        },
        "METAC_O3_DEEP_RESEARCH": {
            "estimated_cost_per_question": roughly_sonnet_3_5_cost * 3
            + guess_at__research_only_bot__search_costs * 5,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="openai/o3-deep-research",
                    temperature=default_temperature,
                ),
                bot_type="research_only",
            ),
            "discontinued": True,
        },
        "METAC_SONAR_DEEP_RESEARCH": {
            "estimated_cost_per_question": 3
            * roughly_sonar_deep_research_cost_per_call,
            "bot": create_bot(
                llm=sonar_deep_research_llm,
                bot_type="research_only",
            ),
        },
        "METAC_EXA_RESEARCH_PRO": {
            "estimated_cost_per_question": roughly_deepseek_r1_cost
            + guess_at__research_only_bot__search_costs,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="exa/exa-research-pro",
                    temperature=default_temperature,
                ),
                bot_type="research_only",
            ),
        },
        "METAC_GEMINI_2_5_PRO_GROUNDING": {
            "estimated_cost_per_question": roughly_sonnet_3_5_cost
            + guess_at__research_only_bot__search_costs,
            "bot": create_bot(
                gemini_grounding_llm,
                bot_type="research_only",
            ),
        },
        "METAC_ASKNEWS_DEEPNEWS": {
            "estimated_cost_per_question": 0,
            "bot": create_bot(
                llm=GeneralLlm(
                    model=deepnews_model,
                    temperature=default_temperature,
                ),
                bot_type="research_only",
            ),
        },
        "METAC_GPT_5_SEARCH": {
            "estimated_cost_per_question": 1.17,
            "bot": create_bot(
                llm=gpt_5_with_search,
                bot_type="research_only",
            ),
        },
        "METAC_GROK_4_LIVE_SEARCH": {
            "estimated_cost_per_question": 3 * roughly_one_call_to_grok_4_llm,
            "bot": create_bot(
                llm=grok_4_search_llm,
                bot_type="research_only",
            ),
        },
        "METAC_SONNET_4_SEARCH": {
            "estimated_cost_per_question": 1.53366,
            "bot": create_bot(
                llm=sonnet_4_search_llm,
                bot_type="research_only",
            ),
        },
        "METAC_DEEPSEEK_R1_EXA_ONLINE_RESEARCH_ONLY": {
            "estimated_cost_per_question": roughly_deepseek_r1_cost
            + guess_at__research_only_bot__search_costs,
            "bot": create_bot(
                llm=deepseek_r1_exa_online_llm,
                bot_type="research_only",
            ),
        },
        ### DeepSeek Research Bots
        "METAC_DEEPSEEK_R1_PLUS_EXA_ONLINE": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                llm=default_research_comparison_forecast_llm,
                researcher=deepseek_r1_exa_online_llm,
            ),
        },
        "METAC_DEEPSEEK_R1_SONNET_4_SEARCH": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                llm=default_research_comparison_forecast_llm,
                researcher=sonnet_4_search_llm,
            ),
        },
        "METAC_DEEPSEEK_R1_XAI_LIVESEARCH": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search
            + roughly_one_call_to_grok_4_llm,
            "bot": create_bot(
                llm=default_research_comparison_forecast_llm,
                researcher=grok_4_search_llm,
            ),
        },
        "METAC_DEEPSEEK_R1_O4_MINI_DEEP_RESEARCH": {
            "estimated_cost_per_question": roughly_deepseek_r1_cost + 1.5 / 3,
            "bot": create_bot(
                llm=default_research_comparison_forecast_llm,
                researcher=o4_mini_deep_research_llm,
            ),
        },
        "METAC_DEEPSEEK_R1_NO_RESEARCH": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                llm=default_research_comparison_forecast_llm,
                researcher="no_research",
            ),
        },
        ### Specialized Bots
        "METAC_GPT_4_1_OPTIMIZED_PROMPT": {
            "estimated_cost_per_question": 0.13871,
            "bot": create_bot(
                GeneralLlm(
                    model="openai/gpt-4.1",
                    temperature=default_temperature,
                ),
                bot_type="gpt_4_1_optimized",
            ),
        },
        "METAC_GPT_4_1_NANO_OPTIMIZED_PROMPT": {
            "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="openai/gpt-4.1-nano",
                    temperature=default_temperature,
                ),
                bot_type="gpt_4_1_optimized",
            ),
        },
        "METAC_GROK_4_TOOLS": {
            "estimated_cost_per_question": None,
            "bot": None,
        },  # Don't have time to implement this, but this is a env variable that exists
        "METAC_GPT_5_HIGH_TOOLS": {
            "estimated_cost_per_question": None,
            "bot": None,
        },  # Don't have time to implement this, but this is a env variable that exists
        "METAC_SONNET_4_HIGH_TOOLS": {
            "estimated_cost_per_question": None,
            "bot": None,
        },  # Don't have time to implement this, but this is a env variable that exists
        ############################ Bots started in Q2 2025 ############################
        "METAC_GEMINI_2_5_PRO_GEMINI_2_5_PRO_GROUNDING": {
            "estimated_cost_per_question": 0.16,
            "bot": create_bot(
                GeneralLlm(
                    model=gemini_2_5_pro,
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
                    model=gemini_2_5_pro,
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
                    model=gemini_2_5_pro,
                    temperature=default_temperature,
                    timeout=gemini_default_timeout,
                ),
                researcher=GeneralLlm(
                    model="exa/exa"
                ),  # NOTE: Used to be exa-pro but that got deprecated
            ),
        },
        "METAC_DEEPSEEK_R1_SONAR_PRO": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher=GeneralLlm(
                    model="perplexity/sonar-pro",
                    **default_perplexity_settings,
                ),
            ),
        },
        "METAC_DEEPSEEK_R1_SONAR": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher=GeneralLlm(
                    model="perplexity/sonar",
                    **default_perplexity_settings,
                ),
            ),
        },
        "METAC_DEEPSEEK_R1_SONAR_DEEP_RESEARCH": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search
            + roughly_sonar_deep_research_cost_per_call,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher=sonar_deep_research_llm,
            ),
        },
        "METAC_DEEPSEEK_R1_SONAR_REASONING_PRO": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher=GeneralLlm(
                    model="perplexity/sonar-reasoning-pro",
                    **default_perplexity_settings,
                ),
            ),
        },
        "METAC_DEEPSEEK_R1_SONAR_REASONING": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
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
                default_research_comparison_forecast_llm,
                researcher=GeneralLlm(
                    model="openai/gpt-4o-search-preview", temperature=None
                ),
            ),
        },
        "METAC_DEEPSEEK_R1_GEMINI_2_5_PRO_GROUNDING": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher=gemini_grounding_llm,
            ),
        },
        "METAC_DEEPSEEK_R1_EXA_SMART_SEARCHER": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher="smart-searcher/openrouter/deepseek/deepseek-r1",
            ),
        },
        "METAC_DEEPSEEK_R1_ASK_EXA_PRO": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher=GeneralLlm(
                    model="exa/exa"
                ),  # Used to be "exa-pro" till this got deprecated
            ),
        },
        "METAC_DEEPSEEK_R1_DEEPNEWS": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher=deepnews_model,
            ),
        },
        "METAC_O3_HIGH_TOKEN": {
            "estimated_cost_per_question": 0.16,
            "bot": create_bot(
                GeneralLlm(
                    model="o3",
                    temperature=1,
                    reasoning_effort="high",
                    timeout=300,
                    **flex_price_settings,
                ),
            ),
        },
        "METAC_O3_TOKEN": {
            "estimated_cost_per_question": 0.16 * 0.8,
            "bot": create_bot(
                GeneralLlm(
                    model="o3",
                    temperature=1,
                    reasoning_effort="medium",
                    **flex_price_settings,
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
                    **flex_price_settings,
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
                    **flex_price_settings,
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
                    model="openrouter/google/gemini-2.5-flash",  # NOTE: This was updated from "gemini/gemini-2.5-flash-preview-04-17" in Fall 2025
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
                    model="gpt-4o-2024-08-06",  # NOTE: This bot used to be just "gpt-4o" in q2 2025 and before so changed between the 3 versions as the API Updated
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
            "discontinued": True,
        },
        "METAC_CLAUDE_3_5_SONNET_20240620_TOKEN": {
            "estimated_cost_per_question": roughly_sonnet_3_5_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="anthropic/claude-3-5-sonnet-20240620",
                    temperature=default_temperature,
                ),
            ),
            "discontinued": True,
        },
        "METAC_GEMINI_2_5_PRO_PREVIEW_TOKEN": {
            "estimated_cost_per_question": roughly_gemini_2_5_pro_preview_cost,
            "bot": create_bot(
                GeneralLlm(
                    model=gemini_2_5_pro,  # NOTE: Switched from preview to regular pro mid Q2
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
        "METAC_DEEPSEEK_R1_TOKEN": deepseek_r1_bot,
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

    # Run basic validation/sanity checks on the bots
    modes = list(mode_base_bot_mapping.keys())
    bots: list[ForecastBot] = [mode_base_bot_mapping[key]["bot"] for key in modes]
    for mode, bot in zip(modes, bots):
        if "sonar" in mode.lower():
            researcher = bot.get_llm("researcher", "llm")
            if "only" in mode.lower():
                researcher = bot.get_llm("default", "llm")

            researcher_is_perplexity = researcher.model.startswith("perplexity/")
            forecaster_is_perplexity = bot.get_llm("default", "llm").model.startswith(
                "perplexity/"
            )

            assert researcher_is_perplexity or forecaster_is_perplexity
            if researcher_is_perplexity:
                assert (
                    researcher.litellm_kwargs["web_search_options"][
                        "search_context_size"
                    ]
                    == "high"
                ), f"Researcher {researcher.model} is perplexity but does not have high search context size for {mode}"
                assert (
                    researcher.litellm_kwargs["reasoning_effort"] == "high"
                ), f"Researcher {researcher.model} is not set to high reasoning effort for {mode}"
        elif "grounding" in mode.lower():
            researcher = bot.get_llm("researcher", "llm")
            forecaster = bot.get_llm("default", "llm")
            researcher_is_google = researcher.model.startswith(
                "gemini/"
            ) or researcher.model.startswith("openrouter/google/")
            forecaster_is_google = forecaster.model.startswith("openrouter/google/")
            assert researcher_is_google or forecaster_is_google
            if researcher_is_google:
                assert len(researcher.litellm_kwargs["tools"]) == 1
        elif "deepseek" in mode.lower():
            researcher = bot.get_llm("researcher", "llm")
            forecaster = bot.get_llm("default", "llm")
            researcher_is_deepseek = researcher.model.startswith("openrouter/deepseek/")
            forecaster_is_deepseek = forecaster.model.startswith("openrouter/deepseek/")
            assert researcher_is_deepseek or forecaster_is_deepseek

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
