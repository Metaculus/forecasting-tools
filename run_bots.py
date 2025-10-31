"""
This is the main file used to run the bots in the Metaulus AI Competition.

It is run by workflows in the .github/workflows directory.
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Literal

import dotenv
import pendulum
from pydantic import BaseModel

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import DateQuestion, MetaculusQuestion
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
from forecasting_tools.helpers.metaculus_api import ApiFilter, MetaculusApi
from forecasting_tools.helpers.structure_output import DEFAULT_STRUCTURE_OUTPUT_MODEL

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


default_for_skipping_questions = False
default_for_publish_to_metaculus = True
default_for_using_summary = False
default_num_forecasts_for_research_only_bot = 3
structure_output_model = DEFAULT_STRUCTURE_OUTPUT_MODEL


class ScheduleConfig:
    regular_forecast_interval_days: int = 2
    min_main_site_forecast_interval_days: int = 4

    _window_length_hrs = 2
    _US_morning_hour = 4  # 4am MT
    _US_afternoon_hour = 12  # 12pm MT
    UTC_morning_hour = _US_morning_hour + 7
    UTC_afternoon_hour = _US_afternoon_hour + 7

    default_max_main_site_questions_per_run = 30
    main_site_months_ahead_to_check = 4

    @classmethod
    def is_interval_day(cls, time: datetime | None = None) -> bool:
        time = time or pendulum.now(tz="UTC")
        value = time.day % cls.regular_forecast_interval_days == 0
        return value

    @classmethod
    def is_morning_window(cls, time: datetime | None = None) -> bool:
        time = time or pendulum.now(tz="UTC")
        hour = time.hour
        max_hour = cls.UTC_morning_hour + cls._window_length_hrs
        value = cls.UTC_morning_hour <= hour < max_hour
        return value

    @classmethod
    def is_afternoon_window(cls, time: datetime | None = None) -> bool:
        time = time or pendulum.now(tz="UTC")
        value = (
            cls.UTC_afternoon_hour
            <= time.hour
            < cls.UTC_afternoon_hour + cls._window_length_hrs
        )
        return value


class AllowedTourn(Enum):
    MINIBENCH = MetaculusApi.CURRENT_MINIBENCH_ID
    MAIN_AIB = MetaculusApi.CURRENT_AI_COMPETITION_ID
    MAIN_SITE = "main-site"
    METACULUS_CUP = MetaculusApi.CURRENT_METACULUS_CUP_ID
    GULF_BREEZE = 32810  # https://www.metaculus.com/tournament/GB/
    DEMOCRACY_THREAT_INDEX = (
        32829  # https://www.metaculus.com/index/us-democracy-threat/
    )
    # When representing a tournament, these should be valid slugs


class TournConfig:
    everything = [t for t in AllowedTourn]
    aib_only = [AllowedTourn.MAIN_AIB, AllowedTourn.MINIBENCH]
    main_site_tourns = [
        AllowedTourn.MAIN_SITE,
        AllowedTourn.GULF_BREEZE,
        AllowedTourn.DEMOCRACY_THREAT_INDEX,
    ]
    aib_and_site = aib_only.copy() + main_site_tourns.copy()
    every_x_days_tourns = [AllowedTourn.METACULUS_CUP]
    experimental = []
    none = []

    forecasts_per_main_site_question: int = 5


class RunBotConfig(BaseModel):
    mode: str
    bot: ForecastBot | None
    estimated_cost_per_question: float | None
    tournaments: list[AllowedTourn]

    model_config = {"arbitrary_types_allowed": True}


async def configure_and_run_bot(
    mode: str,
    max_questions_for_run: int = ScheduleConfig.default_max_main_site_questions_per_run,
) -> list[ForecastReport | BaseException]:
    bot_config = get_default_bot_dict()[mode]
    questions = await get_questions_for_config(
        bot_config, max_questions=max_questions_for_run
    )
    bot = bot_config.bot

    assert isinstance(
        bot, ForecastBot
    ), f"Bot {mode} is not a ForecastBot, it is a {type(bot)}"
    logger.info(f"LLMs for bot are: {bot.make_llm_dict()}")
    all_reports = []
    batch_size = 10
    batches = [
        questions[i : i + batch_size] for i in range(0, len(questions), batch_size)
    ]
    for batch in batches:
        reports = await bot.forecast_questions(batch, return_exceptions=True)
        all_reports.extend(reports)

    for i, question_report in enumerate(zip(questions, all_reports)):
        question, report = question_report
        if isinstance(report, BaseException) and "TimeoutError" in str(report):
            new_report = await bot.forecast_question(question, return_exceptions=True)
            all_reports[i] = new_report

    bot.log_report_summary(all_reports)

    return all_reports


async def get_questions_for_config(
    bot_config: RunBotConfig, max_questions: int
) -> list[MetaculusQuestion]:
    if max_questions < 1:
        raise ValueError(f"max questions ({max_questions}) must be at least 1")
    mode = bot_config.mode
    allowed_tournaments = list(set(bot_config.tournaments))
    aib_tourns = [t for t in allowed_tournaments if t in TournConfig.aib_only]
    regularly_forecast_tourns = [
        t for t in allowed_tournaments if t in TournConfig.every_x_days_tourns
    ]
    main_site_tourns = [
        t for t in allowed_tournaments if t in TournConfig.main_site_tourns
    ]

    mode_parts = mode.split("+")
    if len(mode_parts) > 1:
        suffix = mode_parts[1]
        assert suffix in [t.value for t in allowed_tournaments]

    main_site_override = (
        os.getenv("FORECAST_ON_MAIN_SITE_ALWAYS", "false").lower() == "true"
    )
    every_x_days_override = (
        os.getenv(
            "FORECAST_ON_REGULARLY_FORECASTED_TOURNAMENTS_ALWAYS", "false"
        ).lower()
        == "true"
    )

    should_forecast_on_main_site = (
        ScheduleConfig.is_interval_day() and ScheduleConfig.is_afternoon_window()
    ) or main_site_override
    should_forecast_on__every_x_days__questions = (
        ScheduleConfig.is_interval_day() and ScheduleConfig.is_morning_window()
    ) or every_x_days_override

    questions: list[MetaculusQuestion] = []
    for tournament in aib_tourns:
        questions.extend(_get_aib_questions(tournament))

    if should_forecast_on__every_x_days__questions:
        questions.extend(_get__every_x_days__questions(regularly_forecast_tourns))

    if should_forecast_on_main_site:
        main_site_questions = await _get_questions_for_main_site(main_site_tourns)
        questions.extend(main_site_questions)

    non_date_questions = [q for q in questions if not isinstance(q, DateQuestion)]

    return non_date_questions[
        :max_questions
    ]  # Note that the order questions are prioritized matter.


def _get_aib_questions(tournament: AllowedTourn) -> list[MetaculusQuestion]:
    aib_questions = MetaculusApi.get_all_open_questions_from_tournament(
        tournament.value
    )
    filtered_questions = []
    for question in aib_questions:
        if not question.already_forecasted:
            filtered_questions.append(question)
    return filtered_questions


def _get__every_x_days__questions(
    tournaments: list[AllowedTourn],
) -> list[MetaculusQuestion]:
    tournament_questions: list[MetaculusQuestion] = []
    for tournament in tournaments:
        tournament_questions += MetaculusApi.get_all_open_questions_from_tournament(
            tournament.value
        )

    filtered_questions = []
    for question in tournament_questions:
        last_forecast_time = question.timestamp_of_my_last_forecast
        should_forecast = (
            last_forecast_time is None
            or last_forecast_time
            < pendulum.now(tz="UTC").subtract(
                days=ScheduleConfig.regular_forecast_interval_days
            )
        )
        if should_forecast:
            filtered_questions.append(question)
    return filtered_questions


async def _get_questions_for_main_site(
    main_site_tourns: list[AllowedTourn],
    months_ahead_to_check: int = ScheduleConfig.main_site_months_ahead_to_check,
) -> list[MetaculusQuestion]:
    site_questions: list[MetaculusQuestion] = []
    if AllowedTourn.MAIN_SITE in main_site_tourns:
        target_months_from_now = pendulum.now(tz="UTC").add(
            days=31 * months_ahead_to_check
        )
        site_questions += await MetaculusApi.get_questions_matching_filter(
            ApiFilter(
                is_in_main_feed=True,
                allowed_statuses=["open"],
                group_question_mode="unpack_subquestions",
                scheduled_resolve_time_lt=target_months_from_now,
            ),
            num_questions=10_000,  # big enough to attempt to get everything available
            error_if_question_target_missed=False,
        )

    other_tourns = [t for t in main_site_tourns if t != AllowedTourn.MAIN_SITE]
    for tournament in other_tourns:
        site_questions += MetaculusApi.get_all_open_questions_from_tournament(
            tournament.value,
            group_question_mode="unpack_subquestions",
        )

    filtered_questions = []
    for question in site_questions:
        last_forecast_time = question.timestamp_of_my_last_forecast
        assert question.close_time is not None
        assert question.open_time is not None
        open_lifetime = (
            question.close_time - question.open_time
        )  # Choose close time over scheduled resolution time so we can actually it the 5 forecasts in the open window
        forecast_every_x_days = round(
            max(
                (open_lifetime / TournConfig.forecasts_per_main_site_question).days,
                ScheduleConfig.min_main_site_forecast_interval_days,
            )
        )
        should_forecast = (
            last_forecast_time is None
            or last_forecast_time
            < pendulum.now(tz="UTC").subtract(days=forecast_every_x_days)
        )
        if should_forecast:
            filtered_questions.append(question)
    return filtered_questions


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
                "parser": structure_output_model,
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
            "parser": structure_output_model,
        },
        extra_metadata_in_explanation=True,
    )
    return default_bot


def make_claude_thinking_settings(thinking_tokens: int, max_tokens: int) -> dict:
    return {
        "temperature": 1,
        "thinking": {
            "type": "enabled",
            "budget_tokens": thinking_tokens,
        },
        "max_tokens": max_tokens,
        "timeout": 160,
    }


def get_default_bot_dict() -> dict[str, RunBotConfig]:  # NOSONAR
    """
    Each entry in the dict has a key which is the environment variable set in the project secrets, and also used in the Workflows that run the bots.

    Note that many of the workflow entries are commented out in order to disable them.
    An entry being marked "none" for tournaments will also be disabled.

    Anything that uses the "roughly" cost value (other than the original model the variable matches to)
    is estimated value and was not measured directly. These estimates were derived from Litellm's pricing functionality.
    Also, a lot of pricing is probably outdated. Unless otherwise stated, costs are "cost per question".

    Useful Links:
    OpenRouter reasoning control settings: https://openrouter.ai/docs/use-cases/reasoning-tokens#controlling-reasoning-tokens

    Comment last updated: 10/11/2025
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
    sonnet_4_5_name = "anthropic/claude-sonnet-4-5-20250929"
    gemini_2_5_pro = "openrouter/google/gemini-2.5-pro"  # Used to be gemini-2.5-pro-preview (though automatically switched to regular pro when preview was deprecated)
    gemini_default_timeout = 120
    deepnews_model = "asknews/deep-research/high-depth/claude-sonnet-4-5-20250929"  # Switched from claude-sonnet-4-20250514 in Nov 2025
    roughly_sonnet_4_cost = 0.25190

    default_perplexity_settings: dict = {
        "web_search_options": {"search_context_size": "high"},
        "reasoning_effort": "high",
    }
    flex_price_settings: dict = {"service_tier": "flex"}
    claude_thinking_settings_16k: dict = make_claude_thinking_settings(
        thinking_tokens=16000, max_tokens=32000
    )
    claude_thinking_settings_32k: dict = make_claude_thinking_settings(
        thinking_tokens=32000, max_tokens=64000
    )

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
        **claude_thinking_settings_16k,
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
        ############################ Bots started in October 2025 ############################
        "METAC_CLAUDE_4_5_SONNET_HIGH": {
            "estimated_cost_per_question": roughly_sonnet_4_cost * 2,
            "bot": create_bot(
                llm=GeneralLlm(
                    model=sonnet_4_5_name,
                    **claude_thinking_settings_32k,
                ),
            ),
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
        },
        "METAC_CLAUDE_4_5_SONNET": {
            "estimated_cost_per_question": roughly_sonnet_4_cost,
            "bot": create_bot(
                llm=GeneralLlm(
                    model=sonnet_4_5_name,
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_QWEN_3_MAX": {
            "estimated_cost_per_question": roughly_sonnet_3_5_cost / 2,
            "bot": create_bot(
                GeneralLlm(
                    model="openrouter/qwen/qwen3-max",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_DEEPSEEK_3_2_REASONING": {
            "estimated_cost_per_question": guess_at_deepseek_v3_1_cost * 1.2,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="openrouter/deepseek/deepseek-v3.2-exp",
                    temperature=default_temperature,
                    reasoning={
                        "enabled": True,
                    },
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_DEEPSEEK_3_2": {
            "estimated_cost_per_question": guess_at_deepseek_v3_1_cost,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="openrouter/deepseek/deepseek-v3.2-exp",
                    temperature=default_temperature,
                    reasoning={
                        "enabled": False,
                    },
                ),
            ),
            "tournaments": TournConfig.none,
        },
        "METAC_GROK_4_FAST_HIGH": {
            "estimated_cost_per_question": guess_at_deepseek_v3_1_cost * 1.2,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="xai/grok-4-fast-reasoning-latest",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_GROK_4_FAST": {
            "estimated_cost_per_question": guess_at_deepseek_v3_1_cost,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="xai/grok-4-fast-non-reasoning-latest",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.none,
        },
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
                    # **flex_price_settings,
                ),
            ),
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
        },
        "METAC_GPT_5": {
            "estimated_cost_per_question": 0.19971,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="openai/gpt-5",
                    temperature=default_temperature,
                    timeout=15 * 60,
                    # **flex_price_settings,
                ),
            ),
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
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
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_GPT_5_NANO": {
            "estimated_cost_per_question": roughly_deepseek_r1_cost,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="openai/gpt-5-nano",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_CLAUDE_4_SONNET_HIGH_16K": {
            "estimated_cost_per_question": 0.33980,
            "bot": create_bot(
                llm=GeneralLlm(
                    model=sonnet_4_name,
                    **claude_thinking_settings_16k,
                ),
            ),
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
        },
        "METAC_CLAUDE_4_SONNET": {
            "estimated_cost_per_question": roughly_sonnet_4_cost,
            "bot": create_bot(
                llm=GeneralLlm(
                    model=sonnet_4_name,
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_CLAUDE_4_1_OPUS_HIGH_16K": {
            "estimated_cost_per_question": 1.56,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="anthropic/claude-opus-4-1",
                    **claude_thinking_settings_16k,
                ),
            ),
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
        },
        "METAC_GROK_4": {
            "estimated_cost_per_question": 5 * roughly_one_call_to_grok_4_llm,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="xai/grok-4-latest",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
        },
        "METAC_KIMI_K2": {
            **kimi_k2_basic_bot,
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_KIMI_K2_VARIANCE_TEST": {
            **kimi_k2_basic_bot,
            "tournaments": TournConfig.aib_only,
        },
        "METAC_DEEPSEEK_R1_VARIANCE_TEST": {
            **deepseek_r1_bot,
            "tournaments": TournConfig.aib_only,
        },  # See METAC_DEEPSEEK_R1_TOKEN below
        "METAC_GPT_OSS_120B": {
            "estimated_cost_per_question": roughly_deepseek_r1_cost,
            "bot": create_bot(
                llm=GeneralLlm(
                    model="openrouter/openai/gpt-oss-120b",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
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
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
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
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
        },
        "METAC_DEEPSEEK_V3_1": {
            **deepseek_v3_1_bot,
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_DEEPSEEK_V3_1_VARIANCE_TEST_1": {
            **deepseek_v3_1_bot,
            "tournaments": TournConfig.aib_only,
        },
        "METAC_DEEPSEEK_V3_1_VARIANCE_TEST_2": {
            **deepseek_v3_1_bot,
            "tournaments": TournConfig.aib_only,
        },
        ### Research-Only Bots
        "METAC_O4_MINI_DEEP_RESEARCH": {
            "estimated_cost_per_question": 1.5,  # Top down calculation was 1.5, while bottom up was 0.64674
            "bot": create_bot(
                llm=o4_mini_deep_research_llm,
                bot_type="research_only",
            ),
            "tournaments": TournConfig.experimental,
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
            "tournaments": TournConfig.none,
        },
        "METAC_SONAR_DEEP_RESEARCH": {
            "estimated_cost_per_question": 3
            * roughly_sonar_deep_research_cost_per_call,
            "bot": create_bot(
                llm=sonar_deep_research_llm,
                bot_type="research_only",
            ),
            "tournaments": TournConfig.experimental,
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
            "tournaments": TournConfig.none,
        },
        "METAC_GEMINI_2_5_PRO_GROUNDING": {
            "estimated_cost_per_question": roughly_sonnet_3_5_cost
            + guess_at__research_only_bot__search_costs,
            "bot": create_bot(
                gemini_grounding_llm,
                bot_type="research_only",
            ),
            "tournaments": TournConfig.aib_only,
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
            "tournaments": TournConfig.aib_only,
        },
        "METAC_GPT_5_SEARCH": {
            "estimated_cost_per_question": 1.17,
            "bot": create_bot(
                llm=gpt_5_with_search,
                bot_type="research_only",
            ),
            "tournaments": TournConfig.experimental,
        },
        "METAC_GROK_4_LIVE_SEARCH": {
            "estimated_cost_per_question": 3 * roughly_one_call_to_grok_4_llm,
            "bot": create_bot(
                llm=grok_4_search_llm,
                bot_type="research_only",
            ),
            "tournaments": TournConfig.aib_only,
        },
        "METAC_SONNET_4_SEARCH": {
            "estimated_cost_per_question": 1.53366,
            "bot": create_bot(
                llm=sonnet_4_search_llm,
                bot_type="research_only",
            ),
            "tournaments": TournConfig.experimental,
        },
        "METAC_DEEPSEEK_R1_EXA_ONLINE_RESEARCH_ONLY": {
            "estimated_cost_per_question": roughly_deepseek_r1_cost
            + guess_at__research_only_bot__search_costs,
            "bot": create_bot(
                llm=deepseek_r1_exa_online_llm,
                bot_type="research_only",
            ),
            "tournaments": TournConfig.aib_only,
        },
        ### DeepSeek Research Bots
        "METAC_DEEPSEEK_R1_PLUS_EXA_ONLINE": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                llm=default_research_comparison_forecast_llm,
                researcher=deepseek_r1_exa_online_llm,
            ),
            "tournaments": TournConfig.aib_only,
        },
        "METAC_DEEPSEEK_R1_SONNET_4_SEARCH": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                llm=default_research_comparison_forecast_llm,
                researcher=sonnet_4_search_llm,
            ),
            "tournaments": TournConfig.experimental,
        },
        "METAC_DEEPSEEK_R1_XAI_LIVESEARCH": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search
            + roughly_one_call_to_grok_4_llm,
            "bot": create_bot(
                llm=default_research_comparison_forecast_llm,
                researcher=grok_4_search_llm,
            ),
            "tournaments": TournConfig.aib_only,
        },
        "METAC_DEEPSEEK_R1_O4_MINI_DEEP_RESEARCH": {
            "estimated_cost_per_question": roughly_deepseek_r1_cost + 1.5 / 3,
            "bot": create_bot(
                llm=default_research_comparison_forecast_llm,
                researcher=o4_mini_deep_research_llm,
            ),
            "tournaments": TournConfig.experimental,
        },
        "METAC_DEEPSEEK_R1_NO_RESEARCH": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                llm=default_research_comparison_forecast_llm,
                researcher="no_research",
            ),
            "tournaments": TournConfig.aib_only + [AllowedTourn.METACULUS_CUP],
        },
        ### Specialized Bots
        "METAC_GPT_4_1_OPTIMIZED_PROMPT": {
            "estimated_cost_per_question": 0.13871,
            "bot": create_bot(
                GeneralLlm(
                    model="openai/gpt-4.1",
                    temperature=default_temperature,
                    timeout=60 * 5,
                ),
                bot_type="gpt_4_1_optimized",
            ),
            "tournaments": TournConfig.aib_only,
        },
        "METAC_GPT_4_1_NANO_OPTIMIZED_PROMPT": {
            "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="openai/gpt-4.1-nano",
                    temperature=default_temperature,
                    timeout=60 * 5,
                ),
                bot_type="gpt_4_1_optimized",
            ),
            "tournaments": TournConfig.aib_only,
        },
        "METAC_GROK_4_TOOLS": {
            "estimated_cost_per_question": None,
            "bot": None,
            "tournaments": TournConfig.none,
        },  # Don't have time to implement this, but this is a env variable that exists
        "METAC_GPT_5_HIGH_TOOLS": {
            "estimated_cost_per_question": None,
            "bot": None,
            "tournaments": TournConfig.none,
        },  # Don't have time to implement this, but this is a env variable that exists
        "METAC_SONNET_4_HIGH_TOOLS": {
            "estimated_cost_per_question": None,
            "bot": None,
            "tournaments": TournConfig.none,
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
            "tournaments": TournConfig.none,
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
            "tournaments": TournConfig.none,
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
            "tournaments": TournConfig.none,
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
            "tournaments": TournConfig.aib_only,
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
            "tournaments": TournConfig.aib_only,
        },
        "METAC_DEEPSEEK_R1_SONAR_DEEP_RESEARCH": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search
            + roughly_sonar_deep_research_cost_per_call,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher=sonar_deep_research_llm,
            ),
            "tournaments": TournConfig.experimental,
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
            "tournaments": TournConfig.aib_only,
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
            "tournaments": TournConfig.aib_only,
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
            "tournaments": TournConfig.aib_only,
        },
        "METAC_DEEPSEEK_R1_GPT_4O_SEARCH_PREVIEW": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher=GeneralLlm(
                    model="openai/gpt-4o-search-preview", temperature=None
                ),
            ),
            "tournaments": TournConfig.aib_only,
        },
        "METAC_DEEPSEEK_R1_GEMINI_2_5_PRO_GROUNDING": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher=gemini_grounding_llm,
            ),
            "tournaments": TournConfig.aib_only,
        },
        "METAC_DEEPSEEK_R1_EXA_SMART_SEARCHER": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher="smart-searcher/openrouter/deepseek/deepseek-r1",
            ),
            "tournaments": TournConfig.none,
        },
        "METAC_DEEPSEEK_R1_ASK_EXA_PRO": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher=GeneralLlm(
                    model="exa/exa"
                ),  # Used to be "exa-pro" till this got deprecated
            ),
            "tournaments": TournConfig.aib_only,
        },
        "METAC_DEEPSEEK_R1_DEEPNEWS": {
            "estimated_cost_per_question": guess_at_deepseek_plus_search,
            "bot": create_bot(
                default_research_comparison_forecast_llm,
                researcher=deepnews_model,
            ),
            "tournaments": TournConfig.experimental,
        },
        "METAC_O3_HIGH_TOKEN": {
            "estimated_cost_per_question": 0.16,
            "bot": create_bot(
                GeneralLlm(
                    model="o3",
                    temperature=1,
                    reasoning_effort="high",
                    timeout=60 * 8,
                    # **flex_price_settings,
                ),
            ),
            "tournaments": TournConfig.none + [AllowedTourn.METACULUS_CUP],
        },
        "METAC_O3_TOKEN": {
            "estimated_cost_per_question": 0.16 * 0.8,
            "bot": create_bot(
                GeneralLlm(
                    model="o3",
                    temperature=1,
                    reasoning_effort="medium",
                    timeout=60 * 8,
                    # **flex_price_settings,
                ),
            ),
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
        },
        "METAC_O4_MINI_HIGH_TOKEN": {
            "estimated_cost_per_question": 0.07,
            "bot": create_bot(
                GeneralLlm(
                    model="o4-mini",
                    temperature=1,
                    reasoning_effort="high",
                    # **flex_price_settings,
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_O4_MINI_TOKEN": {
            "estimated_cost_per_question": 0.043,
            "bot": create_bot(
                GeneralLlm(
                    model="o4-mini",
                    temperature=1,
                    reasoning_effort="medium",
                    # **flex_price_settings,
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_4_1_TOKEN": {
            "estimated_cost_per_question": 0.07,
            "bot": create_bot(
                GeneralLlm(model="gpt-4.1", temperature=default_temperature),
            ),
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_4_1_MINI_TOKEN": {
            "estimated_cost_per_question": 0.015,
            "bot": create_bot(
                GeneralLlm(
                    model="gpt-4.1-mini",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_4_1_NANO_TOKEN": {
            "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="gpt-4.1-nano",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
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
            "tournaments": TournConfig.aib_and_site,
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
            "tournaments": TournConfig.none,
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
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
        },
        "METAC_O1_MINI_TOKEN": {
            "estimated_cost_per_question": roughly_gpt_4o_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="o1-mini",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.none,
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
            "tournaments": TournConfig.none,
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
            "tournaments": TournConfig.none,
        },
        "METAC_GPT_4O_TOKEN": {
            "estimated_cost_per_question": roughly_gpt_4o_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="gpt-4o-2024-08-06",  # NOTE: This bot used to be just "gpt-4o" in q2 2025 and before so changed between the 3 versions as the API Updated
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
        },
        "METAC_GPT_4O_MINI_TOKEN": {
            "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="gpt-4o-mini",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_GPT_3_5_TURBO_TOKEN": {
            "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="gpt-3.5-turbo",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
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
            "tournaments": TournConfig.none,
        },
        "METAC_CLAUDE_3_7_SONNET_LATEST_TOKEN": {
            "estimated_cost_per_question": roughly_sonnet_3_5_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="anthropic/claude-3-7-sonnet-latest",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.none,
        },
        "METAC_CLAUDE_3_5_SONNET_LATEST_TOKEN": {
            "estimated_cost_per_question": roughly_sonnet_3_5_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="anthropic/claude-3-5-sonnet-latest",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.none,  # NOTE: No longer available (model deprecated by Anthropic)
        },
        "METAC_CLAUDE_3_5_SONNET_20240620_TOKEN": {
            "estimated_cost_per_question": roughly_sonnet_3_5_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="anthropic/claude-3-5-sonnet-20240620",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.none,  # NOTE: No longer available (model deprecated by Anthropic)
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
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
        },
        "METAC_GEMINI_2_0_FLASH_TOKEN": {
            "estimated_cost_per_question": 0.05,
            "bot": create_bot(
                GeneralLlm(
                    model="gemini/gemini-2.0-flash-001",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.none,
        },
        "METAC_LLAMA_4_MAVERICK_17B_TOKEN": {
            "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="openrouter/meta-llama/llama-4-maverick",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
        },
        "METAC_QWEN_2_5_MAX_TOKEN": {
            "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="openrouter/qwen/qwen-max",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_DEEPSEEK_R1_TOKEN": {
            **deepseek_r1_bot,
            "tournaments": TournConfig.aib_and_site + [AllowedTourn.METACULUS_CUP],
        },
        "METAC_DEEPSEEK_V3_TOKEN": {
            "estimated_cost_per_question": roughly_gpt_4o_mini_cost,
            "bot": create_bot(
                GeneralLlm(
                    model="openrouter/deepseek/deepseek-chat",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_GROK_3_LATEST_TOKEN": {
            "estimated_cost_per_question": 0.13,
            "bot": create_bot(
                GeneralLlm(
                    model="xai/grok-3-latest",
                    temperature=default_temperature,
                ),
            ),
            "tournaments": TournConfig.aib_and_site,
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
            "tournaments": TournConfig.aib_and_site,
        },
        "METAC_UNIFORM_PROBABILITY_BOT_TOKEN": {
            "estimated_cost_per_question": 0.00,
            "bot": UniformProbabilityBot(
                use_research_summary_to_forecast=default_for_using_summary,
                publish_reports_to_metaculus=default_for_publish_to_metaculus,
                skip_previously_forecasted_questions=default_for_skipping_questions,
            ),
            "tournaments": TournConfig.everything,
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

    mode_to_bot_config = {
        mode: RunBotConfig(**{**bot_config, "mode": mode})
        for mode, bot_config in mode_base_bot_mapping.items()
    }
    return mode_to_bot_config


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
