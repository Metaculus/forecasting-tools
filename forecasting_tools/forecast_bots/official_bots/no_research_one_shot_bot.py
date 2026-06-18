"""A minimal single-shot forecasting bot with no research and no tools.

This bot asks a model to forecast a question directly, with a short
"helpful assistant" framing and a request for a JSON forecast. It performs no
research phase and (when configured with a single prediction per question)
makes exactly one model call per question.
"""

import logging
from datetime import datetime, timezone

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.official_bots.template_bot_2026_summer import (
    SummerTemplateBot2026,
)

logger = logging.getLogger(__name__)

_SYSTEM_FRAMING = "You are a helpful assistant.\n\n"
_REASONING_INSTRUCTION = (
    "Briefly explain your reasoning and provide your forecast as a JSON code block.\n\n"
)
_PERCENTILE_KEYS = ["p05", "p25", "p50", "p75", "p95"]
_EXAMPLE_FRACTIONS = [0.05, 0.25, 0.5, 0.75, 0.95]


class NoResearchOneShotBot(SummerTemplateBot2026):
    """Forecasts each question in a single model call with no research phase.

    The prompts are intentionally minimal: there is no professional-forecaster
    persona and no guided chain-of-thought sub-questions. The model is simply
    asked to reason briefly and return a JSON forecast, which is then parsed by
    the configured parser model.
    """

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        config_dict = super()._llm_config_defaults()
        if "researcher" in config_dict:
            config_dict.pop("researcher")
        if "summarizer" in config_dict:
            config_dict["summarizer"] = None
        return config_dict

    async def run_research(self, question: MetaculusQuestion) -> str:
        return ""

    @staticmethod
    def _question_details(question: MetaculusQuestion) -> str:
        parts: list[str] = []
        if question.background_info:
            parts.append(question.background_info)
        if question.resolution_criteria:
            parts.append(f"Resolution Criteria:\n{question.resolution_criteria}")
        if question.fine_print:
            parts.append(f"Fine Print:\n{question.fine_print}")
        return "\n\n".join(parts)

    @classmethod
    def _header(cls, question: MetaculusQuestion) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return (
            f"**Question:** {question.question_text}\n\n"
            f"**Today's Date:** {today}\n\n"
            f"**Forecasting Window:** opens {question.open_time}, "
            f"closes {question.close_time}\n\n"
            f"**Details:**\n{cls._question_details(question)}\n\n"
        )

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = (
            _SYSTEM_FRAMING
            + self._header(question)
            + "Forecast the probability that this question resolves YES.\n\n"
            + _REASONING_INSTRUCTION
            + "The JSON must be in this exact format:\n"
            '```json\n{"yes": 0.XXX}\n```\n'
            "where 0.XXXX is a float between 0 and 1 representing P(yes)."
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        outcomes_str = ", ".join(f'"{option}"' for option in question.options)
        prompt = (
            _SYSTEM_FRAMING
            + self._header(question)
            + f"Forecast the probability for each outcome. Outcomes: [{outcomes_str}]\n\n"
            + _REASONING_INSTRUCTION
            + "The JSON must map each outcome to its probability. "
            "All values must be non-negative and sum to 1.0. Example format:\n"
            "```json\n"
            + "{\n"
            + "".join(f'  "{option}": 0.XXX,\n' for option in question.options)
            + "}\n```"
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        lo = (
            question.nominal_lower_bound
            if question.nominal_lower_bound is not None
            else question.lower_bound
        )
        hi = (
            question.nominal_upper_bound
            if question.nominal_upper_bound is not None
            else question.upper_bound
        )
        range_desc = f"{lo} to {hi}"
        if question.zero_point is not None:
            range_desc += " (logarithmic scale)"
        examples = [f"{lo + f * (hi - lo):g}" for f in _EXAMPLE_FRACTIONS]
        prompt = (
            _SYSTEM_FRAMING
            + self._header(question)
            + f"Forecast this continuous question. The scale ranges from {range_desc}.\n"
            f"{lower_bound_message} {upper_bound_message}\n\n"
            + _REASONING_INSTRUCTION
            + "Provide percentile estimates as numeric values on the question's scale. "
            'Use keys of the form "p<N>" where N is 1-99. '
            "Values must be strictly increasing. "
            "Set wide intervals - good forecasters account for unknown unknowns.\n"
            "Example:\n"
            "```json\n{\n"
            f"{self._json_example_body(examples)}\n"
            "}\n```"
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        lo_dt = question.lower_bound
        hi_dt = question.upper_bound
        range_desc = f"{lo_dt.date().isoformat()} to {hi_dt.date().isoformat()}"
        span = hi_dt - lo_dt
        examples = [
            f'"{(lo_dt + span * f).date().isoformat()}"' for f in _EXAMPLE_FRACTIONS
        ]
        prompt = (
            _SYSTEM_FRAMING
            + self._header(question)
            + f"Forecast this continuous question. The scale ranges from {range_desc}.\n"
            f"{lower_bound_message} {upper_bound_message}\n\n"
            + _REASONING_INSTRUCTION
            + "Provide percentile estimates as ISO date strings "
            '(e.g. "2025-06-15" or "2025-06-15T14:30:00" - time is optional). '
            'Use keys of the form "p<N>" where N is 1-99. '
            "Dates must be in strictly chronological order. "
            "Set wide intervals - good forecasters account for unknown unknowns.\n"
            "Example:\n"
            "```json\n{\n"
            f"{self._json_example_body(examples)}\n"
            "}\n```"
        )
        return await self._date_prompt_to_forecast(question, prompt)

    @staticmethod
    def _json_example_body(example_values: list[str]) -> str:
        return "\n".join(
            f'  "{key}": {value}{"," if index < len(_PERCENTILE_KEYS) - 1 else ""}'
            for index, (key, value) in enumerate(zip(_PERCENTILE_KEYS, example_values))
        )
