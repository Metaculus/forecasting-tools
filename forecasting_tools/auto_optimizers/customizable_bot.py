from __future__ import annotations

import copy
import logging
from enum import Enum

from pydantic import BaseModel, field_validator

from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AgentTool,
    AiAgent,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.auto_optimizers.prompt_data_models import (
    PromptIdea,
    ResearchTool,
)
from forecasting_tools.auto_optimizers.question_plus_research import (
    QuestionPlusResearch,
    ResearchType,
)
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher
from forecasting_tools.helpers.structure_output import structure_output

logger = logging.getLogger(__name__)


class BinaryPrediction(BaseModel):
    prediction_in_decimal: float

    @field_validator("prediction_in_decimal")
    @classmethod
    def validate_prediction_range(cls, value: float) -> float:
        if 0.001 <= value <= 0.999:
            return value
        elif 0 <= value < 0.001:
            return 0.001
        elif 0.999 < value <= 1:
            return 0.999
        else:
            raise ValueError("Prediction must be between 0 and 1")


class ToolUsageTracker:
    def __init__(self, research_tools: list[ResearchTool]) -> None:
        self.tool_usage: dict[str, int] = {}
        self.tool_limits: dict[str, int | None] = {}

        for research_tool in research_tools:
            tool_name = research_tool.tool.name
            self.tool_usage[tool_name] = 0
            self.tool_limits[tool_name] = research_tool.max_calls

    def increment_usage(self, tool_name: str) -> None:
        if tool_name not in self.tool_usage:
            raise ValueError(f"Tool {tool_name} not found in usage tracker")

        self.tool_usage[tool_name] += 1
        current_usage = self.tool_usage[tool_name]
        max_calls = self.tool_limits[tool_name]

        if max_calls is not None and current_usage > max_calls:
            raise RuntimeError(
                f"Tool {tool_name} has exceeded its maximum calls limit of {max_calls}. "
                f"Current usage: {current_usage}"
            )


class PresetResearchStrategy(Enum):
    SEARCH_ASKNEWS_WITH_QUESTION_TEXT = "SEARCH_ASKNEWS_WITH_QUESTION_TEXT"

    async def run_research(self, question: MetaculusQuestion) -> str:
        logger.debug(f"Running research strategy: {self}")
        if self == PresetResearchStrategy.SEARCH_ASKNEWS_WITH_QUESTION_TEXT:
            return await AskNewsSearcher().get_formatted_news_async(
                question.question_text
            )
        else:
            raise ValueError(f"Unknown research type: {self}")

    @classmethod
    def find_matching_strategy(
        cls, research_strategy_id: str
    ) -> PresetResearchStrategy | None:
        matching_strategy = [
            strategy
            for strategy in PresetResearchStrategy
            if strategy.value == research_strategy_id.strip()
        ]
        if len(matching_strategy) > 1:
            raise ValueError(
                f"Multiple matching research strategies found: {matching_strategy}"
            )
        if len(matching_strategy) == 0:
            return None
        return matching_strategy[0]


class CustomizableBot(ForecastBot):
    """
    A customizable bot that can be used to forecast questions.

    The flow goes:
    1. The bot is given a question
    2. Research is performed
        a. If the question is in the research snapshots, the bot uses the research snapshot.
        b. If the question is not in the research snapshots, the bot uses research prompt to agentically run the research tools.
    3. The bot forecasts the question using the reasoning prompt
    4. The bot returns the forecast

    See ForecastBot for more details.
    Comment Last updated: July 10, 2025
    """

    REQUIRED_REASONING_PROMPT_VARIABLES = [
        "{question_text}",
        "{resolution_criteria}",
        "{today}",
        "{research}",
        "{background_info}",
        "{fine_print}",
    ]
    REQUIRED_RESEARCH_PROMPT_VARIABLES = [
        "{question_text}",
    ]
    OPTIONAL_RESEARCH_PROMPT_VARIABLES = [
        "{resolution_criteria}",
        "{today}",
        "{background_info}",
        "{fine_print}",
    ]
    ALL_POSSIBLE_VARIABLES = list(
        set(
            REQUIRED_REASONING_PROMPT_VARIABLES
            + OPTIONAL_RESEARCH_PROMPT_VARIABLES
            + REQUIRED_RESEARCH_PROMPT_VARIABLES
        )
    )
    RESEARCH_REASONING_SPLIT_STRING = (
        "<<<< RESEARCH PROMPT ABOVE, REASONING PROMPT BELOW >>>>"
    )

    def __init__(
        self,
        research_prompt: str,
        reasoning_prompt: str,
        research_tools: list[ResearchTool],
        cached_research: list[QuestionPlusResearch] | None,
        cached_research_type: ResearchType | None,
        originating_idea: PromptIdea | None,
        parameters_to_exclude_from_config_dict: list[str] | None = [
            "research_snapshots",
            "cached_research",
            "cached_research_type",
        ],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            parameters_to_exclude_from_config_dict=parameters_to_exclude_from_config_dict,
            **kwargs,
        )
        self.reasoning_prompt = reasoning_prompt
        self.research_prompt = research_prompt
        self.research_tools = research_tools
        self.research_type = cached_research_type
        self.originating_idea = originating_idea  # As of May 26, 2025 This parameter is logged in the config for the bot, even if not used here.

        self._validate_cache(cached_research)

        if not self.get_llm("researcher"):
            raise ValueError("Research LLM must be provided")

        self.cached_research = cached_research or []

        self.validate_research_prompt(self.research_prompt)
        self.validate_reasoning_prompt(self.reasoning_prompt)

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        return {
            "default": None,
            "summarizer": None,
            "researcher": None,
        }

    def _create_tracked_tools(self, usage_tracker: ToolUsageTracker) -> list[AgentTool]:
        tracked_tools = []

        for research_tool in self.research_tools:
            original_tool = research_tool.tool
            tracked_tool = copy.deepcopy(original_tool)
            original_on_invoke = tracked_tool.on_invoke_tool

            async def wrapped_on_invoke_tool(
                ctx,
                input_str,
                tool_name=original_tool.name,
                original_func=original_on_invoke,
            ) -> str:
                try:
                    usage_tracker.increment_usage(tool_name)
                    return await original_func(ctx, input_str)
                except Exception as e:
                    logger.warning(f"Error calling tool {tool_name}: {e}")
                    return f"Error calling tool {tool_name}: {e}"

            tracked_tool.on_invoke_tool = wrapped_on_invoke_tool

            tracked_tools.append(tracked_tool)

        return tracked_tools

    async def run_research(self, question: MetaculusQuestion) -> str:
        try:
            return self._get_cached_research(question)
        except ValueError:
            pass

        matching_strategy = PresetResearchStrategy.find_matching_strategy(
            self.research_prompt
        )
        if matching_strategy is not None:
            return await matching_strategy.run_research(question)

        research = await self._run_research_with_tools(question)
        self._handle_if_metaculus_cp_was_used(research)
        return research

    def _get_cached_research(self, question: MetaculusQuestion) -> str:
        matching_snapshots = [
            snapshot
            for snapshot in self.cached_research
            if snapshot.question == question
        ]
        if len(matching_snapshots) > 1:
            raise ValueError(
                f"Expected 1 research snapshot for question {question.page_url}, got {len(matching_snapshots)}"
            )
        if len(matching_snapshots) == 1:
            assert self.research_type is not None
            return matching_snapshots[0].get_research_for_type(self.research_type)
        else:
            raise ValueError(
                f"No cached research found for question {question.page_url}"
            )

    async def _run_research_with_tools(self, question: MetaculusQuestion) -> str:
        research_llm = self.get_llm("researcher")
        if not isinstance(research_llm, str):
            raise ValueError("Research LLM must be a string model name")
        if not self.research_tools:
            raise ValueError(
                "Attempted to use research tools but no research tools available"
            )

        formatted_prompt = self._format_prompt_with_question(
            self.research_prompt, question, None
        )

        usage_tracker = ToolUsageTracker(self.research_tools)
        tracked_tools = self._create_tracked_tools(usage_tracker)

        agent = AiAgent(
            name="Research Agent",
            instructions=formatted_prompt,
            model=AgentSdkLlm(model=research_llm),
            tools=tracked_tools,  # type: ignore
            handoffs=[],
        )

        result = await AgentRunner.run(
            agent,
            "Please follow your instructions given to you.",
            max_turns=25,
        )
        final_output = result.final_output
        if not isinstance(final_output, str):
            raise ValueError(
                f"Expected final output to be a string, got {type(final_output)}"
            )

        return final_output

    def _handle_if_metaculus_cp_was_used(self, research: str) -> None:
        if "Metaculus" in research:
            logger.warning(
                f"There may be a prediction from Metaculus used in a question's research: {research}"
            )

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = self._format_prompt_with_question(
            self.reasoning_prompt, question, research
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")

        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction
        )
        prediction = binary_prediction.prediction_in_decimal

        logger.info(f"Forecasted URL {question.page_url} with prediction: {prediction}")
        if prediction >= 1:
            prediction = 0.999
        if prediction <= 0:
            prediction = 0.001
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        raise NotImplementedError()

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        raise NotImplementedError()

    @classmethod
    def _format_prompt_with_question(
        cls, prompt: str, question: MetaculusQuestion, research: str | None
    ) -> str:
        combined_question_info = (
            question.question_text
            + (question.background_info or "")
            + (question.resolution_criteria or "")
            + (question.fine_print or "")
        )
        for variable in cls.ALL_POSSIBLE_VARIABLES:
            if f"{{{variable}}}" in combined_question_info:
                raise ValueError(
                    f"There is a template variable in the question itself: {variable}. Combined text: {combined_question_info}"
                )

        formatted_prompt = prompt.replace("{question_text}", question.question_text)
        formatted_prompt = formatted_prompt.replace(
            "{background_info}", question.background_info or "None"
        )
        formatted_prompt = formatted_prompt.replace(
            "{resolution_criteria}", question.resolution_criteria or "None"
        )
        formatted_prompt = formatted_prompt.replace(
            "{fine_print}", question.fine_print or "None"
        )
        formatted_prompt = formatted_prompt.replace(
            "{today}", question.date_accessed.strftime("%Y-%m-%d")
        )
        if research is not None:
            formatted_prompt = formatted_prompt.replace("{research}", research)
        return formatted_prompt

    @classmethod
    def split_combined_research_reasoning_prompt(
        cls,
        combined_prompt: str,
    ) -> tuple[str, str]:
        """
        Utility function to split a combined research and reasoning prompt into a separate research prompt and a reasoning prompt.
        Also validates that the template variables are present in the prompts.
        """
        if cls.RESEARCH_REASONING_SPLIT_STRING not in combined_prompt:
            raise ValueError(
                f"Combined prompt does not contain {cls.RESEARCH_REASONING_SPLIT_STRING}. Prompt: {combined_prompt}"
            )
        research_prompt, reasoning_prompt = combined_prompt.split(
            CustomizableBot.RESEARCH_REASONING_SPLIT_STRING
        )
        cls.validate_research_prompt(research_prompt)
        cls.validate_reasoning_prompt(reasoning_prompt)
        return research_prompt, reasoning_prompt

    @classmethod
    def combine_research_reasoning_prompt(
        cls, research_prompt: str, reasoning_prompt: str
    ) -> str:
        return (
            f"{research_prompt}{cls.RESEARCH_REASONING_SPLIT_STRING}{reasoning_prompt}"
        )

    @classmethod
    def validate_combined_research_reasoning_prompt(cls, combined_prompt: str) -> None:

        research_prompt, reasoning_prompt = (
            cls.split_combined_research_reasoning_prompt(combined_prompt)
        )
        cls.validate_research_prompt(research_prompt)
        cls.validate_reasoning_prompt(reasoning_prompt)

    @classmethod
    def validate_research_prompt(cls, research_prompt: str) -> None:
        if PresetResearchStrategy.find_matching_strategy(research_prompt) is not None:
            return

        cls._validate_template_variables(
            research_prompt,
            required_variables=CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES,
        )
        if "{research}" in research_prompt:
            raise ValueError(
                f"Research prompt must not contain {{research}} variable since research has not been found yet. Research prompt: {research_prompt}"
            )

    @classmethod
    def validate_reasoning_prompt(cls, reasoning_prompt: str) -> None:
        cls._validate_template_variables(
            reasoning_prompt,
            required_variables=CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES,
        )
        cls._validate_no_duplicate_variables(reasoning_prompt, "{research}")

    @classmethod
    def _validate_template_variables(
        cls,
        prompt: str,
        required_variables: list[str],
    ) -> None:
        missing_vars = [var for var in required_variables if var not in prompt]
        if missing_vars:
            raise ValueError(
                f"Generated prompt is missing template variables: {missing_vars}. Prompt: {prompt}"
            )

    @classmethod
    def _validate_no_duplicate_variables(
        cls, prompt: str, specific_variable: str | None = None
    ) -> None:
        import re

        variable_pattern = r"\{[^}]+\}"
        variables = re.findall(variable_pattern, prompt)
        variable_counts = {}

        for variable in variables:
            variable_counts[variable] = variable_counts.get(variable, 0) + 1

        if specific_variable is not None:
            if (
                specific_variable in variable_counts
                and variable_counts[specific_variable] > 1
            ):
                raise ValueError(
                    f"Prompt contains duplicate template variable '{specific_variable}' (found {variable_counts[specific_variable]} times). Prompt: {prompt}"
                )
        else:
            duplicates = [var for var, count in variable_counts.items() if count > 1]
            if duplicates:
                raise ValueError(
                    f"Prompt contains duplicate template variables: {duplicates}. Prompt: {prompt}"
                )

    def _validate_cache(
        self, cached_research: list[QuestionPlusResearch] | None
    ) -> None:
        if cached_research:
            unique_questions = list(
                set([snapshot.question.question_text for snapshot in cached_research])
            )
            if len(unique_questions) != len(cached_research):
                raise ValueError("Research snapshots must have unique questions")
            if self.research_type is None:
                raise ValueError(
                    "Research type must be provided if cached research is provided"
                )
        else:
            if self.research_type is not None:
                raise ValueError(
                    "Research type must be None if cached research is None"
                )
