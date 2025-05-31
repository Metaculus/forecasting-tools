import logging

from pydantic import BaseModel, field_validator

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.benchmarking.prompt_data_models import PromptIdea
from forecasting_tools.benchmarking.question_research_snapshot import (
    QuestionResearchSnapshot,
    ResearchType,
)
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_helpers.structure_output import (
    structure_output,
)

logger = logging.getLogger(__name__)


class BinaryPrediction(BaseModel):
    prediction_in_decimal: float

    @field_validator("prediction_in_decimal")
    @classmethod
    def validate_prediction_range(cls, value: float) -> float:
        if value == 0:
            return 0.001
        if value == 1:
            return 0.999

        if value < 0.001:
            raise ValueError("Prediction must be at least 0.001")
        if value > 0.999:
            raise ValueError("Prediction must be at most 0.999")
        return value


class CustomizableBot(ForecastBot):
    def __init__(
        self,
        prompt: str,
        research_snapshots: list[QuestionResearchSnapshot],
        research_type: ResearchType,
        originating_idea: PromptIdea | None,
        parameters_to_exclude_from_config_dict: list[str] | None = [
            "research_snapshots"
        ],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            parameters_to_exclude_from_config_dict=parameters_to_exclude_from_config_dict,
            **kwargs,
        )
        self.prompt = prompt
        self.research_snapshots = research_snapshots
        self.research_type = research_type
        self.originating_idea = originating_idea  # As of May 26, 2025 This parameter is logged in the config for the bot, even if not used here.

        unique_questions = list(
            set(
                [
                    snapshot.question.question_text
                    for snapshot in research_snapshots
                ]
            )
        )
        if len(unique_questions) != len(research_snapshots):
            raise ValueError("Research snapshots must have unique questions")

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        return {
            "default": None,
            "summarizer": None,
            "researcher": None,
        }

    async def run_research(self, question: MetaculusQuestion) -> str:
        matching_snapshots = [
            snapshot
            for snapshot in self.research_snapshots
            if snapshot.question == question
        ]
        if len(matching_snapshots) != 1:
            raise ValueError(
                f"Expected 1 research snapshot for question {question.page_url}, got {len(matching_snapshots)}"
            )
        return matching_snapshots[0].get_research_for_type(self.research_type)

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        required_variables = [
            "{question_text}",
            "{resolution_criteria}",
            "{today}",
            "{research}",
        ]

        for required_variable in required_variables:
            if required_variable not in self.prompt:
                raise ValueError(
                    f"Prompt {self.prompt} does not contain {required_variable}"
                )
        prompt = self.prompt.format(
            question_text=question.question_text,
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            today=question.date_accessed.strftime("%Y-%m-%d"),
            research=research,
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")

        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction
        )
        prediction = binary_prediction.prediction_in_decimal

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction}"
        )
        if prediction >= 1:
            prediction = 0.999
        if prediction <= 0:
            prediction = 0.001
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        raise NotImplementedError()

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        raise NotImplementedError()
