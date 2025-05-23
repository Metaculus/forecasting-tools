import logging
from datetime import datetime

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


class CustomizableBot(ForecastBot):
    def __init__(
        self,
        prompt: str,
        research_snapshots: list[QuestionResearchSnapshot],
        research_type: ResearchType,
        today: datetime,
        exclude_from_config_dict: list[str] | None = ["research_snapshots"],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args, exclude_from_config_dict=exclude_from_config_dict, **kwargs
        )
        self.prompt = prompt
        self.research_snapshots = research_snapshots
        self.research_type = research_type
        self.today = today

        unique_questions = list(
            set([snapshot.question for snapshot in research_snapshots])
        )
        if len(unique_questions) != len(research_snapshots):
            raise ValueError("Research snapshots must have unique questions")

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
        prompt = self.prompt.format(
            question_text=question.question_text,
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            today=self.today.strftime("%Y-%m-%d"),
            research=research,
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        prediction: float = await structure_output(reasoning, float)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction}"
        )
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
