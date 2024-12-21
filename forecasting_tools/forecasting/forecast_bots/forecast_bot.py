import asyncio
import inspect
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Coroutine, cast

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecasting.helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ForecastReport,
    ReasonedPrediction,
    ResearchWithPredictions,
)
from forecasting_tools.forecasting.questions_and_reports.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    NumericDistribution,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecasting.questions_and_reports.report_organizer import (
    ReportOrganizer,
)
from forecasting_tools.util import async_batching

logger = logging.getLogger(__name__)


class ForecastBot(ABC):

    def __init__(
        self,
        *,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        skip_questions_that_error: bool = True,
    ) -> None:
        assert (
            research_reports_per_question > 0
        ), "Must run at least one research report"
        assert (
            predictions_per_research_report > 0
        ), "Must run at least one prediction"
        self.research_reports_per_question = research_reports_per_question
        self.predictions_per_research_report = predictions_per_research_report
        self.use_research_summary_to_forecast = (
            use_research_summary_to_forecast
        )
        self.folder_to_save_reports_to = folder_to_save_reports_to
        self.publish_reports_to_metaculus = publish_reports_to_metaculus
        self.skip_previously_forecasted_questions = (
            skip_previously_forecasted_questions
        )
        self.skip_questions_that_error = skip_questions_that_error

    def get_config(self) -> dict[str, str]:
        params = inspect.signature(self.__init__).parameters
        return {
            name: str(getattr(self, name))
            for name in params.keys()
            if name != "self" and name != "kwargs" and name != "args"
        }

    async def forecast_on_tournament(
        self,
        tournament_id: int,
    ) -> list[ForecastReport]:
        questions = MetaculusApi.get_all_open_questions_from_tournament(
            tournament_id
        )
        return await self.forecast_questions(questions)

    async def forecast_question(
        self,
        question: MetaculusQuestion,
    ) -> ForecastReport:
        assert (
            not self.skip_previously_forecasted_questions
        ), "Skipping questions is not supported for single question forecasts"
        reports = await self.forecast_questions([question])
        if len(reports) == 0:
            raise ValueError(
                "No reports found. There was probably an error lower in the system. Check logs."
            )
        return reports[0]

    async def forecast_questions(
        self,
        questions: list[MetaculusQuestion],
    ) -> list[ForecastReport]:
        if self.skip_previously_forecasted_questions:
            unforecasted_questions = [
                question
                for question in questions
                if not question.already_forecasted
            ]
            if len(questions) != len(unforecasted_questions):
                logger.info(
                    f"Skipping {len(questions) - len(unforecasted_questions)} previously forecasted questions"
                )
            questions = unforecasted_questions
        reports: list[ForecastReport] = []
        reports = await self._run_coroutines_and_error_if_configured(
            [self._run_individual_question(question) for question in questions]
        )
        if self.folder_to_save_reports_to:
            file_path = self.__create_file_path_to_save_to(questions)
            ForecastReport.save_object_list_to_file_path(reports, file_path)
        await self._run_coroutines_and_error_if_configured(
            [report.publish_report_to_metaculus() for report in reports]
        )
        return reports

    @abstractmethod
    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Researches a question and returns markdown report
        """
        raise NotImplementedError("Subclass must implement this method")

    async def summarize_research(
        self, question: MetaculusQuestion, research: str
    ) -> str:
        return f"{research[:2500]}..."

    async def _run_individual_question(
        self, question: MetaculusQuestion
    ) -> ForecastReport:
        with MonetaryCostManager() as cost_manager:
            start_time = time.time()
            prediction_tasks = [
                self._research_and_make_predictions(question)
                for _ in range(self.research_reports_per_question)
            ]
            research_with_predictions_units = (
                await self._run_coroutines_and_error_if_configured(
                    prediction_tasks
                )
            )
            if len(research_with_predictions_units) == 0:
                raise ValueError("All research reports/predictions failed")
            report_type = ReportOrganizer.get_report_type_for_question_type(
                type(question)
            )
            all_predictions = [
                reasoned_prediction.prediction_value
                for research_prediction_collection in research_with_predictions_units
                for reasoned_prediction in research_prediction_collection.predictions
            ]
            aggregated_prediction = await report_type.aggregate_predictions(
                all_predictions,
                question,
            )
            end_time = time.time()
            time_spent_in_minutes = (end_time - start_time) / 60
            final_cost = cost_manager.current_usage

        unified_explanation = self._create_unified_explanation(
            question,
            research_with_predictions_units,
            aggregated_prediction,
            final_cost,
            time_spent_in_minutes,
        )
        return report_type(
            question=question,
            prediction=aggregated_prediction,
            explanation=unified_explanation,
            price_estimate=final_cost,
            minutes_taken=time_spent_in_minutes,
        )

    async def _research_and_make_predictions(
        self, question: MetaculusQuestion
    ) -> ResearchWithPredictions:
        research = await self.run_research(question)
        summary_report = await self.summarize_research(question, research)
        research_to_use = (
            research
            if self.use_research_summary_to_forecast
            else summary_report
        )

        if isinstance(question, BinaryQuestion):
            forecast_function = lambda q, r: self._run_forecast_on_binary(q, r)
        elif isinstance(question, MultipleChoiceQuestion):
            forecast_function = (
                lambda q, r: self._run_forecast_on_multiple_choice(q, r)
            )
        elif isinstance(question, NumericQuestion):
            forecast_function = lambda q, r: self._run_forecast_on_numeric(
                q, r
            )
        elif isinstance(question, DateQuestion):
            raise NotImplementedError("Date questions not supported yet")
        else:
            raise ValueError(f"Unknown question type: {type(question)}")

        tasks = cast(
            list[Coroutine[Any, Any, ReasonedPrediction[Any]]],
            [
                forecast_function(question, research_to_use)
                for _ in range(self.predictions_per_research_report)
            ],
        )
        reasoned_predictions, _ = (
            async_batching.run_coroutines_while_removing_and_logging_exceptions(
                tasks
            )
        )
        if len(reasoned_predictions) == 0:
            raise ValueError("All predictions failed")

        return ResearchWithPredictions(
            research_report=research,
            summary_report=summary_report,
            predictions=reasoned_predictions,
        )

    async def _run_coroutines_and_error_if_configured(
        self, coroutines: list[Coroutine[Any, Any, Any]]
    ) -> list[Any]:
        if self.skip_questions_that_error:
            outputs, _ = (
                async_batching.run_coroutines_while_removing_and_logging_exceptions(
                    coroutines
                )
            )
            return outputs
        else:
            return await asyncio.gather(*coroutines)

    @abstractmethod
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        raise NotImplementedError("Subclass must implement this method")

    @abstractmethod
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        raise NotImplementedError("Subclass must implement this method")

    def _create_unified_explanation(
        self,
        question: MetaculusQuestion,
        research_prediction_collections: list[ResearchWithPredictions],
        aggregated_prediction: Any,
        final_cost: float,
        time_spent_in_minutes: float,
    ) -> str:
        report_type = ReportOrganizer.get_report_type_for_question_type(
            type(question)
        )

        all_summaries = []
        all_core_research = []
        all_forecaster_rationales = []
        for i, collection in enumerate(research_prediction_collections):
            summary = self._format_and_expand_research_summary(
                i + 1, report_type, collection
            )
            core_research_for_collection = self._format_main_research(
                i + 1, collection
            )
            forecaster_rationales_for_collection = (
                self._format_forecaster_rationales(i + 1, collection)
            )
            all_summaries.append(summary)
            all_core_research.append(core_research_for_collection)
            all_forecaster_rationales.append(
                forecaster_rationales_for_collection
            )

        combined_summaries = "\n".join(all_summaries)
        combined_research_reports = "\n".join(all_core_research)
        combined_rationales = "\n".join(all_forecaster_rationales)
        full_explanation_without_summary = clean_indents(
            f"""
            # SUMMARY
            *Question*: {question.question_text}
            *Final Prediction*: {report_type.make_readable_prediction(aggregated_prediction)}
            *Total Cost*: ${round(final_cost,2)}
            *Time Spent*: {round(time_spent_in_minutes, 2)} minutes

            {combined_summaries}

            # RESEARCH
            {combined_research_reports}

            # FORECASTS
            {combined_rationales}
            """
        )
        return full_explanation_without_summary

    @classmethod
    def _format_and_expand_research_summary(
        cls,
        report_number: int,
        report_type: type[ForecastReport],
        predicted_research: ResearchWithPredictions,
    ) -> str:
        forecaster_prediction_bullet_points = ""
        for j, forecast in enumerate(predicted_research.predictions):
            readable_prediction = report_type.make_readable_prediction(
                forecast.prediction_value
            )
            forecaster_prediction_bullet_points += (
                f"*Forecaster {j + 1}*: {readable_prediction}\n"
            )

        new_summary = clean_indents(
            f"""
            ## Report {report_number} Summary
            ### Forecasts
            {forecaster_prediction_bullet_points}

            ### Research Summary
            {predicted_research.summary_report}
            """
        )
        return new_summary

    @classmethod
    def _format_main_research(
        cls, report_number: int, predicted_research: ResearchWithPredictions
    ) -> str:
        markdown = predicted_research.research_report
        lines = markdown.split("\n")
        modified_content = ""

        # Add Report number to all headings
        for line in lines:
            if line.startswith("## "):
                line = f"## R{report_number}: {line[3:]}"
            modified_content += line + "\n"
        return modified_content

    def _format_forecaster_rationales(
        self, report_number: int, collection: ResearchWithPredictions
    ) -> str:
        rationales = []
        for j, forecast in enumerate(collection.predictions):
            new_rationale = clean_indents(
                f"""
                ## R{report_number}: Forecaster {j + 1} Reasoning
                {forecast.reasoning}
                """
            )
            rationales.append(new_rationale)
        return "\n".join(rationales)

    def __create_file_path_to_save_to(
        self, questions: list[MetaculusQuestion]
    ) -> str:
        assert (
            self.folder_to_save_reports_to is not None
        ), "Folder to save reports to is not set"
        now_as_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        folder_path = self.folder_to_save_reports_to

        if not folder_path.endswith("/"):
            folder_path += "/"

        return f"{folder_path}Forecasts-for-{now_as_string}--{len(questions)}-questions.json"
