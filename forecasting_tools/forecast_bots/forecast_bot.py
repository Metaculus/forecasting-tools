import asyncio
import inspect
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Coroutine, Sequence, TypeVar, cast, overload

from exceptiongroup import ExceptionGroup
from pydantic import BaseModel

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.data_models.data_organizer import (
    DataOrganizer,
    PredictionTypes,
)
from forecasting_tools.data_models.forecast_report import (
    ForecastReport,
    ReasonedPrediction,
    ResearchWithPredictions,
)
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi

T = TypeVar("T")

logger = logging.getLogger(__name__)


class ScratchPad(BaseModel):
    """
    Context object that is available while forecasting on a question
    You can keep tally's, todos, notes, or other organizational information here
    that other parts of the forecasting bot needs to access

    You will want to inherit from this class to add additional attributes
    """

    question: MetaculusQuestion
    note_entries: dict[str, str] = {}


class ForecastBot(ABC):
    """
    Base class for all forecasting bots.
    """

    def __init__(
        self,
        *,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
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
        self._scratch_pads: list[ScratchPad] = []
        self._scratch_pad_lock = asyncio.Lock()

    def get_config(self) -> dict[str, str]:
        params = inspect.signature(self.__init__).parameters
        return {
            name: str(getattr(self, name))
            for name in params.keys()
            if name != "self" and name != "kwargs" and name != "args"
        }

    @overload
    async def forecast_on_tournament(
        self,
        tournament_id: int | str,
        return_exceptions: bool = False,
    ) -> list[ForecastReport]: ...

    @overload
    async def forecast_on_tournament(
        self,
        tournament_id: int | str,
        return_exceptions: bool = True,
    ) -> list[ForecastReport | BaseException]: ...

    async def forecast_on_tournament(
        self,
        tournament_id: int | str,
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]:
        questions = MetaculusApi.get_all_open_questions_from_tournament(
            tournament_id
        )
        return await self.forecast_questions(questions, return_exceptions)

    @overload
    async def forecast_question(
        self,
        question: MetaculusQuestion,
        return_exceptions: bool = False,
    ) -> ForecastReport: ...

    @overload
    async def forecast_question(
        self,
        question: MetaculusQuestion,
        return_exceptions: bool = True,
    ) -> ForecastReport | BaseException: ...

    async def forecast_question(
        self,
        question: MetaculusQuestion,
        return_exceptions: bool = False,
    ) -> ForecastReport | BaseException:
        assert (
            not self.skip_previously_forecasted_questions
        ), "Skipping questions is not supported for single question forecasts"
        reports = await self.forecast_questions([question], return_exceptions)
        assert len(reports) == 1, f"Expected 1 report, got {len(reports)}"
        return reports[0]

    @overload
    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: bool = False,
    ) -> list[ForecastReport]: ...

    @overload
    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: bool = True,
    ) -> list[ForecastReport | BaseException]: ...

    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]:
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
        reports: list[ForecastReport | BaseException] = []
        reports = await asyncio.gather(
            *[
                self._run_individual_question_with_error_propagation(question)
                for question in questions
            ],
            return_exceptions=return_exceptions,
        )
        if self.folder_to_save_reports_to:
            non_exception_reports = [
                report
                for report in reports
                if not isinstance(report, BaseException)
            ]
            questions_as_list = list(questions)
            file_path = self._create_file_path_to_save_to(questions_as_list)
            ForecastReport.save_object_list_to_file_path(
                non_exception_reports, file_path
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
        logger.info(f"Summarizing research for question: {question.page_url}")
        default_summary_size = 2500
        default_summary = f"{research[:default_summary_size]}..."

        if len(research) < default_summary_size:
            return research

        if os.getenv("OPENAI_API_KEY"):
            model = GeneralLlm(model="gpt-4o-mini", temperature=0.3)
        elif os.getenv("METACULUS_TOKEN"):
            model = GeneralLlm(model="metaculus/gpt-4o-mini", temperature=0.3)
        else:
            return default_summary

        try:
            prompt = clean_indents(
                f"""
                Please summarize the following research in 1-2 paragraphs. The report tries to help answer the following question:
                {question.question_text}

                Only summarize the research. Do not answer the question. Just say what the research says w/o any opinions added.

                The research is:
                {research}
                """
            )
            summary = await model.invoke(prompt)
            return summary
        except Exception as e:
            logger.debug(
                f"Could not summarize research. Defaulting to first {default_summary_size} characters: {e}"
            )
            return default_summary

    async def _run_individual_question_with_error_propagation(
        self, question: MetaculusQuestion
    ) -> ForecastReport:
        try:
            return await self._run_individual_question(question)
        except Exception as e:
            error_message = (
                f"Error while processing question url: '{question.page_url}'"
            )
            logger.error(f"{error_message}: {e}")
            self._reraise_exception_with_prepended_message(e, error_message)
            assert (
                False
            ), "This is to satisfy type checker. The previous function should raise an exception"

    async def _run_individual_question(
        self, question: MetaculusQuestion
    ) -> ForecastReport:
        scratchpad = await self._initialize_scratchpad(question)
        async with self._scratch_pad_lock:
            self._scratch_pads.append(scratchpad)
        with MonetaryCostManager() as cost_manager:
            start_time = time.time()
            prediction_tasks = [
                self._research_and_make_predictions(question)
                for _ in range(self.research_reports_per_question)
            ]
            valid_prediction_set, research_errors, exception_group = (
                await self._gather_results_and_exceptions(prediction_tasks)
            )
            if research_errors:
                logger.warning(
                    f"Encountered errors while researching: {research_errors}"
                )
            if len(valid_prediction_set) == 0:
                assert exception_group, "Exception group should not be None"
                self._reraise_exception_with_prepended_message(
                    exception_group,
                    f"All {self.research_reports_per_question} research reports/predictions failed",
                )
            prediction_errors = [
                error
                for prediction_set in valid_prediction_set
                for error in prediction_set.errors
            ]
            all_errors = research_errors + prediction_errors

            report_type = DataOrganizer.get_report_type_for_question_type(
                type(question)
            )
            all_predictions = [
                reasoned_prediction.prediction_value
                for research_prediction_collection in valid_prediction_set
                for reasoned_prediction in research_prediction_collection.predictions
            ]
            aggregated_prediction = await self._aggregate_predictions(
                all_predictions,
                question,
            )
            end_time = time.time()
            time_spent_in_minutes = (end_time - start_time) / 60
            final_cost = cost_manager.current_usage

        unified_explanation = self._create_unified_explanation(
            question,
            valid_prediction_set,
            aggregated_prediction,
            final_cost,
            time_spent_in_minutes,
        )
        report = report_type(
            question=question,
            prediction=aggregated_prediction,
            explanation=unified_explanation,
            price_estimate=final_cost,
            minutes_taken=time_spent_in_minutes,
            errors=all_errors,
        )
        if self.publish_reports_to_metaculus:
            await report.publish_report_to_metaculus()
        await self._remove_scratchpad(question)
        return report

    async def _aggregate_predictions(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        if not predictions:
            raise ValueError("Cannot aggregate empty list of predictions")
        prediction_types = {type(pred) for pred in predictions}
        if len(prediction_types) > 1:
            raise TypeError(
                f"All predictions must be of the same type. Found types: {prediction_types}"
            )
        report_type = DataOrganizer.get_report_type_for_question_type(
            type(question)
        )
        aggregate = await report_type.aggregate_predictions(
            predictions, question
        )
        return aggregate

    async def _research_and_make_predictions(
        self, question: MetaculusQuestion
    ) -> ResearchWithPredictions[PredictionTypes]:
        research = await self.run_research(question)
        summary_report = await self.summarize_research(question, research)
        research_to_use = (
            summary_report
            if self.use_research_summary_to_forecast
            else research
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
        valid_predictions, errors, exception_group = (
            await self._gather_results_and_exceptions(tasks)
        )
        if errors:
            logger.warning(f"Encountered errors while predicting: {errors}")
        if len(valid_predictions) == 0:
            assert exception_group, "Exception group should not be None"
            self._reraise_exception_with_prepended_message(
                exception_group,
                "Error while running research and predictions",
            )
        return ResearchWithPredictions(
            research_report=research,
            summary_report=summary_report,
            errors=errors,
            predictions=valid_predictions,
        )

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
        aggregated_prediction: PredictionTypes,
        final_cost: float,
        time_spent_in_minutes: float,
    ) -> str:
        report_type = DataOrganizer.get_report_type_for_question_type(
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

        for line in lines:
            if line.startswith("#"):
                heading_level = len(line) - len(line.lstrip("#"))
                content = line[heading_level:].lstrip()
                new_heading_level = max(3, heading_level + 2)
                line = f"{'#' * new_heading_level} {content}"
            modified_content += line + "\n"
        final_content = (
            f"## Report {report_number} Research\n{modified_content}"
        )
        return final_content

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

    def _create_file_path_to_save_to(
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

    async def _gather_results_and_exceptions(
        self, coroutines: list[Coroutine[Any, Any, T]]
    ) -> tuple[list[T], list[str], ExceptionGroup | None]:
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        valid_results = [
            result
            for result in results
            if not isinstance(result, BaseException)
        ]
        error_messages = []
        exceptions = []
        for error in results:
            if isinstance(error, BaseException):
                error_messages.append(f"{error.__class__.__name__}: {error}")
                exceptions.append(error)
        exception_group = (
            ExceptionGroup(f"Errors: {error_messages}", exceptions)
            if exceptions
            else None
        )
        return valid_results, error_messages, exception_group

    def _reraise_exception_with_prepended_message(
        self, exception: Exception | ExceptionGroup, message: str
    ) -> None:
        if isinstance(exception, ExceptionGroup):
            raise ExceptionGroup(
                f"{message}: {exception.message}", exception.exceptions
            )
        else:
            raise type(exception)(f"{message}: {exception}") from exception

    async def _initialize_scratchpad(
        self, question: MetaculusQuestion
    ) -> ScratchPad:
        new_scratchpad = ScratchPad(question=question)
        return new_scratchpad

    async def _remove_scratchpad(self, question: MetaculusQuestion) -> None:
        async with self._scratch_pad_lock:
            self._scratch_pads = [
                scratchpad
                for scratchpad in self._scratch_pads
                if scratchpad.question != question
            ]

    async def _get_scratchpad(self, question: MetaculusQuestion) -> ScratchPad:
        async with self._scratch_pad_lock:
            for scratchpad in self._scratch_pads:
                if scratchpad.question == question:
                    return scratchpad
        raise ValueError(
            f"No scratchpad found for question: ID: {question.id_of_post} Text: {question.question_text}"
        )

    @staticmethod
    def log_report_summary(
        forecast_reports: list[ForecastReport | BaseException],
    ) -> None:
        valid_reports = [
            report
            for report in forecast_reports
            if isinstance(report, ForecastReport)
        ]
        exceptions = [
            report
            for report in forecast_reports
            if isinstance(report, BaseException)
        ]
        minor_exceptions = [
            report.errors for report in valid_reports if report.errors
        ]

        for report in valid_reports:
            question_summary = clean_indents(
                f"""
                URL: {report.question.page_url}
                Errors: {report.errors}
                Summary:
                {report.summary}
                ---------------------------------------------------------
            """
            )
            logger.info(question_summary)

        if exceptions:
            raise RuntimeError(
                f"{len(exceptions)} errors occurred while forecasting: {exceptions}"
            )
        if minor_exceptions:
            logger.error(
                f"{len(minor_exceptions)} minor exceptions occurred while forecasting: {minor_exceptions}"
            )
