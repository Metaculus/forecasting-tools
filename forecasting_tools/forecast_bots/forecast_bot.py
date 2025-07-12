import asyncio
import inspect
import json
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Coroutine, Literal, Sequence, TypeVar, cast, overload

from exceptiongroup import ExceptionGroup
from pydantic import BaseModel

from forecasting_tools.ai_models.agent_wrappers import general_trace_or_span
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.data_models.data_organizer import DataOrganizer, PredictionTypes
from forecasting_tools.data_models.forecast_report import (
    ForecastReport,
    ReasonedPrediction,
    ResearchWithPredictions,
)
from forecasting_tools.data_models.markdown_tree import MarkdownTree
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.helpers.metaculus_api import MetaculusApi

T = TypeVar("T")

logger = logging.getLogger(__name__)


class Notepad(BaseModel):
    """
    Context object that is available while forecasting on a question and that persists
    across multiple forecasts on the same question.

    You can keep tally's, todos, notes, or other organizational information here
    that other parts of the forecasting bot needs to access

    You can inherit from this class to add additional attributes

    A notepad for a question within a forecast bot can be obtained by calling `self._get_notepad(question)`
    """

    question: MetaculusQuestion
    total_research_reports_attempted: int = 0
    total_predictions_attempted: int = 0
    note_entries: dict[str, Any] = {}


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
        llms: (
            dict[str, str | GeneralLlm | None] | None
        ) = None,  # Default LLMs are used if llms is set to None
        enable_summarize_research: bool = True,
        parameters_to_exclude_from_config_dict: list[str] | None = None,
    ) -> None:
        assert (
            research_reports_per_question > 0
        ), "Must run at least one research report"
        assert predictions_per_research_report > 0, "Must run at least one prediction"
        if use_research_summary_to_forecast and not enable_summarize_research:
            raise ValueError(
                "Cannot use research summary to forecast if summarize_research is False"
            )
        self.research_reports_per_question = research_reports_per_question
        self.predictions_per_research_report = predictions_per_research_report
        self.use_research_summary_to_forecast = use_research_summary_to_forecast
        self.folder_to_save_reports_to = folder_to_save_reports_to
        self.publish_reports_to_metaculus = publish_reports_to_metaculus
        self.skip_previously_forecasted_questions = skip_previously_forecasted_questions
        self.parameters_to_exclude_from_config_dict = (
            parameters_to_exclude_from_config_dict or []
        )
        self.enable_summarize_research = enable_summarize_research
        self._note_pads: list[Notepad] = []
        self._note_pad_lock = asyncio.Lock()
        self._llms = llms or self._llm_config_defaults()

        for purpose, llm in self._llm_config_defaults().items():
            if purpose not in self._llms:
                logger.warning(
                    f"User forgot to set an llm for purpose: '{purpose}'. Using default llm: '{llm.model if isinstance(llm, GeneralLlm) else llm}'"
                )
                self._llms[purpose] = llm

        for purpose, llm in self._llms.items():
            if purpose not in self._llm_config_defaults():
                logger.warning(
                    f"There is no default for llm: '{purpose}'."
                    f"Please override and add it to the {self._llm_config_defaults.__name__} method"
                )

        logger.debug(f"LLMs at initialization for bot are: {self.make_llm_dict()}")

    @overload
    async def forecast_on_tournament(
        self,
        tournament_id: int | str,
        return_exceptions: Literal[False] = False,
    ) -> list[ForecastReport]: ...

    @overload
    async def forecast_on_tournament(
        self,
        tournament_id: int | str,
        return_exceptions: Literal[True] = True,
    ) -> list[ForecastReport | BaseException]: ...

    @overload
    async def forecast_on_tournament(
        self,
        tournament_id: int | str,
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]: ...

    async def forecast_on_tournament(
        self,
        tournament_id: int | str,
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]:
        questions = MetaculusApi.get_all_open_questions_from_tournament(tournament_id)
        return await self.forecast_questions(questions, return_exceptions)

    @overload
    async def forecast_question(
        self,
        question: MetaculusQuestion,
        return_exceptions: Literal[False] = False,
    ) -> ForecastReport: ...

    @overload
    async def forecast_question(
        self,
        question: MetaculusQuestion,
        return_exceptions: Literal[True] = True,
    ) -> ForecastReport | BaseException: ...

    @overload
    async def forecast_question(
        self,
        question: MetaculusQuestion,
        return_exceptions: bool = False,
    ) -> ForecastReport | BaseException: ...

    async def forecast_question(
        self,
        question: MetaculusQuestion,
        return_exceptions: bool = False,
    ) -> ForecastReport | BaseException:
        if self.skip_previously_forecasted_questions:
            logger.warning(
                "Setting skip_previously_forecasted_questions to True might not be intended if forecasting one question at a time"
            )
        reports = await self.forecast_questions([question], return_exceptions)
        assert len(reports) == 1, f"Expected 1 report, got {len(reports)}"
        return reports[0]

    @overload
    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: Literal[False] = False,
    ) -> list[ForecastReport]: ...

    @overload
    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: Literal[True] = True,
    ) -> list[ForecastReport | BaseException]: ...

    @overload
    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]: ...

    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]:
        if self.skip_previously_forecasted_questions:
            unforecasted_questions = [
                question for question in questions if not question.already_forecasted
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
                report for report in reports if not isinstance(report, BaseException)
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
        raise NotImplementedError("Subclass should implement this method")

    async def summarize_research(
        self, question: MetaculusQuestion, research: str
    ) -> str:
        if not self.enable_summarize_research:
            return "Summarize research was disabled for this run"

        try:
            logger.info(f"Summarizing research for question: {question.page_url}")
            model = self.get_llm("summarizer", "llm")
            prompt = clean_indents(
                f"""
                Please summarize the following research in 1-2 paragraphs. The research tries to help answer the following question:
                {question.question_text}

                Only summarize the research. Do not answer the question. Just say what the research says w/o any opinions added.
                At the end mention what websites/sources were used (and copy links verbatim if possible)

                The research is:
                {research}
                """
            )
            summary = await model.invoke(prompt)
            return summary
        except Exception as e:
            if self.use_research_summary_to_forecast:
                raise e  # If the summary is needed for research, then returning the normal error message as the research will confuse the AI
            logger.warning(f"Could not summarize research. {e}")
            return f"{e.__class__.__name__} exception while summarizing research"

    def get_config(self) -> dict[str, Any]:
        params = inspect.signature(self.__init__).parameters

        config: dict[str, Any] = {}
        for name in params.keys():
            if (
                name == "self"
                or name == "kwargs"
                or name == "args"
                or name == "llms"
                or name in self.parameters_to_exclude_from_config_dict
            ):
                continue
            value = getattr(self, name)
            try:
                if isinstance(value, BaseModel):
                    config[name] = value.model_dump()
                elif (
                    isinstance(value, list)
                    and len(value) > 0
                    and isinstance(value[0], BaseModel)
                ):
                    config[name] = [item.model_dump() for item in value]
                else:
                    json.dumps({name: value})
                    config[name] = value
            except Exception:
                config[name] = str(value)

        llm_dict = self.make_llm_dict()
        config["llms"] = llm_dict
        return config

    def make_llm_dict(self) -> dict[str, str | dict[str, Any] | None]:
        llm_dict: dict[str, str | dict[str, Any] | None] = {}
        for key, value in self._llms.items():
            if isinstance(value, GeneralLlm):
                llm_dict[key] = value.to_dict()
            else:
                llm_dict[key] = value
        return llm_dict

    async def _run_individual_question_with_error_propagation(
        self, question: MetaculusQuestion
    ) -> ForecastReport:
        with general_trace_or_span(
            f"{self.__class__.__name__} - Question: {question.page_url}"
        ):
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
        notepad = await self._initialize_notepad(question)
        async with self._note_pad_lock:
            self._note_pads.append(notepad)
        with MonetaryCostManager() as cost_manager:
            start_time = time.time()
            prediction_tasks = [
                self._research_and_make_predictions(question)
                for _ in range(self.research_reports_per_question)
            ]
            valid_prediction_set, research_errors, exception_group = (
                await self._gather_results_and_exceptions(prediction_tasks)  # type: ignore
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
        await self._remove_notepad(question)
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
            logger.warning(
                f"Predictions have different types. Types: {prediction_types}. "
                "This may cause problems when aggregating."
            )
        report_type = DataOrganizer.get_report_type_for_question_type(type(question))
        aggregate = await report_type.aggregate_predictions(predictions, question)
        return aggregate

    async def _research_and_make_predictions(
        self, question: MetaculusQuestion
    ) -> ResearchWithPredictions[PredictionTypes]:
        notepad = await self._get_notepad(question)
        notepad.total_research_reports_attempted += 1
        research = await self.run_research(question)
        summary_report = await self.summarize_research(question, research)
        research_to_use = (
            summary_report if self.use_research_summary_to_forecast else research
        )

        tasks = cast(
            list[Coroutine[Any, Any, ReasonedPrediction[Any]]],
            [
                self._make_prediction(question, research_to_use)
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

    async def _make_prediction(
        self, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction[PredictionTypes]:
        notepad = await self._get_notepad(question)
        notepad.total_predictions_attempted += 1

        if isinstance(question, BinaryQuestion):
            forecast_function = lambda q, r: self._run_forecast_on_binary(q, r)
        elif isinstance(question, MultipleChoiceQuestion):
            forecast_function = lambda q, r: self._run_forecast_on_multiple_choice(q, r)
        elif isinstance(question, NumericQuestion):
            forecast_function = lambda q, r: self._run_forecast_on_numeric(q, r)
        elif isinstance(question, DateQuestion):
            raise NotImplementedError("Date questions not supported yet")
        else:
            raise ValueError(f"Unknown question type: {type(question)}")

        prediction = await forecast_function(question, research)
        return prediction  # type: ignore

    @abstractmethod
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        raise NotImplementedError("Subclass should implement this method")

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
        report_type = DataOrganizer.get_report_type_for_question_type(type(question))

        all_summaries = []
        all_core_research = []
        all_forecaster_rationales = []
        for i, collection in enumerate(research_prediction_collections):
            summary = self._format_and_expand_research_summary(
                i + 1, report_type, collection
            )
            core_research_for_collection = self._format_main_research(i + 1, collection)
            forecaster_rationales_for_collection = self._format_forecaster_rationales(
                i + 1, collection
            )
            all_summaries.append(summary)
            all_core_research.append(core_research_for_collection)
            all_forecaster_rationales.append(forecaster_rationales_for_collection)

        combined_summaries = "\n".join(all_summaries)
        combined_research_reports = "\n".join(all_core_research)
        combined_rationales = "\n".join(all_forecaster_rationales)
        full_explanation_without_summary = clean_indents(
            f"""
            # SUMMARY
            *Question*: {question.question_text}
            *Final Prediction*: {report_type.make_readable_prediction(aggregated_prediction)}
            *Total Cost*: ${round(final_cost,4)}
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
        sections = MarkdownTree.turn_markdown_into_report_sections(markdown)
        modified_content = MarkdownTree.report_sections_to_markdown(sections, 3)
        final_content = f"## Report {report_number} Research\n{modified_content}"
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

    def _create_file_path_to_save_to(self, questions: list[MetaculusQuestion]) -> str:
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
            result for result in results if not isinstance(result, BaseException)
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
            raise RuntimeError(
                f"{message}: {exception.__class__.__name__} - {str(exception)}"
            ) from exception

    async def _initialize_notepad(self, question: MetaculusQuestion) -> Notepad:
        new_notepad = Notepad(question=question)
        return new_notepad

    async def _remove_notepad(self, question: MetaculusQuestion) -> None:
        async with self._note_pad_lock:
            self._note_pads = [
                notepad for notepad in self._note_pads if notepad.question != question
            ]

    async def _get_notepad(self, question: MetaculusQuestion) -> Notepad:
        async with self._note_pad_lock:
            for notepad in self._note_pads:
                if notepad.question == question:
                    return notepad
        raise ValueError(
            f"No notepad found for question: ID: {question.id_of_post} Text: {question.question_text}"
        )

    @classmethod
    def log_report_summary(
        cls,
        forecast_reports: Sequence[ForecastReport | BaseException],
        raise_errors: bool = True,
    ) -> None:
        valid_reports = [
            report for report in forecast_reports if isinstance(report, ForecastReport)
        ]

        full_summary = "\n"
        full_summary += "-" * 100 + "\n"

        for report in valid_reports:
            try:
                first_rationale = report.first_rationale
            except Exception as e:
                first_rationale = f"Failed to get first rationale: {e}"
            question_summary = clean_indents(
                f"""
                URL: {report.question.page_url}
                Errors: {report.errors}
                <<<<<<<<<<<<<<<<<<<< Summary >>>>>>>>>>>>>>>>>>>>>
                {report.summary}

                <<<<<<<<<<<<<<<<<<<< First Rationale >>>>>>>>>>>>>>>>>>>>>
                {first_rationale[:10000]}
                -------------------------------------------------------------------------------------------
            """
            )
            full_summary += question_summary + "\n"

        full_summary += f"Bot: {cls.__name__}\n"
        for report in forecast_reports:
            if isinstance(report, ForecastReport):
                short_summary = f"✅ URL: {report.question.page_url} | Minor Errors: {len(report.errors)}"
            else:
                exception_message = (
                    str(report)
                    if len(str(report)) < 1000
                    else f"{str(report)[:500]}...{str(report)[-500:]}"
                )
                short_summary = f"❌ Exception: {report.__class__.__name__} | Message: {exception_message}"
            full_summary += short_summary + "\n"

        total_cost = sum(
            report.price_estimate if report.price_estimate else 0
            for report in valid_reports
        )
        average_minutes = (
            (
                sum(
                    report.minutes_taken if report.minutes_taken else 0
                    for report in valid_reports
                )
                / len(valid_reports)
            )
            if valid_reports
            else 0
        )
        average_cost = total_cost / len(valid_reports) if valid_reports else 0
        full_summary += "\nStats for passing reports:\n"
        full_summary += f"Total cost estimated: ${total_cost:.5f}\n"
        full_summary += f"Average cost per question: ${average_cost:.5f}\n"
        full_summary += (
            f"Average time spent per question: {average_minutes:.4f} minutes\n"
        )
        full_summary += "-" * 100 + "\n\n\n"
        logger.info(full_summary)

        exceptions = [
            report for report in forecast_reports if isinstance(report, BaseException)
        ]
        minor_exceptions = [
            error for report in valid_reports for error in report.errors or []
        ]

        if exceptions:
            for exc in exceptions:
                logger.error(
                    "Exception occurred during forecasting:\n%s",
                    "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                )
            if raise_errors:
                raise RuntimeError(
                    f"{len(exceptions)} errors occurred while forecasting: {exceptions}"
                )
        elif minor_exceptions:
            logger.error(
                f"{len(minor_exceptions)} minor exceptions occurred while forecasting: {minor_exceptions}"
            )

    @overload
    def get_llm(
        self,
        purpose: str = "default",
        guarantee_type: None = None,
    ) -> str | GeneralLlm: ...

    @overload
    def get_llm(
        self,
        purpose: str = "default",
        guarantee_type: Literal["llm"] = "llm",
    ) -> GeneralLlm: ...

    @overload
    def get_llm(
        self,
        purpose: str = "default",
        guarantee_type: Literal["string_name"] = "string_name",
    ) -> str: ...

    def get_llm(
        self,
        purpose: str = "default",
        guarantee_type: Literal["llm", "string_name"] | None = None,
    ) -> GeneralLlm | str:
        if purpose not in self._llms:
            raise ValueError(
                f"Unknown llm requested from llm dict for purpose: '{purpose}'"
            )

        llm = self._llms[purpose]
        if llm is None:
            raise ValueError(
                f"LLM is undefined for purpose: {purpose}. It was probably not defined in defaults."
            )
        return_value = None

        if guarantee_type is None:
            return_value = llm
        elif guarantee_type == "llm":
            if isinstance(llm, GeneralLlm):
                return_value = llm
            else:
                return_value = GeneralLlm(model=llm)
        elif guarantee_type == "string_name":
            if isinstance(llm, str):
                return_value = llm
            else:
                logger.warning(
                    f"Converting GeneralLlm to string llm name: {llm.model} for purpose: {purpose}. This means any settings for the GeneralLlm will be ignored."
                )
                return_value = llm.model
        else:
            raise ValueError(f"Unknown guarantee_type: {guarantee_type}")

        return return_value

    def set_llm(self, llm: GeneralLlm | str | None, purpose: str = "default") -> None:
        if purpose not in self._llms:
            raise ValueError(f"Unknown llm purpose: {purpose}")
        self._llms[purpose] = llm

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        """
        Returns a dictionary of default llms for the bot.
        The keys are the purpose of the llm and the values are the llms (model name or GeneralLlm object).
        Consider adding:
        - reasoner
        - fermi-estimator
        - judge
        - etc.
        """

        if os.getenv("OPENAI_API_KEY"):
            main_default_llm = GeneralLlm(model="gpt-4o", temperature=0.3)
        elif os.getenv("ANTHROPIC_API_KEY"):
            main_default_llm = GeneralLlm(
                model="claude-3-7-sonnet-latest", temperature=0.3
            )
        elif os.getenv("OPENROUTER_API_KEY"):
            main_default_llm = GeneralLlm(
                model="openrouter/openai/gpt-4o", temperature=0.3
            )
        elif os.getenv("METACULUS_TOKEN"):
            main_default_llm = GeneralLlm(model="metaculus/gpt-4o", temperature=0.3)
        else:
            main_default_llm = GeneralLlm(model="gpt-4o", temperature=0.3)

        if os.getenv("OPENAI_API_KEY"):
            summarizer = GeneralLlm(model="gpt-4o-mini", temperature=0.3)
        elif os.getenv("METACULUS_TOKEN"):
            summarizer = GeneralLlm(model="metaculus/gpt-4o-mini", temperature=0.3)
        else:
            summarizer = GeneralLlm(model="gpt-4o-mini", temperature=0.3)

        if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
            researcher = "asknews/news-summaries"
        elif os.getenv("PERPLEXITY_API_KEY"):
            researcher = GeneralLlm(model="perplexity/sonar-pro", temperature=0.1)
        elif os.getenv("OPENROUTER_API_KEY"):
            researcher = GeneralLlm(
                model="openrouter/perplexity/sonar-reasoning", temperature=0.1
            )
        elif os.getenv("EXA_API_KEY"):
            researcher = f"smart-searcher/{main_default_llm.model}"
        else:
            researcher = GeneralLlm(model="perplexity/sonar-pro", temperature=0.1)

        return {
            "default": main_default_llm,
            "summarizer": summarizer,
            "researcher": researcher,
        }
