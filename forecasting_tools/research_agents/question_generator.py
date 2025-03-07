from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot2025,
)
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class SimpleQuestion(BaseModel, Jsonable):
    question_text: str
    resolution_criteria: str
    fine_print: str
    background_information: str
    expected_resolution_date: datetime
    question_type: Literal["binary", "numeric", "multiple_choice"] = "binary"
    options: list[str] = Field(default_factory=list)

    @classmethod
    def full_questions_to_simple_questions(
        cls, full_questions: list[MetaculusQuestion]
    ) -> list[SimpleQuestion]:
        simple_questions = []
        for question in full_questions:
            assert question.question_text is not None
            assert question.resolution_criteria is not None
            assert question.background_info is not None
            assert question.scheduled_resolution_time is not None
            assert question.fine_print is not None

            if isinstance(question, NumericQuestion):
                question_type = "numeric"
                options = []
            elif isinstance(question, BinaryQuestion):
                question_type = "binary"
                options = []
            elif isinstance(question, MultipleChoiceQuestion):
                question_type = "multiple_choice"
                options = question.options
            else:
                raise ValueError(f"Unknown question type: {type(question)}")

            simple_question = SimpleQuestion(
                question_text=question.question_text,
                resolution_criteria=question.resolution_criteria,
                fine_print=question.fine_print,
                background_information=question.background_info,
                expected_resolution_date=question.scheduled_resolution_time,
                question_type=question_type,
                options=options,
            )
            simple_questions.append(simple_question)
        return simple_questions

    @classmethod
    def simple_questions_to_full_questions(
        cls, simple_questions: list[SimpleQuestion]
    ) -> list[MetaculusQuestion]:
        full_questions = []
        for question in simple_questions:
            if question.question_type == "binary":
                full_question = BinaryQuestion(
                    question_text=question.question_text,
                    background_info=question.background_information,
                    resolution_criteria=question.resolution_criteria,
                    fine_print=question.fine_print,
                    scheduled_resolution_time=question.expected_resolution_date,
                )
            elif question.question_type == "numeric":
                full_question = NumericQuestion(
                    question_text=question.question_text,
                    background_info=question.background_information,
                    resolution_criteria=question.resolution_criteria,
                    fine_print=question.fine_print,
                    upper_bound=1000000000,
                    lower_bound=-1000000000,
                    open_upper_bound=True,
                    open_lower_bound=True,
                    scheduled_resolution_time=question.expected_resolution_date,
                )
            elif question.question_type == "multiple_choice":
                full_question = MultipleChoiceQuestion(
                    question_text=question.question_text,
                    background_info=question.background_information,
                    resolution_criteria=question.resolution_criteria,
                    fine_print=question.fine_print,
                    options=question.options,
                    scheduled_resolution_time=question.expected_resolution_date,
                )
            else:
                raise ValueError(
                    f"Unknown question type: {question.question_type}"
                )
            full_questions.append(full_question)
        return full_questions


class QuestionGenerator:
    FIELD_DESCRIPTIONS = clean_indents(
        """
        - question_text: A clear question about a future event
        - resolution_criteria: Specific criteria for how the question will resolve. If possible include a link to a status page (e.g. a website with a live number or condition that is easy to resolve)
        - fine_print: Additional information covering every edge case that could happen. This should reduce the change of an ambiguous resolution to 0. Resolution criteria + fine print should pass the clairvoyance test such that after the event happens there is no debate about whether it happened or not.
        - background_information: Relevant context and historical information to help understand the question
        - expected_resolution_date: The date when the question is expected to resolve
        - question_type: The type of question, either binary, numeric, or multiple_choice based on how the forecaster should answer (with yes/no, a number, or a choice from a list)
        - options: The options for the question, only used for multiple_choice questions. Empty list for other question types.
        """
    )

    def __init__(
        self,
        model: GeneralLlm | str = "o1",
        forecaster: ForecastBot | None = None,
        researcher: SmartSearcher | None = None,
    ) -> None:
        if isinstance(model, str):
            self.model = GeneralLlm(model=model, temperature=1, timeout=120)
        else:
            self.model = model

        if forecaster is None:
            self.forecaster = Q1TemplateBot2025(
                research_reports_per_question=1,
                predictions_per_research_report=5,
                publish_reports_to_metaculus=False,
            )
        else:
            self.forecaster = forecaster

        if researcher is None:
            self.smart_searcher = SmartSearcher(
                model=self.model,
                num_searches_to_run=5,
                num_sites_per_search=10,
                use_brackets_around_citations=False,
            )
        else:
            self.smart_searcher = researcher

        self.example_full_questions = DataOrganizer.load_questions_from_file_path(
            "forecasting_tools/research_agents/q3_q4_quarterly_questions.json"
        )
        self.example_simple_questions = (
            SimpleQuestion.full_questions_to_simple_questions(
                self.example_full_questions
            )
        )
        self.random_example_question_sample = random.sample(
            self.example_simple_questions, 10
        )

    async def generate_questions(
        self,
        number_of_questions: int = 3,
        topic: str = "",  # e.g. "Lithuanian elections"
        resolve_before_date: datetime = datetime.now() + timedelta(days=30),
        resolve_after_date: datetime = datetime.now(),
    ) -> list[SimpleQuestion]:
        if resolve_before_date <= resolve_after_date:
            raise ValueError(
                "resolve_before_date must be after resolve_after_date"
            )
        if number_of_questions < 1:
            raise ValueError("number_of_questions must be positive")

        num_weeks_till_resolution = (
            resolve_before_date - datetime.now()
        ).days / 7

        if topic == "":
            about_prompt = "The questions must be about general diverse hot news items (they should not all be in the same industry/field/etc.)"
        else:
            about_prompt = f"The questions must be about: {topic}"

        prompt = clean_indents(
            f"""
            # Instructions
            Search the web and make {number_of_questions} forecasting questions.
            {about_prompt}

            Questions should resolve between {resolve_after_date} and {resolve_before_date} (end date is {num_weeks_till_resolution} weeks from now).

            Please create {number_of_questions} questions following the same format:
            Pay especially close attention to making sure that the questions are uncertain:
            - For binary, probabilities should be between 10% and 90%
            - For numeric, the range should not be an obvious number (i.e. one that will not change)
            - For multiple choice, probability for each option should not be more than 90% or less than 5%

            # Field descriptions:
            {self.FIELD_DESCRIPTIONS}

            # Examples
            Here are some example questions:
            {self.random_example_question_sample}

            # Schema
            Return only a list of dictionaries in valid JSON format. Use markdown for each question field (e.g. dashes for bullet points). Always return a list of questions (even if it's a list of one question).
            {SmartSearcher.get_schema_format_instructions_for_pydantic_type(SimpleQuestion)}
            """
        )

        logger.debug(f"Question Generation Prompt\n{prompt}")
        logger.info(f"Attempting to generate {number_of_questions} questions")

        final_questions = []
        max_iterations = 3
        iteration = 0
        questions_needed = number_of_questions

        # Create a forecast bot instance for evaluating question certainty
        while iteration < max_iterations and questions_needed > 0:
            iteration += 1
            logger.info(
                f"Starting iteration {iteration} of question generation"
            )

            new_questions = (
                await self.smart_searcher.invoke_and_return_verified_type(
                    prompt,
                    list[SimpleQuestion],
                )
            )

            refined_questions = await self.refine_questions(new_questions)

            date_filtered_questions = [
                question
                for question in refined_questions
                if resolve_before_date
                > question.expected_resolution_date
                > resolve_after_date
            ]
            uncertainty_filtered_questions = (
                await self._filter_questions_by_certainty(
                    date_filtered_questions
                )
            )

            removed_questions = [
                question
                for question in refined_questions
                if question not in uncertainty_filtered_questions
            ]
            logger.info(
                f"Removed {len(removed_questions)} questions that didn't match criteria: {[removed_question.question_text for removed_question in removed_questions]}"
            )
            logger.debug(
                f"Removed questions: {[removed_question for removed_question in removed_questions]}"
            )

            final_questions.extend(uncertainty_filtered_questions)

            questions_needed = number_of_questions - len(final_questions)

            if questions_needed <= 0:
                break

            logger.info(
                f"Need {questions_needed} more questions. Starting next iteration."
            )

        logger.info(
            f"Generated {len(final_questions)} valid questions after {iteration} iterations"
        )

        return final_questions

    async def _filter_questions_by_certainty(
        self, questions: list[SimpleQuestion]
    ) -> list[SimpleQuestion]:
        """Filters out questions that are too certain based on forecasts."""
        uncertain_questions = []

        # Convert simple questions to MetaculusQuestion format
        full_questions = SimpleQuestion.simple_questions_to_full_questions(
            questions
        )

        for i, question in enumerate(full_questions):
            try:
                # Run forecast on question
                forecast_report = await self.forecaster.forecast_question(
                    question
                )

                # Check if question is sufficiently uncertain
                is_uncertain = self._is_forecast_uncertain(forecast_report)

                if is_uncertain:
                    uncertain_questions.append(questions[i])
                else:
                    logger.info(
                        f"Removed too certain question: {questions[i].question_text}"
                    )
            except Exception as e:
                logger.warning(
                    f"Error forecasting question {questions[i].question_text}: {str(e)}"
                )
                # If we can't forecast it, we'll keep it to be safe
                uncertain_questions.append(questions[i])

        return uncertain_questions

    def _is_forecast_uncertain(self, forecast_report: ForecastReport) -> bool:
        """Determines if a forecast shows sufficient uncertainty."""
        question = forecast_report.question
        prediction = forecast_report.prediction

        if isinstance(question, BinaryQuestion):
            # For binary questions, check if probability is between 10% and 90%
            probability = prediction
            is_uncertain = 0.1 <= probability <= 0.9
        elif isinstance(question, NumericQuestion):
            is_uncertain = True
        elif isinstance(question, MultipleChoiceQuestion):
            # For multiple choice, no option should have >90% or <5% probability
            option_probs = prediction.options
            for option in option_probs:
                if option.probability > 0.9 or option.probability < 0.05:
                    is_uncertain = False
                    break
            is_uncertain = True
        return is_uncertain

    async def refine_questions(
        self, questions: list[SimpleQuestion]
    ) -> list[SimpleQuestion]:
        tasks = []
        for question in questions:
            prompt = clean_indents(
                f"""
                # Instructions
                The below question has not been reviewed yet and the resolution criteria may need improvement.

                Here is the question:
                {question.model_dump_json()}

                Please improve the fine print and ideally add a link to it (only if there is a clear place that could help resolve the question).
                Look for clear places that could help resolve the question.
                You have to be more than 100% confident that the resolution criteria/fine print will be unambiguous in retrospect.
                Walk through ways that this could go wrong such as:
                - The resolution source doesn't update
                - The resolution source retracts or changes information
                - One of your assumptions was wrong
                - A key date changes

                # Field descriptions:
                {self.FIELD_DESCRIPTIONS}

                # Examples
                Here are some example questions with good resolution criteria:
                {self.random_example_question_sample}

                # Schema
                Return only a single dictionary in valid JSON format. Use markdown for each question field (e.g. dashes for bullet points).
                {SmartSearcher.get_schema_format_instructions_for_pydantic_type(SimpleQuestion)}
                """
            )

            logger.debug(f"Refining question: {question.question_text}")
            tasks.append(
                self.smart_searcher.invoke_and_return_verified_type(
                    prompt, SimpleQuestion
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        refined_questions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error refining question: {result}")
                refined_questions.append(questions[i])
            else:
                refined_questions.append(result)

        return refined_questions
