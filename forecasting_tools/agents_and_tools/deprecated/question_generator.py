from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timedelta, timezone

import typeguard

from forecasting_tools.agents_and_tools.question_generators.generated_question import (
    GeneratedQuestion,
)
from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.agents_and_tools.research.smart_searcher import SmartSearcher
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.data_organizer import DataOrganizer, ReportTypes
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot2025,
)

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    Question writing guidelines:
    https://www.metaculus.com/question-writing/
    https://metaculus.notion.site/Public-Facing-Question-Writing-Guide-9e7374d638e749a2ae40b093ce619a9a?pvs=73
    """

    def __init__(
        self,
        model: GeneralLlm | str = "gpt-4o",
        forecaster: ForecastBot | None = None,
        researcher: SmartSearcher | None = None,
        max_iterations: int = 3,
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
            "forecasting_tools/agents_and_tools/question_generators/q3_q4_quarterly_questions.json"
        )
        self.example_simple_questions = (
            SimpleQuestion.full_questions_to_simple_questions(
                self.example_full_questions
            )
        )
        self.random_example_question_sample = random.sample(
            self.example_simple_questions, 10
        )
        self.max_iterations = max_iterations

    async def generate_questions(
        self,
        number_of_questions: int = 3,
        topic: str = "",  # e.g. "Lithuanian elections"
        resolve_before_date: datetime = datetime.now() + timedelta(days=30),
        resolve_after_date: datetime = datetime.now(),
    ) -> list[GeneratedQuestion]:
        if resolve_before_date <= resolve_after_date:
            raise ValueError("resolve_before_date must be after resolve_after_date")
        if number_of_questions < 1:
            raise ValueError("number_of_questions must be positive")

        resolve_before_date = resolve_before_date.astimezone(timezone.utc)
        resolve_after_date = resolve_after_date.astimezone(timezone.utc)

        logger.info(f"Attempting to generate {number_of_questions} questions")

        final_questions: list[GeneratedQuestion] = []
        iteration = 0
        questions_needed = number_of_questions

        while iteration < self.max_iterations and questions_needed > 0:
            logger.info(f"Starting iteration {iteration + 1} of question generation")
            new_questions = await self._generate_draft_questions(
                number_of_questions,
                topic,
                resolve_before_date,
                resolve_after_date,
            )
            new_questions_with_forecasts = await self._add_forecast_to_questions(
                new_questions
            )
            final_questions.extend(new_questions_with_forecasts)
            logger.debug(
                f"Generated {len(new_questions_with_forecasts)} new questions for iteration {iteration + 1}: {new_questions_with_forecasts}"
            )

            number_bad_questions = len(
                [
                    question
                    for question in final_questions
                    if not question.is_within_date_range(
                        resolve_before_date, resolve_after_date
                    )
                    or not question.is_uncertain
                ]
            )
            questions_needed = (
                number_of_questions - len(final_questions) + number_bad_questions
            )
            logger.info(
                f"At iteration {iteration + 1}, there are {number_bad_questions} bad questions (not within date range or not uncertain) out of {len(final_questions)} questions generated and {questions_needed} questions left to generate"
            )

            if questions_needed <= 0:
                break
            iteration += 1

        logger.info(
            f"Generated {len(final_questions)} questions after {iteration + 1} iterations"
        )
        logger.debug(f"Final questions: {final_questions}")
        return final_questions

    async def _generate_draft_questions(
        self,
        number_of_questions: int,
        topic: str,
        resolve_before_date: datetime,
        resolve_after_date: datetime,
    ) -> list[SimpleQuestion]:
        num_weeks_till_resolution = (
            resolve_before_date.astimezone(timezone.utc)
            - datetime.now().astimezone(timezone.utc)
        ).days / 7

        if not topic:
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
            - For numeric, the range should not be an obvious number (i.e. there needs to be uncertainty)
            - For multiple choice, probability for each option should not be more than 80% or less than 5%

            # Field descriptions:
            {SimpleQuestion.get_field_descriptions()}

            # Examples
            Here are some example questions:
            {self.random_example_question_sample}

            # Schema
            Return only a list of dictionaries in valid JSON format. Use markdown for each question field (e.g. dashes for bullet points). Always return a list of questions (even if it's a list of one question).
            {SmartSearcher.get_schema_format_instructions_for_pydantic_type(SimpleQuestion)}
            """
        )

        questions = await self.smart_searcher.invoke_and_return_verified_type(
            prompt, list[SimpleQuestion]
        )
        for question in questions:
            if question.question_type == "numeric":
                assert question.max_value is not None
                assert question.min_value is not None
                distance = question.max_value - question.min_value
                buffer_room = distance * 0.9
                if question.open_lower_bound is True:
                    question.min_value -= buffer_room
                if question.open_upper_bound is True:
                    question.max_value += buffer_room
        return questions

    async def _refine_questions(
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

                Before giving your final answer in a json code block, please walk through at least 3 possible situations that could happen and how you would resolve them.
                Compare to ways that similar things have happened in the past that surprised people.

                # Field descriptions:
                {SimpleQuestion.get_field_descriptions()}

                # Examples
                Here are some example questions with good resolution criteria:
                {self.random_example_question_sample}

                # Schema
                After your reasoning please return only a single dictionary in valid JSON code blockformat. Use markdown for each question field (e.g. dashes for bullet points).
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

    async def _add_forecast_to_questions(
        self, questions: list[SimpleQuestion]
    ) -> list[GeneratedQuestion]:
        extended_questions = []

        # Convert simple questions to MetaculusQuestion format
        metaculus_questions = SimpleQuestion.simple_questions_to_metaculus_questions(
            questions
        )

        for simple_question, metaculus_question in zip(questions, metaculus_questions):
            try:
                forecast_report = await self.forecaster.forecast_question(
                    metaculus_question
                )
                forecast_report = typeguard.check_type(forecast_report, ReportTypes)
                error_message = None
            except Exception as e:
                logger.warning(
                    f"Error forecasting question {simple_question.question_text}: {str(e)}"
                )
                forecast_report = None
                error_message = str(e)

            extended_questions.append(
                GeneratedQuestion(
                    **simple_question.model_dump(),
                    forecast_report=forecast_report,
                    error_message=error_message,
                )
            )

        return extended_questions
