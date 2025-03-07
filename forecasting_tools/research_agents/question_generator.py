import asyncio
import logging
import random
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class SimpleQuestion(BaseModel, Jsonable):
    question_text: str = Field(
        description="A clear question about a future event",
    )
    resolution_criteria: str = Field(
        description="Specific criteria for how the question will resolve",
    )
    fine_print: str = Field(
        description="Additional information covering every edge case that could happen. This should reduce the change of an ambiguous resolution to 0"
    )
    background_information: str = Field(
        description="Relevant context and historical information to help understand the question"
    )
    expected_resolution_date: datetime = Field(
        description="The date when the question is expected to resolve"
    )


class QuestionGenerator:
    FIELD_DESCRIPTIONS = clean_indents(
        """
        - question_text: A clear question about a future event
        - resolution_criteria: Specific criteria for how the question will resolve
        - fine_print: Additional information covering every edge case that could happen. This should reduce the change of an ambiguous resolution to 0
        - background_information: Relevant context and historical information to help understand the question
        - expected_resolution_date: The date when the question is expected to resolve
        """
    )

    def __init__(
        self,
        model: GeneralLlm | str = "o1",
        researcher: SmartSearcher | None = None,
    ):
        if isinstance(model, str):
            self.model = GeneralLlm(model=model, temperature=1)
        else:
            self.model = model

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
            self.full_questions_to_simple_questions(
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

            Field descriptions:
            {self.FIELD_DESCRIPTIONS}

            Please create {number_of_questions} questions following the same format:
            Pay especially close attention to the resolution criteria:
            Resolution criteria are highly specific way to resolve this that will always be super obvious in retrospect.
            Resolution criteria + fine print should pass the clairvoyance test such that after the event happens there is no debate about whether it happened or not.
            It should be meaningful and pertain to the intent of the question.
            Ideally you give a link for where this information can be found.

            # Examples
            Here are some example questions:
            {self.random_example_question_sample}

            # Schema
            Return only a list of dictionaries in valid JSON format. Use markdown for each question field (e.g. dashes for bullet points).
            {SmartSearcher.get_schema_format_instructions_for_pydantic_type(SimpleQuestion)}
            """
        )

        logger.debug(f"Question Generation Prompt\n{prompt}")
        logger.info(f"Attempting to generate {number_of_questions} questions")

        questions = await self.smart_searcher.invoke_and_return_verified_type(
            prompt, list[SimpleQuestion]
        )
        logger.info(
            f"Generated {len(questions)} questions: {[question.question_text for question in questions]}"
        )

        refined_questions = await self.refine_questions(questions)
        logger.info(f"Refined {len(refined_questions)} questions")
        logger.debug(f"Questions: {questions}")

        invalid_questions = [
            question
            for question in refined_questions
            if not (
                resolve_before_date
                > question.expected_resolution_date
                > resolve_after_date
            )
        ]
        for question in invalid_questions:
            logger.warning(
                f"Question {question.question_text} has an expected resolution date ({question.expected_resolution_date}) that is not between {resolve_after_date} and {resolve_before_date}"
            )
        return refined_questions

    async def refine_questions(
        self, questions: list[SimpleQuestion]
    ) -> list[SimpleQuestion]:
        """Refine the resolution criteria for each question to make them more precise."""
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

                Field descriptions:
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
            simple_question = SimpleQuestion(
                question_text=question.question_text,
                resolution_criteria=question.resolution_criteria,
                fine_print=question.fine_print,
                background_information=question.background_info,
                expected_resolution_date=question.scheduled_resolution_time,
            )
            simple_questions.append(simple_question)
        return simple_questions
