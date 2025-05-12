import random
from datetime import datetime, timedelta, timezone

from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher


class QuestionOperationalizer:
    def __init__(
        self,
        model: str = "gpt-4o",
    ) -> None:
        self.smart_searcher = SmartSearcher(
            model=model,
            num_searches_to_run=5,
            num_sites_per_search=10,
            use_brackets_around_citations=False,
        )
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

    async def question_title_to_simple_question(
        self,
        question_title: str,
        resolve_before_date: datetime = datetime.now(timezone.utc)
        + timedelta(days=30),
        resolve_after_date: datetime = datetime.now(timezone.utc),
    ) -> SimpleQuestion:
        num_weeks_till_resolution = (
            resolve_before_date.astimezone(timezone.utc)
            - datetime.now().astimezone(timezone.utc)
        ).days / 7

        prompt = clean_indents(
            f"""
            # Instructions
            Take the following question title and turn it into a full forecasting question, following the format below.

            Question title: {question_title}

            The question should resolve between {resolve_after_date} and {resolve_before_date} (end date is {num_weeks_till_resolution} weeks from now).

            Please create a full question following the same format:
            Pay especially close attention to making sure that the question is uncertain:
            - For binary, probabilities should be between 10% and 90%
            - For numeric, the range should not be an obvious number (i.e. there needs to be uncertainty)
            - For multiple choice, probability for each option should not be more than 80% or less than 5%

            # Field descriptions:
            {SimpleQuestion.get_field_descriptions()}

            # Examples
            Here are some example questions:
            {self.random_example_question_sample}

            # Schema
            Return only a single dictionary in valid JSON format. Use markdown for each question field (e.g. dashes for bullet points).
            {self.smart_searcher.llm.get_schema_format_instructions_for_pydantic_type(SimpleQuestion)}
            """
        )

        questions = await self.smart_searcher.invoke_and_return_verified_type(
            prompt, list[SimpleQuestion]
        )
        if not questions:
            raise ValueError("No question generated from the title.")
        return questions[0]
