from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Literal

import typeguard
from faker import Faker
from pydantic import BaseModel, Field, field_validator, model_validator

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.data_organizer import (
    DataOrganizer,
    ReportTypes,
)
from forecasting_tools.data_models.multiple_choice_report import (
    MultipleChoiceReport,
)
from forecasting_tools.data_models.numeric_report import NumericReport
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot2025,
)
from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher
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
    open_upper_bound: bool | None = None
    open_lower_bound: bool | None = None
    max_value: float | None = None
    min_value: float | None = None

    @field_validator("expected_resolution_date", mode="after")
    @classmethod
    def ensure_utc_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @model_validator(
        mode="after",
    )
    def validate_question_type_fields(self: SimpleQuestion) -> SimpleQuestion:
        if self.question_type == "numeric":
            assert (
                self.max_value is not None
            ), "Upper bound must be provided for numeric questions"
            assert (
                self.min_value is not None
            ), "Lower bound must be provided for numeric questions"
            assert (
                self.open_upper_bound is not None
            ), "Open upper bound must be provided for numeric questions"
            assert (
                self.open_lower_bound is not None
            ), "Open lower bound must be provided for numeric questions"
        else:
            assert (
                self.max_value is None
            ), "Upper bound must not be provided for non-numeric questions"
            assert (
                self.min_value is None
            ), "Lower bound must not be provided for non-numeric questions"
            assert (
                self.open_upper_bound is None
            ), "Open upper bound must not be provided for non-numeric questions"
            assert (
                self.open_lower_bound is None
            ), "Open lower bound must not be provided for non-numeric questions"

        if self.question_type == "multiple_choice":
            assert (
                len(self.options) > 0
            ), "Options must be provided for multiple choice questions"
        else:
            assert (
                len(self.options) == 0
            ), "Options must not be provided for non-multiple choice questions"
        return self

    @classmethod
    def full_questions_to_simple_questions(
        cls, full_questions: list[MetaculusQuestion]
    ) -> list[SimpleQuestion]:
        simple_questions = []
        for question in full_questions:
            if isinstance(question, DateQuestion):
                # TODO: Give more direct support for date questions
                continue

            assert question.question_text is not None
            assert question.resolution_criteria is not None
            assert question.background_info is not None
            assert question.scheduled_resolution_time is not None
            assert question.fine_print is not None

            if isinstance(question, NumericQuestion):
                # TODO: Give more direct support for date questions
                question_type = "numeric"
                options = []
                upper_bound = question.upper_bound
                lower_bound = question.lower_bound
                open_upper_bound = question.open_upper_bound
                open_lower_bound = question.open_lower_bound
            elif isinstance(question, BinaryQuestion):
                question_type = "binary"
                options = []
                upper_bound = None
                lower_bound = None
                open_upper_bound = None
                open_lower_bound = None
            elif isinstance(question, MultipleChoiceQuestion):
                question_type = "multiple_choice"
                options = question.options
                upper_bound = None
                lower_bound = None
                open_upper_bound = None
                open_lower_bound = None
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
                max_value=upper_bound,
                min_value=lower_bound,
                open_upper_bound=open_upper_bound,
                open_lower_bound=open_lower_bound,
            )
            simple_questions.append(simple_question)
        return simple_questions

    @classmethod
    def simple_questions_to_metaculus_question(
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
                assert question.max_value is not None
                assert question.min_value is not None
                assert question.open_upper_bound is not None
                assert question.open_lower_bound is not None
                full_question = NumericQuestion(
                    question_text=question.question_text,
                    background_info=question.background_information,
                    resolution_criteria=question.resolution_criteria,
                    fine_print=question.fine_print,
                    upper_bound=question.max_value,
                    lower_bound=question.min_value,
                    open_upper_bound=question.open_upper_bound,
                    open_lower_bound=question.open_lower_bound,
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

    def is_within_date_range(
        self, resolve_before_date: datetime, resolve_after_date: datetime
    ) -> bool:

        return (
            resolve_before_date.astimezone(timezone.utc)
            >= self.expected_resolution_date.astimezone(timezone.utc)
            >= resolve_after_date.astimezone(timezone.utc)
        )


class GeneratedQuestion(SimpleQuestion):
    forecast_report: ReportTypes | None = None
    error_message: str | None = None

    @property
    def is_uncertain(self) -> bool:
        """Determines if a forecast shows sufficient uncertainty."""
        report = self.forecast_report
        if report is None or isinstance(report, Exception):
            return False

        if isinstance(report, BinaryReport):
            # For binary questions, check if probability is between 10% and 90%
            probability = report.prediction
            is_uncertain = 0.1 <= probability <= 0.9
        elif isinstance(report, NumericReport):
            is_uncertain = True
        elif isinstance(report, MultipleChoiceReport):
            # For multiple choice, no option should have >90% or <5% probability
            for option in report.prediction.predicted_options:
                if option.probability > 0.8 or option.probability < 0.05:
                    is_uncertain = False
                    break
            else:
                is_uncertain = True
        else:
            is_uncertain = False
        return is_uncertain


class QuestionGenerator:
    """
    Question writing guidelines:
    https://www.metaculus.com/question-writing/
    https://metaculus.notion.site/Public-Facing-Question-Writing-Guide-9e7374d638e749a2ae40b093ce619a9a?pvs=73
    """

    FIELD_DESCRIPTIONS = clean_indents(
        """
        - question_text: A clear question about a future event
        - resolution_criteria: Specific criteria for how the question will resolve. If possible include a link to a status page (e.g. a website with a live number or condition that is easy to resolve). Mention the units/scale expected (give an example like "a value of $1.2 million of income will resolve as '1.2'")
        - fine_print: Additional information covering *every* edge case that could happen. There should be no chance of an ambiguous resolution. Resolution criteria + fine print should pass the clairvoyance test such that after the event happens there is no debate about whether it happened or not no matter how it resolves.
        - background_information: Relevant context and historical information to help understand the question
        - expected_resolution_date: The date when the question is expected to resolve
        - question_type: The type of question, either binary, numeric, or multiple_choice based on how the forecaster should answer (with yes/no, a number, or a choice from a list)
        - options: The options for the question, only used for multiple_choice questions. Empty list for other question types.
        - open_upper_bound: Whether there can be a value higher than upper bound (e.g. if the value is a percentag, 100 is the max the bound is closed, but number of certifications in a population has an open upper bound), only used for numeric questions.
        - open_lower_bound: Whether there can be a value lower than lower bound (e.g. distances can't be negative the bound is closed at 0, but profit margins can be negative so the bound is open), only used for numeric questions.
        - max_value: The max value that the question can be. If bound is closed then choose the max number. If bound is open then pick a really really big number. Only used for numeric questions. (e.g. 100 for a percentage, 1000 for a number of certifications from an small org, 100000 for a number of new houses built in a large city in a year)
        - min_value: The min value that the question can be. If bound is closed then choose the min number. If bound is open then pick a really really negative number. Only used for numeric questions. (e.g. 0 for a percentage, 0 for a number of certifications from a small org, -10000000 for a medium company net profit)
        """
    )

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
        self.max_iterations = max_iterations

    async def generate_questions(
        self,
        number_of_questions: int = 3,
        topic: str = "",  # e.g. "Lithuanian elections"
        resolve_before_date: datetime = datetime.now() + timedelta(days=30),
        resolve_after_date: datetime = datetime.now(),
    ) -> list[GeneratedQuestion]:
        if resolve_before_date <= resolve_after_date:
            raise ValueError(
                "resolve_before_date must be after resolve_after_date"
            )
        if number_of_questions < 1:
            raise ValueError("number_of_questions must be positive")

        resolve_before_date = resolve_before_date.astimezone(timezone.utc)
        resolve_after_date = resolve_after_date.astimezone(timezone.utc)

        logger.info(f"Attempting to generate {number_of_questions} questions")

        final_questions: list[GeneratedQuestion] = []
        iteration = 0
        questions_needed = number_of_questions

        while iteration < self.max_iterations and questions_needed > 0:
            logger.info(
                f"Starting iteration {iteration + 1} of question generation"
            )
            new_questions = await self._generate_draft_questions(
                number_of_questions,
                topic,
                resolve_before_date,
                resolve_after_date,
            )
            new_questions_with_forecasts = (
                await self._add_forecast_to_questions(new_questions)
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
                number_of_questions
                - len(final_questions)
                + number_bad_questions
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
            {self.FIELD_DESCRIPTIONS}

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
                {self.FIELD_DESCRIPTIONS}

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
        metaculus_questions = (
            SimpleQuestion.simple_questions_to_metaculus_question(questions)
        )

        for simple_question, metaculus_question in zip(
            questions, metaculus_questions
        ):
            try:
                forecast_report = await self.forecaster.forecast_question(
                    metaculus_question
                )
                forecast_report = typeguard.check_type(
                    forecast_report, ReportTypes
                )
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


class TopicGenerator:

    @classmethod
    async def generate_random_topic(
        cls,
        model: GeneralLlm | SmartSearcher | str = "gpt-4o",
        number_of_topics: int = 10,
        additional_instructions: str = "",
    ) -> list[str]:
        if isinstance(model, str):
            model = GeneralLlm(model=model, temperature=1, timeout=40)

        fake = Faker(
            [
                "en_US",
                "ja_JP",
                "de_DE",
                "en_GB",
                "fr_FR",
                "es_ES",
                "it_IT",
                "pt_BR",
                "ru_RU",
                "zh_CN",
                "ar_EG",
                "hi_IN",
                "ko_KR",
            ]
        )

        random_text = clean_indents(
            f"""
            Job: {fake.job()}
            Country 1: {fake.country()}
            Country 2: {fake.country()}
            State/Province (not necessarily in above country): {fake.state()}
            City (not necessarily in above state): {fake.city()}
            Word: {fake.word()}
            Sentence: {fake.sentence()}
            Paragraph: {fake.paragraph()}
            Text: {fake.text(max_nb_chars=50)}
            News headline: {fake.sentence().rstrip('.')}
            Company ticker symbol: {fake.lexify(text='???', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
            """
        )

        prompt = clean_indents(
            f"""
            # Instructions
            Using the ideas below (some of which are abstract or randomly generated)
            come up with {number_of_topics} topics for a forecasting questions about the future.
            These will be used to make questions for superforecasters
            Make sure all ideas come from the material below (do not copy the initial ideas)
            Make sure to put everything in English (except for proper nouns)

            Try to choose something interesting and meaningful.

            {additional_instructions}

            Return your response as a list of dictionaries with a "topic" key.

            # Example result w/o citations:
            ```json
            [
                {{"topic": "Lithuanian politics"}},
                {{"topic": "Gun violence in Olklahoma"}},
                {{"topic": "News on Japenese elections"}},
                {{"topic": "Sports results and news in Russian Hockey"}},
                {{"topic": "Number of new houses built in the US"}},
                {{"topic": "Mining jobs in Canada and related news"}},
                {{"topic": "News related to company with ticker symbol XYZ"}},
            ]
            ```

            # Example result w/ citations:
            ```json
            [
                {{"topic": "March Madness", "citations": "[1] [2] [7]"}},
                {{"topic": "Japenese Obon Festival", "citations": "[3] [8] [9]"}},
                {{"topic": "National Hocky tournament in Russia", "citations": "[4] [11] [12]"}},
                {{"topic": "Current Housing Crisis in Oklahoma", "citations": "[13] [14] [15]"}},
                {{"topic": "Corona Outbreak in Europe", "citations": "[13] [14] [15]"}},
                {{"topic": "Recent AI initiative of the TransFord Institute", "citations": "[19] [20] [21]"}},
            ]
            ```

            # Material to adapt:
            {random_text}

            Now please generate a list of topics (in json format) that could be interesting and meaningful.
            """
        )

        topic_dicts = await model.invoke_and_return_verified_type(
            prompt, list[dict[str, str]]
        )
        final_topics = []
        for topic in topic_dicts:
            text = topic["topic"]
            citations = topic.get("citations", "")
            final_topics.append(f"{text} {citations}")

        return final_topics

    @classmethod
    async def generate_random_news_items(
        cls,
        model: GeneralLlm | str = "gpt-4o",
        number_of_items: int = 10,
    ) -> list[str]:
        num_topics = 2
        num_news_items_per_topic = number_of_items // num_topics

        topics = await cls.generate_random_topic(
            model=model,
            additional_instructions=(
                "Pick topics related to breaking news"
                " (e.g. if your material is related to basketball"
                " and march madness is happening choose this as a topic)."
                " Add citations to show the topic is recent and relevant."
                " Consider searching for 'latest news in <place>' or 'news related to <upcoming holidays/tournaments/events>'."
                f" Today is {datetime.now().strftime('%Y-%m-%d')} if you already know of something specific in an area to find juice."
            ),
            number_of_topics=num_topics,
        )

        results = await asyncio.gather(
            *[
                cls.topic_to_news_item(
                    topic,
                    number_of_items=num_news_items_per_topic,
                    model=model,
                )
                for topic in topics
            ]
        )
        news_items = []
        for topic_results in results:
            news_items.extend(topic_results)

        return news_items

    @classmethod
    async def topic_to_news_item(
        cls,
        topic: str,
        number_of_items: int = 5,
        model: GeneralLlm | str = "gpt-4o",
    ) -> list[str]:
        if isinstance(model, str):
            model = GeneralLlm(model=model, temperature=1, timeout=40)

        ask_news_results = await AskNewsSearcher().get_formatted_news_async(
            topic
        )
        prompt = clean_indents(
            f"""
            # Instructions
            Please extract {number_of_items} news items from the following text:

            Return your response as a list of strings with the related url

            # Example result:
            ```json
            [
                {{"topic": "Senator Joe Shmoe joins the race for presidency in the US", "url": "https://www.nyt.com/breaking-news/us/senator-joe-shmoe-joins-the-race-for-presidency"}},
                {{"topic": "Russia attacks Ukraine with new drone technology", "url": "https://www.bbc.com/events/russia-attacks-ukraine"}},
                {{"topic": "Nicuragua start first nuclear power plant", "url": "https://www.nuclearnews.com/events/nicuragua-starts-first-nuclear-power-plant"}},
                {{"topic": "Deadly outbreak of disease spreading through Europe", "url": "https://www.outbreaknews.com/events/deadly-outbreak-of-disease-spreading-through-europe"}},
                {{"topic": "Chinese officials visit Lebanon to discuss trade", "url": "https://www.tradeinkorea.com/events/chinese-officials-visit-lebanon-to-discuss-trade"}},
            ]
            ```

            # Search results
            {ask_news_results}

            # Final instructions
            Now return a json list of topics please
            """
        )
        topic_dicts = await model.invoke_and_return_verified_type(
            prompt, list[dict[str, str]]
        )
        topics = [
            f"{topic_dict['topic']} [link]({topic_dict['url']})"
            for topic_dict in topic_dicts
        ]
        return topics
