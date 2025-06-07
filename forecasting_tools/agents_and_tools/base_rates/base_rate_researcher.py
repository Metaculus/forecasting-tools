from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, field_validator

from forecasting_tools.agents_and_tools.base_rates.estimator import Estimator
from forecasting_tools.agents_and_tools.base_rates.niche_list_researcher import (
    FactCheckedItem,
    NicheListResearcher,
)
from forecasting_tools.agents_and_tools.deprecated.configured_llms import (
    BasicLlm,
)
from forecasting_tools.agents_and_tools.deprecated.general_researcher import (
    GeneralResearcher,
)
from forecasting_tools.agents_and_tools.deprecated.question_responder import (
    QuestionResponder,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class BaseRateResearcher(QuestionResponder):
    """
    Takes a question like
    > "What are the chances per year of SpaceX launching a failed rocket?"

    It then finds a historical rate and makes a future prediction based on that rate
    """

    NAME = "Historical/Future Rate Analysis"
    DESCRIPTION_OF_WHEN_TO_USE = "Use this responder when online information is needed about historical rates, historical occurrences, and future probabilities"

    def __init__(self, question: str) -> None:
        super().__init__(question)
        is_valid = asyncio.run(self.__is_valid_question())
        if not is_valid:
            raise ValueError(
                f"The question doesn't seem to be about base rates: {question}"
            )
        self.__start_date: datetime | None = None
        self.__end_date: datetime | None = None
        self.__general_search_information: str | None = None

    async def respond_with_markdown(self) -> str:
        try:
            report = await self.make_base_rate_report()
            return report.markdown_report
        except Exception as e:
            logger.error(
                f"Error while making base rate report for question: {self.question}. Doing general search instead. Error: {e}"
            )
            await asyncio.sleep(15)
            back_up_report = await GeneralResearcher(
                self.question
            ).respond_with_markdown()
            back_up_report = f"Deep Base Rate research was attempted but failed. Here is a general search report instead:\n\n{back_up_report}"
            return back_up_report

    async def make_base_rate_report(self) -> BaseRateReport:
        logger.info(
            f"Starting to make base rate report for question: {self.question}"
        )
        with MonetaryCostManager() as cost_manager:
            report = await self.__make_base_rate_report(cost_manager)
            logger.info(
                f"Made markdown report. Cost ${cost_manager.current_usage:.2f}. Text: {report.markdown_report[:1000]}..."
            )
        return report

    async def __make_base_rate_report(
        self, cost_manager: MonetaryCostManager
    ) -> BaseRateReport:
        await self.__populate_start_and_end_date()
        await self.__populate_general_search_information()
        assert self.__start_date
        assert self.__end_date

        numerator_class = await self.__get_numerator_ref_class()
        denominator_type_decision = await self.__run_hits_per_day_decision(
            numerator_class
        )
        denominator_class = await self.__get_denominator_reference_class(
            numerator_class,
            denominator_type_decision,
        )
        assert numerator_class.start_date == denominator_class.start_date
        assert numerator_class.end_date == denominator_class.end_date

        historical_rate = numerator_class.count / denominator_class.count
        final_cost = cost_manager.current_usage
        historical_rate_string = self.__create_historical_rate_string(
            historical_rate,
            denominator_type_decision.answer,
            numerator_class,
            denominator_class,
        )
        date_range_string = self.__create_date_range_string(numerator_class)

        markdown_report = clean_indents(
            f"""
            **Cost of base rate report**: ${final_cost:.2f}

            ### Results
            - {historical_rate_string}
            - Numerator Size: {numerator_class.count} {numerator_class.hit_description_with_dates_included}
            - Denominator Size: {denominator_class.count} {denominator_class.hit_description_with_dates_included}
            - {date_range_string}

            ### Background Information
            {self.__general_search_information}

            ### Numerator
            **Size Found**: {numerator_class.count} | **Hit Definition**: {numerator_class.hit_definition}

            **Explanation**:
            {numerator_class.reasoning}

            ### Denominator
            **Size Found**: {denominator_class.count} | **Hit Definition**: {denominator_class.hit_definition}

            **Explanation**:
            {denominator_class.reasoning}
            """
        )

        return BaseRateReport(
            question=self.question,
            historical_rate=historical_rate,
            numerator_reference_class=numerator_class,
            denominator_reference_class=denominator_class,
            denominator_type=denominator_type_decision.answer,
            start_date=self.__start_date,
            end_date=self.__end_date,
            markdown_report=markdown_report,
            price_estimate=final_cost,
        )

    async def __populate_start_and_end_date(self) -> None:
        prompt = clean_indents(
            f"""
            You are an AI assistant tasked with determining the start and end dates for a historical base rate analysis.

            The question being analyzed is:
            {self.question}

            Today is:
            {datetime.now().strftime("%Y-%m-%d")}

            Please determine the most appropriate start and end dates for this analysis. The end date should typically be the current date unless the question specifies otherwise. The start date should be a reasonable point in the past that captures relevant historical data.

            Provide your answer as a JSON object with the following format:
            {{
                "reasoning": "Your step-by-step reasoning here..."
                "start_date": "YYYY-MM-DD",
                "end_date": "YYYY-MM-DD",
            }}

            Return only the JSON object and nothing else.

            Example: If you were asked "How often has SpaceX launched rockets over the last 5 years?"
            {{
                "reasoning": "1. The question asks about the last 5 years...\n2. Today's date is 2023-04-15...\n3. Therefore, I've set the start date to 5 years ago and the end date to today...",
                "start_date": "2018-01-01",
                "end_date": "2023-04-15",
            }}
            """
        )

        model = BasicLlm(temperature=0)
        response = await model.invoke_and_return_verified_type(prompt, dict)

        start_date = datetime.strptime(response["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(response["end_date"], "%Y-%m-%d")

        self.__start_date = start_date
        self.__end_date = end_date

        logger.info(
            f"Determined date range for question '{self.question}': {start_date} to {end_date}. Reasoning: {response['reasoning']}"
        )

    async def __populate_general_search_information(self) -> None:
        general_search_information: str = await GeneralResearcher(
            self.question
        ).respond_with_markdown()
        self.__general_search_information = general_search_information
        logger.info(f"General Info: {general_search_information[:1000]}...")

    async def __get_numerator_ref_class(self) -> ReferenceClassWithCount:
        assert self.__general_search_information
        prompt = clean_indents(
            f"""
            # Setup
            You are a superforecaster trying to find the a base rate of how often something has happened in the past.

            {self.__create_ref_class_instruction_prompt()}

            # Your Turn
            Now please find a good reference class for the question: {self.question}
            Remember to return only the json and nothing else
            """
        )
        numerator_ref_class = await self.__call_model_expecting_ref_class(
            prompt
        )
        numerator_ref_class_with_size: ReferenceClassWithCount = (
            await self.__find_size_of_ref_class(numerator_ref_class)
        )
        logger.info(
            f"Numerator: {str(numerator_ref_class_with_size)[:1000]}..."
        )
        return numerator_ref_class_with_size

    async def __call_model_expecting_ref_class(
        self, prompt: str
    ) -> ReferenceClass:
        assert self.__start_date
        assert self.__end_date
        model = BasicLlm(temperature=0)
        reference_class = await model.invoke_and_return_verified_type(
            prompt, dict
        )
        hit_definition: str = reference_class["hit_definition"]
        search_query: str = reference_class["search_query"]
        return ReferenceClass(
            start_date=self.__start_date,
            end_date=self.__end_date,
            hit_definition=hit_definition,
            hit_description_with_dates_included=search_query,
        )

    async def __find_size_of_ref_class(
        self, reference_class: ReferenceClass
    ) -> ReferenceClassWithCount:
        estimated_reference_class = (
            await self.__find_size_of_ref_class_through_estimation(
                reference_class
            )
        )
        if (
            estimated_reference_class.count
            < NicheListResearcher.MAX_ITEMS_IN_LIST
        ):
            new_reference_class = estimated_reference_class
            try:
                new_reference_class = (
                    await self.__find_size_of_ref_class_through_list_generator(
                        reference_class
                    )
                )
            except Exception as e:
                logger.info(
                    f"Failed to generate exhaustive list for search query: {reference_class.hit_description_with_dates_included}. Error: {e}"
                )
                new_reference_class = estimated_reference_class
        else:
            new_reference_class = estimated_reference_class
        return new_reference_class

    async def __find_size_of_ref_class_through_estimation(
        self, reference_class: ReferenceClass
    ) -> ReferenceClassWithCount:
        estimator = Estimator(
            type_of_thing_to_estimate=reference_class.hit_definition,
            previous_research=self.__general_search_information,
        )
        size, explanation = await estimator.estimate_size()
        reference_class = ReferenceClassWithCount(
            **reference_class.model_dump(),
            count=size,
            reasoning=explanation,
        )
        return reference_class

    async def __find_size_of_ref_class_through_list_generator(
        self, reference_class: ReferenceClass
    ) -> ReferenceClassWithCount:
        items_found = await NicheListResearcher(
            reference_class.hit_description_with_dates_included
        ).research_niche_reference_class(return_invalid_items=True)
        correct_items = [item for item in items_found if item.is_valid]
        markdown_of_items = (
            FactCheckedItem.make_markdown_with_valid_and_invalid_lists(
                items_found
            )
        )
        markdown_report = clean_indents(
            f"""
            I found {len(items_found)} items that are '{reference_class.hit_definition}'.
            {markdown_of_items}
            """
        )

        new_reference_class = ReferenceClassWithCount(
            **reference_class.model_dump(),
            count=len(correct_items),
            reasoning=markdown_report,
        )
        return new_reference_class

    async def __run_hits_per_day_decision(
        self, numerator_reference_class: ReferenceClass
    ) -> EventOrDayDecision:
        choosing_event_or_days_prompt = clean_indents(
            f"""
            You are answering the question: {self.question}

            When predicting the future related to this question, would it be more useful to know:
            1) '{numerator_reference_class.hit_description_with_dates_included}' divided by days as a percentage?
            2) '{numerator_reference_class.hit_description_with_dates_included}' divided by a class of events as a percentage?

            For instance when predicting whether Apple will get sued related to a recent lawsuit, it is more useful to know how often Apple has been successfully sued for patent violations per time they are sued (event) than per day.
            However it is more useful to know how often a meteorite hits the US per day since there is no clear event that causes a meteorite to hit the US.
            Your answer should only be either {DenominatorOption.PER_DAY.name} or {DenominatorOption.PER_EVENT.name}.

            Please give your answer as a json. Return only the json and nothing else.
            For example:
            {{
                "thinking": "...Do your thinking here...",
                "answer": "{DenominatorOption.PER_DAY.name}"
            }}
            """
        )
        model = BasicLlm(temperature=0)
        response = await model.invoke_and_return_verified_type(
            choosing_event_or_days_prompt, dict
        )
        reasoning = response["thinking"]
        string_answer = response["answer"]

        assert isinstance(string_answer, str)
        if DenominatorOption.PER_DAY.name in string_answer:
            real_answer = DenominatorOption.PER_DAY
        elif DenominatorOption.PER_EVENT.name in string_answer:
            real_answer = DenominatorOption.PER_EVENT
        else:
            raise ValueError(
                f"Could not understand the answer: {string_answer}"
            )
        per_day_or_event_decision = EventOrDayDecision(
            prompt=choosing_event_or_days_prompt,
            reasoning=reasoning,
            answer=real_answer,
        )
        logger.info(
            f"Found hits per day decision: {per_day_or_event_decision}"
        )
        return per_day_or_event_decision

    async def __get_denominator_reference_class(
        self,
        numerator_ref_class: ReferenceClass,
        denominator_type: EventOrDayDecision,
    ) -> ReferenceClassWithCount:
        assert self.__general_search_information
        if denominator_type.answer == DenominatorOption.PER_DAY:
            days_between_start_and_end = (
                numerator_ref_class.end_date - numerator_ref_class.start_date
            ).days
            return ReferenceClassWithCount(
                start_date=numerator_ref_class.start_date,
                end_date=numerator_ref_class.end_date,
                hit_definition="A day within the start date to end date period",
                hit_description_with_dates_included=f"Days between {numerator_ref_class.readable_start_date} and {numerator_ref_class.readable_end_date}",
                count=days_between_start_and_end,
                reasoning="I calculated the number of days between the start and end date using code.",
            )

        prompt = clean_indents(
            f"""
            # Setup
            You are a superforecaster trying to find the a base rate of how often something has happened in the past.

            You just finished a conversation with a researcher who decided that it would be best to model a question based on the number of events that happen over a larger set of events rather than over a set of days.
            Below is your conversation

            # Conversation
            ## The original instructions to the researcher
            {denominator_type.prompt}

            ## The researcher's response
            Answer: {denominator_type.answer}
            Reasoning: {denominator_type.reasoning}

            {self.__create_ref_class_instruction_prompt()}

            # Your Turn
            Now please find a good reference class that can be used as the denominator for the other reference class '{numerator_ref_class.hit_description_with_dates_included}'
            Remember to return only the json and nothing else
            """
        )
        denominator_ref_class = await self.__call_model_expecting_ref_class(
            prompt
        )
        denominator_ref_class_with_size = await self.__find_size_of_ref_class(
            denominator_ref_class
        )
        logger.info(
            f"Denominator reference class: {str(denominator_ref_class_with_size)[:1000]}..."
        )
        return denominator_ref_class_with_size

    async def __is_valid_question(self) -> bool:
        prompt = clean_indents(
            f"""
            You are an AI assistant tasked with determining if a question is valid for historical base rate analysis and future prediction.

            A valid question is one that is about base rates or how often something has happened in the past. Remember, be loose with your definition. We are just trying to remove clearly off topic questions, or prompt leaking.

            Examples of valid questions:
            - How often has SpaceX launched rockets over the last 5 years?
            - How often have meteorites hit the US?
            - How often has someone successfully sued Apple regarding violated patents?
            - How often has SpaceX launched rockets over the last 5 years? Using their launches per year, what is the chance they will launch a rocket by Dec 30 2025?

            Examples of invalid questions:
            - Who is Abraham Lincoln? (Not about base rates)
            - Will the world end in 2100? (About the future)

            The question to evaluate is:
            {self.question}

            Please analyze the question and determine if it's valid. Respond with a JSON object containing your reasoning and a boolean indicating validity. Return only the JSON object and nothing else.

            Example response:
            {{
                "reasoning": "Lets evaluate the criteria step by step..."
                "is_valid": true,
            }}

            """
        )
        model = BasicLlm(temperature=0)
        response = await model.invoke_and_return_verified_type(prompt, dict)

        logger.debug(
            f"Question is {'valid' if response['is_valid'] else 'invalid'} for '{self.question}'. Reasoning: {response['reasoning']}"
        )
        return response["is_valid"]

    def __create_ref_class_instruction_prompt(self) -> str:
        assert self.__start_date
        assert self.__end_date
        return clean_indents(
            f"""
            # General Instructions
            Your next step is to do deeper research. You need to a good description of a reference class to answer this question.
            A good reference class has a start date (or event), an end date (or event), and a clear definition of what counts as a hit.
            Make sure you exactly mirror the words in the hit_definition and search_query.
            Please give me a json object with a good reference class to search for. Give me the json object and nothing but the json. Do not give any other words, just the json.

            # What you know
            - Today is: {datetime.now().strftime("%Y-%m-%d")}
            - The question is: {self.question}
            - The start date is: {self.__start_date}
            - The end date is: {self.__end_date}

            # Examples
            {[example.prompt_string for example in self.get_reference_class_examples()]}
            """
        )

    @classmethod
    def get_reference_class_examples(cls) -> list[ReferenceClassExample]:
        mock_today_datetime = datetime(2023, 2, 2)
        mock_today_string = "Feb 2, 2023"
        return [
            ReferenceClassExample(
                question="How often has someone successfully sued Apple regarding violated patents?",
                start_date="April 1, 1976",
                end_date=mock_today_string,
                hit_definition="Court cases where someone successfully sues Apple for patent violations",
                search_query="Court cases where someone successfully sued Apple for patent violations between Apple's founding and Dec 31, 2023",
                note="You are allowed to give a start date that is not a date, but it must be a clear event that can be looked up.",
                today=mock_today_datetime,
            ),
            ReferenceClassExample(
                question="How often have meteorites hit the US?",
                start_date="Jan 1, 1600",
                end_date="Dec 31, 2023",
                hit_definition="A meteorite hits the US",
                search_query="Times that Meteorites hit the US between Jan 1, 1600 and Dec 31, 2023",
                note="With the above example, you had to guess at a date that made sense. Something like 1800 would also make sense, but 'The beginning of the universe' would not make sense.",
                today=mock_today_datetime,
            ),
            ReferenceClassExample(
                question="How often has SpaceX launched rockets over the last 5 years?",
                start_date="Feb 2, 2018",
                end_date=mock_today_string,
                hit_definition="SpaceX launches a rocket",
                search_query=f"Times that SpaceX launched a rocket between Feb 2, 2018 and {mock_today_string}",
                today=mock_today_datetime,
            ),
        ]

    def __create_date_range_string(
        self, numerator_class: ReferenceClassWithCount
    ) -> str:
        days_in_period = (
            numerator_class.end_date - numerator_class.start_date
        ).days
        months_in_period = days_in_period / 30
        years_in_period = days_in_period / 365
        return f"Date Range: {numerator_class.readable_start_date} to {numerator_class.readable_end_date} | {days_in_period} days, {months_in_period:.2f} months, {years_in_period:.2f} years"

    def __create_historical_rate_string(
        self,
        historical_rate: float,
        denominator_type: DenominatorOption,
        numerator_class: ReferenceClassWithCount,
        denominator_class: ReferenceClassWithCount,
    ) -> str:
        if denominator_type == DenominatorOption.PER_DAY:
            per_day_chances = historical_rate
            per_month_chances = 1 - (1 - per_day_chances) ** 30
            per_year_chances = 1 - (1 - per_day_chances) ** 365
            return f"Chances of {numerator_class.hit_definition}: {round(per_day_chances * 100, 4)}% per day for days in the date range (Using this rate we can extrapolate to {round(per_month_chances * 100, 2)}% per month, {round(per_year_chances * 100, 2)}% per year)"
        elif denominator_type == DenominatorOption.PER_EVENT:
            return f"Chances of {numerator_class.hit_definition}: {round(historical_rate * 100, 4)}% per {denominator_class.hit_definition}"
        else:
            raise ValueError(
                f"Unsupported denominator type: {denominator_type}"
            )


class ReferenceClass(BaseModel):
    start_date: datetime
    end_date: datetime
    hit_definition: str
    hit_description_with_dates_included: str

    @property
    def readable_start_date(self) -> str:
        return self.start_date.strftime("%b %d, %Y")

    @property
    def readable_end_date(self) -> str:
        return self.end_date.strftime("%b %d, %Y")


class ReferenceClassWithCount(ReferenceClass):
    count: int
    reasoning: str

    @field_validator("count")
    def count_must_be_non_negative(cls, value):
        if value < 0:
            raise ValueError("Count must be greater than or equal to 0")
        return value


class DenominatorOption(str, Enum):
    PER_DAY = "per day"
    PER_EVENT = "per event"


class EventOrDayDecision(BaseModel):
    prompt: str
    answer: DenominatorOption
    reasoning: str

    @property
    def answer_as_string(self) -> str:
        return f"{self.answer.value}"


class BaseRateReport(BaseModel, Jsonable):
    question: str
    historical_rate: float
    start_date: datetime
    end_date: datetime
    numerator_reference_class: ReferenceClassWithCount
    denominator_reference_class: ReferenceClassWithCount
    denominator_type: DenominatorOption
    markdown_report: str
    price_estimate: float | None = None


class ReferenceClassExample(BaseModel):
    question: str
    start_date: str
    end_date: str
    hit_definition: str
    search_query: str
    note: str | None = None
    today: datetime

    @property
    def prompt_string(self) -> str:
        return clean_indents(
            f"""
            Assuming:
            - Today is {self.today}
            - You were asked "{self.question}".
            - The start date is {self.start_date}
            - The end date is {self.end_date}
            You would put down:
            {{
                "hit_definition": "{self.hit_definition}"
                "search_query": "{self.search_query}"
            }}
            NOTICE: {self.note}
            """
        )
