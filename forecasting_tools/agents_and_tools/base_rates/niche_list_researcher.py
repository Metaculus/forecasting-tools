from __future__ import annotations

import asyncio
import json
import logging

from pydantic import BaseModel, field_validator

from forecasting_tools.agents_and_tools.base_rates.deduplicator import Deduplicator
from forecasting_tools.agents_and_tools.deprecated.configured_llms import BasicLlm
from forecasting_tools.agents_and_tools.research.smart_searcher import SmartSearcher
from forecasting_tools.util import async_batching
from forecasting_tools.util.misc import (
    clean_indents,
    extract_url_from_markdown_link,
    is_markdown_citation,
)

logger = logging.getLogger(__name__)


class Criteria(BaseModel):
    short_name: str
    description: str


class CriteriaAssessment(Criteria):
    validity_assessment: str
    is_valid_or_unknown: bool | None
    citation_proving_assessment: str | None

    @field_validator("citation_proving_assessment")
    def validate_citation_format(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not is_markdown_citation(v):
            raise ValueError(
                "Citation must be in the markdown friendly format [number](url)"
            )
        return v

    @property
    def url_proving_assessment(self) -> str | None:
        if not self.citation_proving_assessment:
            return None
        return extract_url_from_markdown_link(self.citation_proving_assessment)


class FactCheck(BaseModel):
    criteria_assessments: list[CriteriaAssessment]

    @property
    def is_valid(self) -> bool:
        number_unknown = 0
        for criteria in self.criteria_assessments:
            if criteria.is_valid_or_unknown == False:
                return False
            elif criteria.is_valid_or_unknown == None:
                number_unknown += 1
        percentage_unknown = number_unknown / len(self.criteria_assessments)
        if percentage_unknown > 0.5:
            return False
        return True


class InitialListItem(BaseModel):
    item_name: str
    description: str
    is_uncertain: bool | None = None
    initial_citations: list[str] | None = None

    @property
    def name_plus_description(self) -> str:
        return f"**{self.item_name}**: {self.description}"

    @classmethod
    def make_markdown_with_name_and_description(
        cls, items: list[InitialListItem]
    ) -> str:
        return "\n".join(f"- {item.name_plus_description}" for item in items)


class FactCheckedItem(InitialListItem):
    fact_check: FactCheck
    type_description: str

    @property
    def is_valid(self) -> bool:
        return self.fact_check.is_valid

    @property
    def supporting_urls(self) -> list[str]:
        return [
            criteria.citation_proving_assessment
            for criteria in self.fact_check.criteria_assessments
            if criteria.citation_proving_assessment
        ]

    @property
    def one_line_fact_check_summary(self) -> str:
        summary_parts = [self.item_name]

        for criteria in self.fact_check.criteria_assessments:
            emoji = (
                "✅"
                if criteria.is_valid_or_unknown == True
                else "❌" if criteria.is_valid_or_unknown == False else "❓"
            )
            if criteria.citation_proving_assessment:
                url = criteria.url_proving_assessment
                assert url is not None
                escaped_url = url.replace("(", "%28").replace(")", "%29")
                summary_parts.append(f"{emoji}[{criteria.short_name}]({escaped_url})")
            else:
                summary_parts.append(f"{emoji}{criteria.short_name}")

        return " | ".join(summary_parts)

    @classmethod
    def make_markdown_with_fact_check_items(cls, items: list[FactCheckedItem]) -> str:
        items_with_bullet_points = [
            f"- {item.one_line_fact_check_summary}" for item in items
        ]
        return "\n".join(items_with_bullet_points)

    @classmethod
    def make_markdown_with_valid_and_invalid_lists(
        cls, items: list[FactCheckedItem]
    ) -> str:
        valid_items = [item for item in items if item.is_valid]
        invalid_items = [item for item in items if not item.is_valid]
        valid_items_markdown = f"Valid instances:\n{cls.make_markdown_with_fact_check_items(valid_items)}\n\n"
        invalid_items_markdown = f"Invalid instances:\n{cls.make_markdown_with_fact_check_items(invalid_items)}\n\n"
        if len(invalid_items) == 0:
            return valid_items_markdown
        else:
            return f"{valid_items_markdown}\n\n{invalid_items_markdown}"


class NicheListResearcher:
    """
    Finds all instances of a thing and then fact checks each one to make sure it is valid.
    Type of thing to generate is a string like "US court cases between date X and Y where Z happened", "times when a country has done X", etc
    """

    MAX_ITEMS_IN_LIST = 30

    def __init__(
        self,
        type_of_thing_to_generate: str,
        reject_inputs_expected_to_exceed_max_size: bool = False,
    ):
        self.reject_inputs_expected_to_exceed_max_size = (
            reject_inputs_expected_to_exceed_max_size
        )
        self.type_of_thing_to_generate = type_of_thing_to_generate
        self.number_of_fact_checks_per_item = 1
        self.num_llm_calls_to_find_new_items = 2
        self.num_internet_searches_to_find_new_items = 3

    async def research_niche_reference_class(
        self, return_invalid_items: bool = False
    ) -> list[FactCheckedItem]:
        if self.reject_inputs_expected_to_exceed_max_size:
            await self.__check_list_is_short_enough()
        initial_items = await self.__brainstorm_all_possible_items()
        deduplicated_list = await self.__deduplicate_list(initial_items)
        fact_checked_list = await self.__fact_check_list(deduplicated_list)
        if return_invalid_items:
            return fact_checked_list
        return [result for result in fact_checked_list if result.is_valid]

    async def __check_list_is_short_enough(self) -> None:
        model = BasicLlm(temperature=0)
        prompt = clean_indents(
            f"""
            If you were to make a full list of all '{self.type_of_thing_to_generate}' how long do you think that list would be?

            If you think there will be more than {self.MAX_ITEMS_IN_LIST}. Please say YES. Otherwise, please say NO. Please give this answer in all caps. Only give one instance of YES or NO in all caps.

            Lets take this step by step:
            1) Come up with a few examples of '{self.type_of_thing_to_generate}'.
            2) Come up with all the different categories of '{self.type_of_thing_to_generate}' you can think of.
            3) Give an estimate of how many different categories of '{self.type_of_thing_to_generate}' you think there are.
            4) Give your answer as "YES" or "NO" about whether you think a full list will be longer than {self.MAX_ITEMS_IN_LIST} items.
            """
        )
        fails_test = await model.invoke_and_check_for_boolean_keyword(prompt)
        if fails_test:
            raise ValueError(
                f"List of '{self.type_of_thing_to_generate}' will probably has too many items to be accurate to generate"
            )

    async def __brainstorm_all_possible_items(
        self,
    ) -> list[InitialListItem]:
        example_items = [
            InitialListItem(
                item_name="Item 1",
                description="This item happened in year X in situation Z, etc...",
                is_uncertain=False,
                initial_citations=["[1]", "[5]"],
            ),
            InitialListItem(
                item_name="Item 2",
                description="This item happened in year X in situation Z, etc...",
                is_uncertain=True,
                initial_citations=None,
            ),
        ]
        prompt = clean_indents(
            f"""
            # Instructions
            You are a expert scholar and advanced researcher trying to track down all instances of "{self.type_of_thing_to_generate}" for a paper you are publishing on the topic. Find and come up with an exhaustive list of all instances of "{self.type_of_thing_to_generate}". You will be given some search results to help you. When you do please try to intuit and get EVERYTHING you can from it. You will sort through the list later and check validity, but you want to pick up anything you can possibly find
            Give your list as a list of json. Include no other words. Just the json. Include any caveats or notes in the description field.

            # Tips
            Make sure to consider (or attempt searching) for lists that you would find on Wikipedia as this is a good place to find long lists of examples of things.

            # Compensation
            A random sample of your work will is being evaluated by a manager. You will be given a $500 bonus if he cannot find an valid instance of "{self.type_of_thing_to_generate}" that is not in your list (i.e. if you get a fully ehaugtive list even if some are invalid upon later review).

            # Example
            {json.dumps([item.model_dump() for item in example_items])}

            Now list out as many examples of "{self.type_of_thing_to_generate}" as you can.
            Make sure your citations have quotes around them.
            """
        )
        smart_model = BasicLlm(temperature=0.8)
        internet_model = SmartSearcher(
            temperature=1,
            use_brackets_around_citations=False,
            num_searches_to_run=3,
            num_sites_per_search=5,
        )

        regular_calls = [
            smart_model.invoke_and_return_verified_type(prompt, list[InitialListItem])
            for _ in range(self.num_llm_calls_to_find_new_items)
        ]
        internet_calls = [
            internet_model.invoke_and_return_verified_type(
                prompt, list[InitialListItem]
            )
            for _ in range(self.num_internet_searches_to_find_new_items)
        ]
        ask_ai_coroutines = regular_calls + internet_calls
        non_errored_responses, _ = (
            async_batching.run_coroutines_while_removing_and_logging_exceptions(
                ask_ai_coroutines
            )
        )

        if len(non_errored_responses) == 0:
            raise RuntimeError("Could not generate any items for exhaustive list")
        log_message = "\n---\n".join(
            [
                InitialListItem.make_markdown_with_name_and_description(response)
                for response in non_errored_responses
            ]
        )
        logger.info(f"Initial lists generated:\n {log_message}")
        combined_list = [
            item for response in non_errored_responses for item in response
        ]
        return combined_list

    async def __deduplicate_list(
        self, possibly_duplicated_items: list[InitialListItem]
    ) -> list[InitialListItem]:
        name_item_dict: dict[str, InitialListItem] = {}
        for item in possibly_duplicated_items:
            if item.item_name not in name_item_dict:
                name_item_dict[item.item_name] = item
        uniquely_named_items = list(name_item_dict.values())

        strings_to_check = [item.name_plus_description for item in uniquely_named_items]
        deduplicated_strings = await Deduplicator.deduplicate_list_in_batches(
            strings_to_check,
            f"I am a superforecaster trying to find all instances of '{self.type_of_thing_to_generate}' in order to construct a base rate how how many times this thing occurs per year. Don't decide if the items match this type of thing, just deduplicate, as I will be doing fact checking later. When picking between duplicates, choose ones that include more specifics that can help it be fact checked.",
        )

        deduplicated_items = []
        for item in uniquely_named_items:
            if item.name_plus_description in deduplicated_strings:
                deduplicated_items.append(item)
        return deduplicated_items

    async def __fact_check_list(
        self, list_to_check: list[InitialListItem]
    ) -> list[FactCheckedItem]:
        if len(list_to_check) > self.MAX_ITEMS_IN_LIST:
            raise RuntimeError(
                f"Too many items generated to fact check {len(list_to_check)}"
            )

        criteria_list = await self.__generate_criteria()
        fact_check_coroutines = [
            self.__fact_check_item(item.item_name, criteria_list)
            for item in list_to_check
        ]
        fact_check_results = await asyncio.gather(*fact_check_coroutines)
        list_items: list[FactCheckedItem] = []
        for initial_item, fact_check_result in zip(list_to_check, fact_check_results):
            list_items.append(
                FactCheckedItem(
                    **initial_item.model_dump(),
                    fact_check=fact_check_result,
                    type_description=self.type_of_thing_to_generate,
                )
            )

        return list_items

    async def __generate_criteria(self) -> list[Criteria]:
        _, example_thing_to_generate = (
            self.__get_example_list_item_and_thing_to_generate()
        )
        example_criteria = self.__get_example_criteria()

        prompt = clean_indents(
            f"""
            You are an fact-checker, and you are about to fact check a list of '{self.type_of_thing_to_generate}'.
            Generate 2-5 criteria that can be used to fact-check each item in the list.
            Each criterion should have a short name and a description.

            Ensure that the criteria are specific to '{self.type_of_thing_to_generate}' and can be used to verify the validity of each item.

            For example, if the type of thing to generate was "{example_thing_to_generate}", the criteria might look like this:
            {json.dumps([c.model_dump() for c in example_criteria])}

            Please give your response in JSON format as a list of dictionaries. Give only the list of JSON, and nothing else.
            Now, please generate criteria specific to '{self.type_of_thing_to_generate}'.
            """
        )

        model = BasicLlm(temperature=0.2)
        criteria_list = await model.invoke_and_return_verified_type(
            prompt, list[Criteria]
        )

        return criteria_list

    async def __fact_check_item(
        self, item: str, criteria_list: list[Criteria]
    ) -> FactCheck:
        try:
            example_item, example_thing_to_generate = (
                self.__get_example_list_item_and_thing_to_generate()
            )
            example_criteria_assessments = self.__get_example_criteria_assessments()

            prompt = clean_indents(
                f"""
                ## Intro
                You are a super genius expert fact-checker verifying information about "{self.type_of_thing_to_generate}".
                You'll be given an item and a list of criteria to check it against:

                ## Instructions
                For each criterion, provide:
                1. A brief assessment of the item's validity according to this criterion
                2. Determine if the item is valid (true), invalid (false), or if there's not enough information to decide (null).
                3. If possible, provide the citation number to where you found the information.
                4. Make sure to give an exact copy of the short name and description of the criterion.
                5. Give your response in JSON format as a list of objects. Do not give any other information, just the list.

                ## Verification
                You are being paid hourly, and a random sample of your work will be verified by your peers. If the verfiication passes, you will get a bonus of $500 this month, so please be very careful and thorough.

                ## Example
                Here's an example of how to format your response, based on a different input (Note the information here isn't necessarily accurate):
                Example item: {example_item}
                Example claim: "{example_item}" is an instance of "{example_thing_to_generate}"
                Example output:
                {json.dumps([assessment.model_dump() for assessment in example_criteria_assessments])}

                ## Your Turn
                Item: {item}
                Claim: "{item}" is an instance of "{self.type_of_thing_to_generate}"

                Criteria:
                {json.dumps([criteria.model_dump() for criteria in criteria_list])}


                Now, please provide your assessment for the given item and criteria.
                Provide only the JSON list in your response, nothing else.
                """
            )
            model = SmartSearcher(
                temperature=0.8,
                use_brackets_around_citations=False,
                num_searches_to_run=3,
                num_sites_per_search=4,
            )
            fact_check_tasks = [
                model.invoke_and_return_verified_type(prompt, list[CriteriaAssessment])
                for _ in range(self.number_of_fact_checks_per_item)
            ]
            fact_check_results = await asyncio.gather(*fact_check_tasks)
            fact_checks = [
                FactCheck(criteria_assessments=assessments)
                for assessments in fact_check_results
            ]
            unified_fact_check = self.__unify_fact_checks(fact_checks)
            logger.info(
                f"Fact checked item: {item}. Is valid: {unified_fact_check.is_valid}. Assessments: {[assessment.model_dump() for assessment in unified_fact_check.criteria_assessments]}"
            )
        except Exception as e:
            logger.exception(f"Error fact checking item {item}: {e}")
            criteria_assessments = [
                CriteriaAssessment(
                    short_name=criteria.short_name,
                    description=criteria.description,
                    validity_assessment="Error while fact checking",
                    is_valid_or_unknown=None,
                    citation_proving_assessment=None,
                )
                for criteria in criteria_list
            ]
            return FactCheck(criteria_assessments=criteria_assessments)
        return unified_fact_check

    @classmethod
    def __unify_fact_checks(cls, fact_checks: list[FactCheck]) -> FactCheck:
        number_of_criteria = len(fact_checks[0].criteria_assessments)
        assert all(
            len(fact_check.criteria_assessments) == number_of_criteria
            for fact_check in fact_checks
        ), "All fact checks must have the same number of criteria assessments"
        unified_criteria_assessments: list[CriteriaAssessment] = []
        for i in range(number_of_criteria):
            all_assessments_of_single_criteria = [
                fact_check.criteria_assessments[i] for fact_check in fact_checks
            ]
            unified_criteria_assessment = cls.__combined_criteria_assessments(
                all_assessments_of_single_criteria
            )
            logger.info(
                f"For criteria {unified_criteria_assessment.short_name}: Initially got {[assessment.is_valid_or_unknown for assessment in all_assessments_of_single_criteria]}. Unified to {unified_criteria_assessment.is_valid_or_unknown}"
            )
            unified_criteria_assessments.append(unified_criteria_assessment)
        return FactCheck(criteria_assessments=unified_criteria_assessments)

    @classmethod
    def __combined_criteria_assessments(
        cls, criteria_assessments: list[CriteriaAssessment]
    ) -> CriteriaAssessment:
        assert all(
            assessment.short_name == criteria_assessments[0].short_name
            for assessment in criteria_assessments
        ), "All criteria assessments must have the same short name"
        yes_votes = 0
        no_votes = 0
        null_votes = 0
        for assessment in criteria_assessments:
            if assessment.is_valid_or_unknown is True:
                yes_votes += 1
            elif assessment.is_valid_or_unknown is False:
                no_votes += 1
            else:
                null_votes += 1
        if yes_votes > no_votes and yes_votes > null_votes:
            valid = True
        elif no_votes > yes_votes and no_votes > null_votes:
            valid = False
        else:
            valid = None
        assessment_with_matching_validity_type = [
            assessment
            for assessment in criteria_assessments
            if assessment.is_valid_or_unknown == valid
        ][0]
        return assessment_with_matching_validity_type

    @staticmethod
    def __get_example_criteria() -> list[Criteria]:
        assessments = NicheListResearcher.__get_example_criteria_assessments()
        return [
            Criteria(
                short_name=assessment.short_name,
                description=assessment.description,
            )
            for assessment in assessments
        ]

    @staticmethod
    def __get_example_list_item_and_thing_to_generate() -> tuple[str, str]:
        return (
            "TechInnovate vs Apple",
            "Times Apple was sued by someone for patent violations between 2000 and 2024",
        )

    @staticmethod
    def __get_example_criteria_assessments() -> list[CriteriaAssessment]:
        return [
            CriteriaAssessment(
                short_name="Date Range",
                description="Ensure that the lawsuit and its resolution occurred between 2000 and 2024",
                validity_assessment="The lawsuit was filed in 2015, which is within the specified date range.",
                is_valid_or_unknown=True,
                citation_proving_assessment="[1](https://www.techinnovate-v-apple.com/lawsuit-details)",
            ),
            CriteriaAssessment(
                short_name="Apple Sued",
                description="Verify it was a lawsuit suing Apple",
                validity_assessment="The item incorrectly states that Apple was sued. In fact, Apple was the plaintiff in this case, suing TechInnovate for patent infringement.",
                is_valid_or_unknown=False,
                citation_proving_assessment="[17](https://www.courtrecords.gov/cases/apple-v-techinnovate-2015)",
            ),
            CriteriaAssessment(
                short_name="Successful",
                description="Verify that the lawsuit resulted in a court ruling or settlement in favor of the plaintiff against Apple",
                validity_assessment="There is insufficient information to determine the outcome of the lawsuit. The case is still ongoing as of the last available update.",
                is_valid_or_unknown=None,
                citation_proving_assessment=None,
            ),
            CriteriaAssessment(
                short_name="Patent Violation",
                description="Confirm that the lawsuit specifically involved patent infringement claims against Apple",
                validity_assessment="The lawsuit does indeed involve patent infringement claims, but they are related to display technology, not touchscreen technology as stated in the item.",
                is_valid_or_unknown=True,
                citation_proving_assessment="[7](https://www.ip-watch.org/apple-techinnovate-lawsuit?p=3#patent-details)",
            ),
        ]
