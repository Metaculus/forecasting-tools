from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from forecasting_tools.agents_and_tools.base_rates.deduplicator import Deduplicator
from forecasting_tools.agents_and_tools.deprecated.configured_llms import (
    AdvancedLlm,
    BasicLlm,
)
from forecasting_tools.agents_and_tools.deprecated.research_coordinator import (
    ResearchCoordinator,
)
from forecasting_tools.agents_and_tools.research.smart_searcher import SmartSearcher
from forecasting_tools.helpers.metaculus_api import MetaculusQuestion
from forecasting_tools.util import async_batching
from forecasting_tools.util.misc import (
    clean_indents,
    extract_url_from_markdown_link,
    is_markdown_citation,
)

logger = logging.getLogger(__name__)


class KeyFactorsResearcher:

    @classmethod
    async def find_and_sort_key_factors(
        cls,
        metaculus_question: MetaculusQuestion,
        num_key_factors_to_return: int = 5,
        num_questions_to_research_with: int = 26,
    ) -> list[ScoredKeyFactor]:
        num_background_questions = num_questions_to_research_with // 2
        num_base_rate_questions = (
            num_questions_to_research_with - num_background_questions
        )
        background_key_factors = await cls.__find_background_key_factors(
            num_background_questions, metaculus_question
        )
        base_rate_key_factors = await cls.__find_base_rate_key_factors(
            num_base_rate_questions, metaculus_question
        )
        combined_key_factors = background_key_factors + base_rate_key_factors
        scored_key_factors = await cls.__score_key_factor_list(
            metaculus_question, combined_key_factors
        )
        sorted_key_factors = sorted(
            scored_key_factors, key=lambda x: x.score, reverse=True
        )
        top_key_factors = sorted_key_factors[: num_key_factors_to_return * 2]
        prioritized_key_factors = await cls.__prioritize_key_factors(
            metaculus_question, top_key_factors, num_key_factors_to_return
        )
        deduplicated_key_factors = await cls.__deduplicate_key_factors(
            prioritized_key_factors, metaculus_question
        )
        logger.info(
            f"Found {len(deduplicated_key_factors)} final key factors (prioritized, deduplicated and filtering for top scores)"
        )
        return deduplicated_key_factors

    @classmethod
    async def __find_background_key_factors(
        cls,
        num_background_questions: int,
        metaculus_question: MetaculusQuestion,
    ) -> list[KeyFactor]:
        research_manager = ResearchCoordinator(metaculus_question)
        background_questions = await research_manager.brainstorm_background_questions(
            num_background_questions
        )
        background_key_factors = await cls.__find_key_factors_for_questions(
            background_questions
        )
        return background_key_factors

    @classmethod
    async def __find_base_rate_key_factors(
        cls,
        num_base_rate_questions: int,
        metaculus_question: MetaculusQuestion,
    ) -> list[KeyFactor]:
        research_manager = ResearchCoordinator(metaculus_question)
        base_rate_questions = await research_manager.brainstorm_base_rate_questions(
            num_base_rate_questions
        )
        base_rate_key_factors = await cls.__find_key_factors_for_questions(
            base_rate_questions
        )
        return base_rate_key_factors

    @classmethod
    async def __find_key_factors_for_questions(
        cls, questions: list[str]
    ) -> list[KeyFactor]:
        key_factor_tasks = [
            cls.__find_key_factors_for_question(question) for question in questions
        ]
        key_factors, _ = (
            async_batching.run_coroutines_while_removing_and_logging_exceptions(
                key_factor_tasks
            )
        )
        flattened_key_factors = [
            factor for sublist in key_factors for factor in sublist
        ]
        logger.info(
            f"Found {len(flattened_key_factors)} key factors. Now scoring them."
        )
        return flattened_key_factors

    @classmethod
    async def __find_key_factors_for_question(
        cls, question_text: str
    ) -> list[KeyFactor]:
        prompt = clean_indents(
            f"""
            You are a top tier expert and assistant to a superforecaster.

            Analyze the following question and provide key factors that could influence the outcome of the larger question.
            Include base rates, pros (factors supporting a positive outcome), and cons (factors supporting a negative outcome).
            Each factor should be a single sentence and include a citation.

            When making a key factor, try to be very specific if you can.
            - If you have the ability to use a number/stat/etc. please do so.
            - If you can state a date something happened, please do so.
            - Also when it makes sense, quote what you care about, rather than paraphrasing.
            - Include the source in your key factor, trying to pick a name that people know and can assign credence to(e.g. "The Guardian says that ..." or "A study found that ...").
            - Include enough context so that the key factor can make sense on its own (i.e. don't use pronouns like "it" or "they" without specifying what "it" or "they" is).

            Please pay attention to publish dates, and don't put down any key factors that are outdated (put emphasis on this).

            Question: {question_text}

            Provide your answer as a list of JSON objects, each representing a KeyFigure with the following format:
            {{
                "text": "The key factor statement",
                "factor_type": "base_rate" or "pro" or "con",
                "citation": "citation number in brackets(e.g. [1])",
                "source_publish_date": "YYYY-MM-DD" (or null if unknown)
            }}

            Return only the list of JSON objects and nothing else.
            """
        )

        smart_searcher = SmartSearcher(
            use_brackets_around_citations=False,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        key_figures = await smart_searcher.invoke_and_return_verified_type(
            prompt, list[KeyFactor]
        )

        return key_figures

    @classmethod
    async def __deduplicate_key_factors(
        cls,
        key_factors: list[ScoredKeyFactor],
        metaculus_question: MetaculusQuestion,
    ) -> list[ScoredKeyFactor]:
        strings_to_check = [factor.text for factor in key_factors]
        prompt_context = (
            "You are an assistant to a superforecaster trying to get a list of "
            "key factors to help answer a question on Metaculus. "
            "You 1) want to deduplicate any that say the same thing "
            "(thus worthless to read twice)"
            "and 2) want to remove anything that was already in the "
            "question's background information (duplicating background knowledge)"
            f"\n\nQuestion: {metaculus_question.give_question_details_as_markdown()}"
        )
        deduplicated_strings = await Deduplicator.deduplicate_list_in_batches(
            strings_to_check,
            prompt_context=prompt_context,
        )
        deduplicated_factors: list[ScoredKeyFactor] = []
        for factor in key_factors:
            if factor.text in deduplicated_strings:
                deduplicated_factors.append(factor)
        return deduplicated_factors

    @classmethod
    async def __score_key_factor_list(
        cls,
        metaculus_question: MetaculusQuestion,
        key_factors: list[KeyFactor],
    ) -> list[ScoredKeyFactor]:
        scoring_coroutines = [
            cls.__score_key_factor(metaculus_question.question_text, factor)
            for factor in key_factors
        ]
        scored_factors, _ = (
            async_batching.run_coroutines_while_removing_and_logging_exceptions(
                scoring_coroutines
            )
        )
        return scored_factors

    @classmethod
    async def __score_key_factor(
        cls, question: str, key_factor: KeyFactor
    ) -> ScoredKeyFactor:
        pydantic_prompt = BasicLlm.get_schema_format_instructions_for_pydantic_type(
            ScoreCard
        )
        prompt = clean_indents(
            f"""
            # Instructions
            You are a superforecaster and an expert at evaluating the importance and relevance of key factors in forecasting questions.
            Your job is to score the quality of a key factor using a score card.

            # Score Card Format
            {pydantic_prompt}

            # Grading Scale for {ScoreCardGrade.__class__.__name__}
            - {ScoreCardGrade.VERY_BAD.value}: Generally poor quality
            - {ScoreCardGrade.BAD.value}: Below average quality
            - {ScoreCardGrade.OK.value}: Below average quality
            - {ScoreCardGrade.GOOD.value}: Above average quality
            - {ScoreCardGrade.VERY_GOOD.value}: Exceptional quality

            # Your Turn
            Please score the following key factor:
            Question: {question}
            Key Factor: {key_factor.text}
            Citation: {key_factor.citation}
            Source Publish Date: {key_factor.source_publish_date.strftime("%Y-%m-%d") if key_factor.source_publish_date else "Unknown"}

            Remember please provide some reasoning, then return a json object following the format specified in the instructions.
            """
        )

        model = BasicLlm(temperature=0)
        score_card = await model.invoke_and_return_verified_type(prompt, ScoreCard)
        logger.info(
            f"Score: {score_card.calculated_score} for key factor: {key_factor.text}: {score_card}"
        )

        return ScoredKeyFactor(
            **key_factor.model_dump(),
            score_card=score_card,
        )

    @classmethod
    async def __prioritize_key_factors(
        cls,
        metaculus_question: MetaculusQuestion,
        key_factors_to_compare: list[ScoredKeyFactor],
        num_factors_to_return: int,
    ) -> list[ScoredKeyFactor]:
        assert len(key_factors_to_compare) < 25, "Too many key factors to compare"
        assert len(key_factors_to_compare) >= num_factors_to_return
        prompt = clean_indents(
            f"""
            You are a superforecaster analyzing key factors for the following question in triple backticks:
            ```
            {metaculus_question.give_question_details_as_markdown()}
            ```

            I have a list of key factors that could influence this question. Each factor has been scored based on various criteria.
            Your task is to select the {num_factors_to_return} most important and diverse factors that would be most useful for forecasting this question.

            Consider:
            1. The overall quality and score of each factor
            2. The complementary nature of the information (avoid redundant information)
            3. The practical usefulness for making a forecast

            Key Factors:
            {[f"{i}. {factor.display_text}" for i, factor in enumerate(key_factors_to_compare)]}

            Return only a list of numbers corresponding to the factors you select, in order of importance. For example: [3, 1, 4, 0]
            """
        )

        model = AdvancedLlm()
        selected_indices = await model.invoke_and_return_verified_type(
            prompt, list[int]
        )
        assert (
            len(selected_indices) == num_factors_to_return
        ), "Not enough factors selected"
        return [key_factors_to_compare[i] for i in selected_indices]


class KeyFactor(BaseModel):
    text: str
    factor_type: KeyFactorType
    citation: str
    source_publish_date: datetime | None

    @field_validator("citation")
    def validate_citation_format(cls, v: str) -> str:
        if not is_markdown_citation(v):
            raise ValueError(
                "Citation must be in the markdown friendly format [number](url)"
            )
        return v

    @property
    def url(self) -> str:
        return extract_url_from_markdown_link(self.citation)


class ScoredKeyFactor(KeyFactor):
    score_card: ScoreCard

    @property
    def score(self) -> int:
        return self.score_card.calculated_score

    @property
    def display_text(self) -> str:
        return f"{self.text} [Source Published on {self.source_publish_date.strftime('%Y-%m-%d') if self.source_publish_date else 'Unknown'}]({self.url})"

    @classmethod
    def turn_key_factors_into_markdown_list(
        cls, key_factors: list[ScoredKeyFactor]
    ) -> str:
        return "\n".join([f"- {factor.display_text}" for factor in key_factors])


class KeyFactorType(str, Enum):
    BASE_RATE = "base_rate"
    PRO = "pro"
    CON = "con"


class ScoreCardGrade(str, Enum):
    VERY_BAD = "very_bad"
    BAD = "bad"
    OK = "ok"
    GOOD = "good"
    VERY_GOOD = "very_good"

    @property
    def grade_as_number(self) -> int:
        score_dict = {
            ScoreCardGrade.VERY_BAD: 1,
            ScoreCardGrade.BAD: 2,
            ScoreCardGrade.OK: 3,
            ScoreCardGrade.GOOD: 4,
            ScoreCardGrade.VERY_GOOD: 5,
        }
        return score_dict[self]


class ScoreCard(BaseModel):
    recency: ScoreCardGrade = Field(
        ...,
        description="How recent was the key factor published relative to how fast information related to this question becomes outdated?",
    )
    relevance: ScoreCardGrade = Field(
        ..., description="How relevant is this key factor to the question?"
    )
    specificness: ScoreCardGrade = Field(
        ...,
        description="How specific is this key factor? Is it vague and thus not useful?",
    )
    predictive_power_and_applicability: ScoreCardGrade = Field(
        ...,
        description="How much weight would this key factor have when predicting the outcome of the question? Would it be useful in forecasting?",
    )
    reputable_source: ScoreCardGrade = Field(
        ..., description="Is this key factor from a reputable source?"
    )
    is_outdated: bool = Field(..., description="Is this key factor outdated?")
    includes_number: bool = Field(
        ...,
        description="Does this key factor mention a number other than a date?",
    )
    includes_date: bool = Field(
        ...,
        description="Does this key factor mention a date/year other than the publish date?",
    )
    is_key_person_quote: bool = Field(
        ...,
        description="Is this key factor a quote from a key decision maker or person related to the question?",
    )
    overall_quality: ScoreCardGrade = Field(
        ...,
        description="Given the above, rate the overall quality of the key factor, and whether a forecaster should use it in forecasting?",
    )

    @property
    def calculated_score(self) -> int:
        final_score = 0
        final_score += 1 * self.recency.grade_as_number
        final_score += 1 * self.relevance.grade_as_number
        final_score += 1 * self.specificness.grade_as_number
        final_score += 2 * self.predictive_power_and_applicability.grade_as_number
        final_score += 1 * self.reputable_source.grade_as_number
        final_score += 2 * self.overall_quality.grade_as_number
        final_score += 5 if self.includes_number else 0
        final_score += 3 if self.includes_date else 0
        final_score += 3 if self.is_key_person_quote else 0

        final_score *= 0.5 if self.is_outdated else 1
        final_score = round(final_score)
        return final_score
