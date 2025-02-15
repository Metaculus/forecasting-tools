import logging
import os
from datetime import datetime

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.exa_searcher import ExaSearcher
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastBot,
)
from forecasting_tools.forecasting.helpers.asknews_searcher import (
    AskNewsSearcher,
)
from forecasting_tools.forecasting.helpers.prediction_extraction import (
    extract_final_percentage,
    extract_numeric_distribution_from_list_of_percentile_number_and_probability,
    extract_option_list_with_percentage_afterwards,
)
from forecasting_tools.forecasting.helpers.smart_searcher import SmartSearcher
from forecasting_tools.forecasting.questions_and_reports.forecast_report import (
    ReasonedPrediction,
)
from forecasting_tools.forecasting.questions_and_reports.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.forecasting.questions_and_reports.numeric_report import (
    NumericDistribution,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

logger = logging.getLogger(__name__)


class Q1TemplateBot(ForecastBot):
    """
    This is a copy of the template bot for Q1 2025 Metaculus AI Tournament.
    The official bots on the leaderboard use AskNews in Q1.

    The main entry point of this bot is 'forecast_on_tournament' in the parent class.
    """

    async def run_research(self, question: MetaculusQuestion) -> str:
        research = ""
        if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
            research = AskNewsSearcher().get_formatted_news(
                question.question_text
            )
        elif os.getenv("EXA_API_KEY"):
            research = await self._call_exa_smart_searcher(
                question.question_text
            )
        elif os.getenv("PERPLEXITY_API_KEY"):
            research = await self._call_perplexity(question.question_text)
        else:
            research = ""
        logger.info(f"Found Research for {question.page_url}:\n{research}")
        return research

    async def _call_perplexity(self, question: str) -> str:
        system_prompt = clean_indents(
            """
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.
            """
        )
        model = GeneralLlm(
            model="perplexity/sonar-pro",  # Regular sonar is cheaper, but does only 1 search.
            temperature=0.1,
            system_prompt=system_prompt,
        )
        response = await model.invoke(question)
        return response

    async def _call_exa_smart_searcher(self, question: str) -> str:
        if os.getenv("OPENAI_API_KEY") is None:
            searcher = ExaSearcher(
                include_highlights=True,
                num_results=10,
            )
            highlights = (
                await searcher.invoke_for_highlights_in_relevance_order(
                    question
                )
            )
            prioritized_highlights = highlights[:10]
            combined_highlights = ""
            for i, highlight in enumerate(prioritized_highlights):
                combined_highlights += f'[Highlight {i+1}]:\nTitle: {highlight.source.title}\nURL: {highlight.source.url}\nText: "{highlight.highlight_text}"\n\n'
            response = combined_highlights
        else:
            searcher = SmartSearcher(
                model=self._get_final_decision_llm(),
                temperature=0,
                num_searches_to_run=2,
                num_sites_per_search=10,
            )
            prompt = (
                "You are an assistant to a superforecaster. The superforecaster will give"
                "you a question they intend to forecast on. To be a great assistant, you generate"
                "a concise but detailed rundown of the most relevant news, including if the question"
                "would resolve Yes or No based on current information. You do not produce forecasts yourself."
                f"\n\nThe question is: {question}"
            )
            response = await searcher.invoke(prompt)

        return response

    def _get_final_decision_llm(self) -> GeneralLlm:
        model = None
        if os.getenv("OPENAI_API_KEY"):
            model = GeneralLlm(model="gpt-4o", temperature=0.3)
        elif os.getenv("METACULUS_TOKEN"):
            model = GeneralLlm(model="metaculus/gpt-4o", temperature=0.3)
        elif os.getenv("ANTHROPIC_API_KEY"):
            model = GeneralLlm(
                model="claude-3-5-sonnet-20241022", temperature=0.3
            )
        else:
            model = GeneralLlm(model="gpt-4o", temperature=0.3)
        return model

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self._get_final_decision_llm().invoke(prompt)
        prediction = extract_final_percentage(
            reasoning, max_prediction=1, min_prediction=0
        )
        logger.info(
            f"Forecasted {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self._get_final_decision_llm().invoke(prompt)
        prediction = extract_option_list_with_percentage_afterwards(
            reasoning, question.options
        )
        logger.info(
            f"Forecasted {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self._get_final_decision_llm().invoke(prompt)
        prediction = extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question
        )
        logger.info(
            f"Forecasted {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message
