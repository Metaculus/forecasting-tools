import asyncio
import logging
from datetime import datetime

from forecasting_tools.agents_and_tools.question_generators.question_decomposer import (
    QuestionDecomposer,
)
from forecasting_tools.agents_and_tools.question_generators.question_operationalizer import (
    QuestionOperationalizer,
)
from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.official_bots.q2_template_bot import (
    Q2TemplateBot2025,
)
from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher
from forecasting_tools.helpers.prediction_extractor import PredictionExtractor
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)


class Q2TemplateBotWithDecompositionV1(Q2TemplateBot2025):
    """
    Runs forecasts on decomposed sub questions separately
    """

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        gemini_grounded_model = GeneralLlm.grounded_model(
            model="openrouter/google/gemini-2.5-pro-preview",
            temperature=0.3,
        )
        gemini_model = GeneralLlm(
            model="openrouter/google/gemini-2.5-pro-preview",
            temperature=0.3,
        )
        return {
            "default": gemini_model,
            "summarizer": "gpt-4o",
            "decomposer": gemini_grounded_model,
            "researcher": gemini_grounded_model,
        }

    async def run_research(self, question: MetaculusQuestion) -> str:
        ask_news_research = await AskNewsSearcher().get_formatted_news_async(
            question.question_text
        )

        question_context = clean_indents(
            f"""
            Here are more details for the original question:

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}
            """
        )

        model = self.get_llm("decomposer", "llm")
        decomposition_result = await QuestionDecomposer().decompose_into_questions_deep(
            model=model,
            fuzzy_topic_or_question=question.question_text,
            number_of_questions=5,
            related_research=ask_news_research,
            additional_context=question_context,
        )
        logger.info(f"Decomposition result: {decomposition_result}")

        operationalize_tasks = [
            QuestionOperationalizer(model=model).operationalize_question(
                question_title=question,
                related_research=None,
            )
            for question in decomposition_result.questions
        ]
        operationalized_questions = await asyncio.gather(*operationalize_tasks)

        metaculus_questions = SimpleQuestion.simple_questions_to_metaculus_questions(
            operationalized_questions
        )
        sub_predictor = Q2TemplateBot2025(
            llms=self._llms,
            predictions_per_research_report=5,
            research_reports_per_question=1,
        )
        forecasts = await sub_predictor.forecast_questions(
            metaculus_questions, return_exceptions=True
        )

        formatted_forecasts = ""
        sub_question_bullets = ""
        for forecast in forecasts:
            if isinstance(forecast, BaseException):
                logger.error(f"Error forecasting on question: {forecast}")
                continue
            formatted_forecasts += f"QUESTION: {forecast.question.question_text}\n\n"
            formatted_forecasts += f"PREDICTION: {forecast.make_readable_prediction(forecast.prediction)}\n\n"
            formatted_forecasts += f"SUMMARY: {forecast.summary}\n\n----------------------------------------\n\n"
            sub_question_bullets += f"- {forecast.question.question_text}\n"
        research = clean_indents(
            f"""
            ==================== NEWS ====================

            {ask_news_research}

            ==================== FORECAST HISTORY ====================

            Below are some related questions you have forecasted on before:

            {sub_question_bullets if sub_question_bullets else "<No related questions>"}

            Below are some forecasts you have made on these question:

            {formatted_forecasts if formatted_forecasts else "<No previous forecasts>"}

            ==================== END ====================
            """
        )
        logger.info(research)
        return research


class Q2TemplateBotWithDecompositionV2(Q2TemplateBot2025):
    """
    Runs forecasts on all decomposed questions simultaneously
    """

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        gemini_grounded_model = GeneralLlm.grounded_model(
            model="openrouter/google/gemini-2.5-pro-preview",
            temperature=0.3,
        )
        gemini_model = GeneralLlm(
            model="openrouter/google/gemini-2.5-pro-preview",
            temperature=0.3,
        )
        return {
            "default": gemini_model,
            "summarizer": "gpt-4o-mini",
            "decomposer": gemini_grounded_model,
            "researcher": "asknews/news-summaries",
        }

    async def run_research(self, question: MetaculusQuestion) -> str:
        additional_context = clean_indents(
            f"""
            Here are more details for the original question:

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}
            """
        )
        model = self.get_llm("decomposer", "llm")
        decomposition_result = await QuestionDecomposer().decompose_into_questions_deep(
            model=model,
            fuzzy_topic_or_question=question.question_text,
            number_of_questions=5,
            related_research=None,
            additional_context=additional_context,
        )
        sub_questions = decomposition_result.questions

        # all_questions_to_research = [question.question_text] + sub_questions
        # research_tasks = []
        # for question_title in all_questions_to_research:
        #     prompt = clean_indents(
        #         f"""
        #         You are an assistant to a superforecaster.
        #         The superforecaster will give you a question they intend to forecast on.
        #         To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
        #         You do not produce forecasts yourself.

        #         Question:
        #         {question_title}
        #         """
        #     )
        #     task = self.get_llm("researcher", "llm").invoke(prompt)
        #     research_tasks.append(task)

        # research: list[str] = []
        # # for task in research_tasks:
        # #     await asyncio.sleep(15)
        # #     new_research = await task
        # #     research.append(
        # #         new_research
        # #     )  # TODO: Make this parallel (this is a hack to avoid perplexity rate limits)
        # research: list[str] = await asyncio.gather(*research_tasks)
        # combined_research = "\n".join(research)

        researcher = self.get_llm("researcher")
        assert researcher == "asknews/news-summaries"
        combined_research = await AskNewsSearcher().get_formatted_news_async(
            question.question_text
        )
        note_pad = await self._get_notepad(question)
        note_pad.note_entries[combined_research] = sub_questions

        return combined_research

    async def _get_sub_questions_as_bullets(
        self, question: MetaculusQuestion, research: str
    ) -> str:
        scratch_pad = await self._get_notepad(question)
        sub_questions = scratch_pad.note_entries[research]
        formatted_sub_questions = ""
        for sub_question in sub_questions:
            formatted_sub_questions += f"- {sub_question}\n"
        return formatted_sub_questions

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        formatted_sub_questions = await self._get_sub_questions_as_bullets(
            question, research
        )
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
            (e) Your forecasts for all the following sub-questions
            {formatted_sub_questions}

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=0.99, min_prediction=0.01
        )
        logger.info(f"Forecasted URL {question.page_url} with prediction: {prediction}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        formatted_sub_questions = await self._get_sub_questions_as_bullets(
            question, research
        )
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
            (d) Your forecasts for all the following sub-questions
            {formatted_sub_questions}

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )
        logger.info(f"Forecasted URL {question.page_url} with prediction: {prediction}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        formatted_sub_questions = await self._get_sub_questions_as_bullets(
            question, research
        )
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

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.
            (g) Your forecasts for all the following sub-questions
            {formatted_sub_questions}

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
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
