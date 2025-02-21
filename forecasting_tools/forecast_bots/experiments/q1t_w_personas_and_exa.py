import logging
import random
from datetime import datetime

import typeguard
from pydantic import BaseModel

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ScratchPad
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot2025,
)
from forecasting_tools.forecast_helpers.prediction_extractor import (
    PredictionExtractor,
)
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher

logger = logging.getLogger(__name__)


class Persona(BaseModel):
    occupation: str
    expertise_areas: list[str]
    background: str

    def create_persona_description(self) -> str:
        return clean_indents(
            f"""
            Your past history and expertise:
            - Occupation: {self.occupation}
            - Expertise areas: {self.expertise_areas}
            - Background: {self.background}
            """
        )


class PersonaScratchpad(ScratchPad):
    question: MetaculusQuestion
    personas: list[Persona | None]

    def get_random_persona(self) -> Persona | None:
        random_persona = random.choice(self.personas)
        return random_persona


class Q1TemplateWithPersonasAndExa(Q1TemplateBot2025):

    def __init__(
        self,
        *,
        research_reports_per_question: int = 5,
        predictions_per_research_report: int = 5,
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = True,
    ) -> None:
        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
        )
        self.scratch_pads: list[PersonaScratchpad] = []

    def _get_final_decision_llm(self) -> GeneralLlm:
        return GeneralLlm(model="gpt-4o", temperature=0.3)

    async def _initialize_scratchpad(
        self, question: MetaculusQuestion
    ) -> PersonaScratchpad:
        personas = await self._create_personas(question)
        personas_with_none = personas + [None]
        new_scratchpad = PersonaScratchpad(
            question=question, personas=personas_with_none
        )
        return new_scratchpad

    async def run_research(self, question: MetaculusQuestion) -> str:
        searcher = SmartSearcher(
            temperature=0.3,
            num_searches_to_run=3,
            num_sites_per_search=10,
        )
        persona_message = await self._get_persona_message(question)
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster. The superforecaster will give
            you a question they intend to forecast on. To be a great assistant, you generate
            a concise but detailed rundown of the most relevant news. You do not produce forecasts yourself.

            {persona_message}

            The question is:
            {question.question_text}

            Background information:
            {question.background_info if question.background_info else "No background information provided."}

            Resolution criteria:
            {question.resolution_criteria if question.resolution_criteria else "No resolution criteria provided."}

            Fine print:
            {question.fine_print if question.fine_print else "No fine print provided."}

            Look for what else you would be need to know to forecast well on this question.
            """
        )
        response = await searcher.invoke(prompt)
        return response

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        persona_message = await self._get_persona_message(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            {persona_message}

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
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )
        reasoning = f"Persona:\n{persona_message}\n\nReasoning:\n{reasoning}"
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        persona_message = await self._get_persona_message(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            {persona_message}

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
        prediction = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )
        reasoning = f"Persona:\n{persona_message}\n\nReasoning:\n{reasoning}"
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        persona_message = await self._get_persona_message(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            {persona_message}

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
        reasoning = f"Persona:\n{persona_message}\n\nReasoning:\n{reasoning}"
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _create_personas(
        self, question: MetaculusQuestion
    ) -> list[Persona]:
        number_of_personas = (
            self.research_reports_per_question
            * self.predictions_per_research_report
            + self.research_reports_per_question
        )
        prompt = clean_indents(
            f"""
            You are one of Philip Tetlock's superforecasters trying to put together a panel of experts to forecast on a question.
            The question is:
            {question.question_text}

            Please come up with {number_of_personas} different personas of experts that would be relevant to this question.
            Return your answer as a list of Persona objects. Remember to give a JSON list even if there is only one item.
            {GeneralLlm.get_schema_format_instructions_for_pydantic_type(Persona)}
            """
        )
        personas = await GeneralLlm(
            model="gpt-4o", temperature=0.8
        ).invoke_and_return_verified_type(prompt, list[Persona])
        logger.debug(f"Personas Chosen:\n{personas}")
        if len(personas) != number_of_personas:
            logger.warning(
                f"Expected {number_of_personas} personas, but got {len(personas)}"
            )
        return personas

    async def _get_persona_message(self, question: MetaculusQuestion) -> str:
        scratch_pad = await self._get_scratchpad(question)
        persona = scratch_pad.get_random_persona()
        if persona is None:
            persona_message = ""
        else:
            persona_message = persona.create_persona_description()
        if len(scratch_pad.personas) > 1:
            try:
                scratch_pad.personas.remove(persona)
            except ValueError:
                logger.error(
                    f"Could not remove persona '{persona}' from scratchpad"
                )
        return persona_message

    async def _get_scratchpad(
        self, question: MetaculusQuestion
    ) -> PersonaScratchpad:
        scratchpad = await super()._get_scratchpad(question)
        return typeguard.check_type(scratchpad, PersonaScratchpad)
