from datetime import datetime

from forecasting_tools.agents_and_tools.deprecated.research_coordinator import (
    ResearchCoordinator,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.questions import BinaryQuestion, MetaculusQuestion
from forecasting_tools.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot2024,
)
from forecasting_tools.helpers.prediction_extractor import PredictionExtractor
from forecasting_tools.util.misc import clean_indents


class Q4VeritasBot(Q3TemplateBot2024):
    FINAL_DECISION_LLM = GeneralLlm(model="gpt-4o", temperature=0.7)

    def __init__(
        self,
        research_reports_per_question: int = 3,
        predictions_per_research_report: int = 5,
        use_research_summary_to_forecast: bool = True,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        number_of_background_questions_to_ask: int = 5,
        number_of_base_rate_questions_to_ask: int = 5,
        number_of_base_rates_to_do_deep_research_on: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            **kwargs,
        )
        self.number_of_background_questions_to_ask = (
            number_of_background_questions_to_ask
        )
        self.number_of_base_rate_questions_to_ask = number_of_base_rate_questions_to_ask
        self.number_of_base_rates_to_do_deep_research_on = (
            number_of_base_rates_to_do_deep_research_on
        )

    async def run_research(self, question: MetaculusQuestion) -> str:
        research_manager = ResearchCoordinator(question)
        combined_markdown = await research_manager.create_full_markdown_research_report(
            self.number_of_background_questions_to_ask,
            self.number_of_base_rate_questions_to_ask,
            self.number_of_base_rates_to_do_deep_research_on,
        )
        return combined_markdown

    async def summarize_research(
        self, question: MetaculusQuestion, research: str
    ) -> str:
        research_coordinator = ResearchCoordinator(question)
        summary_report = await research_coordinator.summarize_full_research_report(
            research
        )
        return summary_report

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        assert isinstance(question, BinaryQuestion), "Question must be a BinaryQuestion"
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.
            Your interview question is:
            {question.question_text}

            Background information:
            {question.background_info if question.background_info else "No background information provided."}

            Resolution criteria:
            {question.resolution_criteria if question.resolution_criteria else "No resolution criteria provided."}

            Fine print:
            {question.fine_print if question.fine_print else "No fine print provided."}


            Your research assistant says:
            ```
            {research}
            ```

            Today is {datetime.now().strftime("%Y-%m-%d")}.


            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) What the outcome would be if nothing changed.
            (c) The most important factors that will influence a successful/unsuccessful resolution.
            (d) What do you not know that should give you pause and lower confidence? Remember people are statistically overconfident.
            (e) What you would forecast if you were to only use historical precedent (i.e. how often this happens in the past) without any current information.
            (f) What you would forecast if there was only a quarter of the time left.
            (g) What you would forecast if there was 4x the time left.

            You write your rationale and then the last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        gpt_forecast = await self.FINAL_DECISION_LLM.invoke(prompt)
        prediction = PredictionExtractor.extract_last_percentage_value(
            gpt_forecast, max_prediction=0.95, min_prediction=0.05
        )
        reasoning = (
            gpt_forecast
            + "\nThe original forecast may have been clamped between 5% and 95%."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
