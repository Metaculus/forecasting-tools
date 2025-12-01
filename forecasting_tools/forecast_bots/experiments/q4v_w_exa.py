from forecasting_tools.agents_and_tools.research.smart_searcher import SmartSearcher
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.other.q4_veritas_bot import Q4VeritasBot
from forecasting_tools.util.misc import clean_indents


class Q4VeritasWithExa(Q4VeritasBot):

    def __init__(
        self,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        number_of_background_questions_to_ask: int = 0,
        number_of_base_rate_questions_to_ask: int = 0,
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
            number_of_background_questions_to_ask=number_of_background_questions_to_ask,
            number_of_base_rate_questions_to_ask=number_of_base_rate_questions_to_ask,
            number_of_base_rates_to_do_deep_research_on=number_of_base_rates_to_do_deep_research_on,
            **kwargs,
        )

    async def run_research(self, question: MetaculusQuestion) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question.question_text}
            """
        )

        response = await SmartSearcher(temperature=0.1).invoke(prompt)
        return response
