from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.forecasting.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot,
)
from forecasting_tools.forecasting.helpers.smart_searcher import SmartSearcher
from forecasting_tools.forecasting.questions_and_reports.questions import (
    MetaculusQuestion,
)


class Q3TemplateWithExa(Q3TemplateBot):

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

        response = await SmartSearcher(
            temperature=0.1, num_searches_to_run=2, num_sites_per_search=10
        ).invoke(prompt)
        return response