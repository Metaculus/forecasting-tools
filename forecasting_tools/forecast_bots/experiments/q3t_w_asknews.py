from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot2024,
)
from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher


class Q3TemplateWithAskNews(Q3TemplateBot2024):

    async def run_research(self, question: MetaculusQuestion) -> str:
        response = await AskNewsSearcher().get_formatted_news_async(
            question.question_text
        )
        return response
