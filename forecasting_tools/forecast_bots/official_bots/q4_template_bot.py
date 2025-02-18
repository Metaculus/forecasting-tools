from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot,
)
from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher


class Q4TemplateBot(Q3TemplateBot):
    """
    Q4 Template Bot was the same as Q3 other than switching out for AskNews
    """

    async def run_research(self, question: MetaculusQuestion) -> str:
        news = AskNewsSearcher().get_formatted_news(question.question_text)
        return news
