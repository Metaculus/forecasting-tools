from forecasting_tools.forecasting.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot,
)
from forecasting_tools.forecasting.helpers.asknews_searcher import (
    AskNewsSearcher,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    MetaculusQuestion,
)


class Q4TemplateBot(Q3TemplateBot):
    """
    Q4 Template Bot was the same as Q3 other than switching out for AskNews
    """

    async def run_research(self, question: MetaculusQuestion) -> str:
        news = AskNewsSearcher.get_formatted_news(question.question_text)
        return news
