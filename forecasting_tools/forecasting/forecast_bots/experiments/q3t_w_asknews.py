from forecasting_tools.forecasting.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot,
)
from forecasting_tools.forecasting.helpers.asknews_searcher import (
    AskNewsSearcher,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    MetaculusQuestion,
)


class Q3TemplateWithAskNews(Q3TemplateBot):

    async def run_research(self, question: MetaculusQuestion) -> str:
        response = AskNewsSearcher().get_formatted_news(question.question_text)
        return response
