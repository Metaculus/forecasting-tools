from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastBot,
)
from forecasting_tools.forecasting.forecast_bots.main_bot import MainBot
from forecasting_tools.forecasting.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot,
)
from forecasting_tools.forecasting.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot,
)
from forecasting_tools.forecasting.forecast_bots.official_bots.q4_template_bot import (
    Q4TemplateBot,
)
from forecasting_tools.forecasting.forecast_bots.official_bots.q4_veritas_bot import (
    Q4VeritasBot,
)
from forecasting_tools.forecasting.forecast_bots.template_bot import (
    TemplateBot,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    MetaculusQuestion,
)
from forecasting_tools.forecasting.questions_and_reports.report_organizer import (
    ReportOrganizer,
)


def get_all_official_bot_classes() -> list[type[ForecastBot]]:
    return [
        MainBot,
        TemplateBot,
        Q1TemplateBot,
        Q3TemplateBot,
        Q4TemplateBot,
        Q4VeritasBot,
    ]


def get_all_bots_for_doing_cheap_tests() -> list[ForecastBot]:
    return [
        TemplateBot(),
    ]


def get_all_bot_question_type_pairs_for_cheap_tests() -> (
    list[tuple[type[MetaculusQuestion], ForecastBot]]
):
    return [
        (question_type, bot)
        for question_type in ReportOrganizer.get_all_question_types()
        for bot in get_all_bots_for_doing_cheap_tests()
    ]
