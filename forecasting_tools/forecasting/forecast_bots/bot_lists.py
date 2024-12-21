from forecasting_tools.forecasting.forecast_bots.forecast_bot import (
    ForecastBot,
)
from forecasting_tools.forecasting.forecast_bots.main_bot import MainBot
from forecasting_tools.forecasting.forecast_bots.template_bot import (
    TemplateBot,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    MetaculusQuestion,
)
from forecasting_tools.forecasting.questions_and_reports.report_organizer import (
    ReportOrganizer,
)


def get_all_bot_classes() -> list[type[ForecastBot]]:
    return [
        MainBot,
        TemplateBot,
    ]


def get_bots_for_cheap_tests() -> list[ForecastBot]:
    return [
        TemplateBot(),
    ]


def get_cheap_bot_question_type_pairs() -> (
    list[tuple[type[MetaculusQuestion], ForecastBot]]
):
    return [
        (question_type, bot)
        for question_type in ReportOrganizer.get_all_question_types()
        for bot in get_bots_for_cheap_tests()
    ]
