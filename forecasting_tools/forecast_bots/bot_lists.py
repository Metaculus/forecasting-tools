from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.community.q1_veritas_bot import (
    Q1VeritasBot,
)
from forecasting_tools.forecast_bots.community.q4_veritas_bot import (
    Q4VeritasBot,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.main_bot import MainBot
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot2025,
)
from forecasting_tools.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot2024,
)
from forecasting_tools.forecast_bots.official_bots.q4_template_bot import (
    Q4TemplateBot2024,
)
from forecasting_tools.forecast_bots.template_bot import TemplateBot


def get_all_important_bot_classes() -> list[type[ForecastBot]]:
    return [
        MainBot,
        TemplateBot,
        Q1TemplateBot2025,
        Q3TemplateBot2024,
        Q4TemplateBot2024,
        Q4VeritasBot,
        Q1VeritasBot,
    ]


def get_all_bots_for_doing_cheap_tests() -> list[ForecastBot]:
    return [TemplateBot()]


def get_all_bot_question_type_pairs_for_cheap_tests() -> (
    list[tuple[type[MetaculusQuestion], ForecastBot]]
):
    question_type_and_bot_pairs = []
    for question_type in DataOrganizer.get_all_question_types():
        for bot in get_all_bots_for_doing_cheap_tests():
            try:  # Skip questions that don't have a report type
                report_type = DataOrganizer.get_report_type_for_question_type(
                    question_type
                )
                assert report_type is not None
            except Exception:
                continue
            question_type_and_bot_pairs.append((question_type, bot))
    return question_type_and_bot_pairs
