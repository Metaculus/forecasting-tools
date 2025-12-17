from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.experiments.q1_veritas_bot import Q1VeritasBot
from forecasting_tools.forecast_bots.experiments.q2t_w_decomposition import (
    Q2TemplateBotWithDecompositionV1,
    Q2TemplateBotWithDecompositionV2,
)
from forecasting_tools.forecast_bots.experiments.q4_veritas_bot import Q4VeritasBot
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.main_bot import MainBot
from forecasting_tools.forecast_bots.official_bots.gpt_4_1_optimized_bot import (
    GPT41OptimizedBot,
)
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot2025,
)
from forecasting_tools.forecast_bots.official_bots.q2_template_bot import (
    Q2TemplateBot2025,
)
from forecasting_tools.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot2024,
)
from forecasting_tools.forecast_bots.official_bots.q4_template_bot import (
    Q4TemplateBot2024,
)
from forecasting_tools.forecast_bots.official_bots.research_only_bot_2025_fall import (
    FallResearchOnlyBot2025,
)
from forecasting_tools.forecast_bots.official_bots.template_bot_2025_fall import (
    FallTemplateBot2025,
)
from forecasting_tools.forecast_bots.official_bots.template_bot_2026_spring import (
    SpringTemplateBot2026,
)
from forecasting_tools.forecast_bots.official_bots.uniform_probability_bot import (
    UniformProbabilityBot,
)
from forecasting_tools.forecast_bots.template_bot import TemplateBot


def get_all_important_bot_classes() -> list[type[ForecastBot]]:
    return [
        MainBot,
        TemplateBot,
        Q2TemplateBot2025,
        Q1TemplateBot2025,
        Q4TemplateBot2024,
        Q3TemplateBot2024,
        FallTemplateBot2025,
        FallResearchOnlyBot2025,
        Q1VeritasBot,
        Q4VeritasBot,
        UniformProbabilityBot,
        Q2TemplateBotWithDecompositionV1,
        Q2TemplateBotWithDecompositionV2,
        FallResearchOnlyBot2025,
        SpringTemplateBot2026,
        GPT41OptimizedBot,
    ]


def get_all_bots_for_doing_cheap_tests() -> list[ForecastBot]:
    return [
        TemplateBot(llms={"default": "openrouter/openai/gpt-4.1-nano"}),
        FallResearchOnlyBot2025(llms={"default": "openrouter/perplexity/sonar"}),
        GPT41OptimizedBot(llms={"default": "openrouter/openai/gpt-4.1-nano"}),
        UniformProbabilityBot(),
    ]


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
