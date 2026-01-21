import nest_asyncio

from forecasting_tools.agents_and_tools.base_rates.base_rate_researcher import (
    BaseRateResearcher as BaseRateResearcher,
)
from forecasting_tools.agents_and_tools.base_rates.estimator import (
    Estimator as Estimator,
)
from forecasting_tools.agents_and_tools.base_rates.niche_list_researcher import (
    FactCheckedItem as FactCheckedItem,
)
from forecasting_tools.agents_and_tools.base_rates.niche_list_researcher import (
    NicheListResearcher as NicheListResearcher,
)
from forecasting_tools.agents_and_tools.deprecated.question_generator import (
    QuestionGenerator as QuestionGenerator,
)
from forecasting_tools.agents_and_tools.other.data_analyzer import (
    DataAnalyzer as DataAnalyzer,
)
from forecasting_tools.agents_and_tools.question_generators.question_decomposer import (
    QuestionDecomposer as QuestionDecomposer,
)
from forecasting_tools.agents_and_tools.question_generators.question_operationalizer import (
    QuestionOperationalizer as QuestionOperationalizer,
)
from forecasting_tools.agents_and_tools.question_generators.topic_generator import (
    TopicGenerator as TopicGenerator,
)
from forecasting_tools.agents_and_tools.research.computer_use import (
    ComputerUse as ComputerUse,
)
from forecasting_tools.agents_and_tools.research.key_factors_researcher import (
    KeyFactorsResearcher as KeyFactorsResearcher,
)
from forecasting_tools.agents_and_tools.research.key_factors_researcher import (
    ScoredKeyFactor as ScoredKeyFactor,
)
from forecasting_tools.agents_and_tools.research.smart_searcher import (
    SmartSearcher as SmartSearcher,
)
from forecasting_tools.ai_models.ai_utils.openai_utils import (
    VisionMessageData as VisionMessageData,
)
from forecasting_tools.ai_models.deprecated_model_classes.claude35sonnet import (
    Claude35Sonnet as Claude35Sonnet,
)
from forecasting_tools.ai_models.deprecated_model_classes.deepseek_r1 import (
    DeepSeekR1 as DeepSeekR1,
)
from forecasting_tools.ai_models.deprecated_model_classes.gpt4o import Gpt4o as Gpt4o
from forecasting_tools.ai_models.deprecated_model_classes.gpt4ovision import (
    Gpt4oVision as Gpt4oVision,
)
from forecasting_tools.ai_models.deprecated_model_classes.metaculus4o import (
    Gpt4oMetaculusProxy as Gpt4oMetaculusProxy,
)
from forecasting_tools.ai_models.deprecated_model_classes.perplexity import (
    Perplexity as Perplexity,
)
from forecasting_tools.ai_models.exa_searcher import ExaSearcher as ExaSearcher
from forecasting_tools.ai_models.general_llm import GeneralLlm as GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager as MonetaryCostManager,
)
from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import (
    RefreshingBucketRateLimiter as RefreshingBucketRateLimiter,
)
from forecasting_tools.auto_optimizers.bot_optimizer import BotOptimizer as BotOptimizer
from forecasting_tools.cp_benchmarking.benchmark_displayer import (
    run_benchmark_streamlit_page as run_benchmark_streamlit_page,
)
from forecasting_tools.cp_benchmarking.benchmark_for_bot import (
    BenchmarkForBot as BenchmarkForBot,
)
from forecasting_tools.cp_benchmarking.benchmarker import Benchmarker as Benchmarker
from forecasting_tools.data_models.binary_report import (
    BinaryPrediction as BinaryPrediction,
)
from forecasting_tools.data_models.binary_report import BinaryReport as BinaryReport
from forecasting_tools.data_models.coherence_link import (
    DetailedCoherenceLink as DetailedCoherenceLink,
)
from forecasting_tools.data_models.conditional_report import (
    ConditionalReport as ConditionalReport,
)
from forecasting_tools.data_models.data_organizer import DataOrganizer as DataOrganizer
from forecasting_tools.data_models.forecast_report import (
    ForecastReport as ForecastReport,
)
from forecasting_tools.data_models.forecast_report import (
    ReasonedPrediction as ReasonedPrediction,
)
from forecasting_tools.data_models.multiple_choice_report import (
    MultipleChoiceReport as MultipleChoiceReport,
)
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOption as PredictedOption,
)
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOptionList as PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import (
    DatePercentile as DatePercentile,
)
from forecasting_tools.data_models.numeric_report import DateReport as DateReport
from forecasting_tools.data_models.numeric_report import (
    DiscreteReport as DiscreteReport,
)
from forecasting_tools.data_models.numeric_report import (
    NumericDistribution as NumericDistribution,
)
from forecasting_tools.data_models.numeric_report import NumericReport as NumericReport
from forecasting_tools.data_models.questions import BinaryQuestion as BinaryQuestion
from forecasting_tools.data_models.questions import Category as Category
from forecasting_tools.data_models.questions import DateQuestion as DateQuestion
from forecasting_tools.data_models.questions import DiscreteQuestion as DiscreteQuestion
from forecasting_tools.data_models.questions import (
    MetaculusQuestion as MetaculusQuestion,
)
from forecasting_tools.data_models.questions import (
    MultipleChoiceQuestion as MultipleChoiceQuestion,
)
from forecasting_tools.data_models.questions import NumericQuestion as NumericQuestion

# Fix for: https://docs.pydantic.dev/latest/errors/usage_errors/#class-not-fully-defined
ConditionalReport.model_rebuild()
BinaryReport.model_rebuild()
MultipleChoiceReport.model_rebuild()
ForecastReport.model_rebuild()
NumericReport.model_rebuild()
DiscreteReport.model_rebuild()
DateReport.model_rebuild()
from forecasting_tools.data_models.conditional_models import (
    ConditionalPrediction as ConditionalPrediction,
)
from forecasting_tools.data_models.conditional_models import (
    PredictionAffirmed as PredictionAffirmed,
)
from forecasting_tools.data_models.data_organizer import (
    PredictionTypes as PredictionTypes,
)
from forecasting_tools.data_models.numeric_report import Percentile as Percentile
from forecasting_tools.data_models.questions import (
    ConditionalQuestion as ConditionalQuestion,
)
from forecasting_tools.data_models.questions import QuestionState as QuestionState
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot as ForecastBot
from forecasting_tools.forecast_bots.forecast_bot import Notepad as Notepad
from forecasting_tools.forecast_bots.main_bot import MainBot as MainBot
from forecasting_tools.forecast_bots.official_bots.gpt_4_1_optimized_bot import (
    GPT41OptimizedBot as GPT41OptimizedBot,
)
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot2025 as Q1TemplateBot2025,
)
from forecasting_tools.forecast_bots.official_bots.q2_template_bot import (
    Q2TemplateBot2025 as Q2TemplateBot2025,
)
from forecasting_tools.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot2024 as Q3TemplateBot2024,
)
from forecasting_tools.forecast_bots.official_bots.q4_template_bot import (
    Q4TemplateBot2024 as Q4TemplateBot2024,
)
from forecasting_tools.forecast_bots.official_bots.research_only_bot_2025_fall import (
    FallResearchOnlyBot2025 as FallResearchOnlyBot2025,
)
from forecasting_tools.forecast_bots.official_bots.template_bot_2025_fall import (
    FallTemplateBot2025 as FallTemplateBot2025,
)
from forecasting_tools.forecast_bots.official_bots.template_bot_2026_spring import (
    SpringTemplateBot2026 as SpringTemplateBot2026,
)
from forecasting_tools.forecast_bots.official_bots.uniform_probability_bot import (
    UniformProbabilityBot as UniformProbabilityBot,
)
from forecasting_tools.forecast_bots.template_bot import (
    ResearchOnlyBot as ResearchOnlyBot,
)
from forecasting_tools.forecast_bots.template_bot import TemplateBot as TemplateBot
from forecasting_tools.helpers.asknews_searcher import (
    AskNewsSearcher as AskNewsSearcher,
)
from forecasting_tools.helpers.metaculus_api import MetaculusApi as MetaculusApi
from forecasting_tools.helpers.metaculus_api import MetaculusClient as MetaculusClient
from forecasting_tools.helpers.metaculus_client import ApiFilter as ApiFilter
from forecasting_tools.helpers.prediction_extractor import (
    PredictionExtractor as PredictionExtractor,
)
from forecasting_tools.helpers.structure_output import (
    structure_output as structure_output,
)
from forecasting_tools.util.misc import clean_indents as clean_indents

nest_asyncio.apply()
