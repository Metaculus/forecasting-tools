from types import SimpleNamespace

from forecasting_tools.agents_and_tools.research.exa_quote_searcher import (
    ExaQuoteSearcher,
)
from forecasting_tools.ai_models.exa_searcher import ExaSource
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.public_baselines.population_baseline_bot import (
    BinaryPopulationForecast,
    DiscoveredSource,
    MultipleChoicePopulationForecast,
    NumericPercentile,
    NumericPopulationForecast,
    OptionProbability,
)
from forecasting_tools.forecast_bots.public_baselines.public_sentiment_bot import (
    PublicSentimentBaselineBot,
)


class _FakeAgent:
    def __init__(self, output: object) -> None:
        self._output = output

    async def run(self, prompt: str, usage_limits: object = None) -> SimpleNamespace:
        return SimpleNamespace(output=self._output)


def _make_bot_returning(output: object) -> PublicSentimentBaselineBot:
    bot = PublicSentimentBaselineBot()
    bot._build_agent = lambda output_type, branch_llm=None: _FakeAgent(output)  # type: ignore
    return bot


def _sample_sources() -> list[DiscoveredSource]:
    return [
        DiscoveredSource(
            name="YouGov poll (Jun 2026)",
            represents="US adults",
            url="https://example.com/poll",
            implied_forecast="~60% expect yes",
            confidence="medium",
            note="Topline support translated to probability.",
        )
    ]


async def test_binary_forecast_converts_and_clamps() -> None:
    forecast = BinaryPopulationForecast(
        population_summary="The public leans yes.",
        sources=_sample_sources(),
        aggregate_probability=0.995,
        aggregate_rationale="Weighted toward the poll.",
    )
    bot = _make_bot_returning(forecast)
    question = BinaryQuestion(question_text="Will it happen?")

    prediction = await bot._run_forecast_on_binary(question, "")

    assert prediction.prediction_value == 0.99
    assert "YouGov poll" in prediction.reasoning
    assert "Sources sampled" in prediction.reasoning


async def test_multiple_choice_maps_and_normalizes_options() -> None:
    forecast = MultipleChoicePopulationForecast(
        population_summary="Split opinion.",
        sources=_sample_sources(),
        option_probabilities=[
            OptionProbability(option_name="Option A", probability=0.6),
            OptionProbability(option_name="Option B", probability=0.6),
        ],
        aggregate_rationale="Most lean A or B.",
    )
    bot = _make_bot_returning(forecast)
    question = MultipleChoiceQuestion(
        question_text="Which option?",
        options=["Option A", "Option B", "Option C"],
    )

    prediction = await bot._run_forecast_on_multiple_choice(question, "")

    assert isinstance(prediction.prediction_value, PredictedOptionList)
    option_names = [
        option.option_name for option in prediction.prediction_value.predicted_options
    ]
    assert option_names == ["Option A", "Option B", "Option C"]
    total = sum(
        option.probability for option in prediction.prediction_value.predicted_options
    )
    assert abs(total - 1.0) < 0.01


async def test_numeric_forecast_builds_distribution() -> None:
    forecast = NumericPopulationForecast(
        population_summary="Public expects a mid value.",
        sources=_sample_sources(),
        percentiles=[
            NumericPercentile(percentile=0.1, value=10),
            NumericPercentile(percentile=0.2, value=20),
            NumericPercentile(percentile=0.4, value=40),
            NumericPercentile(percentile=0.6, value=60),
            NumericPercentile(percentile=0.8, value=80),
            NumericPercentile(percentile=0.9, value=90),
        ],
        aggregate_rationale="Spread across plausible values.",
    )
    bot = _make_bot_returning(forecast)
    question = NumericQuestion(
        question_text="How many?",
        upper_bound=100,
        lower_bound=0,
        open_upper_bound=True,
        open_lower_bound=True,
    )

    prediction = await bot._run_forecast_on_numeric(question, "")

    assert isinstance(prediction.prediction_value, NumericDistribution)
    assert len(prediction.prediction_value.declared_percentiles) == 6


async def test_make_prediction_round_robins_branch_models() -> None:
    bot = PublicSentimentBaselineBot()
    question = BinaryQuestion(question_text="Will it happen?")
    notepad = await bot._initialize_notepad(question)
    bot._note_pads.append(notepad)

    used_models: list[str] = []

    async def fake_binary(
        question: BinaryQuestion, research: str, branch_llm: object = None
    ) -> ReasonedPrediction[float]:
        used_models.append(branch_llm.model)  # type: ignore[union-attr]
        return ReasonedPrediction(prediction_value=0.5, reasoning="x")

    bot._run_forecast_on_binary = fake_binary  # type: ignore[assignment]

    for _ in range(len(bot.branch_llms)):
        await bot._make_prediction(question, "")

    assert used_models == [branch.model for branch in bot.branch_llms]


def test_exa_quote_searcher_formats_summary_and_top_quotes() -> None:
    searcher = ExaQuoteSearcher(num_quotes_per_source=2)
    source = ExaSource(
        original_query="poll",
        auto_prompt_string=None,
        title="Poll shows majority support",
        url="https://example.com/poll",
        text=None,
        author="Jane Doe",
        published_date=None,
        score=0.9,
        highlights=["a low-scoring quote", "a high-scoring quote"],
        highlight_scores=[0.1, 0.9],
        summary="A representative poll about the topic.",
    )

    formatted = searcher._format_sources("poll", [source])

    assert "Poll shows majority support" in formatted
    assert "https://example.com/poll" in formatted
    assert "A representative poll about the topic." in formatted
    assert "a high-scoring quote" in formatted


def test_comment_includes_population_framing_and_sources() -> None:
    from forecasting_tools.data_models.forecast_report import (
        ReasonedPrediction,
        ResearchWithPredictions,
    )

    bot = PublicSentimentBaselineBot()
    question = BinaryQuestion(question_text="Will it happen?")
    collection = ResearchWithPredictions(
        research_report="",
        summary_report="",
        errors=[],
        predictions=[
            ReasonedPrediction(
                prediction_value=0.6,
                reasoning="| 1 | YouGov poll | US adults | 60% | medium | note |",
            )
        ],
    )

    comment = bot._create_comment(question, [collection], 0.6, 0.0, 0.0)

    assert "BASELINE FORECAST" in comment
    assert "general public" in comment
    assert "YouGov poll" in comment
