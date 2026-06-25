"""Baseline bot estimating the forecast implied by credible news outlets."""

from forecasting_tools.forecast_bots.public_baselines.population_baseline_bot import (
    PopulationBaselineBot,
    PopulationSpec,
)


class CredibleNewsBaselineBot(PopulationBaselineBot):
    """Estimates the forecast implied by a sample of credible news outlets."""

    population_spec = PopulationSpec(
        name="credible news outlets",
        short_name="credible news",
        target_description=(
            "established, fact-checked news organisations with strong reputations for "
            "accuracy and editorial standards (e.g. Reuters, AP, BBC, The New York "
            "Times, The Wall Street Journal, The Economist, Financial Times, Bloomberg, "
            "Nature/Science news), sampled across outlets rather than relying on any "
            "single one"
        ),
        sampling_method=(
            "Imagine collecting the most recent reporting on this question from a "
            "basket of credible, mainstream-to-high-quality outlets and asking what "
            "forecast their coverage collectively implies. News outlets rarely state "
            "explicit probabilities, so infer the implied forecast from how they frame "
            "the situation: which outcome is treated as the default/expected one, the "
            "hedging language used ('likely', 'unlikely', 'on track', 'in doubt'), and "
            "which scenarios are given the most weight. Prefer reporting and analysis "
            "over opinion columns."
        ),
        source_guidance=(
            "Prioritise straight news reporting and data journalism from reputable "
            "outlets and wire services. Use the publication dates to weight toward the "
            "most current framing. Avoid low-credibility or partisan tabloid sources."
        ),
        interpretation_guidance=(
            "Map editorial framing onto a concrete forecast: e.g. coverage describing "
            "an outcome as 'widely expected' implies a high probability, 'facing "
            "long odds' implies a low one. Account for newsroom tendencies toward "
            "drama, novelty, and balance ('both sides') that can distort the implied "
            "forecast."
        ),
    )
