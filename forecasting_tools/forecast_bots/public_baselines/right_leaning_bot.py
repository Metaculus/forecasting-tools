"""Baseline bot estimating what the political right would forecast."""

from forecasting_tools.forecast_bots.public_baselines.population_baseline_bot import (
    PopulationBaselineBot,
    PopulationSpec,
)


class RightLeaningBaselineBot(PopulationBaselineBot):
    """Estimates the forecast of a sample of right-leaning figures and outlets."""

    population_spec = PopulationSpec(
        name="the political right",
        short_name="right",
        target_description=(
            "right-leaning / conservative public figures, commentators, and media "
            "outlets (e.g. Fox News, The Wall Street Journal opinion page, National "
            "Review, The Telegraph, The Free Press, prominent conservative politicians "
            "and writers), sampled across the center-right to the further right"
        ),
        sampling_method=(
            "Imagine sampling a representative set of right-leaning voices and asking "
            "each what they would predict. Their forecasts are shaped by conservative "
            "priors and the issues their side emphasises. Weight across the spectrum "
            "from the establishment center-right to the populist/further right, and "
            "base the leaning on the question's relevant country/context."
        ),
        source_guidance=(
            "Prioritise reporting, op-eds, and commentary from right-leaning outlets "
            "and figures, and right-leaning framing of polls and events. Identify the "
            "source's lean explicitly when recording it."
        ),
        interpretation_guidance=(
            "Reflect how this side's worldview shapes its predictions: motivated "
            "reasoning toward outcomes it favours or fears, distinct trusted sources, "
            "and characteristic framings of risk and blame. Faithfully represent the "
            "right's expected forecast even where it diverges from neutral analysis, "
            "and note internal disagreement between the center-right and the further "
            "right."
        ),
    )
