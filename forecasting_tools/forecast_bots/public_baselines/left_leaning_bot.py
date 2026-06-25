"""Baseline bot estimating what the political left would forecast."""

from forecasting_tools.forecast_bots.public_baselines.population_baseline_bot import (
    PopulationBaselineBot,
    PopulationSpec,
)


class LeftLeaningBaselineBot(PopulationBaselineBot):
    """Estimates the forecast of a sample of left-leaning figures and outlets."""

    population_spec = PopulationSpec(
        name="the political left",
        short_name="left",
        target_description=(
            "left-leaning / progressive public figures, commentators, and media "
            "outlets (e.g. MSNBC, The Guardian, The Nation, Vox, Mother Jones, "
            "prominent progressive politicians and writers), sampled across the "
            "center-left to the further left"
        ),
        sampling_method=(
            "Imagine sampling a representative set of left-leaning voices and asking "
            "each what they would predict. Their forecasts are shaped by progressive "
            "priors and the issues their side emphasises. Weight across the spectrum "
            "from the establishment center-left to the activist left, and base the "
            "leaning on the question's relevant country/context."
        ),
        source_guidance=(
            "Prioritise reporting, op-eds, and commentary from left-leaning outlets and "
            "figures, and left-leaning framing of polls and events. Identify the "
            "source's lean explicitly when recording it."
        ),
        interpretation_guidance=(
            "Reflect how this side's worldview shapes its predictions: motivated "
            "reasoning toward outcomes it favours or fears, distinct trusted sources, "
            "and characteristic framings of risk and blame. Faithfully represent the "
            "left's expected forecast even where it diverges from neutral analysis, and "
            "note internal disagreement between the center-left and the further left."
        ),
    )
