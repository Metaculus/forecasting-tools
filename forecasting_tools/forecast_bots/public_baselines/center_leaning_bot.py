"""Baseline bot estimating what the political center would forecast."""

from forecasting_tools.forecast_bots.public_baselines.population_baseline_bot import (
    PopulationBaselineBot,
    PopulationSpec,
)


class CenterLeaningBaselineBot(PopulationBaselineBot):
    """Estimates the forecast of a sample of centrist figures and outlets."""

    population_spec = PopulationSpec(
        name="the political center",
        short_name="center",
        target_description=(
            "centrist / moderate public figures, commentators, and media outlets "
            "(e.g. Reuters, AP, the news pages of major papers, moderate and "
            "independent voices, centrist think tanks), excluding strongly partisan "
            "left or right sources"
        ),
        sampling_method=(
            "Imagine sampling a representative set of moderate, non-partisan-leaning "
            "voices and asking each what they would predict. Centrists tend to weight "
            "mainstream expert consensus, official data, and 'both sides' framings, and "
            "to avoid the strong directional priors of either wing. Sample across "
            "center-left-of-neutral to center-right-of-neutral moderates."
        ),
        source_guidance=(
            "Prioritise wire services, straight news, centrist columnists, and "
            "non-partisan analysts and think tanks. When using a source, note why it "
            "qualifies as centrist rather than partisan."
        ),
        interpretation_guidance=(
            "Reflect the centrist tendency to anchor on consensus and official "
            "forecasts, to split the difference between partisan narratives, and to be "
            "cautious about extreme outcomes. Represent this measured framing rather "
            "than either wing's view."
        ),
    )
