"""Baseline bot estimating what the general public would forecast."""

from forecasting_tools.forecast_bots.public_baselines.population_baseline_bot import (
    PopulationBaselineBot,
    PopulationSpec,
)


class PublicSentimentBaselineBot(PopulationBaselineBot):
    """Estimates the forecast of a representative sample of the general public."""

    population_spec = PopulationSpec(
        name="the general public",
        short_name="general public",
        target_description=(
            "ordinary members of the general public (not specialists), spread across "
            "ages, regions, education levels, and political affiliations, weighted "
            "toward the population most relevant to the question (e.g. the relevant "
            "country's adults for a national question, or a global cross-section for a "
            "global one)"
        ),
        sampling_method=(
            "Imagine handing this question to a demographically representative panel "
            "of everyday people and recording their gut predictions. Most members are "
            "not closely following the topic, so their answers are driven by general "
            "impressions, recent headlines they happened to see, vibes, hope and fear, "
            "and simple heuristics rather than careful base-rate analysis. Weight by "
            "how the broad population actually skews, not by how the most engaged or "
            "online subgroups skew."
        ),
        source_guidance=(
            "Prioritise opinion polls, surveys, prediction-poll/'wisdom of the crowd' "
            "results, Google Trends, and broadly representative public-sentiment data. "
            "Treat viral social-media reactions as evidence of mood but down-weight "
            "them since they over-represent the highly engaged."
        ),
        interpretation_guidance=(
            "Remember well-documented lay-forecasting tendencies: anchoring on the "
            "current vivid narrative, optimism/pessimism and wishful thinking, poor "
            "calibration (over-confidence on familiar topics, excess uncertainty on "
            "unfamiliar ones), scope insensitivity, and recency bias. Reflect these in "
            "the implied forecast rather than correcting them."
        ),
    )
