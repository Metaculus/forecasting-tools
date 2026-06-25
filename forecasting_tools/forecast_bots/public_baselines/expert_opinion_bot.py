"""Baseline bot estimating what topic experts would forecast."""

from forecasting_tools.forecast_bots.public_baselines.population_baseline_bot import (
    PopulationBaselineBot,
    PopulationSpec,
)


class ExpertOpinionBaselineBot(PopulationBaselineBot):
    """Estimates the forecast of a representative sample of relevant experts."""

    population_spec = PopulationSpec(
        name="subject-matter experts on the question's topic",
        short_name="topic experts",
        target_description=(
            "credentialed specialists whose professional field is directly relevant to "
            "the question (e.g. epidemiologists for a disease question, central-bank "
            "economists for a rates question, election scientists for an election "
            "question), sampled across institutions and viewpoints rather than a single "
            "school of thought"
        ),
        sampling_method=(
            "First identify which 1-3 fields of expertise are most relevant to the "
            "question. Then imagine polling a representative cross-section of recognised "
            "experts in those fields. Experts reason from domain models, base rates, "
            "and current data, and tend to be better calibrated than the public, but "
            "they also have characteristic blind spots (over-reliance on existing "
            "models, herding around consensus, and slowness to update on regime "
            "changes). Sample across competing expert camps where the field is divided."
        ),
        source_guidance=(
            "Prioritise peer-reviewed research, expert surveys and elicitations, "
            "official forecasts from expert bodies (IMF, IPCC, CBO, central banks, "
            "WHO, etc.), analyst consensus, and named expert commentary. Down-weight "
            "non-expert punditry."
        ),
        interpretation_guidance=(
            "Translate technical findings, model outputs, and expert statements into a "
            "concrete forecast for this question's resolution criteria. Where experts "
            "disagree, represent the spread of expert opinion rather than collapsing to "
            "a single view, and note the consensus position separately from outliers."
        ),
    )
