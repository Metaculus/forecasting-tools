"""Build a table of how a bot's forecasts turned out.

One row per question in a tournament: the bot's latest forecast, how the
question resolved, and the scores Metaculus computed for it.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel

from forecasting_tools.data_models.leaderboard import Leaderboard
from forecasting_tools.helpers.metaculus_client import ApiFilter, MetaculusClient

logger = logging.getLogger(__name__)

CONTINUOUS_TYPES = ("numeric", "date", "discrete")
MAX_QUESTIONS_PER_TOURNAMENT = 1000
LATE_RESOLUTION_ALLOWANCE = timedelta(days=14)


class QuestionScores(BaseModel):
    """Official scores from ``my_forecasts.score_data``."""

    peer_score: float | None = None
    baseline_score: float | None = None
    spot_peer_score: float | None = None
    spot_baseline_score: float | None = None
    coverage: float | None = None
    weighted_coverage: float | None = None
    relative_legacy_score: float | None = None


class QuestionOutcome(BaseModel):
    """A question, the bot's forecast on it, and the result."""

    post_id: int
    question_id: int
    title: str | None
    url: str
    question_type: str | None
    status: str | None
    resolution: str | None
    actual_resolve_time: datetime | None
    spot_scoring_time: datetime | None
    options: list[str] | None
    group_label: str | None = None
    conditional_branch: str | None = None
    forecasted: bool
    forecast: dict[str, Any] | None
    forecast_time: datetime | None
    scores: QuestionScores | None
    project_id: int | None
    project_slug: str | None
    project_name: str | None

    @property
    def was_annulled(self) -> bool:
        return self.resolution == "annulled"


class OutcomeTable(BaseModel):
    """A set of questions, the bot's forecasts on them, and its standing."""

    generated_at: datetime
    user_id: int
    questions: list[QuestionOutcome]
    project_id: int | None = None
    project_slug: str | None = None
    project_name: str | None = None
    leaderboard: Leaderboard | None = None


def scale_internal(location: float, scaling: dict) -> float | None:
    """Map an internal [0, 1] location onto the question's own units."""
    range_min, range_max = scaling.get("range_min"), scaling.get("range_max")
    zero_point = scaling.get("zero_point")
    if range_min is None or range_max is None:
        return None
    if zero_point is not None:
        ratio = (range_max - zero_point) / (range_min - zero_point)
        return range_min + (range_max - range_min) * (ratio**location - 1) / (ratio - 1)
    return range_min + location * (range_max - range_min)


def cdf_median(cdf: list[float]) -> float:
    """The internal [0, 1] location where the CDF crosses 0.5."""
    for index, value in enumerate(cdf):
        if value >= 0.5:
            if index == 0:
                return 0.0
            below = cdf[index - 1]
            fraction = 0.0 if value == below else (0.5 - below) / (value - below)
            return (index - 1 + fraction) / (len(cdf) - 1)
    return 1.0


def summarize_forecast(
    question_type: str | None,
    forecast_values: list[float] | None,
    options: list[str] | None,
    scaling: dict,
) -> dict[str, Any] | None:
    """Reduce raw ``forecast_values`` to a readable summary of the forecast."""
    if not forecast_values:
        return None
    if question_type == "binary":
        probability = (
            forecast_values[-1] if len(forecast_values) == 2 else forecast_values[0]
        )
        return {"probability_yes": probability}
    if question_type == "multiple_choice":
        return {"probabilities": dict(zip(options or [], forecast_values))}
    if question_type in CONTINUOUS_TYPES:
        median = scale_internal(cdf_median(forecast_values), scaling)
        summary: dict[str, Any] = {"median": median}
        if question_type == "date" and median is not None:
            summary["median_iso"] = datetime.fromtimestamp(
                median, tz=timezone.utc
            ).isoformat()
        return summary
    return {"values": forecast_values}


def _question_outcome(
    post_json: dict, question_json: dict, **extra: Any
) -> QuestionOutcome:
    post_id = post_json["id"]
    question_type = question_json.get("type")
    options = question_json.get("options")
    scaling = question_json.get("scaling") or {}
    project = (post_json.get("projects") or {}).get("default_project") or {}
    my_forecasts = question_json.get("my_forecasts") or {}
    latest = my_forecasts.get("latest") or {}
    score_data = my_forecasts.get("score_data") or None
    return QuestionOutcome(
        post_id=post_id,
        question_id=question_json["id"],
        title=question_json.get("title") or post_json.get("title"),
        url=f"https://www.metaculus.com/questions/{post_id}/",
        question_type=question_type,
        status=question_json.get("status") or post_json.get("status"),
        resolution=question_json.get("resolution"),
        actual_resolve_time=question_json.get("actual_resolve_time"),
        spot_scoring_time=question_json.get("spot_scoring_time"),
        options=options,
        forecasted=bool(latest),
        forecast=summarize_forecast(
            question_type, latest.get("forecast_values"), options, scaling
        ),
        forecast_time=latest.get("start_time"),
        scores=QuestionScores(**score_data) if score_data else None,
        project_id=project.get("id"),
        project_slug=project.get("slug"),
        project_name=project.get("name"),
        **extra,
    )


def outcomes_from_post(post_json: dict) -> list[QuestionOutcome]:
    """One outcome per question on a post, unpacking groups and conditionals."""
    if post_json.get("notebook"):
        return []
    if post_json.get("question"):
        return [_question_outcome(post_json, post_json["question"])]
    group = post_json.get("group_of_questions")
    if group:
        return [
            _question_outcome(post_json, question, group_label=question.get("label"))
            for question in group.get("questions", [])
        ]
    conditional = post_json.get("conditional")
    if conditional:
        return [
            _question_outcome(
                post_json, conditional[branch], conditional_branch=branch.split("_")[1]
            )
            for branch in ("question_yes", "question_no")
            if conditional.get(branch)
        ]
    return []


def get_outcomes_for_posts(
    post_ids: list[int], client: MetaculusClient | None = None
) -> list[QuestionOutcome]:
    """
    Fetch outcomes for specific posts, one request each.

    :param post_ids: post ids, as they appear in question urls
    :param client: client to fetch with, created from the environment if not given
    """
    client = client or MetaculusClient()
    outcomes: list[QuestionOutcome] = []
    for post_id in post_ids:
        fetched = client.get_question_by_post_id(
            post_id, group_question_mode="unpack_subquestions"
        )
        questions = fetched if isinstance(fetched, list) else [fetched]
        outcomes.extend(outcomes_from_post(questions[0].api_json))
    return outcomes


def get_recently_resolved_outcomes(
    days: int, client: MetaculusClient | None = None
) -> list[QuestionOutcome]:
    """
    Fetch the bot's forecasts on questions that resolved in the last few days.

    :param days: how far back to look
    :param client: client to fetch with, created from the environment if not given
    """
    client = client or MetaculusClient()
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    api_filter = ApiFilter(
        allowed_statuses=["resolved"],
        is_previously_forecasted_by_user=True,
        scheduled_resolve_time_gt=cutoff - LATE_RESOLUTION_ALLOWANCE,
        group_question_mode="unpack_subquestions",
    )
    listed_questions = asyncio.run(
        client.get_questions_matching_filter(
            api_filter,
            num_questions=MAX_QUESTIONS_PER_TOURNAMENT,
            error_if_question_target_missed=False,
        )
    )
    post_ids = list(dict.fromkeys(question.id_of_post for question in listed_questions))
    logger.info(f"Building outcomes for {len(post_ids)} posts resolved since {cutoff}")
    outcomes = get_outcomes_for_posts(post_ids, client)  # type: ignore[arg-type]
    return [
        outcome
        for outcome in outcomes
        if outcome.actual_resolve_time is not None
        and outcome.actual_resolve_time >= cutoff
    ]


def get_tournament_outcomes(
    tournament: int | str,
    client: MetaculusClient | None = None,
    forecasted_only: bool = False,
) -> OutcomeTable:
    """
    Fetch the questions in a tournament and how the bot did on each.

    :param tournament: tournament slug or id
    :param client: client to fetch with, created from the environment if not given
    :param forecasted_only: skip questions the bot never forecast
    """
    client = client or MetaculusClient()
    api_filter = ApiFilter(
        allowed_tournaments=[tournament],
        group_question_mode="unpack_subquestions",
        is_previously_forecasted_by_user=True if forecasted_only else None,
    )
    listed_questions = asyncio.run(
        client.get_questions_matching_filter(
            api_filter,
            num_questions=MAX_QUESTIONS_PER_TOURNAMENT,
            error_if_question_target_missed=False,
        )
    )
    post_ids = list(dict.fromkeys(question.id_of_post for question in listed_questions))
    logger.info(f"Building outcomes for {len(post_ids)} posts in {tournament}")

    outcomes = get_outcomes_for_posts(post_ids, client)  # type: ignore[arg-type]
    project_id = outcomes[0].project_id if outcomes else None
    leaderboard = (
        client.get_project_leaderboard(project_id) if project_id is not None else None
    )
    return OutcomeTable(
        generated_at=datetime.now(tz=timezone.utc),
        project_id=project_id,
        project_slug=outcomes[0].project_slug if outcomes else None,
        project_name=outcomes[0].project_name if outcomes else None,
        user_id=client.get_current_user_id(),
        leaderboard=leaderboard,
        questions=outcomes,
    )
