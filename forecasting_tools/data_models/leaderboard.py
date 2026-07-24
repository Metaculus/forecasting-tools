from __future__ import annotations

from pydantic import BaseModel


class LeaderboardEntry(BaseModel):
    """One forecaster's standing on a leaderboard.

    Community aggregates appear as entries with no user and an
    ``aggregation_method`` set.
    """

    user_id: int | None
    username: str | None
    aggregation_method: str | None
    rank: int | None
    score: float | None
    coverage: float | None
    contribution_count: int | None
    medal: str | None
    prize: float | None
    excluded: bool

    @classmethod
    def from_metaculus_api_json(cls, entry_json: dict) -> LeaderboardEntry:
        user = entry_json.get("user") or {}
        return cls(
            user_id=user.get("id"),
            username=user.get("username"),
            aggregation_method=entry_json.get("aggregation_method"),
            rank=entry_json["rank"],
            score=entry_json["score"],
            coverage=entry_json["coverage"],
            contribution_count=entry_json["contribution_count"],
            medal=entry_json["medal"],
            prize=entry_json["prize"],
            excluded=entry_json["excluded"],
        )


class Leaderboard(BaseModel):
    """A project's leaderboard. ``user_entry`` is None if you have no standing on it."""

    project_id: int
    project_name: str | None
    project_slug: str | None
    score_type: str
    finalized: bool
    entries: list[LeaderboardEntry]
    user_entry: LeaderboardEntry | None

    @classmethod
    def from_metaculus_api_json(cls, leaderboard_json: dict) -> Leaderboard:
        user_entry = leaderboard_json.get("userEntry")
        return cls(
            project_id=leaderboard_json["project_id"],
            project_name=leaderboard_json.get("project_name"),
            project_slug=leaderboard_json.get("project_slug"),
            score_type=leaderboard_json["score_type"],
            finalized=leaderboard_json["finalized"],
            entries=[
                LeaderboardEntry.from_metaculus_api_json(entry)
                for entry in leaderboard_json.get("entries") or []
            ],
            user_entry=(
                LeaderboardEntry.from_metaculus_api_json(user_entry)
                if user_entry
                else None
            ),
        )
