from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from forecasting_tools.bot_review import outcomes as outcomes_module
from forecasting_tools.bot_review.outcomes import (
    cdf_median,
    get_recently_resolved_outcomes,
    outcomes_from_post,
    scale_internal,
    summarize_forecast,
)


def make_binary_post(**question_overrides) -> dict:
    question = {
        "id": 101,
        "type": "binary",
        "title": "Will X happen?",
        "status": "resolved",
        "resolution": "yes",
        "actual_resolve_time": "2026-07-01T00:00:00Z",
        "spot_scoring_time": "2026-06-15T00:00:00Z",
        "my_forecasts": {
            "latest": {
                "forecast_values": [0.3, 0.7],
                "start_time": "2026-06-10T00:00:00Z",
            },
            "score_data": {
                "peer_score": 5.0,
                "baseline_score": 10.0,
                "coverage": 0.9,
                "spot_peer_score": -2.5,
            },
        },
    }
    question.update(question_overrides)
    return {
        "id": 1,
        "title": "Will X happen?",
        "question": question,
        "projects": {
            "default_project": {"id": 33064, "slug": "a-tournament", "name": "A"}
        },
    }


class TestScaling:
    def test_linear(self):
        scaling = {"range_min": 0, "range_max": 100, "zero_point": None}
        assert scale_internal(0.5, scaling) == 50

    def test_log(self):
        # range 1..100 with zero_point 0 -> ratio 100, so the midpoint is 10
        scaling = {"range_min": 1, "range_max": 100, "zero_point": 0}
        assert math.isclose(scale_internal(0.5, scaling), 10.0)

    def test_missing_range(self):
        assert scale_internal(0.5, {}) is None

    def test_cdf_median_interpolates(self):
        assert math.isclose(cdf_median([0.0, 0.25, 0.75, 1.0]), 0.5)

    def test_cdf_median_endpoints(self):
        assert cdf_median([0.6, 0.8, 1.0]) == 0.0
        assert cdf_median([0.0, 0.1, 0.2]) == 1.0


class TestSummarizeForecast:
    def test_binary(self):
        assert summarize_forecast("binary", [0.3, 0.7], None, {}) == {
            "probability_yes": 0.7
        }

    def test_multiple_choice_zips_options(self):
        summary = summarize_forecast("multiple_choice", [0.2, 0.8], ["A", "B"], {})
        assert summary == {"probabilities": {"A": 0.2, "B": 0.8}}

    def test_numeric_median_is_scaled(self):
        scaling = {"range_min": 0, "range_max": 100, "zero_point": None}
        summary = summarize_forecast("numeric", [0.0, 0.25, 0.75, 1.0], None, scaling)
        assert summary is not None
        assert math.isclose(summary["median"], 50.0)

    def test_date_gets_an_iso_median(self):
        scaling = {"range_min": 0, "range_max": 86400, "zero_point": None}
        summary = summarize_forecast("date", [0.0, 0.5, 1.0], None, scaling)
        assert summary is not None
        assert summary["median_iso"].startswith("1970-01-01T12:00")

    def test_no_forecast(self):
        assert summarize_forecast("binary", None, None, {}) is None


class TestOutcomesFromPost:
    def test_binary_post(self):
        (outcome,) = outcomes_from_post(make_binary_post())
        assert outcome.post_id == 1
        assert outcome.question_id == 101
        assert outcome.forecasted
        assert outcome.forecast == {"probability_yes": 0.7}
        assert outcome.resolution == "yes"
        assert outcome.scores is not None
        assert outcome.scores.peer_score == 5.0
        assert outcome.scores.spot_peer_score == -2.5
        assert outcome.project_slug == "a-tournament"
        assert outcome.url.endswith("/questions/1/")

    def test_resolved_question_with_null_resolution(self):
        post = make_binary_post(
            type="multiple_choice",
            resolution=None,
            options=["0", "1"],
            my_forecasts={"latest": {"forecast_values": [0.4, 0.6]}, "score_data": {}},
        )
        (outcome,) = outcomes_from_post(post)
        assert outcome.status == "resolved"
        assert outcome.resolution is None
        assert outcome.forecast == {"probabilities": {"0": 0.4, "1": 0.6}}
        assert outcome.scores is None

    def test_annulled_question(self):
        post = make_binary_post(resolution="annulled", my_forecasts={})
        (outcome,) = outcomes_from_post(post)
        assert outcome.was_annulled
        assert not outcome.forecasted

    def test_unforecasted_question(self):
        (outcome,) = outcomes_from_post(make_binary_post(my_forecasts={}))
        assert not outcome.forecasted
        assert outcome.forecast is None
        assert outcome.scores is None

    def test_group_post_expands_to_one_row_per_subquestion(self):
        subquestion = make_binary_post()["question"]
        post = {
            "id": 2,
            "title": "Group",
            "projects": {},
            "group_of_questions": {
                "questions": [
                    dict(subquestion, id=201, label="a"),
                    dict(subquestion, id=202, label="b"),
                ]
            },
        }
        outcomes = outcomes_from_post(post)
        assert [o.question_id for o in outcomes] == [201, 202]
        assert [o.group_label for o in outcomes] == ["a", "b"]
        assert all(o.post_id == 2 for o in outcomes)

    def test_conditional_post_expands_to_both_branches(self):
        child = make_binary_post()["question"]
        post = {
            "id": 3,
            "projects": {},
            "conditional": {
                "question_yes": dict(child, id=301),
                "question_no": dict(child, id=302),
            },
        }
        outcomes = outcomes_from_post(post)
        assert [o.conditional_branch for o in outcomes] == ["yes", "no"]

    def test_notebooks_are_skipped(self):
        assert outcomes_from_post({"id": 4, "notebook": {}}) == []


class TestRecentlyResolved:
    def outcome_resolved(self, resolve_time: datetime | None):
        (outcome,) = outcomes_from_post(make_binary_post())
        return outcome.model_copy(update={"actual_resolve_time": resolve_time})

    def test_keeps_only_questions_resolved_inside_the_window(self):
        now = datetime.now(tz=timezone.utc)
        recent = self.outcome_resolved(now - timedelta(days=2))
        old = self.outcome_resolved(now - timedelta(days=40))
        never = self.outcome_resolved(None)

        client = MagicMock()
        client.get_questions_matching_filter = AsyncMock(return_value=[])
        with patch.object(
            outcomes_module,
            "get_outcomes_for_posts",
            return_value=[recent, old, never],
        ):
            kept = get_recently_resolved_outcomes(7, client)

        assert kept == [recent]

    def test_asks_the_api_for_resolved_questions_it_forecast(self):
        client = MagicMock()
        client.get_questions_matching_filter = AsyncMock(return_value=[])
        with patch.object(outcomes_module, "get_outcomes_for_posts", return_value=[]):
            get_recently_resolved_outcomes(7, client)

        api_filter = client.get_questions_matching_filter.call_args.args[0]
        assert api_filter.allowed_statuses == ["resolved"]
        assert api_filter.is_previously_forecasted_by_user
        assert api_filter.scheduled_resolve_time_gt is not None

    def test_the_api_window_reaches_further_back_than_the_cutoff(self):
        # a question scheduled before the cutoff can still resolve inside it
        client = MagicMock()
        client.get_questions_matching_filter = AsyncMock(return_value=[])
        with patch.object(outcomes_module, "get_outcomes_for_posts", return_value=[]):
            get_recently_resolved_outcomes(7, client)

        api_filter = client.get_questions_matching_filter.call_args.args[0]
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=7)
        assert api_filter.scheduled_resolve_time_gt < cutoff
