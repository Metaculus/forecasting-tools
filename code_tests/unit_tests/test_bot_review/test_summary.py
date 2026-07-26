from __future__ import annotations

from datetime import datetime, timezone

from forecasting_tools.bot_review.outcomes import (
    OutcomeTable,
    QuestionOutcome,
    QuestionScores,
)
from forecasting_tools.bot_review.summary import build_summary
from forecasting_tools.data_models.leaderboard import Leaderboard, LeaderboardEntry


def make_question(
    post_id: int,
    spot_peer: float | None = 0.0,
    baseline: float | None = 0.0,
    peer: float | None = None,
    scored: bool = True,
    forecasted: bool = True,
    resolution: str | None = "yes",
    project: str = "CupA",
    status: str = "resolved",
) -> QuestionOutcome:
    return QuestionOutcome(
        post_id=post_id,
        question_id=post_id + 100,
        title=f"Q{post_id}",
        url=f"https://www.metaculus.com/questions/{post_id}/",
        question_type="binary",
        status=status,
        resolution=resolution,
        actual_resolve_time=None,
        spot_scoring_time=None,
        options=None,
        forecasted=forecasted,
        forecast={"probability_yes": 0.5} if forecasted else None,
        forecast_time=None,
        scores=(
            QuestionScores(
                spot_peer_score=spot_peer,
                baseline_score=baseline,
                peer_score=peer,
                coverage=0.5,
            )
            if scored
            else None
        ),
        project_id=1,
        project_slug=project.lower(),
        project_name=project,
    )


def make_leaderboard(
    rank: int = 3, entries: int = 50, score_type: str = "spot_peer_tournament"
) -> Leaderboard:
    def entry(user_id: int | None, rank: int | None) -> LeaderboardEntry:
        return LeaderboardEntry(
            user_id=user_id,
            username=f"bot{user_id}",
            aggregation_method=None,
            rank=rank,
            score=30.0,
            coverage=4.0,
            contribution_count=4,
            medal=None,
            prize=None,
            excluded=False,
        )

    return Leaderboard(
        project_id=1,
        project_name="TestCup",
        project_slug="testcup",
        score_type=score_type,
        finalized=False,
        entries=[entry(i, i) for i in range(1, entries + 1)],
        user_entry=entry(7, rank),
    )


def make_table(**overrides) -> OutcomeTable:
    fields = {
        "generated_at": datetime(2026, 7, 23, tzinfo=timezone.utc),
        "user_id": 7,
        "project_name": "TestCup",
        "leaderboard": make_leaderboard(),
        "questions": [
            make_question(1, 20.0, 40.0),
            make_question(2, -15.0, -30.0),
            make_question(3, 25.0, 50.0),
            make_question(4, 0.0, 5.0),
            make_question(5, scored=False, resolution="annulled"),
            make_question(6, scored=False, forecasted=False, resolution=None),
        ],
    }
    fields.update(overrides)
    return OutcomeTable(**fields)


def test_counts():
    report = build_summary(make_table())
    assert "Questions: 6" in report
    assert "Forecasted: 5" in report
    assert "Scored: 4" in report
    assert "Forecasted but unscored: 1 (1 resolved without a score)" in report


def test_single_project_still_shows_breakdown():
    report = build_summary(make_table())
    assert "### By tournament" in report
    assert "- CupA: 6 questions (4 scored)" in report


def test_multi_project_breakdown_counts_questions_and_scored():
    questions = make_table().questions
    for index in (0, 4):
        questions[index] = questions[index].model_copy(
            update={"project_name": "CupB", "project_slug": "cupb"}
        )
    report = build_summary(make_table(questions=questions))
    assert "- CupA: 4 questions (3 scored)" in report
    assert "- CupB: 2 questions (1 scored)" in report


def test_annulled_listed_with_resolution():
    report = build_summary(make_table())
    assert "Q5" in report and "resolution: annulled" in report


def test_best_worst_ordering():
    report = build_summary(make_table(), top_n=2)
    best = report.split("## Best")[1].split("## Worst")[0]
    worst = report.split("## Worst")[1]
    assert best.index("Q3") < best.index("Q1")
    assert worst.index("Q2") < worst.index("Q4")
    assert "Q2" not in best
    assert "Q3" not in worst


def test_signed_formatting_and_leaderboard():
    report = build_summary(make_table())
    assert "+25.0 spot peer" in report
    assert "-15.0 spot peer" in report
    assert "Rank: **3 / 50** (spot_peer_tournament)" in report


def test_a_peer_tournament_is_ranked_on_peer_score():
    questions = [
        make_question(1, spot_peer=90.0, peer=-5.0),
        make_question(2, spot_peer=-90.0, peer=10.0),
    ]
    report = build_summary(
        make_table(
            questions=questions,
            leaderboard=make_leaderboard(score_type="peer_tournament"),
        )
    )
    assert "## Best 2 (by peer score)" in report
    assert report.split("## Best")[1].index("Q2") < report.split("## Best")[1].index(
        "Q1"
    )
    assert "+10.0 peer" in report


def test_an_unknown_score_type_falls_back_to_spot_peer():
    report = build_summary(
        make_table(leaderboard=make_leaderboard(score_type="comment_insight"))
    )
    assert "## Best 4 (by spot peer score)" in report


def test_mean_coverage_over_scored():
    report = build_summary(make_table())
    assert "Mean coverage: 0.50 (over 4 scored questions)" in report


def test_open_unscored_question_not_listed_as_resolved():
    questions = make_table().questions
    questions[4] = questions[4].model_copy(update={"status": "open"})
    report = build_summary(make_table(questions=questions))
    assert "Forecasted but unscored: 1 (0 resolved without a score)" in report
    assert "Q5" not in report


def test_no_leaderboard_entry():
    assert "No leaderboard entry found." in build_summary(make_table(leaderboard=None))


def test_questions_without_a_spot_peer_score_are_not_ranked():
    questions = [make_question(1, spot_peer=None, baseline=40.0), make_question(2, 5.0)]
    report = build_summary(make_table(questions=questions))
    assert "Scored: 2" in report
    assert "## Best 1 (by spot peer score)" in report
    assert "Q1" not in report.split("## Best")[1]


def test_a_table_saved_to_json_renders_the_same_report():
    table = make_table()
    reloaded = OutcomeTable.model_validate_json(table.model_dump_json())
    assert build_summary(reloaded) == build_summary(table)


def test_no_scored_questions():
    questions = [make_question(1, scored=False, status="open")]
    report = build_summary(make_table(questions=questions, leaderboard=None))
    assert "## Best 0 (by spot peer score)" in report
    assert "Mean coverage" not in report
