"""Turn an outcome table into a markdown report.

Covers the bot's leaderboard standing, how many questions it forecast and
scored, and the questions it did best and worst on.
"""

from __future__ import annotations

from collections import Counter

from forecasting_tools.bot_review.outcomes import OutcomeTable, QuestionOutcome

RANK_METRIC = "spot_peer_score"


def _signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}"


def _project_label(outcome: QuestionOutcome) -> str:
    return outcome.project_name or outcome.project_slug or str(outcome.project_id)


def _question_line(rank: int, outcome: QuestionOutcome) -> str:
    assert outcome.scores is not None
    return (
        f"{rank}. {_signed(outcome.scores.spot_peer_score)} spot-peer · "
        f"{_signed(outcome.scores.baseline_score)} baseline · "
        f"{outcome.question_type} · [{outcome.title}]({outcome.url})"
    )


def _overall_lines(table: OutcomeTable, scored: list[QuestionOutcome]) -> list[str]:
    lines = ["## Overall", ""]
    leaderboard = table.leaderboard
    entry = leaderboard.user_entry if leaderboard else None
    if leaderboard is not None and entry is not None:
        lines += [
            (
                f"- Rank: **{entry.rank} / {len(leaderboard.entries)}** "
                f"({leaderboard.score_type})"
            ),
            (
                f"- Score: **{entry.score:.2f}**"
                if entry.score is not None
                else "- Score: n/a"
            ),
            f"- Medal: {entry.medal or 'none'}",
        ]
    else:
        lines.append("- No leaderboard entry found.")
    coverages = [
        outcome.scores.coverage
        for outcome in scored
        if outcome.scores is not None and outcome.scores.coverage is not None
    ]
    if coverages:
        mean_coverage = sum(coverages) / len(coverages)
        lines.append(
            f"- Mean coverage: {mean_coverage:.2f} (over {len(coverages)} scored questions)"
        )
    return lines


def build_summary(table: OutcomeTable, top_n: int = 10) -> str:
    """
    Render an outcome table as a markdown report.

    :param table: the questions and standing to report on
    :param top_n: how many questions to list under best and worst
    """
    questions = table.questions
    forecasted = [outcome for outcome in questions if outcome.forecasted]
    scored = [outcome for outcome in forecasted if outcome.scores is not None]
    unscored = [outcome for outcome in forecasted if outcome.scores is None]
    resolved_unscored = [
        outcome for outcome in unscored if outcome.status == "resolved"
    ]
    ranked = sorted(
        (
            outcome
            for outcome in scored
            if outcome.scores is not None
            and getattr(outcome.scores, RANK_METRIC) is not None
        ),
        key=lambda outcome: getattr(outcome.scores, RANK_METRIC),
        reverse=True,
    )

    lines = [
        f"# Review: {table.project_name or 'forecast outcomes'}",
        "",
        f"user {table.user_id} — generated {table.generated_at.isoformat()}",
        "",
    ]
    lines += _overall_lines(table, scored)
    lines += [
        "",
        "## Participation",
        "",
        f"- Questions: {len(questions)}",
        f"- Forecasted: {len(forecasted)}",
        f"- Scored: {len(scored)}",
        (
            f"- Forecasted but unscored: {len(unscored)} "
            f"({len(resolved_unscored)} resolved without a score)"
        ),
    ]
    for outcome in resolved_unscored:
        lines.append(
            f"  - [{outcome.title}]({outcome.url}) — resolution: {outcome.resolution}"
        )

    lines += ["", "### By tournament", ""]
    for label, count in Counter(_project_label(o) for o in questions).most_common():
        scored_here = sum(1 for o in scored if _project_label(o) == label)
        lines.append(f"- {label}: {count} questions ({scored_here} scored)")

    lines += ["", f"## Best {min(top_n, len(ranked))} (by spot peer score)", ""]
    lines += [_question_line(i, o) for i, o in enumerate(ranked[:top_n], 1)]
    lines += ["", f"## Worst {min(top_n, len(ranked))} (by spot peer score)", ""]
    lines += [_question_line(i, o) for i, o in enumerate(reversed(ranked[-top_n:]), 1)]

    return "\n".join(lines) + "\n"
