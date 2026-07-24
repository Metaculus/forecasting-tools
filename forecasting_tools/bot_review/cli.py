"""Command line entry point for reviewing a bot's forecasts."""

from __future__ import annotations

import argparse
import json
import logging

from dotenv import load_dotenv

from forecasting_tools.bot_review.outcomes import (
    QuestionOutcome,
    get_outcomes_for_posts,
    get_recently_resolved_outcomes,
    get_tournament_outcomes,
)
from forecasting_tools.helpers.metaculus_client import MetaculusClient


def _print_outcome(outcome: QuestionOutcome) -> None:
    print(f"{outcome.post_id} {outcome.question_type or '':16} {outcome.title or ''}")
    print(f"    status={outcome.status} resolution={outcome.resolution}")
    print(f"    forecast={outcome.forecast}")
    print(f"    scores={outcome.scores}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review how a bot's forecasts did")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tournament", help="tournament slug or id")
    source.add_argument("--post", type=int, nargs="+", metavar="POST_ID")
    source.add_argument(
        "--resolved-since",
        type=int,
        metavar="DAYS",
        help="your questions that resolved in the last DAYS days, any tournament",
    )
    parser.add_argument(
        "--include-unforecasted",
        action="store_true",
        help="also include questions in the tournament that the bot never forecast",
    )
    parser.add_argument("--output", help="write the full table to this json file")
    parser.add_argument(
        "--seconds-between-requests",
        type=float,
        default=0.7,
        help="delay between Metaculus requests",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    load_dotenv()

    client = MetaculusClient(
        sleep_seconds_between_requests=args.seconds_between_requests
    )
    print(f"\nreviewing forecasts made by user {client.get_current_user_id()}")
    if args.post:
        outcomes = get_outcomes_for_posts(args.post, client)
        report = None
    elif args.resolved_since:
        outcomes = get_recently_resolved_outcomes(args.resolved_since, client)
        report = None
    else:
        report = get_tournament_outcomes(
            args.tournament,
            client=client,
            forecasted_only=not args.include_unforecasted,
        )
        outcomes = report.questions
        print(f"{report.project_name} ({report.project_id})")
        print(f"leaderboard: {report.leaderboard_entry}")

    scored = [outcome for outcome in outcomes if outcome.scores]
    forecasted = [outcome for outcome in outcomes if outcome.forecasted]
    print(
        f"questions {len(outcomes)} | forecasted {len(forecasted)} | scored {len(scored)}\n"
    )
    for outcome in outcomes:
        _print_outcome(outcome)

    if args.output:
        if report is not None:
            json_text = report.model_dump_json(indent=2)
        else:
            json_text = json.dumps(
                [outcome.model_dump(mode="json") for outcome in outcomes], indent=2
            )
        with open(args.output, "w") as file:
            file.write(json_text)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
