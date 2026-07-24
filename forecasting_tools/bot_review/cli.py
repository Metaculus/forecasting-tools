"""Command line entry point for reviewing a bot's forecasts."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv

from forecasting_tools.bot_review.outcomes import (
    OutcomeTable,
    get_outcomes_for_posts,
    get_recently_resolved_outcomes,
    get_tournament_outcomes,
)
from forecasting_tools.bot_review.summary import build_summary
from forecasting_tools.helpers.metaculus_client import MetaculusClient


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
    parser.add_argument("--summary", help="write the markdown report to this file")
    parser.add_argument(
        "--top", type=int, default=10, help="how many best and worst questions to list"
    )
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
    user_id = client.get_current_user_id()
    print(f"\nreviewing forecasts made by user {user_id}")
    if args.tournament:
        table = get_tournament_outcomes(
            args.tournament,
            client=client,
            forecasted_only=not args.include_unforecasted,
        )
    else:
        if args.post:
            outcomes = get_outcomes_for_posts(args.post, client)
            project_name = "selected questions"
        else:
            outcomes = get_recently_resolved_outcomes(args.resolved_since, client)
            project_name = f"questions resolved in the last {args.resolved_since} days"
        table = OutcomeTable(
            generated_at=datetime.now(tz=timezone.utc),
            user_id=user_id,
            questions=outcomes,
            project_name=project_name,
        )

    report = build_summary(table, top_n=args.top)
    print(report)

    if args.output:
        with open(args.output, "w") as file:
            file.write(table.model_dump_json(indent=2))
        print(f"wrote {args.output}")
    if args.summary:
        with open(args.summary, "w") as file:
            file.write(report)
        print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
