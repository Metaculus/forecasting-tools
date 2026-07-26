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
from forecasting_tools.bot_review.traces import attach_traces, get_trace
from forecasting_tools.helpers.metaculus_client import MetaculusClient


def _write_and_print(table: OutcomeTable, args: argparse.Namespace) -> None:
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
    source.add_argument(
        "--from-json", metavar="FILE", help="re-render a table saved with --output"
    )
    source.add_argument(
        "--show",
        type=int,
        metavar="POST_ID",
        help="print part of the bot's report on one post",
    )
    parser.add_argument(
        "--section",
        default="research",
        choices=["summary", "research", "forecasts"],
        help="which section --show prints",
    )
    parser.add_argument(
        "--forecaster", metavar="KEY", help="print one rationale, e.g. R1:F2"
    )
    parser.add_argument(
        "--comment",
        type=int,
        metavar="ID",
        help="which run --show reads, from a trace's comment_id (default: the latest)",
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

    if args.from_json:
        with open(args.from_json) as file:
            table = OutcomeTable.model_validate_json(file.read())
        print(f"\nreviewing forecasts made by user {table.user_id}")
        _write_and_print(table, args)
        return

    client = MetaculusClient(
        sleep_seconds_between_requests=args.seconds_between_requests
    )
    if args.show:
        print(get_trace(args.show, args.section, args.forecaster, args.comment, client))
        return

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

    attach_traces(table, client)
    _write_and_print(table, args)


if __name__ == "__main__":
    main()
