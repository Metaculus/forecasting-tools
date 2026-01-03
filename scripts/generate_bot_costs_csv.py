"""
Script to generate a CSV table of bot costs per tournament.

IMPORTANT: Please note that the observed cost of the bots ends up being roughly 2x the estimated costs in Fall 2025 AIB.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_bots import AllowedTourn, get_default_bot_dict

TOURNAMENT_SLUG_MAP: dict[AllowedTourn, str] = {
    AllowedTourn.MINIBENCH: "minibench",
    AllowedTourn.MAIN_AIB: "main_aib",
    AllowedTourn.MAIN_SITE: "main_site",
    AllowedTourn.METACULUS_CUP: "metaculus_cup",
    AllowedTourn.GULF_BREEZE: "gulf_breeze",
    AllowedTourn.DEMOCRACY_THREAT_INDEX: "democracy_threat_index",
}


def generate_bot_costs_csv(output_path: str | None = None) -> list[dict]:
    bot_dict = get_default_bot_dict()
    all_tournaments = list(AllowedTourn)

    rows: list[dict] = []
    for bot_name, config in bot_dict.items():
        cost_per_question = config.estimated_cost_per_question or 0
        bot_tournaments = config.tournaments

        row = {
            "bot_name": bot_name,
            "cost_per_question": cost_per_question,
        }

        for tournament in all_tournaments:
            slug = TOURNAMENT_SLUG_MAP.get(tournament, tournament.name.lower())
            if tournament in bot_tournaments:
                row[slug] = cost_per_question
            else:
                row[slug] = 0

        rows.append(row)

    rows.sort(key=lambda x: (x["cost_per_question"], x["bot_name"]))

    if output_path:
        fieldnames = ["bot_name", "cost_per_question"] + [
            TOURNAMENT_SLUG_MAP.get(t, t.name.lower()) for t in all_tournaments
        ]
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written to {output_path}")

    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No bots found")
        return

    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        row_line = " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
        print(row_line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate bot costs CSV")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV file path. If not provided, prints to stdout.",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print a formatted table to stdout (in addition to CSV if -o is provided)",
    )

    args = parser.parse_args()

    rows = generate_bot_costs_csv(args.output)

    if args.print_table or not args.output:
        print_table(rows)
