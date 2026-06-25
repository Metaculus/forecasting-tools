"""
Scheduled, agent-driven version of the analyze-bot-failures skill.

This orchestrates the whole skill end to end so it can run unattended on a
cron (GitHub Actions or a local crontab):

1. Runs the deterministic failure-log parser (``analyze_bot_run_failures.py``)
   to produce ``report.md`` / ``failures.json`` / raw logs.
2. Hands the report to a headless Cursor agent (``cursor-agent``) which follows
   the investigation/triage steps of the skill and writes a prioritized
   markdown summary.
3. Delivers that summary to Slack (incoming webhook) and/or email (SMTP), and
   always prints it to stdout so it shows up in the job logs.

Environment variables:
    GITHUB_TOKEN / GH_TOKEN   token for downloading job logs (required)
    CURSOR_API_KEY            auth for the headless cursor-agent (required for the agent step)
    CURSOR_AGENT_MODEL        model slug for cursor-agent (default: "auto")
    SLACK_WEBHOOK_URL         if set, the summary is posted to this Slack webhook
    SMTP_HOST/SMTP_PORT       if set (with the fields below), the summary is emailed
    SMTP_USER/SMTP_PASSWORD
    EMAIL_FROM/EMAIL_TO       comma-separated recipient list
    GITHUB_SERVER_URL/GITHUB_REPOSITORY/GITHUB_RUN_ID
                              used to link back to the triggering Actions run

Example:
    poetry run python scripts/skills/analyze_bot_failures/run_scheduled_analysis.py --since 1d
"""

import argparse
import json
import logging
import os
import signal
import smtplib
import subprocess
import sys
import urllib.request
from email.message import EmailMessage
from pathlib import Path

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_SCRIPT = SCRIPT_DIR / "analyze_bot_run_failures.py"
SKILL_FILE = SCRIPT_DIR / "SKILL.md"
DEFAULT_OUTPUT_DIR = Path("logs/workflow_failure_analysis")
AGENT_REPORT_FILENAME = "agent_report.md"
SLACK_MESSAGE_CHAR_LIMIT = 3500
EMAIL_SUBJECT = "Bot forecasting failure report"


def run_failure_log_analysis(since: str, output_dir: Path) -> None:
    logger.info("Running failure-log parser (since=%s)", since)
    subprocess.run(
        [
            sys.executable,
            str(ANALYSIS_SCRIPT),
            "--since",
            since,
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )


def find_latest_analysis_dir(output_dir: Path) -> Path:
    timestamped_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
    if not timestamped_dirs:
        raise RuntimeError(f"No analysis output found under {output_dir}")
    return max(timestamped_dirs, key=lambda path: path.name)


def count_parsed_failures(analysis_dir: Path) -> int:
    failures_json = analysis_dir / "failures.json"
    if not failures_json.exists():
        return 0
    return len(json.loads(failures_json.read_text()))


def build_agent_prompt(analysis_dir: Path, agent_report_path: Path) -> str:
    return f"""You are running the "analyze-bot-failures" skill unattended on a schedule.

The deterministic log parser has already run. Its output is here:
- Report: `{analysis_dir / "report.md"}`
- Machine-readable failures: `{analysis_dir / "failures.json"}`
- Raw per-job logs: `{analysis_dir / "raw_logs"}/`

The full skill instructions are at `{SKILL_FILE}`. Read it, then carry out
steps 2-5 (read the report, spot-check 2-3 raw logs for the most frequent
signatures, triage transient noise vs real bugs, map likely-real bugs to code
by reading the relevant files, and identify recurring question-id skip
candidates).

Hard constraints for this unattended run:
- DO NOT modify any code, skip lists, or workflows. This is read-only
  investigation. The only file you may write is the report described below.
- Be efficient: a handful of targeted file reads and at most a few raw-log
  spot-checks. Do not attempt to fix anything.

When done, WRITE your findings as markdown to exactly this path:
`{agent_report_path}`

Structure the report so it is useful as a Slack/email digest:
1. One-line health summary (how many failed jobs / failures, overall severity).
2. Failure counts by category and by bot (from report.md).
3. Real bugs found: each with a short evidence excerpt + code location
   (`path:line`) + a concrete proposed fix. If none, say so explicitly.
4. Transient noise: what to ignore and why.
5. Question skip candidates: recurring question IDs with counts, if any.
Keep it tight and skimmable. Lead with what a human needs to act on.

After writing the file, reply with just the word DONE."""


def run_cursor_agent(
    prompt: str, model: str, workspace: Path, timeout_seconds: int
) -> None:
    command = [
        "cursor-agent",
        "--print",
        "--force",
        "--output-format",
        "text",
        "--model",
        model,
        prompt,
    ]
    logger.info("Invoking cursor-agent (model=%s, timeout=%ss)", model, timeout_seconds)
    process = subprocess.Popen(
        command,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
        logger.info("cursor-agent output:\n%s", stdout)
    except subprocess.TimeoutExpired:
        logger.warning(
            "cursor-agent did not exit within %ss (known headless hang); "
            "terminating and using whatever report it already wrote.",
            timeout_seconds,
        )
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()


def read_report_or_fallback(agent_report_path: Path, analysis_dir: Path) -> str:
    if agent_report_path.exists() and agent_report_path.read_text().strip():
        return agent_report_path.read_text().strip()
    logger.warning(
        "Agent did not produce %s; falling back to the deterministic report.",
        agent_report_path,
    )
    deterministic_report = analysis_dir / "report.md"
    if deterministic_report.exists():
        return (
            "NOTE: the agent investigation step produced no report; "
            "showing the raw deterministic report instead.\n\n"
            + deterministic_report.read_text().strip()
        )
    return "No report could be generated."


def build_run_link() -> str | None:
    server = os.getenv("GITHUB_SERVER_URL")
    repository = os.getenv("GITHUB_REPOSITORY")
    run_id = os.getenv("GITHUB_RUN_ID")
    if server and repository and run_id:
        return f"{server}/{repository}/actions/runs/{run_id}"
    return None


def post_to_slack(report: str, webhook_url: str) -> None:
    run_link = build_run_link()
    header = "*Bot forecasting failure report*"
    if run_link:
        header += f" (<{run_link}|full logs & artifacts>)"
    body = report
    if len(body) > SLACK_MESSAGE_CHAR_LIMIT:
        body = body[:SLACK_MESSAGE_CHAR_LIMIT] + "\n…(truncated — see full artifacts)"
    payload = json.dumps({"text": f"{header}\n\n{body}"}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        logger.info("Posted report to Slack (status %s)", response.status)


def send_email(report: str) -> None:
    smtp_host = os.getenv("SMTP_HOST")
    email_from = os.getenv("EMAIL_FROM")
    email_to = os.getenv("EMAIL_TO")
    if not (smtp_host and email_from and email_to):
        return
    message = EmailMessage()
    message["Subject"] = EMAIL_SUBJECT
    message["From"] = email_from
    message["To"] = email_to
    run_link = build_run_link()
    suffix = f"\n\nFull logs & artifacts: {run_link}" if run_link else ""
    message.set_content(report + suffix)

    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
        smtp.starttls()
        if smtp_user and smtp_password:
            smtp.login(smtp_user, smtp_password)
        smtp.send_message(message)
    logger.info("Emailed report to %s", email_to)


def deliver_report(report: str) -> None:
    print("\n===== REPORT =====\n")
    print(report)
    print("\n==================\n")
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if webhook_url:
        post_to_slack(report, webhook_url)
    send_email(report)


def run_scheduled_analysis(
    since: str,
    output_dir: Path,
    model: str,
    agent_timeout_seconds: int,
    skip_agent: bool,
) -> None:
    run_failure_log_analysis(since, output_dir)
    analysis_dir = find_latest_analysis_dir(output_dir)
    failure_count = count_parsed_failures(analysis_dir)
    logger.info("Parsed %s failures in %s", failure_count, analysis_dir)

    if failure_count == 0:
        deliver_report(
            f"No bot job failures found in the last `{since}`. All green. ✅"
        )
        return

    if skip_agent or not os.getenv("CURSOR_API_KEY"):
        logger.warning(
            "Skipping agent step (skip_agent=%s, CURSOR_API_KEY set=%s); "
            "delivering the deterministic report.",
            skip_agent,
            bool(os.getenv("CURSOR_API_KEY")),
        )
        deliver_report((analysis_dir / "report.md").read_text().strip())
        return

    agent_report_path = analysis_dir / AGENT_REPORT_FILENAME
    prompt = build_agent_prompt(analysis_dir, agent_report_path)
    run_cursor_agent(
        prompt=prompt,
        model=model,
        workspace=Path.cwd(),
        timeout_seconds=agent_timeout_seconds,
    )
    report = read_report_or_fallback(agent_report_path, analysis_dir)
    deliver_report(report)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Run the analyze-bot-failures skill end to end on a schedule"
    )
    parser.add_argument("--since", default="1d")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=os.getenv("CURSOR_AGENT_MODEL", "auto"))
    parser.add_argument("--agent-timeout-seconds", type=int, default=1500)
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Skip the cursor-agent step and deliver the deterministic report only",
    )
    args = parser.parse_args()

    run_scheduled_analysis(
        since=args.since,
        output_dir=args.output_dir,
        model=args.model,
        agent_timeout_seconds=args.agent_timeout_seconds,
        skip_agent=args.skip_agent,
    )


if __name__ == "__main__":
    main()
