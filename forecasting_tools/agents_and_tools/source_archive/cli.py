"""Command-line interface for the source archive.

    # See the resolved configuration (secrets masked)
    python -m forecasting_tools.agents_and_tools.source_archive.cli check

    # Capture every URL in a manifest and upload to S3 (uses WEB_ARCHIVE_S3_BUCKET)
    python -m forecasting_tools.agents_and_tools.source_archive.cli capture run.jsonl

    # Same, but store to a local folder instead of S3 (no AWS needed)
    python -m forecasting_tools.agents_and_tools.source_archive.cli capture run.jsonl --local ./archive

If installed via the ``source-archive`` extra, the ``source-archive`` console
command is equivalent to ``python -m ...cli``.
"""

from __future__ import annotations

import argparse
import sys

from forecasting_tools.agents_and_tools.source_archive import manifest as manifest_io
from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig
from forecasting_tools.agents_and_tools.source_archive.content_store import ContentStore
from forecasting_tools.agents_and_tools.source_archive.fetchers import (
    build_default_fetcher,
)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _mask(value: str | None) -> str:
    if not value:
        return "(unset)"
    if len(value) <= 6:
        return "***"
    return f"{value[:3]}…{value[-2:]}"


def _make_blob_store(config: ArchiveConfig, local_dir: str | None, bucket: str | None):
    if local_dir:
        from forecasting_tools.agents_and_tools.source_archive.storage import (
            LocalBlobStore,
        )

        return LocalBlobStore(local_dir)
    bucket = bucket or config.s3_bucket
    if not bucket:
        sys.exit(
            "No S3 bucket configured. Set WEB_ARCHIVE_S3_BUCKET (or pass --bucket), "
            "or use --local DIR to store to the filesystem."
        )
    from forecasting_tools.agents_and_tools.source_archive.storage import S3BlobStore

    return S3BlobStore(bucket, config=config)


def _cmd_check(config: ArchiveConfig) -> int:
    print("Source-archive configuration (secrets masked):")
    print(f"  S3 bucket            : {config.s3_bucket or '(unset)'}")
    print(f"  S3 prefix            : {config.s3_prefix}")
    print(f"  AWS profile          : {config.aws_profile or '(default chain)'}")
    print(f"  AWS region           : {config.aws_region or '(default)'}")
    print(f"  Firecrawl API key    : {_mask(config.firecrawl_api_key)}")
    print(f"  Firecrawl proxy mode : {config.firecrawl_proxy}")
    print(f"  Hyperbrowser API key : {_mask(config.hyperbrowser_api_key)}")
    print(f"  Hyperbrowser proxy   : {config.hyperbrowser_use_proxy}")
    print(f"  CloakBrowser module  : {config.cloakbrowser_import}")
    print(f"  PDF max pages        : {config.pdf_max_pages}")
    print(f"  TTL (days)           : {config.ttl_days}")
    print(f"  Screenshot format    : {config.screenshot_format}")
    print(f"  Screenshot max height: {config.screenshot_max_height}")
    return 0


def _cmd_capture(args, config: ArchiveConfig) -> int:
    from forecasting_tools.agents_and_tools.source_archive.manifest import unique_urls
    from forecasting_tools.agents_and_tools.source_archive.pipeline import (
        capture_urls_concurrent,
    )

    records = manifest_io.read_file(args.manifest)

    overrides = {}
    if getattr(args, "no_hyperbrowser", False):
        overrides["hyperbrowser_api_key"] = None
    if getattr(args, "concurrency", None):
        overrides["concurrency"] = args.concurrency
    if overrides:
        config = config.model_copy(update=overrides)
    if "hyperbrowser_api_key" in overrides:
        print("Hyperbrowser fallback DISABLED for this run.")

    store = ContentStore(_make_blob_store(config, args.local, args.bucket), config)

    urls = list(unique_urls(records))
    if args.limit:
        urls = urls[: args.limit]
    target = args.local or f"s3://{args.bucket or config.s3_bucket}/{config.s3_prefix}"
    print(
        f"Capturing {len(urls)} URL(s) at concurrency {config.concurrency} -> {target}"
    )

    summary = capture_urls_concurrent(urls, store, config, build_default_fetcher)
    print(summary)

    from forecasting_tools.agents_and_tools.source_archive import cost as cost_mod

    run_cost = cost_mod.estimate_run_cost(summary, config, run_id=args.run_id)
    print(run_cost)

    run_id = args.run_id or (records[0].run_id if records else None)
    if run_id:
        from forecasting_tools.agents_and_tools.source_archive import reports

        reports.write_run_report(store.blobs, run_id, summary, config)
        print(f"Wrote run outcomes -> {config.s3_prefix}/reports/{run_id}.json")
        cost_mod.write_cost_report(store.blobs, run_id, run_cost, config)
        print(f"Wrote cost report -> {config.s3_prefix}/reports/{run_id}_cost.json")

    # Failures leave no cache entry, so re-running retries exactly them. Write a
    # retry manifest (with provenance) so coming back — e.g. with hyperbrowser
    # re-enabled — is one command over only the sites that still need it.
    failed = {
        o.url for o in summary.outcomes if o.status in ("quality_failed", "error")
    }
    if failed:
        from forecasting_tools.agents_and_tools.source_archive.ingest import (
            dedupe_records,
        )

        retry_records = dedupe_records(r for r in records if r.url in failed)
        retry_path = args.retry_out or f"{run_id or 'run'}_needs_retry.jsonl"
        manifest_io.write_file(retry_path, retry_records)
        print(
            f"{len(failed)} URL(s) failed -> retry manifest {retry_path}\n"
            f"  come back later with:  source-archive capture {retry_path} "
            f"--run-id {run_id or '<run-id>'}   (hyperbrowser on by default)"
        )

    if args.upload_manifest:
        if not run_id:
            sys.exit("--upload-manifest needs --run-id (no run_id found in records)")
        manifest_io.write_blob(store.blobs, run_id, records, config)
        print(f"Uploaded manifest -> {config.s3_prefix}/manifests/{run_id}.jsonl")
    return 0


def _cmd_ingest_traces(args, config: ArchiveConfig) -> int:
    from forecasting_tools.agents_and_tools.source_archive.ingest import (
        dedupe_records,
        harvest_run,
    )

    run_id = args.run_id  # None -> derived from the run dir name
    records = harvest_run(args.run_dir, run_id=run_id, bot=args.bot)
    if args.dedupe:
        records = dedupe_records(records)
    run_id = run_id or (records[0].run_id if records else None)
    print(f"Ingested {len(records)} citation record(s) from traces in {args.run_dir}")

    out_path = args.out or f"{run_id or 'traces'}.jsonl"
    if not args.upload or args.out:
        manifest_io.write_file(out_path, records)
        print(f"Wrote manifest -> {out_path}")
    if args.upload:
        if not run_id:
            sys.exit("--upload needs a run id (pass --run-id; none found in records)")
        store = _make_blob_store(config, None, args.bucket)
        manifest_io.write_blob(store, run_id, records, config)
        print(f"Uploaded manifest -> {config.s3_prefix}/manifests/{run_id}.jsonl")
    return 0


def _cmd_catalog(args, config: ArchiveConfig) -> int:
    from forecasting_tools.agents_and_tools.source_archive.catalog import write_catalog

    store = _make_blob_store(config, args.local, args.bucket)
    target = args.local or f"s3://{args.bucket or config.s3_bucket}/{config.s3_prefix}"
    print(f"Building catalog from manifests + index -> {target}/catalog/")
    summary = write_catalog(store, config)
    print(summary)
    print(f"Open {config.s3_prefix}/catalog/index.html")
    return 0


def _cmd_harvest_db(args, config: ArchiveConfig) -> int:
    from forecasting_tools.agents_and_tools.source_archive.ingest import (
        MetaculusDbHarvester,
        dedupe_records,
        resolve_dsn,
    )

    dsn = resolve_dsn(args.dsn)
    include_private = not args.public_only
    harvester = MetaculusDbHarvester.from_dsn(dsn)
    if args.post:
        records = harvester.harvest_post(
            args.post, run_id=args.run_id, include_private=include_private
        )
        run_id = args.run_id or f"metaculus-db-post-{args.post}"
    else:
        records = harvester.harvest_recent(
            days=args.days,
            limit=args.limit,
            run_id=args.run_id,
            include_private=include_private,
        )
        run_id = args.run_id or f"metaculus-db-recent-{args.days}d"
    if args.dedupe:
        records = dedupe_records(records)
    print(f"Harvested {len(records)} citation record(s) from the Metaculus DB")

    out_path = args.out or f"{run_id}.jsonl"
    if not args.upload or args.out:
        manifest_io.write_file(out_path, records)
        print(f"Wrote manifest -> {out_path}")
    if args.upload:
        store = _make_blob_store(config, None, args.bucket)
        manifest_io.write_blob(store, run_id, records, config)
        print(f"Uploaded manifest -> {config.s3_prefix}/manifests/{run_id}.jsonl")
    return 0


def _cmd_coverage(args, config: ArchiveConfig) -> int:
    from pathlib import Path

    from forecasting_tools.agents_and_tools.source_archive import reports
    from forecasting_tools.agents_and_tools.source_archive.catalog import build_sources
    from forecasting_tools.agents_and_tools.source_archive.coverage import (
        MODES,
        coverage_from_sources,
    )

    store = _make_blob_store(config, args.local, args.bucket)
    sources = build_sources(store, config)  # read manifests + index once
    outcomes = reports.read_outcomes(store, config) or None
    modes = MODES if args.mode == "both" else (args.mode,)
    for mode in modes:
        report = coverage_from_sources(sources, mode, outcomes)
        print(report)
        print()
        if args.csv:
            Path(f"{args.csv}_{mode}.csv").write_text(report.to_csv())
            print(f"Wrote {args.csv}_{mode}.csv")
            if report.missing_urls:
                Path(f"{args.csv}_{mode}_missing.txt").write_text(
                    "\n".join(report.missing_urls)
                )
                print(f"Wrote {args.csv}_{mode}_missing.txt")
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        prog="source-archive",
        description="Capture HTML + screenshot + markdown for the URLs a "
        "forecasting bot cited, and store them with provenance.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="print the resolved configuration (secrets masked)")

    cap = sub.add_parser("capture", help="capture all URLs in a citation manifest")
    cap.add_argument("manifest", help="path to a citation manifest (.jsonl)")
    cap.add_argument(
        "--local", metavar="DIR", help="store to this directory instead of S3"
    )
    cap.add_argument(
        "--bucket", help="override the S3 bucket (default: WEB_ARCHIVE_S3_BUCKET)"
    )
    cap.add_argument(
        "--upload-manifest",
        action="store_true",
        help="also upload the manifest itself to manifests/<run_id>.jsonl",
    )
    cap.add_argument("--run-id", help="run id for the uploaded manifest")
    cap.add_argument(
        "--no-hyperbrowser",
        action="store_true",
        help="disable the Hyperbrowser fallback for this run (others still run)",
    )
    cap.add_argument(
        "--retry-out",
        metavar="FILE",
        help="where to write the failed-URL retry manifest "
        "(default: <run_id>_needs_retry.jsonl)",
    )
    cap.add_argument(
        "--concurrency",
        type=int,
        metavar="N",
        help="parallel browser workers for this run (overrides WEB_ARCHIVE_CONCURRENCY)",
    )
    cap.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="capture only the first N URLs (chunk a big manifest; resume via cache)",
    )

    ing = sub.add_parser(
        "ingest-traces",
        help="build a manifest from a traced bot run directory (bot_*/q_*/traces_*.jsonl)",
    )
    ing.add_argument("run_dir", help="path to a traced run directory")
    ing.add_argument(
        "--out", metavar="FILE", help="write the manifest to this .jsonl file"
    )
    ing.add_argument("--run-id", help="run id (default: the run dir's name)")
    ing.add_argument(
        "--bot",
        help="bot name for a flat (no bot_*/) layout (default: the run dir's name)",
    )
    ing.add_argument(
        "--dedupe", action="store_true", help="keep one record per URL (first seen)"
    )
    ing.add_argument(
        "--upload", action="store_true", help="upload the manifest to S3 manifests/"
    )
    ing.add_argument("--bucket", help="override the S3 bucket")

    cat = sub.add_parser(
        "catalog",
        help="generate a coworker-legible HTML/CSV catalog (by question/bot/site)",
    )
    cat.add_argument(
        "--local", metavar="DIR", help="read/write the catalog in this directory"
    )
    cat.add_argument("--bucket", help="override the S3 bucket")

    hdb = sub.add_parser(
        "harvest-db",
        help="read a bot's cited URLs from the platform Postgres database (operator)",
    )
    grp = hdb.add_mutually_exclusive_group(required=True)
    grp.add_argument("--post", help="harvest one post id")
    grp.add_argument("--days", type=int, help="harvest the most recent N days")
    hdb.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap rows when using --days (default: uncapped — a daily sweep wants all)",
    )
    hdb.add_argument(
        "--public-only",
        action="store_true",
        help="read only public comments (default: read all of a bot's comments)",
    )
    hdb.add_argument(
        "--dsn",
        help="libpq DSN or postgresql:// URL. Default resolution: --dsn > "
        "$METACULUS_DB_DSN > macOS Keychain item 'metaculus-db-dsn' > "
        "dbname=metaculus. Prefer the Keychain for the real secret "
        "(a --dsn value lands in shell history).",
    )
    hdb.add_argument("--out", metavar="FILE", help="write the manifest to this .jsonl")
    hdb.add_argument("--run-id", help="run id (default derived from the slice)")
    hdb.add_argument(
        "--dedupe", action="store_true", help="keep one record per URL (first seen)"
    )
    hdb.add_argument(
        "--upload", action="store_true", help="upload the manifest to S3 manifests/"
    )
    hdb.add_argument("--bucket", help="override the S3 bucket")

    cov = sub.add_parser(
        "coverage",
        help="report what %% of cited sources were archived (trace vs comments)",
    )
    cov.add_argument(
        "--mode",
        choices=["trace", "comments", "both"],
        default="both",
        help="which report(s) to print (default: both)",
    )
    cov.add_argument(
        "--csv", metavar="PREFIX", help="write PREFIX_<mode>.csv (+ _missing.txt)"
    )
    cov.add_argument("--local", metavar="DIR", help="read from this directory")
    cov.add_argument("--bucket", help="override the S3 bucket")

    args = parser.parse_args(argv)
    config = ArchiveConfig.from_env()

    if args.command == "check":
        return _cmd_check(config)
    if args.command == "capture":
        return _cmd_capture(args, config)
    if args.command == "ingest-traces":
        return _cmd_ingest_traces(args, config)
    if args.command == "harvest-db":
        return _cmd_harvest_db(args, config)
    if args.command == "catalog":
        return _cmd_catalog(args, config)
    if args.command == "coverage":
        return _cmd_coverage(args, config)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
