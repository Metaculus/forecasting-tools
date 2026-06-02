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
from forecasting_tools.agents_and_tools.source_archive.pipeline import CapturePipeline


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
    print(f"  TTL (days)           : {config.ttl_days}")
    print(f"  Screenshot format    : {config.screenshot_format}")
    print(f"  Screenshot max height: {config.screenshot_max_height}")
    return 0


def _cmd_capture(args, config: ArchiveConfig) -> int:
    records = manifest_io.read_file(args.manifest)
    store = ContentStore(_make_blob_store(config, args.local, args.bucket), config)

    target = args.local or f"s3://{args.bucket or config.s3_bucket}/{config.s3_prefix}"
    print(f"Capturing {len(records)} citation record(s) -> {target}")

    with build_default_fetcher(config) as fetcher:
        pipeline = CapturePipeline(fetcher, store)
        summary = pipeline.run_manifest(records)
    print(summary)

    if args.upload_manifest:
        run_id = args.run_id or (records[0].run_id if records else None)
        if not run_id:
            sys.exit("--upload-manifest needs --run-id (no run_id found in records)")
        manifest_io.write_blob(store.blobs, run_id, records, config)
        print(f"Uploaded manifest -> {config.s3_prefix}/manifests/{run_id}.jsonl")
    return 0


def _cmd_harvest(args, config: ArchiveConfig) -> int:
    from forecasting_tools.agents_and_tools.source_archive.ingest import (
        MetaculusCommentHarvester,
    )

    run_id = args.run_id or f"metaculus-comments-{args.project_id}"
    harvester = MetaculusCommentHarvester()
    records = harvester.harvest_project(args.project_id, run_id=run_id)
    print(
        f"Harvested {len(records)} citation record(s) from project "
        f"{args.project_id}"
    )

    out_path = args.out or f"{run_id}.jsonl"
    if not args.upload or args.out:
        manifest_io.write_file(out_path, records)
        print(f"Wrote manifest -> {out_path}")
    if args.upload:
        store = _make_blob_store(config, None, args.bucket)
        manifest_io.write_blob(store, run_id, records, config)
        print(f"Uploaded manifest -> {config.s3_prefix}/manifests/{run_id}.jsonl")
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

    harv = sub.add_parser(
        "harvest",
        help="harvest cited URLs from bot comments on a Metaculus project",
    )
    harv.add_argument("project_id", help="Metaculus project / tournament id")
    harv.add_argument(
        "--out", metavar="FILE", help="write the manifest to this .jsonl file"
    )
    harv.add_argument(
        "--run-id", help="run id (default: metaculus-comments-<project_id>)"
    )
    harv.add_argument(
        "--upload", action="store_true", help="upload the manifest to S3 manifests/"
    )
    harv.add_argument("--bucket", help="override the S3 bucket")

    args = parser.parse_args(argv)
    config = ArchiveConfig.from_env()

    if args.command == "check":
        return _cmd_check(config)
    if args.command == "capture":
        return _cmd_capture(args, config)
    if args.command == "harvest":
        return _cmd_harvest(args, config)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
