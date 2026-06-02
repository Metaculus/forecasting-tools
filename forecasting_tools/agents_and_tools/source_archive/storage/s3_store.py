"""S3-backed blob store (boto3).

Bucket and credentials come from :class:`ArchiveConfig` / the environment and are
never hardcoded, so this is safe to publish. boto3 is optional and imported
lazily (``pip install forecasting-tools[source-archive]``).
"""

from __future__ import annotations

from forecasting_tools.agents_and_tools.source_archive.config import ArchiveConfig


class S3BlobStore:
    def __init__(
        self, bucket: str, *, config: ArchiveConfig | None = None, client=None
    ):
        if not bucket:
            raise ValueError(
                "S3BlobStore requires a bucket name (set WEB_ARCHIVE_S3_BUCKET)"
            )
        self.bucket = bucket
        self._config = config or ArchiveConfig()
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as e:
                raise ImportError(
                    "boto3 is not installed. Install it with "
                    "`pip install forecasting-tools[source-archive]`."
                ) from e

            session = boto3.Session(
                profile_name=self._config.aws_profile,
                region_name=self._config.aws_region,
            )
            self._client = session.client("s3")
        return self._client

    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        self._get_client().put_object(Bucket=self.bucket, Key=key, Body=data, **extra)

    def get(self, key: str) -> bytes:
        resp = self._get_client().get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._get_client().head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
