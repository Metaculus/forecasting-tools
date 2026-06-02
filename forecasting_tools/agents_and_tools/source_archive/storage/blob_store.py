"""Blob store interface.

The content store and manifest layer depend on this abstraction, not on S3
directly, so they can run offline against :class:`LocalBlobStore`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BlobStore(Protocol):
    def put(
        self, key: str, data: bytes, *, content_type: str | None = None
    ) -> None: ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...
