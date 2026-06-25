"""Filesystem-backed blob store for tests, local dev, and dry runs."""

from __future__ import annotations

from pathlib import Path


class LocalBlobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key

    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list_keys(self, prefix: str = "") -> list[str]:
        if not self.root.exists():
            return []
        keys = [
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*")
            if p.is_file()
        ]
        return sorted(k for k in keys if k.startswith(prefix))
