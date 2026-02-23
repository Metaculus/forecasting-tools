from __future__ import annotations

import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod

from forecasting_tools.agents_and_tools.decision_hub.data_models import (
    AnalysisSession,
    _new_id,
    _utcnow,
)

logger = logging.getLogger(__name__)


class DecisionHubDB(ABC):

    @abstractmethod
    def create_user(self, username: str) -> None: ...

    @abstractmethod
    def get_user(self, username: str) -> dict | None: ...

    @abstractmethod
    def create_session(self, session: AnalysisSession) -> str: ...

    @abstractmethod
    def list_sessions(self, username: str) -> list[AnalysisSession]: ...

    @abstractmethod
    def get_session(self, session_id: str) -> AnalysisSession | None: ...

    @abstractmethod
    def update_session(self, session: AnalysisSession) -> None: ...

    @abstractmethod
    def save_artifact(
        self, session_id: str, step_name: str, data: dict, username: str
    ) -> str: ...

    @abstractmethod
    def load_artifacts(self, session_id: str, step_name: str) -> list[dict]: ...

    @abstractmethod
    def delete_artifact(self, artifact_id: str) -> None: ...

    @abstractmethod
    def update_artifact(self, artifact_id: str, data: dict) -> None: ...


class SQLiteBackend(DecisionHubDB):
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.getenv("DECISION_HUB_DB_PATH", "decision_hub.db")
        self._initialize_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize_tables(self) -> None:
        conn = self._get_connection()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    policy_question TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'in_progress',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (username) REFERENCES users(username)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    username TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY (username) REFERENCES users(username)
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_username
                    ON sessions(username);
                CREATE INDEX IF NOT EXISTS idx_artifacts_session_step
                    ON artifacts(session_id, step_name);
                """
            )
            conn.commit()
        finally:
            conn.close()

    def create_user(self, username: str) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO users (username, created_at) VALUES (?, ?)",
                (username, _utcnow().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_user(self, username: str) -> dict | None:
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def create_session(self, session: AnalysisSession) -> str:
        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT INTO sessions
                   (session_id, username, policy_question, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session.session_id,
                    session.username,
                    session.policy_question,
                    session.status,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )
            conn.commit()
            return session.session_id
        finally:
            conn.close()

    def list_sessions(self, username: str) -> list[AnalysisSession]:
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE username = ? ORDER BY updated_at DESC",
                (username,),
            ).fetchall()
            return [
                AnalysisSession(
                    session_id=row["session_id"],
                    username=row["username"],
                    policy_question=row["policy_question"],
                    status=row["status"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]
        finally:
            conn.close()

    def get_session(self, session_id: str) -> AnalysisSession | None:
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not row:
                return None
            return AnalysisSession(
                session_id=row["session_id"],
                username=row["username"],
                policy_question=row["policy_question"],
                status=row["status"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        finally:
            conn.close()

    def update_session(self, session: AnalysisSession) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                """UPDATE sessions
                   SET policy_question = ?, status = ?, updated_at = ?
                   WHERE session_id = ?""",
                (
                    session.policy_question,
                    session.status,
                    _utcnow().isoformat(),
                    session.session_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def save_artifact(
        self, session_id: str, step_name: str, data: dict, username: str
    ) -> str:
        artifact_id = (
            data.get("artifact_id")
            or data.get("question_id")
            or data.get("forecast_id")
            or _new_id()
        )
        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO artifacts
                   (artifact_id, session_id, step_name, username, data, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    artifact_id,
                    session_id,
                    step_name,
                    username,
                    json.dumps(data),
                    _utcnow().isoformat(),
                ),
            )
            conn.commit()
            return artifact_id
        finally:
            conn.close()

    def load_artifacts(self, session_id: str, step_name: str) -> list[dict]:
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """SELECT data FROM artifacts
                   WHERE session_id = ? AND step_name = ?
                   ORDER BY created_at ASC""",
                (session_id, step_name),
            ).fetchall()
            return [json.loads(row["data"]) for row in rows]
        finally:
            conn.close()

    def delete_artifact(self, artifact_id: str) -> None:
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,))
            conn.commit()
        finally:
            conn.close()

    def update_artifact(self, artifact_id: str, data: dict) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE artifacts SET data = ? WHERE artifact_id = ?",
                (json.dumps(data), artifact_id),
            )
            conn.commit()
        finally:
            conn.close()


def get_database() -> DecisionHubDB:
    backend = os.getenv("DECISION_HUB_DB_BACKEND", "sqlite")
    if backend == "sqlite":
        return SQLiteBackend()
    elif backend == "supabase":
        raise NotImplementedError(
            "Supabase backend not yet implemented. Set DECISION_HUB_DB_BACKEND=sqlite."
        )
    else:
        raise ValueError(f"Unknown database backend: {backend}")
