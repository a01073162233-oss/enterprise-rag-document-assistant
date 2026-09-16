from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registry:
    """SQLite-backed business metadata registry.

    Qdrant is deliberately not used as the business system of record. This keeps
    document lifecycle and chat history independent from vector index rebuilds.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    embedding_provider TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_overlap INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL CHECK(status IN ('processing', 'ready', 'failed')),
                    error TEXT,
                    created_at TEXT NOT NULL,
                    processed_at TEXT,
                    UNIQUE(knowledge_base_id, sha256)
                );

                CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(knowledge_base_id);
                CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    trace_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id, created_at);
                """
            )

    def create_knowledge_base(
        self,
        *,
        name: str,
        description: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        chunk_size: int,
        chunk_overlap: int,
    ) -> dict[str, Any]:
        kb_id = str(uuid.uuid4())
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_bases(
                    id, name, description, embedding_provider, embedding_model,
                    embedding_dimension, chunk_size, chunk_overlap, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kb_id,
                    name,
                    description,
                    embedding_provider,
                    embedding_model,
                    embedding_dimension,
                    chunk_size,
                    chunk_overlap,
                    now,
                    now,
                ),
            )
        result = self.get_knowledge_base(kb_id)
        assert result is not None
        return result

    @staticmethod
    def _kb_query(where: str = "") -> str:
        return f"""
            SELECT kb.*,
                   COUNT(DISTINCT CASE WHEN d.status = 'ready' THEN d.id END) AS document_count,
                   COALESCE(SUM(CASE WHEN d.status = 'ready' THEN d.chunk_count ELSE 0 END), 0) AS chunk_count
            FROM knowledge_bases kb
            LEFT JOIN documents d ON d.knowledge_base_id = kb.id
            {where}
            GROUP BY kb.id
        """

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(self._kb_query() + " ORDER BY kb.updated_at DESC").fetchall()
        return [dict(row) for row in rows]

    def get_knowledge_base(self, kb_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(self._kb_query("WHERE kb.id = ?"), (kb_id,)).fetchone()
        return dict(row) if row else None

    def delete_knowledge_base(self, kb_id: str) -> list[str]:
        with self.connect() as conn:
            paths = [
                row["stored_path"]
                for row in conn.execute(
                    "SELECT stored_path FROM documents WHERE knowledge_base_id = ?", (kb_id,)
                ).fetchall()
            ]
            conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
        return paths

    def find_document_by_hash(self, kb_id: str, sha256: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE knowledge_base_id = ? AND sha256 = ?",
                (kb_id, sha256),
            ).fetchone()
        return self._document_row(row) if row else None

    def create_document(
        self,
        *,
        kb_id: str,
        filename: str,
        stored_path: str,
        content_type: str,
        byte_size: int,
        sha256: str,
    ) -> dict[str, Any]:
        document_id = str(uuid.uuid4())
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(
                    id, knowledge_base_id, filename, stored_path, content_type,
                    byte_size, sha256, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'processing', ?)
                """,
                (document_id, kb_id, filename, stored_path, content_type, byte_size, sha256, now),
            )
            conn.execute("UPDATE knowledge_bases SET updated_at = ? WHERE id = ?", (now, kb_id))
        result = self.get_document(document_id)
        assert result is not None
        return result

    def retry_failed_document(
        self,
        document_id: str,
        *,
        filename: str,
        stored_path: str,
        content_type: str,
        byte_size: int,
        sha256: str,
    ) -> tuple[dict[str, Any], str] | None:
        """Atomically reuse a failed row without leaving a metadata gap."""
        now = utc_now()
        with self.connect() as conn:
            previous = conn.execute(
                "SELECT stored_path, knowledge_base_id FROM documents "
                "WHERE id = ? AND status = 'failed'",
                (document_id,),
            ).fetchone()
            if previous is None:
                return None
            updated = conn.execute(
                """
                UPDATE documents
                SET filename = ?, stored_path = ?, content_type = ?, byte_size = ?,
                    sha256 = ?, page_count = 0, chunk_count = 0,
                    status = 'processing', error = NULL, processed_at = NULL
                WHERE id = ? AND status = 'failed'
                """,
                (
                    filename,
                    stored_path,
                    content_type,
                    byte_size,
                    sha256,
                    document_id,
                ),
            )
            if updated.rowcount != 1:
                return None
            conn.execute(
                "UPDATE knowledge_bases SET updated_at = ? WHERE id = ?",
                (now, previous["knowledge_base_id"]),
            )
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        assert row is not None
        return self._document_row(row), str(previous["stored_path"])

    def update_document(
        self,
        document_id: str,
        *,
        status: str,
        page_count: int = 0,
        chunk_count: int = 0,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = ?, page_count = ?, chunk_count = ?, error = ?, processed_at = ?
                WHERE id = ?
                """,
                (status, page_count, chunk_count, error, now, document_id),
            )
            conn.execute(
                """
                UPDATE knowledge_bases SET updated_at = ?
                WHERE id = (SELECT knowledge_base_id FROM documents WHERE id = ?)
                """,
                (now, document_id),
            )
        return self.get_document(document_id)

    @staticmethod
    def _document_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["knowledge_base_id"] = item.pop("knowledge_base_id")
        return item

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return self._document_row(row) if row else None

    def list_documents(self, kb_id: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if kb_id:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE knowledge_base_id = ? ORDER BY created_at DESC",
                    (kb_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [self._document_row(row) for row in rows]

    def recover_interrupted_documents(self) -> list[dict[str, Any]]:
        """Mark ingestion rows left in-flight by a previous process as failed.

        Ingestion is completed inside the API process, so no worker from an old
        process can still own a ``processing`` row after startup.  Returning the
        rows lets the application also remove any vectors that were upserted
        before the interruption.
        """
        now = utc_now()
        message = "服务在文档处理完成前中断，请重新上传以重试"
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE status = 'processing'"
            ).fetchall()
            if not rows:
                return []
            conn.execute(
                """
                UPDATE documents
                SET status = 'failed', page_count = 0, chunk_count = 0,
                    error = ?, processed_at = ?
                WHERE status = 'processing'
                """,
                (message, now),
            )
            conn.executemany(
                "UPDATE knowledge_bases SET updated_at = ? WHERE id = ?",
                [(now, row["knowledge_base_id"]) for row in rows],
            )
        recovered = [self._document_row(row) for row in rows]
        for item in recovered:
            item.update(
                status="failed",
                page_count=0,
                chunk_count=0,
                error=message,
                processed_at=now,
            )
        return recovered

    def delete_document(self, document_id: str) -> dict[str, Any] | None:
        document = self.get_document(document_id)
        if not document:
            return None
        now = utc_now()
        with self.connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            conn.execute(
                "UPDATE knowledge_bases SET updated_at = ? WHERE id = ?",
                (now, document["knowledge_base_id"]),
            )
        return document

    def ensure_session(self, session_id: str | None, question: str) -> str:
        if session_id:
            with self.connect() as conn:
                exists = conn.execute("SELECT 1 FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
            if exists:
                return session_id
        new_id = str(uuid.uuid4())
        title = question.strip().replace("\n", " ")[:40] or "新会话"
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO chat_sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (new_id, title, now, now),
            )
        return new_id

    def add_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages(id, session_id, role, content, sources_json, trace_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    session_id,
                    role,
                    content,
                    json.dumps(sources or [], ensure_ascii=False),
                    json.dumps(trace or [], ensure_ascii=False),
                    now,
                ),
            )
            conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, session_id))

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete_session(self, session_id: str) -> bool:
        """Delete exactly one chat session and its messages atomically.

        ``chat_messages.session_id`` uses ``ON DELETE CASCADE``.  Returning the
        affected-row result from the same DELETE avoids a read/delete race: if
        two requests target the same session, only the winner reports success.
        """
        with self.connect() as conn:
            deleted = conn.execute(
                "DELETE FROM chat_sessions WHERE id = ?", (session_id,)
            )
        return deleted.rowcount == 1

    def get_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC, rowid ASC",
                (session_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["sources"] = json.loads(item.pop("sources_json"))
            item["trace"] = json.loads(item.pop("trace_json"))
            result.append(item)
        return result
