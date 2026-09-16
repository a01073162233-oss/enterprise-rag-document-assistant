import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.db import Registry


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    value = Registry(tmp_path / "metadata.db")
    value.initialize()
    return value


def create_kb(registry: Registry, name: str = "HR") -> dict[str, object]:
    return registry.create_knowledge_base(
        name=name,
        description="Enterprise policies",
        embedding_provider="local",
        embedding_model="hashing-multilingual-v1",
        embedding_dimension=64,
        chunk_size=300,
        chunk_overlap=50,
    )


def create_document(
    registry: Registry, kb_id: str, sha256: str = "a" * 64
) -> dict[str, object]:
    return registry.create_document(
        kb_id=kb_id,
        filename="handbook.pdf",
        stored_path="/objects/handbook.pdf",
        content_type="application/pdf",
        byte_size=1234,
        sha256=sha256,
    )


def test_registry_knowledge_base_and_document_crud_updates_aggregates(
    registry: Registry,
) -> None:
    kb = create_kb(registry)
    assert registry.get_knowledge_base(str(kb["id"]))["document_count"] == 0

    document = create_document(registry, str(kb["id"]))
    assert document["status"] == "processing"
    updated = registry.update_document(
        str(document["id"]), status="ready", page_count=7, chunk_count=12
    )

    assert updated is not None
    assert updated["page_count"] == 7
    assert updated["chunk_count"] == 12
    aggregate = registry.get_knowledge_base(str(kb["id"]))
    assert aggregate is not None
    assert aggregate["document_count"] == 1
    assert aggregate["chunk_count"] == 12
    assert registry.list_documents(str(kb["id"]))[0]["id"] == document["id"]

    deleted = registry.delete_document(str(document["id"]))
    assert deleted is not None
    assert registry.get_document(str(document["id"])) is None
    assert registry.delete_document(str(document["id"])) is None

    paths = registry.delete_knowledge_base(str(kb["id"]))
    assert paths == []
    assert registry.get_knowledge_base(str(kb["id"])) is None


def test_registry_finds_hash_and_database_constraint_prevents_duplicates(
    registry: Registry,
) -> None:
    kb = create_kb(registry)
    document = create_document(registry, str(kb["id"]), sha256="b" * 64)

    found = registry.find_document_by_hash(str(kb["id"]), "b" * 64)

    assert found is not None
    assert found["id"] == document["id"]
    with pytest.raises(sqlite3.IntegrityError):
        create_document(registry, str(kb["id"]), sha256="b" * 64)


def test_same_hash_is_allowed_in_different_knowledge_bases(registry: Registry) -> None:
    first = create_kb(registry, "HR")
    second = create_kb(registry, "Legal")

    first_document = create_document(registry, str(first["id"]), sha256="c" * 64)
    second_document = create_document(registry, str(second["id"]), sha256="c" * 64)

    assert first_document["id"] != second_document["id"]


def test_registry_session_crud_round_trips_sources_and_trace(registry: Registry) -> None:
    session_id = registry.ensure_session(None, "  年假怎么算？\n请详细说明  ")
    assert registry.ensure_session(session_id, "ignored") == session_id

    registry.add_message(session_id=session_id, role="user", content="年假怎么算？")
    registry.add_message(
        session_id=session_id,
        role="assistant",
        content="依据手册 [S1]",
        sources=[{"id": "S1", "page": 2}],
        trace=[{"id": "retrieve", "status": "completed"}],
    )

    sessions = registry.list_sessions()
    messages = registry.get_session_messages(session_id)

    assert sessions[0]["id"] == session_id
    assert sessions[0]["title"] == "年假怎么算？ 请详细说明"
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["sources"] == [{"id": "S1", "page": 2}]
    assert messages[1]["trace"] == [{"id": "retrieve", "status": "completed"}]


def test_registry_delete_session_cascades_messages_and_is_concurrency_safe(
    registry: Registry,
) -> None:
    session_id = registry.ensure_session(None, "待删除会话")
    registry.add_message(session_id=session_id, role="user", content="问题")
    registry.add_message(session_id=session_id, role="assistant", content="回答")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(registry.delete_session, [session_id, session_id]))

    assert sorted(results) == [False, True]
    assert registry.get_session(session_id) is None
    assert registry.get_session_messages(session_id) == []
    with registry.connect() as conn:
        orphan_count = conn.execute(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    assert orphan_count == 0


def test_registry_recovers_documents_interrupted_during_ingestion(
    registry: Registry,
) -> None:
    kb = create_kb(registry)
    document = create_document(registry, str(kb["id"]))

    recovered = registry.recover_interrupted_documents()

    assert [item["id"] for item in recovered] == [document["id"]]
    stored = registry.get_document(str(document["id"]))
    assert stored is not None
    assert stored["status"] == "failed"
    assert stored["page_count"] == 0
    assert stored["chunk_count"] == 0
    assert "重新上传" in stored["error"]
    assert registry.recover_interrupted_documents() == []


def test_registry_atomically_reuses_only_a_failed_document_for_retry(
    registry: Registry,
) -> None:
    kb = create_kb(registry)
    document = create_document(registry, str(kb["id"]), sha256="d" * 64)
    registry.update_document(
        str(document["id"]),
        status="failed",
        page_count=2,
        chunk_count=3,
        error="temporary outage",
    )

    retried = registry.retry_failed_document(
        str(document["id"]),
        filename="retry.pdf",
        stored_path="/objects/retry.pdf",
        content_type="application/pdf",
        byte_size=4321,
        sha256="d" * 64,
    )

    assert retried is not None
    item, previous_path = retried
    assert item["id"] == document["id"]
    assert item["status"] == "processing"
    assert item["stored_path"] == "/objects/retry.pdf"
    assert item["page_count"] == 0
    assert item["chunk_count"] == 0
    assert item["error"] is None
    assert item["processed_at"] is None
    assert previous_path == "/objects/handbook.pdf"
    assert (
        registry.retry_failed_document(
            str(document["id"]),
            filename="loser.pdf",
            stored_path="/objects/loser.pdf",
            content_type="application/pdf",
            byte_size=1,
            sha256="d" * 64,
        )
        is None
    )
