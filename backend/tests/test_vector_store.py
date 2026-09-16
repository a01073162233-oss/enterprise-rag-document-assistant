from uuid import UUID

import pytest

from app.config import Settings
from app.services.documents import TextChunk
from app.services.vector_store import QdrantVectorStore, VectorStoreError


def chunk(point_id: str, document_id: str, index: int, text: str) -> TextChunk:
    return TextChunk(
        id=str(UUID(point_id)),
        document_id=document_id,
        knowledge_base_id="kb-memory",
        document_name=f"{document_id}.txt",
        page_number=index + 1,
        chunk_index=index,
        text=text,
        content_hash=f"hash-{index}",
    )


@pytest.fixture
def store() -> QdrantVectorStore:
    value = QdrantVectorStore(Settings(), location=":memory:")
    yield value
    value.close()


def test_qdrant_memory_upsert_search_and_delete_document(
    store: QdrantVectorStore,
) -> None:
    chunks = [
        chunk("00000000-0000-0000-0000-000000000001", "leave", 0, "annual leave"),
        chunk("00000000-0000-0000-0000-000000000002", "security", 1, "security"),
        chunk("00000000-0000-0000-0000-000000000003", "leave", 2, "leave days"),
    ]
    store.upsert("kb-memory", chunks, [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])

    assert store.count("kb-memory") == 3
    results = store.dense_search("kb-memory", [1.0, 0.0], limit=3)
    assert results[0]["id"] == chunks[0].id
    assert results[0]["payload"]["document_id"] == "leave"
    assert results[0]["payload"]["page"] == 1
    assert {item["id"] for item in store.scroll_payloads("kb-memory", 10)} == {
        item.id for item in chunks
    }

    store.delete_document("kb-memory", "leave")

    assert store.count("kb-memory") == 1
    remaining = store.scroll_payloads("kb-memory", 10)
    assert [item["payload"]["document_id"] for item in remaining] == ["security"]


def test_qdrant_memory_delete_collection_and_missing_collection_are_idempotent(
    store: QdrantVectorStore,
) -> None:
    assert store.count("missing") == 0
    assert store.dense_search("missing", [1.0, 0.0], 5) == []
    assert store.scroll_payloads("missing", 5) == []
    store.delete_document("missing", "document")

    item = chunk("00000000-0000-0000-0000-000000000004", "one", 0, "content")
    store.upsert("kb-memory", [item], [[1.0, 0.0]])
    store.delete_knowledge_base("kb-memory")

    assert store.count("kb-memory") == 0
    assert store.health()["collections"] == 0


def test_qdrant_rejects_mismatched_batch_and_existing_dimension(
    store: QdrantVectorStore,
) -> None:
    item = chunk("00000000-0000-0000-0000-000000000005", "one", 0, "content")
    with pytest.raises(VectorStoreError, match="数量"):
        store.upsert("kb-memory", [item], [])

    store.upsert("kb-memory", [item], [[1.0, 0.0]])
    with pytest.raises(VectorStoreError, match="维度"):
        store.ensure_collection("kb-memory", 3)


def test_qdrant_rejects_empty_or_inconsistent_vectors(
    store: QdrantVectorStore,
) -> None:
    first = chunk("00000000-0000-0000-0000-000000000006", "one", 0, "one")
    second = chunk("00000000-0000-0000-0000-000000000007", "two", 1, "two")

    with pytest.raises(VectorStoreError, match="维度"):
        store.upsert("kb-memory", [first], [[]])
    with pytest.raises(VectorStoreError, match="维度"):
        store.upsert("kb-memory", [first, second], [[1.0, 0.0], [1.0]])


def test_qdrant_search_can_filter_out_non_ready_document_ids(
    store: QdrantVectorStore,
) -> None:
    ready = chunk("00000000-0000-0000-0000-000000000008", "ready", 0, "ready")
    partial = chunk("00000000-0000-0000-0000-000000000009", "partial", 1, "partial")
    store.upsert("kb-memory", [ready, partial], [[0.7, 0.3], [1.0, 0.0]])

    dense = store.dense_search(
        "kb-memory", [1.0, 0.0], limit=5, document_ids=["ready"]
    )
    scrolled = store.scroll_payloads(
        "kb-memory", limit=5, document_ids=["ready"]
    )

    assert [item["payload"]["document_id"] for item in dense] == ["ready"]
    assert [item["payload"]["document_id"] for item in scrolled] == ["ready"]
    assert store.dense_search("kb-memory", [1.0, 0.0], 5, []) == []
