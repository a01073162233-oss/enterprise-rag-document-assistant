from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.services.documents import TextChunk


class VectorStoreError(RuntimeError):
    pass


def collection_name(knowledge_base_id: str) -> str:
    return f"kb_{knowledge_base_id.replace('-', '')}"


class QdrantVectorStore:
    def __init__(self, settings: Settings, *, location: str | None = None) -> None:
        self.settings = settings
        if location is not None:
            self.client = QdrantClient(location=location)
            self.mode = "memory" if location == ":memory:" else "local"
        elif settings.qdrant_mode == "remote":
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=30,
            )
            self.mode = "remote"
        else:
            Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.qdrant_path))
            self.mode = "local"

    def ensure_collection(self, knowledge_base_id: str, dimension: int) -> None:
        name = collection_name(knowledge_base_id)
        if self.client.collection_exists(name):
            info = self.client.get_collection(name)
            vectors = info.config.params.vectors
            configured_size = getattr(vectors, "size", None)
            if configured_size is not None and configured_size != dimension:
                raise VectorStoreError(
                    f"知识库向量维度为 {configured_size}，当前 Embedding 维度为 {dimension}，请重建索引"
                )
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )

    def upsert(self, knowledge_base_id: str, chunks: list[TextChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("分块数量和向量数量不一致")
        if not chunks:
            return
        dimension = len(vectors[0])
        if dimension <= 0 or any(len(vector) != dimension for vector in vectors):
            raise VectorStoreError("向量维度必须为正数且在同一批次内保持一致")
        self.ensure_collection(knowledge_base_id, dimension)
        points = [
            models.PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "knowledge_base_id": chunk.knowledge_base_id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "page": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "content_hash": chunk.content_hash,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        # Smaller batches keep local-mode memory bounded for large PDFs.
        for start in range(0, len(points), 128):
            self.client.upsert(
                collection_name=collection_name(knowledge_base_id),
                points=points[start : start + 128],
                wait=True,
            )

    def dense_search(
        self,
        knowledge_base_id: str,
        query_vector: list[float],
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        name = collection_name(knowledge_base_id)
        if not self.client.collection_exists(name) or document_ids == []:
            return []
        query_filter = self._document_filter(document_ids)
        response = self.client.query_points(
            collection_name=name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {"id": str(point.id), "score": float(point.score), "payload": point.payload or {}}
            for point in response.points
        ]

    def scroll_payloads(
        self,
        knowledge_base_id: str,
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        name = collection_name(knowledge_base_id)
        if not self.client.collection_exists(name) or document_ids == []:
            return []
        scroll_filter = self._document_filter(document_ids)
        records: list[dict[str, Any]] = []
        offset: models.PointId | None = None
        while len(records) < limit:
            batch_size = min(256, limit - len(records))
            batch, next_offset = self.client.scroll(
                collection_name=name,
                scroll_filter=scroll_filter,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(
                {"id": str(record.id), "payload": record.payload or {}} for record in batch
            )
            if next_offset is None or not batch:
                break
            offset = next_offset
        return records

    @staticmethod
    def _document_filter(document_ids: list[str] | None) -> models.Filter | None:
        if document_ids is None:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(dict.fromkeys(document_ids))),
                )
            ]
        )

    def delete_document(self, knowledge_base_id: str, document_id: str) -> None:
        name = collection_name(knowledge_base_id)
        if not self.client.collection_exists(name):
            return
        self.client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        name = collection_name(knowledge_base_id)
        if self.client.collection_exists(name):
            self.client.delete_collection(name)

    def count(self, knowledge_base_id: str) -> int:
        name = collection_name(knowledge_base_id)
        if not self.client.collection_exists(name):
            return 0
        result = self.client.count(collection_name=name, exact=True)
        return int(result.count)

    def health(self) -> dict[str, Any]:
        collections = self.client.get_collections().collections
        return {"mode": self.mode, "collections": len(collections)}

    def close(self) -> None:
        self.client.close()
