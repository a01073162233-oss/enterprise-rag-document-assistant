from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.config import Settings
from app.db import Registry
from app.services.documents import chunk_document, extract_document
from app.services.embeddings import create_embedding_provider
from app.services.vector_store import QdrantVectorStore


class IngestionService:
    def __init__(self, settings: Settings, registry: Registry, vector_store: QdrantVectorStore) -> None:
        self.settings = settings
        self.registry = registry
        self.vector_store = vector_store

    async def _record_failure(
        self,
        *,
        document_id: str,
        knowledge_base_id: str,
        error: BaseException,
    ) -> dict[str, Any]:
        try:
            await asyncio.to_thread(
                self.vector_store.delete_document,
                knowledge_base_id,
                document_id,
            )
        except Exception:
            pass
        message = (
            "文档处理被中断，请重新上传以重试"
            if isinstance(error, asyncio.CancelledError)
            else (str(error).strip() or type(error).__name__)
        )
        failed = await asyncio.to_thread(
            self.registry.update_document,
            document_id,
            status="failed",
            error=message[:500],
        )
        assert failed is not None
        return failed

    async def ingest(self, document: dict[str, Any], knowledge_base: dict[str, Any]) -> dict[str, Any]:
        document_id = str(document["id"])
        try:
            parsed = await asyncio.to_thread(
                extract_document,
                Path(str(document["stored_path"])),
                max_pages=self.settings.max_pdf_pages,
                max_characters=self.settings.max_document_chars,
            )
            chunks = await asyncio.to_thread(
                chunk_document,
                parsed,
                document_id=document_id,
                knowledge_base_id=str(knowledge_base["id"]),
                document_name=str(document["filename"]),
                chunk_size=int(knowledge_base["chunk_size"]),
                overlap=int(knowledge_base["chunk_overlap"]),
                max_chunks=self.settings.max_chunks_per_document,
            )
            provider = create_embedding_provider(
                self.settings,
                provider=str(knowledge_base["embedding_provider"]),
                model=str(knowledge_base["embedding_model"]),
                dimension=int(knowledge_base["embedding_dimension"]),
            )
            await asyncio.to_thread(
                self.vector_store.ensure_collection,
                str(knowledge_base["id"]),
                int(knowledge_base["embedding_dimension"]),
            )
            batch_size = max(1, self.settings.embedding_batch_size)
            for start in range(0, len(chunks), batch_size):
                chunk_batch = chunks[start : start + batch_size]
                vectors = await provider.embed([chunk.text for chunk in chunk_batch])
                await asyncio.to_thread(
                    self.vector_store.upsert,
                    str(knowledge_base["id"]),
                    chunk_batch,
                    vectors,
                )
            updated = await asyncio.to_thread(
                self.registry.update_document,
                document_id,
                status="ready",
                page_count=parsed.page_count,
                chunk_count=len(chunks),
            )
            assert updated is not None
            return updated
        except asyncio.CancelledError as exc:
            # Client disconnects and server shutdown can cancel an in-flight
            # request.  Persist a retryable terminal state before propagating
            # cancellation; hard process exits are recovered during startup.
            try:
                await asyncio.shield(
                    self._record_failure(
                        document_id=document_id,
                        knowledge_base_id=str(knowledge_base["id"]),
                        error=exc,
                    )
                )
            except Exception:
                pass
            raise
        except Exception as exc:
            # Persist a useful but bounded error; API responses never expose a traceback.
            return await self._record_failure(
                document_id=document_id,
                knowledge_base_id=str(knowledge_base["id"]),
                error=exc,
            )
