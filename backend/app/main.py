from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import secrets
import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import PROJECT_ROOT, settings
from app.db import Registry
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Document,
    HealthResponse,
    KnowledgeBase,
    KnowledgeBaseCreate,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SessionSummary,
)
from app.services.agent import AgentOrchestrator
from app.services.embeddings import provider_identity
from app.services.ingestion import IngestionService
from app.services.llm import LanguageModel
from app.services.retrieval import RetrievalService
from app.services.vector_store import QdrantVectorStore


logger = logging.getLogger("enterprise-rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class RequestBodyTooLarge(Exception):
    pass


class UploadSizeLimitMiddleware:
    """Reject oversized multipart bodies before Starlette spools them to disk."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = str(scope.get("path", ""))
        is_upload = scope.get("type") == "http" and scope.get("method") == "POST" and (
            path == f"{settings.api_prefix}/documents/upload"
            or (
                path.startswith(f"{settings.api_prefix}/knowledge-bases/")
                and path.endswith("/documents")
            )
        )
        if not is_upload:
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    response = JSONResponse(
                        status_code=413,
                        content={"detail": "上传请求总量超过服务端限制"},
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass
        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        await self.app(scope, limited_receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.app_env == "production" and not settings.api_key:
        raise RuntimeError("生产环境必须设置 RAG_API_KEY")
    if settings.app_env == "production" and settings.qdrant_mode != "remote":
        raise RuntimeError("生产环境必须使用 QDRANT_MODE=remote")
    settings.ensure_directories()
    registry = Registry(settings.metadata_db_path)
    registry.initialize()
    vector_store = QdrantVectorStore(settings)
    try:
        # Embedded Qdrant enforces single-process ownership of its directory,
        # making abandoned-row recovery safe.  Remote mode may have other live
        # API workers, so it must not reinterpret their in-flight rows.
        interrupted = (
            registry.recover_interrupted_documents()
            if vector_store.mode == "local"
            else []
        )
        for document in interrupted:
            try:
                vector_store.delete_document(
                    str(document["knowledge_base_id"]), str(document["id"])
                )
            except Exception as exc:
                # Retrieval independently filters out non-ready rows, so a remote
                # cleanup outage cannot expose partial vectors to users.
                logger.warning(
                    "Could not clean vectors for interrupted document %s: %s",
                    document["id"],
                    type(exc).__name__,
                )
        if interrupted:
            logger.warning(
                "Recovered %d interrupted document ingestion(s)", len(interrupted)
            )
        retrieval = RetrievalService(settings, vector_store, registry)
        app.state.registry = registry
        app.state.vector_store = vector_store
        app.state.ingestion = IngestionService(settings, registry, vector_store)
        app.state.agent = AgentOrchestrator(
            settings, registry, retrieval, LanguageModel(settings)
        )
        app.state.retrieval = retrieval
        yield
    finally:
        # An exception raised while the server is shutting down is injected at
        # the ``yield`` point.  Keep the embedded Qdrant lock from leaking in
        # that case so the next local start can open the same data directory.
        vector_store.close()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="PDF ingestion, Qdrant hybrid retrieval, grounded generation and bounded agent workflows.",
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    UploadSizeLimitMiddleware,
    max_bytes=(settings.max_total_upload_mb * 1024 * 1024)
    + (settings.max_upload_files * 64 * 1024)
    + (1024 * 1024),
)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    public_paths = {f"{settings.api_prefix}/health", f"{settings.api_prefix}/settings/public"}
    if settings.api_key and request.url.path.startswith(settings.api_prefix):
        if request.url.path not in public_paths:
            authorization = request.headers.get("authorization", "")
            bearer = authorization[7:] if authorization.lower().startswith("bearer ") else ""
            presented = request.headers.get("x-api-key", "") or bearer
            if not secrets.compare_digest(presented, settings.api_key):
                return JSONResponse(status_code=401, content={"detail": "缺少或无效的 API Key"})
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": "请求参数不合法", "errors": exc.errors()}),
    )


@app.exception_handler(RequestBodyTooLarge)
async def request_too_large_handler(
    _request: Request, _exc: RequestBodyTooLarge
) -> JSONResponse:
    return JSONResponse(
        status_code=413, content={"detail": "上传请求总量超过服务端限制"}
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled request error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "服务内部错误，请稍后重试"})


def registry(request: Request) -> Registry:
    return request.app.state.registry


def vector_store(request: Request) -> QdrantVectorStore:
    return request.app.state.vector_store


def _safe_unlink(stored_path: str) -> None:
    """Delete only files contained by the configured upload directory."""
    candidate = Path(stored_path).resolve()
    upload_root = settings.upload_dir.resolve()
    if upload_root in candidate.parents:
        try:
            candidate.unlink(missing_ok=True)
        except OSError as exc:
            # Metadata/vector deletion has already committed at most call sites;
            # keep that state coherent and leave a clear signal for disk cleanup.
            logger.warning(
                "Could not remove uploaded file %s: %s",
                candidate.name,
                type(exc).__name__,
            )


@app.get(f"{settings.api_prefix}/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    try:
        vector_details = await asyncio.to_thread(vector_store(request).health)
        vector_status = "connected"
    except Exception as exc:
        vector_details = {"error": type(exc).__name__}
        vector_status = "unavailable"
    provider, model, _dimension = provider_identity(settings)
    return HealthResponse(
        status="ok" if vector_status == "connected" else "degraded",
        vector_database=vector_status,
        vector_database_mode=settings.qdrant_mode,
        llm=(
            settings.llm_model
            if settings.llm_enabled
            else (
                settings.llm_fallback_model
                if settings.llm_fallback_configured
                else "extractive-fallback"
            )
        ),
        embedding=f"{provider}:{model}",
        details=vector_details,
    )


@app.get(f"{settings.api_prefix}/settings/public")
async def public_settings() -> dict[str, Any]:
    provider, model, dimension = provider_identity(settings)
    return {
        "app_name": settings.app_name,
        "llm_configured": settings.llm_enabled or settings.llm_fallback_configured,
        "llm_model": (
            settings.llm_model
            if settings.llm_enabled
            else (
                settings.llm_fallback_model
                if settings.llm_fallback_configured
                else "extractive-fallback"
            )
        ),
        "llm_fallback_configured": settings.llm_fallback_configured,
        "llm_fallback_model": (
            settings.llm_fallback_model
            if settings.llm_fallback_configured
            else None
        ),
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "qdrant_mode": settings.qdrant_mode,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "max_upload_mb": settings.max_upload_mb,
        "max_upload_files": settings.max_upload_files,
        "authentication_required": bool(settings.api_key),
    }


@app.get(f"{settings.api_prefix}/knowledge-bases", response_model=list[KnowledgeBase])
async def list_knowledge_bases(request: Request) -> list[dict[str, Any]]:
    return await asyncio.to_thread(registry(request).list_knowledge_bases)


@app.post(
    f"{settings.api_prefix}/knowledge-bases",
    response_model=KnowledgeBase,
    status_code=status.HTTP_201_CREATED,
)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate, request: Request
) -> dict[str, Any]:
    provider, model, dimension = provider_identity(settings)
    item = await asyncio.to_thread(
        registry(request).create_knowledge_base,
        name=payload.name,
        description=payload.description,
        embedding_provider=provider,
        embedding_model=model,
        embedding_dimension=dimension,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    try:
        await asyncio.to_thread(vector_store(request).ensure_collection, item["id"], dimension)
    except Exception:
        await asyncio.to_thread(registry(request).delete_knowledge_base, item["id"])
        raise HTTPException(status_code=503, detail="无法创建向量集合，请检查 Qdrant 配置")
    return item


@app.delete(f"{settings.api_prefix}/knowledge-bases/{{knowledge_base_id}}", status_code=204)
async def delete_knowledge_base(knowledge_base_id: str, request: Request) -> None:
    item = await asyncio.to_thread(
        registry(request).get_knowledge_base, knowledge_base_id
    )
    if not item:
        raise HTTPException(status_code=404, detail="知识库不存在")
    await asyncio.to_thread(vector_store(request).delete_knowledge_base, knowledge_base_id)
    stored_paths = await asyncio.to_thread(
        registry(request).delete_knowledge_base, knowledge_base_id
    )
    for stored_path in stored_paths:
        _safe_unlink(stored_path)


@app.get(f"{settings.api_prefix}/documents", response_model=list[Document])
async def list_documents(
    request: Request,
    knowledge_base_id: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    if knowledge_base_id and not await asyncio.to_thread(
        registry(request).get_knowledge_base, knowledge_base_id
    ):
        raise HTTPException(status_code=404, detail="知识库不存在")
    return await asyncio.to_thread(registry(request).list_documents, knowledge_base_id)


async def _save_and_ingest_files(
    *,
    request: Request,
    knowledge_base_id: str,
    files: list[UploadFile],
) -> list[dict[str, Any]]:
    kb = await asyncio.to_thread(
        registry(request).get_knowledge_base, knowledge_base_id
    )
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if not files:
        raise HTTPException(status_code=400, detail="请选择至少一个文件")
    if len(files) > settings.max_upload_files:
        raise HTTPException(
            status_code=413,
            detail=f"单次最多上传 {settings.max_upload_files} 个文件",
        )

    allowed = {".pdf", ".txt", ".md"}
    safe_content_types = {
        ".pdf": "application/pdf",
        ".txt": "text/plain; charset=utf-8",
        ".md": "text/plain; charset=utf-8",
    }
    results: list[dict[str, Any]] = []
    max_bytes = settings.max_upload_mb * 1024 * 1024
    max_total_bytes = settings.max_total_upload_mb * 1024 * 1024
    declared_total = sum(int(upload.size or 0) for upload in files)
    if declared_total > max_total_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"单次上传总量超过 {settings.max_total_upload_mb} MB 限制",
        )
    target_dir = settings.upload_dir / knowledge_base_id
    target_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    for upload in files:
        safe_name = re.sub(r"[\x00-\x1f\x7f]", "_", Path(upload.filename or "document").name).strip()
        if not safe_name:
            safe_name = "document"
        suffix = Path(safe_name).suffix.lower()
        if suffix not in allowed:
            raise HTTPException(
                status_code=415, detail=f"{safe_name}：仅支持 PDF、TXT 和 Markdown"
            )
        stored_path = target_dir / f"{uuid.uuid4().hex}{suffix}"
        digest_builder = hashlib.sha256()
        byte_size = 0
        signature = b""
        try:
            with stored_path.open("wb") as output:
                while chunk := await upload.read(1024 * 1024):
                    if not signature:
                        signature = chunk[:8]
                    byte_size += len(chunk)
                    total_bytes += len(chunk)
                    if byte_size > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{safe_name}：超过 {settings.max_upload_mb} MB 上传限制",
                        )
                    if total_bytes > max_total_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"单次上传总量超过 {settings.max_total_upload_mb} MB 限制",
                        )
                    digest_builder.update(chunk)
                    await asyncio.to_thread(output.write, chunk)
        except BaseException:
            _safe_unlink(str(stored_path))
            raise
        if not byte_size:
            _safe_unlink(str(stored_path))
            raise HTTPException(status_code=400, detail=f"{safe_name}：文件为空")
        if suffix == ".pdf" and not signature.startswith(b"%PDF"):
            _safe_unlink(str(stored_path))
            raise HTTPException(status_code=415, detail=f"{safe_name}：不是有效的 PDF 文件")
        digest = digest_builder.hexdigest()
        document: dict[str, Any] | None = None
        try:
            existing = await asyncio.to_thread(
                registry(request).find_document_by_hash, knowledge_base_id, digest
            )
        except BaseException:
            _safe_unlink(str(stored_path))
            raise
        if existing:
            if existing["status"] != "failed":
                _safe_unlink(str(stored_path))
                results.append(existing)
                continue

            # Atomically claim the failed row before deleting partial vectors.
            # This prevents a losing concurrent retry from erasing batches that
            # the winning request has already started to upsert.
            retry_task = asyncio.create_task(
                asyncio.to_thread(
                    registry(request).retry_failed_document,
                    str(existing["id"]),
                    filename=safe_name,
                    stored_path=str(stored_path),
                    content_type=safe_content_types[suffix],
                    byte_size=byte_size,
                    sha256=digest,
                )
            )
            try:
                retried = await asyncio.shield(retry_task)
            except asyncio.CancelledError:
                try:
                    retried = await retry_task
                except BaseException:
                    _safe_unlink(str(stored_path))
                else:
                    if retried:
                        cancelled_document, previous_path = retried
                        await asyncio.shield(
                            asyncio.to_thread(
                                registry(request).update_document,
                                str(cancelled_document["id"]),
                                status="failed",
                                error="文档处理被中断，请重新上传以重试",
                            )
                        )
                        _safe_unlink(previous_path)
                    else:
                        _safe_unlink(str(stored_path))
                raise
            except BaseException:
                _safe_unlink(str(stored_path))
                raise
            if retried:
                document, previous_path = retried
                try:
                    await asyncio.to_thread(
                        vector_store(request).delete_document,
                        knowledge_base_id,
                        str(document["id"]),
                    )
                except asyncio.CancelledError:
                    await asyncio.shield(
                        asyncio.to_thread(
                            registry(request).update_document,
                            str(document["id"]),
                            status="failed",
                            error="文档处理被中断，请重新上传以重试",
                        )
                    )
                    _safe_unlink(previous_path)
                    raise
                except Exception as exc:
                    failed = await asyncio.to_thread(
                        registry(request).update_document,
                        str(document["id"]),
                        status="failed",
                        error=f"清理旧索引失败：{type(exc).__name__}",
                    )
                    _safe_unlink(previous_path)
                    if failed is None:
                        raise
                    results.append(failed)
                    continue
                else:
                    _safe_unlink(previous_path)
            else:
                try:
                    duplicate = await asyncio.to_thread(
                        registry(request).find_document_by_hash,
                        knowledge_base_id,
                        digest,
                    )
                except BaseException:
                    _safe_unlink(str(stored_path))
                    raise
                if duplicate:
                    _safe_unlink(str(stored_path))
                    results.append(duplicate)
                    continue
        if document is None:
            create_task = asyncio.create_task(
                asyncio.to_thread(
                    registry(request).create_document,
                    kb_id=knowledge_base_id,
                    filename=safe_name,
                    stored_path=str(stored_path),
                    content_type=safe_content_types[suffix],
                    byte_size=byte_size,
                    sha256=digest,
                )
            )
            try:
                document = await asyncio.shield(create_task)
            except asyncio.CancelledError:
                try:
                    cancelled_document = await create_task
                except BaseException:
                    _safe_unlink(str(stored_path))
                else:
                    await asyncio.shield(
                        asyncio.to_thread(
                            registry(request).update_document,
                            str(cancelled_document["id"]),
                            status="failed",
                            error="文档处理被中断，请重新上传以重试",
                        )
                    )
                raise
            except sqlite3.IntegrityError:
                _safe_unlink(str(stored_path))
                duplicate = await asyncio.to_thread(
                    registry(request).find_document_by_hash, knowledge_base_id, digest
                )
                if duplicate:
                    results.append(duplicate)
                    continue
                raise
            except BaseException:
                _safe_unlink(str(stored_path))
                raise
        processed = await request.app.state.ingestion.ingest(document, kb)
        results.append(processed)
    return results


@app.post(
    f"{settings.api_prefix}/documents/upload",
    response_model=list[Document],
    status_code=status.HTTP_201_CREATED,
)
async def upload_documents(
    request: Request,
    knowledge_base_id: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File(description="PDF、TXT 或 Markdown 文件")],
) -> list[dict[str, Any]]:
    return await _save_and_ingest_files(
        request=request, knowledge_base_id=knowledge_base_id, files=files
    )


@app.post(
    f"{settings.api_prefix}/knowledge-bases/{{knowledge_base_id}}/documents",
    response_model=list[Document],
    status_code=status.HTTP_201_CREATED,
)
async def upload_documents_nested(
    knowledge_base_id: str,
    request: Request,
    files: Annotated[list[UploadFile], File(description="PDF、TXT 或 Markdown 文件")],
) -> list[dict[str, Any]]:
    return await _save_and_ingest_files(
        request=request, knowledge_base_id=knowledge_base_id, files=files
    )


@app.get(f"{settings.api_prefix}/documents/{{document_id}}/file")
async def get_document_file(document_id: str, request: Request) -> FileResponse:
    item = await asyncio.to_thread(registry(request).get_document, document_id)
    if not item:
        raise HTTPException(status_code=404, detail="文档不存在")
    resolved = Path(str(item["stored_path"])).resolve()
    upload_root = settings.upload_dir.resolve()
    if upload_root not in resolved.parents:
        raise HTTPException(status_code=403, detail="文件路径不合法")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="原文件已丢失")
    return FileResponse(
        resolved,
        media_type="application/pdf" if resolved.suffix.lower() == ".pdf" else "text/plain; charset=utf-8",
        filename=str(item["filename"]),
        content_disposition_type="inline" if resolved.suffix.lower() == ".pdf" else "attachment",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox; default-src 'none'",
        },
    )


@app.delete(f"{settings.api_prefix}/documents/{{document_id}}", status_code=204)
async def delete_document(document_id: str, request: Request) -> None:
    item = await asyncio.to_thread(registry(request).get_document, document_id)
    if not item:
        raise HTTPException(status_code=404, detail="文档不存在")
    await asyncio.to_thread(
        vector_store(request).delete_document,
        str(item["knowledge_base_id"]),
        document_id,
    )
    deleted = await asyncio.to_thread(registry(request).delete_document, document_id)
    if deleted:
        _safe_unlink(str(deleted["stored_path"]))


@app.post(f"{settings.api_prefix}/search", response_model=SearchResponse)
async def search(payload: SearchRequest, request: Request) -> SearchResponse:
    if payload.knowledge_base_ids:
        knowledge_bases = await asyncio.gather(
            *[
                asyncio.to_thread(registry(request).get_knowledge_base, kb_id)
                for kb_id in payload.knowledge_base_ids
            ]
        )
        knowledge_bases = [item for item in knowledge_bases if item]
    else:
        knowledge_bases = await asyncio.to_thread(registry(request).list_knowledge_bases)
    hits = await request.app.state.retrieval.search(payload.query, knowledge_bases, payload.top_k)
    return SearchResponse(
        query=payload.query,
        hits=[SearchHit(**hit.as_dict()) for hit in hits],
    )


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    return await request.app.state.agent.run(payload)


@app.get(f"{settings.api_prefix}/chat/sessions", response_model=list[SessionSummary])
async def list_chat_sessions(request: Request) -> list[dict[str, Any]]:
    return await asyncio.to_thread(registry(request).list_sessions)


@app.get(f"{settings.api_prefix}/chat/sessions/{{session_id}}")
async def get_chat_session(session_id: str, request: Request) -> dict[str, Any]:
    session = await asyncio.to_thread(registry(request).get_session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {
        **session,
        "messages": await asyncio.to_thread(
            registry(request).get_session_messages, session_id
        ),
    }


@app.delete(
    f"{settings.api_prefix}/chat/sessions/{{session_id}}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_chat_session(session_id: str, request: Request) -> None:
    deleted = await asyncio.to_thread(registry(request).delete_session, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")


# In production the frontend is normally served by a reverse proxy. Mounting a
# completed local build makes `npm run build` + Uvicorn a convenient single URL.
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
