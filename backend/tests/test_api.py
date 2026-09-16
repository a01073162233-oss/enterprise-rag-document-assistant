import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import main as main_module
from app.config import Settings


def test_public_settings_never_expose_primary_or_fallback_secrets(monkeypatch) -> None:
    primary_secret = "public-primary-secret"
    fallback_secret = "public-fallback-secret"
    monkeypatch.setattr(
        main_module,
        "settings",
        Settings(
            llm_provider="openai-compatible",
            llm_api_key=primary_secret,
            llm_base_url="https://deepseek.invalid/v1",
            llm_model="deepseek-flash",
            llm_fallback_enabled=True,
            llm_fallback_api_key=fallback_secret,
            llm_fallback_base_url="http://127.0.0.1:11434/v1",
            llm_fallback_model="qwen3:4b-instruct",
        ),
    )

    payload = asyncio.run(main_module.public_settings())
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["llm_configured"] is True
    assert payload["llm_fallback_configured"] is True
    assert payload["llm_fallback_model"] == "qwen3:4b-instruct"
    assert primary_secret not in serialized
    assert fallback_secret not in serialized


def test_api_full_local_flow_and_failed_upload_retry(
    tmp_path: Path, monkeypatch
) -> None:
    configured = Settings(
        app_env="test",
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        metadata_db_path=tmp_path / "metadata.db",
        qdrant_mode="local",
        qdrant_path=tmp_path / "qdrant",
        embedding_provider="local",
        embedding_dimension=64,
        llm_provider="disabled",
        llm_fallback_enabled=False,
    )
    monkeypatch.setattr(main_module, "settings", configured)

    with TestClient(main_module.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        created = client.post(
            "/api/knowledge-bases",
            json={"name": "HR", "description": "Policies"},
        )
        assert created.status_code == 201
        knowledge_base_id = created.json()["id"]

        uploaded = client.post(
            "/api/documents/upload",
            data={"knowledge_base_id": knowledge_base_id},
            files={
                "files": (
                    "policy.txt",
                    "Annual leave is 10 days. 员工年假为十天。".encode(),
                    "text/plain",
                )
            },
        )
        assert uploaded.status_code == 201
        ready_document = uploaded.json()[0]
        assert ready_document["status"] == "ready"

        search = client.post(
            "/api/search",
            json={
                "query": "年假多少天",
                "knowledge_base_ids": [knowledge_base_id],
                "top_k": 5,
            },
        )
        assert search.status_code == 200
        assert search.json()["hits"][0]["document_id"] == ready_document["id"]

        chat = client.post(
            "/api/chat",
            json={
                "question": "员工年假是多少天？",
                "knowledge_base_ids": [knowledge_base_id],
            },
        )
        assert chat.status_code == 200
        assert chat.json()["sources"][0]["document_id"] == ready_document["id"]
        assert chat.json()["generation_source"] == "extractive"
        session_id = chat.json()["session_id"]

        session = client.get(f"/api/chat/sessions/{session_id}")
        assert session.status_code == 200
        assert len(session.json()["messages"]) == 2

        deleted_session = client.delete(f"/api/chat/sessions/{session_id}")
        assert deleted_session.status_code == 204
        assert deleted_session.content == b""
        assert client.get(f"/api/chat/sessions/{session_id}").status_code == 404
        duplicate_delete = client.delete(f"/api/chat/sessions/{session_id}")
        assert duplicate_delete.status_code == 404
        assert duplicate_delete.json() == {"detail": "会话不存在"}
        listed_sessions = client.get("/api/chat/sessions")
        assert listed_sessions.status_code == 200
        assert all(item["id"] != session_id for item in listed_sessions.json())

        broken_pdf = b"%PDF-1.7\nnot actually a pdf"
        first_failure = client.post(
            "/api/documents/upload",
            data={"knowledge_base_id": knowledge_base_id},
            files={"files": ("broken.pdf", broken_pdf, "application/pdf")},
        )
        assert first_failure.status_code == 201
        failed_document = first_failure.json()[0]
        assert failed_document["status"] == "failed"

        retried = client.post(
            "/api/documents/upload",
            data={"knowledge_base_id": knowledge_base_id},
            files={"files": ("broken.pdf", broken_pdf, "application/pdf")},
        )
        assert retried.status_code == 201
        assert retried.json()[0]["id"] == failed_document["id"]
        assert retried.json()[0]["status"] == "failed"

        documents = client.get(
            "/api/documents", params={"knowledge_base_id": knowledge_base_id}
        )
        assert documents.status_code == 200
        assert len(documents.json()) == 2
