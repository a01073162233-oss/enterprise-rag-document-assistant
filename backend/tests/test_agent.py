import asyncio
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.db import Registry
from app.schemas import ChatRequest, FallbackReason, GenerationSource
from app.services.agent import AgentOrchestrator
from app.services.llm import LLMResult, LanguageModel
from app.services.retrieval import RetrievalHit


class EmptyRetrieval:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, Any]], int]] = []

    async def search(
        self, query: str, knowledge_bases: list[dict[str, Any]], top_k: int
    ) -> list[Any]:
        self.calls.append((query, knowledge_bases, top_k))
        return []


class ForbiddenLanguageModel:
    async def answer(self, **_: object) -> object:
        raise AssertionError("LLM must not be called without evidence")


class EvidenceRetrieval:
    def __init__(self, knowledge_base_id: str) -> None:
        self.queries: list[str] = []
        self.hit = RetrievalHit(
            id="chunk-1",
            document_id="document-1",
            document_name="员工手册.pdf",
            knowledge_base_id=knowledge_base_id,
            page=3,
            chunk_index=0,
            text="员工每年可享受十天年假。",
            score=0.9,
            dense_score=0.8,
            lexical_score=0.7,
        )

    async def search(
        self, query: str, knowledge_bases: list[dict[str, Any]], top_k: int
    ) -> list[RetrievalHit]:
        self.queries.append(query)
        return [self.hit]


class RecordingLanguageModel:
    def __init__(
        self,
        outputs: list[str],
        *,
        model: str = "recording-llm",
        used_fallback: bool = False,
        generation_source: GenerationSource = "primary",
        fallback_reason: FallbackReason | None = None,
    ) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []
        self.model = model
        self.used_fallback = used_fallback
        self.generation_source = generation_source
        self.fallback_reason = fallback_reason

    async def answer(self, **kwargs: Any) -> LLMResult:
        self.calls.append(kwargs)
        return LLMResult(
            text=self.outputs[len(self.calls) - 1],
            model=self.model,
            used_fallback=self.used_fallback,
            generation_source=self.generation_source,
            fallback_reason=self.fallback_reason,
        )

    @staticmethod
    def extractive_fallback(
        question: str, items: list[dict[str, object]]
    ) -> str:
        return LanguageModel.extractive_fallback(question, items)


def registry_with_kb(tmp_path: Path) -> tuple[Registry, dict[str, Any]]:
    registry = Registry(tmp_path / "metadata.db")
    registry.initialize()
    kb = registry.create_knowledge_base(
        name="HR",
        description="",
        embedding_provider="local",
        embedding_model="hashing-multilingual-v1",
        embedding_dimension=64,
        chunk_size=300,
        chunk_overlap=50,
    )
    return registry, kb


def test_agent_abstains_and_skips_llm_when_retrieval_has_no_evidence(
    tmp_path: Path,
) -> None:
    registry, kb = registry_with_kb(tmp_path)
    retrieval = EmptyRetrieval()
    agent = AgentOrchestrator(
        Settings(llm_provider="disabled"),
        registry,
        retrieval,  # type: ignore[arg-type]
        ForbiddenLanguageModel(),  # type: ignore[arg-type]
    )

    response = asyncio.run(
        agent.run(
            ChatRequest(
                question="公司是否提供家属医疗保险？",
                knowledge_base_ids=[str(kb["id"])],
                top_k=4,
            )
        )
    )

    assert "没有足够依据" in response.answer
    assert response.sources == []
    assert response.model == "evidence-gate"
    assert response.used_fallback is True
    assert [step.id for step in response.trace] == [
        "route",
        "plan",
        "retrieve",
        "evidence_gate",
    ]
    assert len(retrieval.calls) == 1
    assert retrieval.calls[0][2] == 4
    assert [item["role"] for item in registry.get_session_messages(response.session_id)] == [
        "user",
        "assistant",
    ]


def test_agent_catalog_route_lists_only_ready_documents_without_retrieval(
    tmp_path: Path,
) -> None:
    registry, kb = registry_with_kb(tmp_path)
    ready = registry.create_document(
        kb_id=str(kb["id"]),
        filename="员工手册.pdf",
        stored_path="/objects/ready.pdf",
        content_type="application/pdf",
        byte_size=100,
        sha256="1" * 64,
    )
    registry.update_document(str(ready["id"]), status="ready", page_count=8, chunk_count=19)
    registry.create_document(
        kb_id=str(kb["id"]),
        filename="尚未完成.pdf",
        stored_path="/objects/processing.pdf",
        content_type="application/pdf",
        byte_size=100,
        sha256="2" * 64,
    )
    retrieval = EmptyRetrieval()
    agent = AgentOrchestrator(
        Settings(llm_provider="disabled"),
        registry,
        retrieval,  # type: ignore[arg-type]
        ForbiddenLanguageModel(),  # type: ignore[arg-type]
    )

    response = asyncio.run(
        agent.run(
            ChatRequest(
                question="知识库里有什么文档？",
                knowledge_base_ids=[str(kb["id"])],
                agent_mode=True,
            )
        )
    )

    assert response.model == "agent-catalog"
    assert response.used_fallback is False
    assert "员工手册.pdf（8 页，19 个分块）" in response.answer
    assert "尚未完成.pdf" not in response.answer
    assert [step.id for step in response.trace] == ["route", "catalog"]
    assert response.trace[-1].tool == "list_documents"
    assert retrieval.calls == []


def test_catalog_phrase_uses_knowledge_search_when_agent_mode_is_disabled(
    tmp_path: Path,
) -> None:
    registry, kb = registry_with_kb(tmp_path)
    retrieval = EmptyRetrieval()
    agent = AgentOrchestrator(
        Settings(llm_provider="disabled"),
        registry,
        retrieval,  # type: ignore[arg-type]
        ForbiddenLanguageModel(),  # type: ignore[arg-type]
    )

    response = asyncio.run(
        agent.run(
            ChatRequest(
                question="有哪些文档",
                knowledge_base_ids=[str(kb["id"])],
                agent_mode=False,
            )
        )
    )

    assert response.model == "evidence-gate"
    assert [step.id for step in response.trace][:2] == ["route", "plan"]
    assert len(retrieval.calls) == 1


def test_second_turn_loads_registry_history_for_query_planning_and_llm(
    tmp_path: Path,
) -> None:
    registry, kb = registry_with_kb(tmp_path)
    retrieval = EvidenceRetrieval(str(kb["id"]))
    language_model = RecordingLanguageModel(
        ["第一轮答案。[S1]", "第二轮答案。[S1]"]
    )
    agent = AgentOrchestrator(
        Settings(llm_provider="disabled"),
        registry,
        retrieval,  # type: ignore[arg-type]
        language_model,  # type: ignore[arg-type]
    )
    first_question = "公司年假制度如何规定？"
    second_question = "它有多少天？"

    first = asyncio.run(
        agent.run(
            ChatRequest(
                question=first_question,
                knowledge_base_ids=[str(kb["id"])],
            )
        )
    )
    second = asyncio.run(
        agent.run(
            ChatRequest(
                question=second_question,
                knowledge_base_ids=[str(kb["id"])],
                session_id=first.session_id,
            )
        )
    )

    assert second.session_id == first.session_id
    assert retrieval.queries == [
        first_question,
        second_question,
        f"{first_question} {second_question}",
    ]
    assert len(language_model.calls) == 2
    persisted_history = language_model.calls[1]["history"]
    assert [(item.role, item.content) for item in persisted_history] == [
        ("user", first_question),
        ("assistant", "第一轮答案。[S1]"),
    ]
    assert language_model.calls[1]["question"] == second_question
    assert [item["role"] for item in registry.get_session_messages(first.session_id)] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.parametrize(
    "invalid_answer",
    [
        "董事长批准即可享受 99 天年假。",
        "董事长批准即可享受 99 天年假。[S99]",
        "员工每年有十天年假。[S1] 董事长批准后可增加到 99 天。[S99]",
        "员工每年有十天年假。[S1] 董事长批准后可增加到 99 天。[Sfoo]",
        "员工每年有十天年假。[S01] 董事长批准后可增加到 99 天。",
    ],
)
def test_agent_replaces_missing_or_out_of_range_citations_with_grounded_fallback(
    tmp_path: Path, invalid_answer: str
) -> None:
    registry, kb = registry_with_kb(tmp_path)
    retrieval = EvidenceRetrieval(str(kb["id"]))
    language_model = RecordingLanguageModel([invalid_answer])
    agent = AgentOrchestrator(
        Settings(llm_provider="disabled"),
        registry,
        retrieval,  # type: ignore[arg-type]
        language_model,  # type: ignore[arg-type]
    )

    response = asyncio.run(
        agent.run(
            ChatRequest(
                question="员工年假有多少天？",
                knowledge_base_ids=[str(kb["id"])],
            )
        )
    )

    assert response.model == "citation-validation-fallback"
    assert response.used_fallback is True
    assert response.generation_source == "extractive"
    assert "董事长批准" not in response.answer
    assert "99 天" not in response.answer
    assert "员工每年可享受十天年假" in response.answer
    assert "[S1]" in response.answer
    assert "[S99]" not in response.answer
    assert [source.id for source in response.sources] == ["S1"]
    assert response.trace[-1].id == "citations"
    assert "已降级" in response.trace[-1].detail


def test_agent_abstains_when_context_limit_cannot_fit_a_source_header(
    tmp_path: Path,
) -> None:
    registry, kb = registry_with_kb(tmp_path)
    agent = AgentOrchestrator(
        Settings(llm_provider="disabled", max_context_chars=1),
        registry,
        EvidenceRetrieval(str(kb["id"])),  # type: ignore[arg-type]
        ForbiddenLanguageModel(),  # type: ignore[arg-type]
    )

    response = asyncio.run(
        agent.run(
            ChatRequest(
                question="员工年假有多少天？",
                knowledge_base_ids=[str(kb["id"])],
            )
        )
    )

    assert response.model == "evidence-gate"
    assert response.sources == []
    assert "没有足够依据" in response.answer
    assert response.trace[-1].id == "evidence_gate"
    assert "上下文字符上限" in response.trace[-1].detail


@pytest.mark.parametrize(
    ("fallback_reason", "expected_detail"),
    [
        ("primary_failed", "主模型失败，已切换 qwen3:4b-instruct"),
        ("primary_unconfigured", "主模型未配置，已使用 qwen3:4b-instruct"),
    ],
)
def test_agent_reports_secondary_generation_without_calling_it_extractive(
    tmp_path: Path,
    fallback_reason: FallbackReason,
    expected_detail: str,
) -> None:
    registry, kb = registry_with_kb(tmp_path)
    language_model = RecordingLanguageModel(
        ["员工每年可享受十天年假。[S1]"],
        model="qwen3:4b-instruct",
        used_fallback=True,
        generation_source="secondary",
        fallback_reason=fallback_reason,
    )
    agent = AgentOrchestrator(
        Settings(llm_provider="disabled"),
        registry,
        EvidenceRetrieval(str(kb["id"])),  # type: ignore[arg-type]
        language_model,  # type: ignore[arg-type]
    )

    response = asyncio.run(
        agent.run(
            ChatRequest(
                question="员工年假有多少天？",
                knowledge_base_ids=[str(kb["id"])],
            )
        )
    )

    generate_step = next(step for step in response.trace if step.id == "generate")
    assert generate_step.detail == expected_detail
    assert "抽取式" not in generate_step.detail
    assert response.generation_source == "secondary"
    assert response.model == "qwen3:4b-instruct"
    assert response.used_fallback is True
