from __future__ import annotations

import asyncio
import re
import time
import uuid
from typing import Any

from app.config import Settings
from app.db import Registry
from app.schemas import (
    AgentTraceStep,
    ChatRequest,
    ChatResponse,
    CitationSource,
    GenerationSource,
    HistoryMessage,
)
from app.services.llm import LanguageModel
from app.services.prompts import format_evidence
from app.services.retrieval import RetrievalHit, RetrievalService


class AgentOrchestrator:
    """A bounded, observable agent workflow with an explicit tool allowlist."""

    def __init__(
        self,
        settings: Settings,
        registry: Registry,
        retrieval: RetrievalService,
        language_model: LanguageModel,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.retrieval = retrieval
        self.language_model = language_model

    @staticmethod
    def _trace(
        trace: list[AgentTraceStep],
        *,
        step_id: str,
        title: str,
        detail: str,
        started: float,
        tool: str | None = None,
        status: str = "completed",
    ) -> None:
        trace.append(
            AgentTraceStep(
                id=step_id,
                title=title,
                detail=detail,
                duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
                tool=tool,
                status=status,  # type: ignore[arg-type]
            )
        )

    @staticmethod
    def _route(question: str, agent_mode: bool) -> str:
        compact = re.sub(r"\s+", "", question.lower())
        if compact in {"你好", "您好", "嗨", "hello", "hi", "hey"}:
            return "greeting"
        catalog_terms = ("有哪些文档", "有哪些文件", "文档列表", "文件列表", "知识库里有什么")
        if agent_mode and any(term in compact for term in catalog_terms):
            return "document_catalog"
        return "knowledge_search"

    @staticmethod
    def _plan_queries(
        question: str, agent_mode: bool, history: list[HistoryMessage]
    ) -> list[str]:
        queries = [question]
        if not agent_mode:
            return queries
        previous_user = next(
            (item.content for item in reversed(history) if item.role == "user"), ""
        )
        follow_up_markers = ("它", "这", "该", "上述", "前面", "其中", "那么", "呢")
        if previous_user and (
            len(question) <= 24 or any(marker in question for marker in follow_up_markers)
        ):
            queries.append(f"{previous_user} {question}")
        if len(question) < 12:
            return list(dict.fromkeys(queries))[:3]
        if any(marker in question for marker in ("比较", "对比", "分别", "异同")):
            parts = [
                part.strip(" ，。？?")
                for part in re.split(r"(?:与|和|以及|相比|比较)", question)
                if len(part.strip(" ，。？?")) >= 4
            ]
            queries.extend(parts[:2])
        return list(dict.fromkeys(queries))[:3]

    async def run(self, request: ChatRequest) -> ChatResponse:
        run_id = str(uuid.uuid4())
        session_id = await asyncio.to_thread(
            self.registry.ensure_session, request.session_id, request.question
        )
        if request.history:
            history = request.history
        elif request.session_id:
            persisted = await asyncio.to_thread(
                self.registry.get_session_messages, session_id
            )
            history = [
                HistoryMessage(role=item["role"], content=item["content"])
                for item in persisted[-12:]
            ]
        else:
            history = []
        return await self._execute(run_id, session_id, request, history)

    async def _execute(
        self,
        run_id: str,
        session_id: str,
        request: ChatRequest,
        history: list[HistoryMessage],
    ) -> ChatResponse:
        trace: list[AgentTraceStep] = []
        await asyncio.to_thread(
            self.registry.add_message,
            session_id=session_id,
            role="user",
            content=request.question,
        )

        started = time.perf_counter()
        requested_ids = list(dict.fromkeys(request.knowledge_base_ids))
        if requested_ids:
            knowledge_bases = await asyncio.gather(
                *[
                    asyncio.to_thread(self.registry.get_knowledge_base, kb_id)
                    for kb_id in requested_ids
                ]
            )
            knowledge_bases = [kb for kb in knowledge_bases if kb is not None]
        else:
            knowledge_bases = await asyncio.to_thread(self.registry.list_knowledge_bases)
        route = self._route(request.question, request.agent_mode)
        self._trace(
            trace,
            step_id="route",
            title="意图路由",
            detail=f"选择工作流：{route}；知识库范围：{len(knowledge_bases)} 个",
            started=started,
        )

        if route == "greeting":
            return await self._finish(
                run_id,
                session_id,
                "你好！我可以检索已上传的企业 PDF、TXT 或 Markdown 文档，并给出带页码引用的回答。",
                [],
                trace,
                "agent-router",
                False,
            )

        if route == "document_catalog":
            started = time.perf_counter()
            allowed_ids = {str(kb["id"]) for kb in knowledge_bases}
            all_documents = await asyncio.to_thread(self.registry.list_documents)
            documents = [
                item
                for item in all_documents
                if str(item["knowledge_base_id"]) in allowed_ids and item["status"] == "ready"
            ]
            self._trace(
                trace,
                step_id="catalog",
                title="调用文档目录工具",
                detail=f"找到 {len(documents)} 份已就绪文档",
                started=started,
                tool="list_documents",
            )
            answer = (
                "当前可检索的文档有：\n"
                + "\n".join(
                    f"- {doc['filename']}（{doc['page_count']} 页，{doc['chunk_count']} 个分块）"
                    for doc in documents
                )
                if documents
                else "当前选择的知识库还没有已完成索引的文档。"
            )
            return await self._finish(
                run_id, session_id, answer, [], trace, "agent-catalog", False
            )

        if not knowledge_bases:
            return await self._finish(
                run_id,
                session_id,
                "当前没有可用的知识库。请先创建知识库并上传企业文档。",
                [],
                trace,
                "agent-router",
                True,
            )

        started = time.perf_counter()
        queries = self._plan_queries(request.question, request.agent_mode, history)
        self._trace(
            trace,
            step_id="plan",
            title="问题规划",
            detail=f"生成 {len(queries)} 个检索查询，保留原问题中的关键限定词",
            started=started,
        )

        started = time.perf_counter()
        by_id: dict[str, RetrievalHit] = {}
        for query in queries:
            hits = await self.retrieval.search(query, knowledge_bases, max(request.top_k, 4))
            for hit in hits:
                current = by_id.get(hit.id)
                if current is None or hit.score > current.score:
                    by_id[hit.id] = hit
        hits = sorted(by_id.values(), key=lambda item: item.score, reverse=True)[: request.top_k]
        self._trace(
            trace,
            step_id="retrieve",
            title="混合检索",
            detail=f"Dense + BM25 + RRF 检索后选出 {len(hits)} 条证据",
            started=started,
            tool="search_knowledge",
        )

        started = time.perf_counter()
        usable_hits = [hit for hit in hits if hit.text.strip() and hit.score > 0.01]
        items = [hit.as_dict() for hit in usable_hits]
        evidence = format_evidence(items, self.settings.max_context_chars)
        source_count = len(re.findall(r"(?m)^\[S\d+\] 文件：", evidence))
        items = items[:source_count]
        enough_evidence = bool(items)
        self._trace(
            trace,
            step_id="evidence_gate",
            title="证据门控",
            detail=(
                "证据足够，进入受约束生成"
                if enough_evidence
                else (
                    "上下文字符上限无法容纳有效证据"
                    if usable_hits
                    else "没有达到阈值的文档证据"
                )
            ),
            started=started,
        )
        if not enough_evidence:
            return await self._finish(
                run_id,
                session_id,
                "知识库中没有足够依据回答这个问题。请尝试补充相关文档、选择其他知识库，或加入更具体的关键词。",
                [],
                trace,
                "evidence-gate",
                True,
            )

        started = time.perf_counter()
        llm_result = await self.language_model.answer(
            question=request.question,
            evidence=evidence,
            history=history,
            source_items=items,
        )
        self._trace(
            trace,
            step_id="generate",
            title="生成回答",
            detail=(
                "使用本地抽取式降级回答"
                if llm_result.is_extractive
                else (
                    f"主模型失败，已切换 {llm_result.model}"
                    if (
                        llm_result.generation_source == "secondary"
                        and llm_result.fallback_reason == "primary_failed"
                    )
                    else (
                        f"主模型未配置，已使用 {llm_result.model}"
                        if llm_result.generation_source == "secondary"
                        else f"使用 {llm_result.model} 生成"
                    )
                )
            ),
            started=started,
            tool="generate_grounded_answer",
        )

        sources = [
            CitationSource(
                id=f"S{index}",
                document_id=str(item["document_id"]),
                document_name=str(item["document_name"]),
                knowledge_base_id=str(item["knowledge_base_id"]),
                page=int(item["page"]),
                chunk_index=int(item["chunk_index"]),
                excerpt=str(item["text"])[:500],
                score=float(item["score"]),
                dense_score=float(item["dense_score"]),
                lexical_score=float(item["lexical_score"]),
            )
            for index, item in enumerate(items, start=1)
        ]
        started = time.perf_counter()
        answer, citations_valid = self._validate_citations(llm_result.text, len(sources))
        effective_model = llm_result.model
        used_fallback = llm_result.used_fallback
        generation_source = llm_result.generation_source
        if sources and not citations_valid:
            answer = self.language_model.extractive_fallback(request.question, items)
            effective_model = "citation-validation-fallback"
            used_fallback = True
            generation_source = "extractive"
        self._trace(
            trace,
            step_id="citations",
            title="引用校验",
            detail=(
                f"校验 {len(sources)} 个可追溯来源"
                if citations_valid
                else "生成内容缺少有效引用，已降级为可追溯的抽取式回答"
            ),
            started=started,
            tool="validate_citations",
        )
        return await self._finish(
            run_id,
            session_id,
            answer,
            sources,
            trace,
            effective_model,
            used_fallback,
            generation_source,
        )

    @staticmethod
    def _validate_citations(answer: str, source_count: int) -> tuple[str, bool]:
        referenced: list[int] = []
        syntax_valid = True

        def replace(match: re.Match[str]) -> str:
            nonlocal syntax_valid
            token = match.group(0)
            canonical = re.fullmatch(r"\[S([1-9]\d*)\]", token)
            if canonical is None:
                syntax_valid = False
                return ""
            number = int(canonical.group(1))
            referenced.append(number)
            return token if 1 <= number <= source_count else ""

        cleaned = re.sub(r"\[[Ss][^\]\r\n]*\]", replace, answer).strip()
        if source_count == 0:
            return cleaned, syntax_valid and not referenced
        citations_valid = syntax_valid and bool(referenced) and all(
            1 <= number <= source_count for number in referenced
        )
        return cleaned, citations_valid

    async def _finish(
        self,
        run_id: str,
        session_id: str,
        answer: str,
        sources: list[CitationSource],
        trace: list[AgentTraceStep],
        model: str,
        used_fallback: bool,
        generation_source: GenerationSource = "workflow",
    ) -> ChatResponse:
        response = ChatResponse(
            run_id=run_id,
            session_id=session_id,
            answer=answer,
            sources=sources,
            trace=trace,
            model=model,
            used_fallback=used_fallback,
            generation_source=generation_source,
        )
        await asyncio.to_thread(
            self.registry.add_message,
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=[source.model_dump() for source in sources],
            trace=[step.model_dump() for step in trace],
        )
        return response
