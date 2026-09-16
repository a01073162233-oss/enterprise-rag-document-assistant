from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


GenerationSource = Literal["primary", "secondary", "extractive", "workflow"]
FallbackReason = Literal["primary_failed", "primary_unconfigured"]


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("知识库名称不能为空")
        return value


class KnowledgeBase(BaseModel):
    id: str
    name: str
    description: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    chunk_size: int
    chunk_overlap: int
    document_count: int = 0
    chunk_count: int = 0
    created_at: str
    updated_at: str


class Document(BaseModel):
    id: str
    knowledge_base_id: str
    filename: str
    content_type: str
    byte_size: int
    sha256: str
    page_count: int
    chunk_count: int
    status: Literal["processing", "ready", "failed"]
    error: str | None = None
    created_at: str
    processed_at: str | None = None


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    knowledge_base_ids: list[str] = Field(default_factory=list, max_length=10)
    agent_mode: bool = True
    top_k: int = Field(default=6, ge=1, le=20)
    session_id: str | None = None
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("问题不能为空")
        return value


class CitationSource(BaseModel):
    id: str
    document_id: str
    document_name: str
    knowledge_base_id: str
    page: int
    chunk_index: int
    excerpt: str
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0


class AgentTraceStep(BaseModel):
    id: str
    title: str
    detail: str
    status: Literal["completed", "skipped", "failed"] = "completed"
    duration_ms: int = 0
    tool: str | None = None


class ChatResponse(BaseModel):
    run_id: str
    session_id: str
    answer: str
    sources: list[CitationSource]
    trace: list[AgentTraceStep]
    model: str
    used_fallback: bool = False
    generation_source: GenerationSource = "workflow"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    knowledge_base_ids: list[str] = Field(default_factory=list, max_length=10)
    top_k: int = Field(default=10, ge=1, le=50)


class SearchHit(BaseModel):
    id: str
    document_id: str
    document_name: str
    knowledge_base_id: str
    page: int
    chunk_index: int
    text: str
    score: float
    dense_score: float
    lexical_score: float


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class HealthResponse(BaseModel):
    status: str
    vector_database: str
    vector_database_mode: str
    llm: str
    embedding: str
    details: dict[str, Any] = Field(default_factory=dict)


class SessionSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
