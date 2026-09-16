from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Load a small, dependency-free subset of .env syntax."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _path_env(name: str, default: str) -> Path:
    value = Path(_env(name, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = _env("APP_NAME", "企业文档智能问答系统")
    app_env: str = _env("APP_ENV", "development").lower()
    api_key: str = field(default=_env("RAG_API_KEY"), repr=False)
    api_prefix: str = "/api"
    data_dir: Path = _path_env("DATA_DIR", "data")
    upload_dir: Path = _path_env("UPLOAD_DIR", "data/uploads")
    metadata_db_path: Path = _path_env("METADATA_DB_PATH", "data/metadata.db")

    qdrant_mode: str = _env("QDRANT_MODE", "local").lower()
    qdrant_path: Path = _path_env("QDRANT_PATH", "data/qdrant")
    qdrant_url: str = _env("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = field(default=_env("QDRANT_API_KEY"), repr=False)

    embedding_provider: str = _env("EMBEDDING_PROVIDER", "local").lower()
    embedding_api_key: str = field(
        default=_env("EMBEDDING_API_KEY", _env("LLM_API_KEY")), repr=False
    )
    embedding_base_url: str = _env(
        "EMBEDDING_BASE_URL", _env("LLM_BASE_URL", "https://api.openai.com/v1")
    ).rstrip("/")
    embedding_model: str = _env("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimension: int = _int_env("EMBEDDING_DIMENSION", 768)
    embedding_batch_size: int = _int_env("EMBEDDING_BATCH_SIZE", 64)

    llm_provider: str = _env("LLM_PROVIDER", "auto").lower()
    llm_api_key: str = field(default=_env("LLM_API_KEY"), repr=False)
    llm_base_url: str = _env("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    llm_model: str = _env("LLM_MODEL", "gpt-4.1-mini")
    llm_temperature: float = _float_env("LLM_TEMPERATURE", 0.1)
    llm_timeout_seconds: int = _int_env("LLM_TIMEOUT_SECONDS", 90)
    llm_fallback_enabled: bool = _bool_env("LLM_FALLBACK_ENABLED", False)
    llm_fallback_api_key: str = field(
        default=_env("LLM_FALLBACK_API_KEY"), repr=False
    )
    llm_fallback_base_url: str = _env(
        "LLM_FALLBACK_BASE_URL", "http://127.0.0.1:11434/v1"
    ).rstrip("/")
    llm_fallback_model: str = _env(
        "LLM_FALLBACK_MODEL", "qwen3:4b-instruct"
    )
    llm_fallback_timeout_seconds: int = _int_env(
        "LLM_FALLBACK_TIMEOUT_SECONDS", 180
    )

    chunk_size: int = _int_env("CHUNK_SIZE", 900)
    chunk_overlap: int = _int_env("CHUNK_OVERLAP", 150)
    retrieval_top_k: int = _int_env("RETRIEVAL_TOP_K", 6)
    lexical_scan_limit: int = _int_env("LEXICAL_SCAN_LIMIT", 5000)
    max_context_chars: int = _int_env("MAX_CONTEXT_CHARS", 14000)
    max_upload_mb: int = _int_env("MAX_UPLOAD_MB", 50)
    max_upload_files: int = _int_env("MAX_UPLOAD_FILES", 10)
    max_total_upload_mb: int = _int_env("MAX_TOTAL_UPLOAD_MB", 100)
    max_pdf_pages: int = _int_env("MAX_PDF_PAGES", 1000)
    max_document_chars: int = _int_env("MAX_DOCUMENT_CHARS", 2_000_000)
    max_chunks_per_document: int = _int_env("MAX_CHUNKS_PER_DOCUMENT", 2500)
    enable_docs: bool = _bool_env("ENABLE_API_DOCS", True)

    cors_origins_raw: str = _env(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    )

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]

    @property
    def llm_enabled(self) -> bool:
        if self.llm_provider in {"disabled", "fallback", "local-fallback"}:
            return False
        return bool(self.llm_api_key) or "localhost" in self.llm_base_url or "127.0.0.1" in self.llm_base_url

    @property
    def llm_fallback_configured(self) -> bool:
        return bool(
            self.llm_fallback_enabled
            and self.llm_fallback_base_url
            and self.llm_fallback_model
        )

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.qdrant_path, self.metadata_db_path.parent):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
