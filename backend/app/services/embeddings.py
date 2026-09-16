from __future__ import annotations

import asyncio
import hashlib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(slots=True)
class LocalHashEmbedding:
    """Deterministic multilingual feature hashing for a zero-config local demo.

    It is intentionally lightweight rather than a replacement for a trained
    semantic embedding model. Production users should configure a compatible
    embedding endpoint without changing the retrieval/indexing code.
    """

    dimension: int = 768
    name: str = "local"
    model: str = "hashing-multilingual-v1"

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise EmbeddingError("Embedding 维度必须是正整数")

    @staticmethod
    def _features(text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text).lower()
        words = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", normalized)
        word_bigrams: list[str] = []
        for segment in re.split(r"[^a-z0-9_\u3400-\u9fff\s]+", normalized):
            segment_words = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", segment)
            word_bigrams.extend(
                f"w2:{a}_{b}" for a, b in zip(segment_words, segment_words[1:])
            )
        cjk_bigrams: list[str] = []
        cjk_source = re.sub(
            r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", normalized
        )
        for span in re.findall(r"[\u3400-\u9fff]+", cjk_source):
            cjk_bigrams.extend(f"c2:{span[index:index + 2]}" for index in range(len(span) - 1))
        latin_fragments: list[str] = []
        for word in words:
            if len(word) > 2 and word.isascii():
                latin_fragments.extend(f"g3:{word[i:i + 3]}" for i in range(len(word) - 2))
        return [f"t:{token}" for token in words] + word_bigrams + cjk_bigrams + latin_fragments

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        counts = Counter(self._features(text))
        for feature, count in counts.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "little")
            index = number % self.dimension
            sign = 1.0 if (number >> 63) == 0 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(lambda: [self._one(text) for text in texts])


@dataclass(slots=True)
class OpenAICompatibleEmbedding:
    api_key: str
    base_url: str
    model: str
    dimension: int
    batch_size: int = 64
    timeout_seconds: int = 90
    name: str = "openai-compatible"

    async def _batch(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, object] = {"model": self.model, "input": texts}
        # OpenAI text-embedding-3 models support shortened dimensions. Other
        # compatible providers often do not, so only send it for those models.
        if self.dimension and self.model.startswith("text-embedding-3"):
            payload["dimensions"] = self.dimension
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(
                        f"{self.base_url}/embeddings", headers=headers, json=payload
                    )
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict) or not isinstance(body.get("data"), list):
                    raise EmbeddingError("Embedding 服务返回格式不合法")
                indexed: list[tuple[int, list[float]]] = []
                for position, item in enumerate(body["data"]):
                    if not isinstance(item, dict) or not isinstance(
                        item.get("embedding"), list
                    ):
                        raise EmbeddingError("Embedding 服务返回格式不合法")
                    index = item.get("index", position)
                    if not isinstance(index, int) or isinstance(index, bool):
                        raise EmbeddingError("Embedding 服务返回的索引不合法")
                    try:
                        vector = [float(value) for value in item["embedding"]]
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise EmbeddingError(
                            "Embedding 服务返回了非数值向量"
                        ) from exc
                    if len(vector) != self.dimension or not all(
                        math.isfinite(value) for value in vector
                    ):
                        raise EmbeddingError(
                            f"Embedding 维度或数值不合法：配置为 {self.dimension}"
                        )
                    indexed.append((index, vector))
                indexed.sort(key=lambda item: item[0])
                if [index for index, _vector in indexed] != list(range(len(texts))):
                    raise EmbeddingError("Embedding 服务返回的索引不连续")
                vectors = [vector for _index, vector in indexed]
                if len(vectors) != len(texts):
                    raise EmbeddingError("Embedding 服务返回数量与输入不一致")
                return vectors
            except (
                httpx.HTTPError,
                KeyError,
                TypeError,
                ValueError,
                EmbeddingError,
            ) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2**attempt))
        raise EmbeddingError(f"Embedding 服务调用失败：{type(last_error).__name__}") from last_error

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(await self._batch(texts[start : start + self.batch_size]))
        return vectors


def provider_identity(settings: Settings) -> tuple[str, str, int]:
    if settings.embedding_provider in {"local", "hash", "local-hash"}:
        return "local", "hashing-multilingual-v1", settings.embedding_dimension
    return "openai-compatible", settings.embedding_model, settings.embedding_dimension


def create_embedding_provider(
    settings: Settings,
    *,
    provider: str | None = None,
    model: str | None = None,
    dimension: int | None = None,
) -> EmbeddingProvider:
    chosen = (provider or settings.embedding_provider).lower()
    chosen_dimension = settings.embedding_dimension if dimension is None else dimension
    if chosen_dimension <= 0:
        raise EmbeddingError("Embedding 维度必须是正整数")
    if chosen in {"local", "hash", "local-hash"}:
        return LocalHashEmbedding(dimension=chosen_dimension)
    return OpenAICompatibleEmbedding(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model=model or settings.embedding_model,
        dimension=chosen_dimension,
        batch_size=max(1, settings.embedding_batch_size),
        timeout_seconds=max(1, settings.llm_timeout_seconds),
    )
