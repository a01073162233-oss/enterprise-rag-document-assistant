import asyncio
import math

import pytest

from app.services.embeddings import (
    EmbeddingError,
    LocalHashEmbedding,
    OpenAICompatibleEmbedding,
)


def embed(provider: LocalHashEmbedding, *texts: str) -> list[list[float]]:
    return asyncio.run(provider.embed(list(texts)))


@pytest.mark.parametrize(
    ("canonical", "variant"),
    [
        ("Hello WORLD 123", "  hello   ＷＯＲＬＤ  123  "),
        ("公司年假制度", "  公司\u3000年假\t制度  "),
    ],
)
def test_hash_embedding_is_stable_across_case_width_and_whitespace(
    canonical: str, variant: str
) -> None:
    first = LocalHashEmbedding(dimension=128)
    second = LocalHashEmbedding(dimension=128)

    canonical_vector = embed(first, canonical)[0]
    variant_vector = embed(second, variant)[0]

    assert canonical_vector == variant_vector
    assert len(canonical_vector) == 128
    assert math.sqrt(sum(value * value for value in canonical_vector)) == pytest.approx(1.0)


def test_hash_embedding_is_deterministic_and_normalizes_chinese_and_english() -> None:
    provider = LocalHashEmbedding(dimension=256)
    texts = ["员工年假如何计算？", "How is annual leave calculated?"]

    first = embed(provider, *texts)
    second = embed(provider, *texts)

    assert first == second
    assert first[0] != first[1]
    for vector in first:
        assert len(vector) == 256
        assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


def test_hash_embedding_returns_a_zero_vector_for_text_without_features() -> None:
    provider = LocalHashEmbedding(dimension=32)

    vector = embed(provider, " 。！\n")[0]

    assert vector == [0.0] * 32


def test_hash_embedding_rejects_non_positive_dimension() -> None:
    with pytest.raises(EmbeddingError, match="正整数"):
        LocalHashEmbedding(dimension=0)


def test_remote_embedding_retries_and_wraps_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> object:
            raise ValueError("not json")

    class Client:
        calls = 0

        async def __aenter__(self) -> "Client":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, *_: object, **__: object) -> Response:
            self.calls += 1
            return Response()

    async def no_sleep(_: float) -> None:
        return None

    client = Client()
    monkeypatch.setattr(
        "app.services.embeddings.httpx.AsyncClient", lambda **_: client
    )
    monkeypatch.setattr("app.services.embeddings.asyncio.sleep", no_sleep)
    provider = OpenAICompatibleEmbedding(
        api_key="key",
        base_url="https://embedding.invalid/v1",
        model="embedding-model",
        dimension=2,
    )

    with pytest.raises(EmbeddingError, match="调用失败"):
        asyncio.run(provider.embed(["one"]))

    assert client.calls == 3


def test_remote_embedding_validates_every_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> object:
            return {
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [1.0]},
                ]
            }

    class Client:
        async def __aenter__(self) -> "Client":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, *_: object, **__: object) -> Response:
            return Response()

    async def no_sleep(_: float) -> None:
        return None

    client = Client()
    monkeypatch.setattr(
        "app.services.embeddings.httpx.AsyncClient", lambda **_: client
    )
    monkeypatch.setattr("app.services.embeddings.asyncio.sleep", no_sleep)
    provider = OpenAICompatibleEmbedding(
        api_key="key",
        base_url="https://embedding.invalid/v1",
        model="embedding-model",
        dimension=2,
    )

    with pytest.raises(EmbeddingError, match="调用失败"):
        asyncio.run(provider.embed(["one", "two"]))
