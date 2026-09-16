import asyncio
from dataclasses import dataclass, replace
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.services.llm import LanguageModel


@dataclass
class StubResponse:
    body: object
    json_error: ValueError | None = None

    @staticmethod
    def raise_for_status() -> None:
        return None

    def json(self) -> object:
        if self.json_error is not None:
            raise self.json_error
        return self.body


class StubClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.client_options: list[dict[str, Any]] = []

    async def __aenter__(self) -> "StubClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> StubResponse:
        self.calls.append({"url": url, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, StubResponse)
        return outcome


def completion(content: object) -> StubResponse:
    return StubResponse({"choices": [{"message": {"content": content}}]})


def failover_settings(*, enabled: bool = True, fallback_api_key: str = "") -> Settings:
    return Settings(
        llm_provider="openai-compatible",
        llm_api_key="deepseek-secret",
        llm_base_url="https://deepseek.invalid/v1",
        llm_model="deepseek-flash",
        llm_timeout_seconds=7,
        llm_fallback_enabled=enabled,
        llm_fallback_api_key=fallback_api_key,
        llm_fallback_base_url="http://127.0.0.1:11434/v1",
        llm_fallback_model="qwen3:4b-instruct",
        llm_fallback_timeout_seconds=11,
    )


def run_answer(model: LanguageModel):
    return asyncio.run(
        model.answer(
            question="年假是多少天？",
            evidence="[S1] 年假为十天。",
            history=[],
            source_items=[{"text": "员工每年享有十天年假。"}],
        )
    )


def install_client(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[object]
) -> StubClient:
    client = StubClient(outcomes)

    def build_client(**kwargs: Any) -> StubClient:
        client.client_options.append(kwargs)
        return client

    monkeypatch.setattr("app.services.llm.httpx.AsyncClient", build_client)
    return client


def test_primary_success_does_not_call_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    client = install_client(monkeypatch, [completion("主模型回答。[S1]")])

    result = run_answer(LanguageModel(failover_settings()))

    assert result.text == "主模型回答。[S1]"
    assert result.model == "deepseek-flash"
    assert result.generation_source == "primary"
    assert result.used_fallback is False
    assert result.is_extractive is False
    assert [call["url"] for call in client.calls] == [
        "https://deepseek.invalid/v1/chat/completions"
    ]
    assert client.calls[0]["json"]["model"] == "deepseek-flash"
    assert client.client_options == [{"timeout": 7}]


def http_failure() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://deepseek.invalid/v1/chat/completions")
    response = httpx.Response(503, request=request)
    return httpx.HTTPStatusError("primary unavailable", request=request, response=response)


@pytest.mark.parametrize(
    "primary_failure",
    [
        httpx.ConnectError("connection refused"),
        http_failure(),
        StubResponse(None, json_error=ValueError("not json")),
        StubResponse({"not_choices": []}),
        completion("   "),
    ],
    ids=["transport", "http", "invalid-json", "malformed", "empty"],
)
def test_primary_failure_uses_ollama(
    monkeypatch: pytest.MonkeyPatch, primary_failure: object
) -> None:
    client = install_client(
        monkeypatch,
        [primary_failure, completion("Ollama 回答。[S1]")],
    )

    result = run_answer(LanguageModel(failover_settings()))

    assert result.text == "Ollama 回答。[S1]"
    assert result.model == "qwen3:4b-instruct"
    assert result.generation_source == "secondary"
    assert result.fallback_reason == "primary_failed"
    assert result.used_fallback is True
    assert result.is_extractive is False
    assert [call["url"] for call in client.calls] == [
        "https://deepseek.invalid/v1/chat/completions",
        "http://127.0.0.1:11434/v1/chat/completions",
    ]
    assert client.calls[0]["headers"]["Authorization"] == "Bearer deepseek-secret"
    assert "Authorization" not in client.calls[1]["headers"]
    assert client.calls[1]["json"]["model"] == "qwen3:4b-instruct"
    assert client.client_options == [{"timeout": 7}, {"timeout": 11}]


def test_both_providers_fail_then_uses_extractively_grounded_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = install_client(
        monkeypatch,
        [StubResponse({"choices": []}), completion(None)],
    )

    result = run_answer(LanguageModel(failover_settings()))

    assert len(client.calls) == 2
    assert result.model == "extractive-fallback"
    assert result.generation_source == "extractive"
    assert result.used_fallback is True
    assert result.is_extractive is True
    assert "十天年假" in result.text
    assert "[S1]" in result.text


def test_disabled_ollama_fallback_is_not_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = install_client(monkeypatch, [httpx.ConnectError("offline")])

    result = run_answer(LanguageModel(failover_settings(enabled=False)))

    assert len(client.calls) == 1
    assert result.generation_source == "extractive"
    assert result.model == "extractive-fallback"


def test_fallback_can_be_the_only_configured_generation_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = install_client(monkeypatch, [completion("仅 Ollama 回答。[S1]")])
    configured = replace(failover_settings(), llm_provider="disabled")

    result = run_answer(LanguageModel(configured))

    assert [call["url"] for call in client.calls] == [
        "http://127.0.0.1:11434/v1/chat/completions"
    ]
    assert result.generation_source == "secondary"
    assert result.fallback_reason == "primary_unconfigured"
    assert result.model == "qwen3:4b-instruct"
    assert result.used_fallback is True


@pytest.mark.parametrize(
    "unexpected",
    [RuntimeError("programming error"), asyncio.CancelledError()],
    ids=["runtime", "cancelled"],
)
def test_unexpected_errors_do_not_trigger_failover(
    monkeypatch: pytest.MonkeyPatch, unexpected: BaseException
) -> None:
    client = install_client(monkeypatch, [unexpected])

    with pytest.raises(type(unexpected)):
        run_answer(LanguageModel(failover_settings()))

    assert len(client.calls) == 1


def test_provider_secrets_are_isolated_and_never_leak_to_result_or_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    primary_secret = "primary-do-not-leak"
    fallback_secret = "fallback-do-not-leak"
    configured = replace(
        failover_settings(fallback_api_key=fallback_secret),
        llm_api_key=primary_secret,
    )
    client = install_client(
        monkeypatch,
        [
            httpx.ConnectError(f"provider rejected {primary_secret}"),
            httpx.ConnectError(f"provider rejected {fallback_secret}"),
        ],
    )

    result = run_answer(LanguageModel(configured))

    assert client.calls[0]["headers"]["Authorization"] == f"Bearer {primary_secret}"
    assert client.calls[1]["headers"]["Authorization"] == f"Bearer {fallback_secret}"
    assert primary_secret not in repr(client.calls[1]["headers"])
    exposed = f"{result!r}\n{caplog.text}"
    assert primary_secret not in repr(configured)
    assert fallback_secret not in repr(configured)
    assert primary_secret not in exposed
    assert fallback_secret not in exposed
