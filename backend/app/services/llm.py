from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from app.config import Settings
from app.schemas import FallbackReason, GenerationSource, HistoryMessage
from app.services.prompts import SYSTEM_PROMPT, build_answer_prompt


class LLMError(RuntimeError):
    pass


@dataclass(slots=True)
class LLMResult:
    text: str
    model: str
    used_fallback: bool
    generation_source: GenerationSource = "primary"
    fallback_reason: FallbackReason | None = None

    @property
    def is_extractive(self) -> bool:
        return self.generation_source == "extractive"


class LanguageModel:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def _generate(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int,
        messages: list[dict[str, str]],
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": 1400,
        }
        async with httpx.AsyncClient(timeout=max(1, timeout_seconds)) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
        response.raise_for_status()
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError("LLM 返回格式不合法") from exc
        if not isinstance(body, dict):
            raise LLMError("LLM 返回格式不合法")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError("LLM 返回格式不合法")
        first = choices[0]
        if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
            raise LLMError("LLM 返回格式不合法")
        content = first["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM 返回了空回答")
        return content.strip()

    async def answer(
        self,
        *,
        question: str,
        evidence: str,
        history: list[HistoryMessage],
        source_items: list[dict[str, object]],
    ) -> LLMResult:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history[-8:]:
            messages.append({"role": item.role, "content": item.content})
        messages.append({"role": "user", "content": build_answer_prompt(question, evidence)})
        provider_failures = (httpx.HTTPError, LLMError)
        fallback_reason: FallbackReason = "primary_unconfigured"
        if self.settings.llm_enabled:
            try:
                content = await self._generate(
                    base_url=self.settings.llm_base_url,
                    api_key=self.settings.llm_api_key,
                    model=self.settings.llm_model,
                    timeout_seconds=self.settings.llm_timeout_seconds,
                    messages=messages,
                )
            except provider_failures:
                fallback_reason = "primary_failed"
            else:
                return LLMResult(
                    text=content,
                    model=self.settings.llm_model,
                    used_fallback=False,
                    generation_source="primary",
                )

        if self.settings.llm_fallback_configured:
            try:
                content = await self._generate(
                    base_url=self.settings.llm_fallback_base_url,
                    api_key=self.settings.llm_fallback_api_key,
                    model=self.settings.llm_fallback_model,
                    timeout_seconds=self.settings.llm_fallback_timeout_seconds,
                    messages=messages,
                )
            except provider_failures:
                pass
            else:
                return LLMResult(
                    text=content,
                    model=self.settings.llm_fallback_model,
                    used_fallback=True,
                    generation_source="secondary",
                    fallback_reason=fallback_reason,
                )

        # Retrieval remains useful while both configured providers are unavailable.
        return LLMResult(
            text=self.extractive_fallback(question, source_items),
            model="extractive-fallback",
            used_fallback=True,
            generation_source="extractive",
        )

    @staticmethod
    def extractive_fallback(question: str, items: list[dict[str, object]]) -> str:
        if not items:
            return "知识库中没有足够依据来回答这个问题。请补充相关文档，或换一种更具体的问法。"
        query_terms = set(re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]", question.lower()))
        excerpts: list[str] = []
        for index, item in enumerate(items[:4], start=1):
            text = str(item["text"]).replace("\n", " ").strip()
            sentences = [part.strip() for part in re.split(r"(?<=[。！？!?；;])\s*", text) if part.strip()]
            ranked = sorted(
                sentences,
                key=lambda sentence: len(
                    query_terms
                    & set(re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]", sentence.lower()))
                ),
                reverse=True,
            )
            excerpt = (ranked[0] if ranked else text)[:300]
            excerpts.append(f"- {excerpt} [S{index}]")
        return (
            "当前生成式 LLM 暂不可用；以下是知识库中与问题最相关的原文依据：\n\n"
            + "\n".join(excerpts)
            + "\n\n配置或恢复生成式 LLM 后，系统会基于这些证据生成归纳答案。"
        )
