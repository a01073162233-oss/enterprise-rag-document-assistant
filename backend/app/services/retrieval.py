from __future__ import annotations

import asyncio

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.db import Registry
from app.services.embeddings import create_embedding_provider
from app.services.vector_store import QdrantVectorStore


@dataclass(slots=True)
class RetrievalHit:
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "knowledge_base_id": self.knowledge_base_id,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "score": self.score,
            "dense_score": self.dense_score,
            "lexical_score": self.lexical_score,
        }


def lexical_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    latin_tokens = re.findall(r"[a-z0-9_]+", normalized)
    cjk_tokens: list[str] = []
    cjk_source = re.sub(
        r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", normalized
    )
    for span in re.findall(r"[\u3400-\u9fff]+", cjk_source):
        # Single Han characters are too noisy for enterprise retrieval: an
        # unrelated page containing only “时” should not match a question
        # containing “什么时候”.  Bigrams keep exact Chinese phrases precise,
        # while a genuinely one-character span remains searchable.
        if len(span) == 1:
            cjk_tokens.append(span)
        else:
            cjk_tokens.extend(
                span[index : index + 2] for index in range(len(span) - 1)
            )
    return latin_tokens + cjk_tokens


def bm25_scores(query: str, records: list[dict[str, Any]]) -> dict[str, float]:
    query_terms = lexical_tokens(query)
    if not query_terms or not records:
        return {}
    tokenized = [lexical_tokens(str(record["payload"].get("text", ""))) for record in records]
    average_length = sum(len(tokens) for tokens in tokenized) / max(1, len(tokenized))
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    scores: dict[str, float] = {}
    total = len(records)
    k1, b = 1.5, 0.75
    for record, tokens in zip(records, tokenized, strict=True):
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * len(tokens) / max(1.0, average_length)
            )
            score += idf * (frequency * (k1 + 1)) / denominator
        if score > 0:
            scores[str(record["id"])] = score
    return scores


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        vector_store: QdrantVectorStore,
        registry: Registry,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.registry = registry

    async def search(
        self, query: str, knowledge_bases: list[dict[str, Any]], top_k: int
    ) -> list[RetrievalHit]:
        all_hits: list[RetrievalHit] = []
        candidate_limit = max(20, top_k * 4)
        for kb in knowledge_bases:
            documents = await asyncio.to_thread(
                self.registry.list_documents, str(kb["id"])
            )
            ready_document_ids = [
                str(document["id"])
                for document in documents
                if document["status"] == "ready"
            ]
            if not ready_document_ids:
                continue
            provider = create_embedding_provider(
                self.settings,
                provider=str(kb["embedding_provider"]),
                model=str(kb["embedding_model"]),
                dimension=int(kb["embedding_dimension"]),
            )
            vector = (await provider.embed([query]))[0]
            dense = await asyncio.to_thread(
                self.vector_store.dense_search,
                str(kb["id"]),
                vector,
                candidate_limit,
                ready_document_ids,
            )
            records = await asyncio.to_thread(
                self.vector_store.scroll_payloads,
                str(kb["id"]),
                self.settings.lexical_scan_limit,
                ready_document_ids,
            )
            lexical = await asyncio.to_thread(bm25_scores, query, records)
            record_by_id = {str(record["id"]): record for record in records}
            # A zero-similarity nearest neighbour is still returned by a vector
            # index; exclude it so the evidence gate can genuinely abstain.
            meaningful_dense = [item for item in dense if float(item.get("score", 0.0)) > 0.05]
            dense_by_id = {str(item["id"]): item for item in meaningful_dense}
            dense_rank = {
                str(item["id"]): rank for rank, item in enumerate(meaningful_dense, start=1)
            }
            lexical_order = sorted(lexical, key=lexical.get, reverse=True)[:candidate_limit]
            lexical_rank = {point_id: rank for rank, point_id in enumerate(lexical_order, start=1)}
            max_lexical = max(lexical.values(), default=1.0)
            candidate_ids = set(dense_by_id) | set(lexical_order)
            for point_id in candidate_ids:
                raw = dense_by_id.get(point_id) or record_by_id.get(point_id)
                if not raw:
                    continue
                payload = raw.get("payload", {})
                dense_score = float(dense_by_id.get(point_id, {}).get("score", 0.0))
                lexical_score = float(lexical.get(point_id, 0.0))
                rrf = 0.0
                if point_id in dense_rank:
                    rrf += 0.65 / (60 + dense_rank[point_id])
                if point_id in lexical_rank:
                    rrf += 0.35 / (60 + lexical_rank[point_id])
                display_score = max(0.0, min(1.0, rrf * 61))
                all_hits.append(
                    RetrievalHit(
                        id=point_id,
                        document_id=str(payload.get("document_id", "")),
                        document_name=str(payload.get("document_name", "未知文档")),
                        knowledge_base_id=str(kb["id"]),
                        page=int(payload.get("page", 1)),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        text=str(payload.get("text", "")),
                        score=round(display_score, 4),
                        dense_score=round(dense_score, 4),
                        lexical_score=round(lexical_score / max_lexical, 4),
                    )
                )

        all_hits.sort(key=lambda hit: (hit.score, hit.dense_score), reverse=True)
        selected: list[RetrievalHit] = []
        per_document: defaultdict[str, int] = defaultdict(int)
        seen_hashes: set[str] = set()
        for hit in all_hits:
            normalized = re.sub(r"\s+", " ", hit.text).strip()
            fingerprint = normalized[:240]
            if not normalized or fingerprint in seen_hashes or per_document[hit.document_id] >= 3:
                continue
            selected.append(hit)
            seen_hashes.add(fingerprint)
            per_document[hit.document_id] += 1
            if len(selected) >= top_k:
                break
        return selected
