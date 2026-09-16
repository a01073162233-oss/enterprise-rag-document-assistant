from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


class DocumentParseError(ValueError):
    pass


@dataclass(slots=True)
class PageText:
    page_number: int
    text: str


@dataclass(slots=True)
class ParsedDocument:
    pages: list[PageText]
    page_count: int


@dataclass(slots=True)
class TextChunk:
    id: str
    document_id: str
    knowledge_base_id: str
    document_name: str
    page_number: int
    chunk_index: int
    text: str
    content_hash: str


def _normalize_text(value: str) -> str:
    value = value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"(?<=[A-Za-z])-\n(?=[A-Za-z])", "", value)
    value = re.sub(r"[\t\u00a0]+", " ", value)
    value = re.sub(r"[ ]{2,}", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _remove_repeated_margins(pages: list[PageText]) -> list[PageText]:
    """Remove common headers/footers while preserving page attribution."""
    if len(pages) < 3:
        return pages
    candidates: Counter[str] = Counter()
    page_lines: list[list[str]] = []
    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        page_lines.append(lines)
        for line in lines[:2] + lines[-2:]:
            key = re.sub(r"\d+", "#", line).strip().lower()
            if 2 <= len(key) <= 120:
                candidates[key] += 1
    threshold = max(3, math.ceil(len(pages) * 0.6))
    repeated = {line for line, count in candidates.items() if count >= threshold}
    if not repeated:
        return pages
    cleaned: list[PageText] = []
    for page, lines in zip(pages, page_lines, strict=True):
        kept: list[str] = []
        for index, line in enumerate(lines):
            key = re.sub(r"\d+", "#", line).strip().lower()
            is_margin = index < 2 or index >= max(0, len(lines) - 2)
            if is_margin and key in repeated:
                continue
            kept.append(line)
        cleaned.append(PageText(page.page_number, "\n".join(kept).strip()))
    return cleaned


def extract_document(
    path: Path, *, max_pages: int = 1000, max_characters: int = 2_000_000
) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="gb18030")
        normalized = _normalize_text(text)
        if not normalized:
            raise DocumentParseError("文档中没有可提取的文本")
        if len(normalized) > max_characters:
            raise DocumentParseError(
                f"文档文本超过处理上限（最多 {max_characters:,} 个字符）"
            )
        return ParsedDocument(pages=[PageText(1, normalized)], page_count=1)
    if suffix != ".pdf":
        raise DocumentParseError("当前仅支持 PDF、TXT 和 Markdown 文件")

    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DocumentParseError("缺少 PyMuPDF，无法解析 PDF") from exc

    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        raise DocumentParseError("PDF 文件损坏或格式不受支持") from exc
    try:
        if pdf.needs_pass:
            raise DocumentParseError("PDF 已加密，请先移除密码后再上传")
        if pdf.page_count > max_pages:
            raise DocumentParseError(f"PDF 页数超过限制（最多 {max_pages} 页）")
        pages: list[PageText] = []
        character_count = 0
        for index, page in enumerate(pdf):
            text = _normalize_text(page.get_text("text", sort=True))
            character_count += len(text)
            if character_count > max_characters:
                raise DocumentParseError(
                    f"PDF 文本超过处理上限（最多 {max_characters:,} 个字符）"
                )
            pages.append(PageText(page_number=index + 1, text=text))
    finally:
        pdf.close()

    pages = _remove_repeated_margins(pages)
    character_count = sum(len(page.text) for page in pages)
    if character_count < 20:
        raise DocumentParseError(
            "PDF 中未检测到足够的可搜索文本；它可能是扫描件，请先进行 OCR"
        )
    return ParsedDocument(pages=pages, page_count=len(pages))


def _best_boundary(text: str, start: int, target_end: int) -> int:
    if target_end >= len(text):
        return len(text)
    minimum = start + max(80, int((target_end - start) * 0.6))
    window = text[minimum:target_end]
    matches = list(re.finditer(r"[。！？!?；;\.\n]", window))
    if matches:
        return minimum + matches[-1].end()
    return target_end


def chunk_document(
    parsed: ParsedDocument,
    *,
    document_id: str,
    knowledge_base_id: str,
    document_name: str,
    chunk_size: int,
    overlap: int,
    max_chunks: int | None = None,
) -> list[TextChunk]:
    if chunk_size < 100:
        raise ValueError("chunk_size 必须至少为 100")
    overlap = max(0, min(overlap, chunk_size // 2))
    chunks: list[TextChunk] = []
    ordinal = 0
    for page in parsed.pages:
        text = page.text.strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            raw_end = min(len(text), start + chunk_size)
            end = _best_boundary(text, start, raw_end)
            piece = text[start:end].strip()
            if piece:
                digest = hashlib.sha256(piece.encode("utf-8")).hexdigest()
                stable_key = f"{document_id}:{page.page_number}:{ordinal}:{digest}"
                chunks.append(
                    TextChunk(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key)),
                        document_id=document_id,
                        knowledge_base_id=knowledge_base_id,
                        document_name=document_name,
                        page_number=page.page_number,
                        chunk_index=ordinal,
                        text=piece,
                        content_hash=digest,
                    )
                )
                if max_chunks is not None and len(chunks) > max_chunks:
                    raise DocumentParseError(
                        f"文档分块超过处理上限（最多 {max_chunks:,} 个分块）"
                    )
                ordinal += 1
            if end >= len(text):
                break
            next_start = max(start + 1, end - overlap)
            while next_start < end and text[next_start].isspace():
                next_start += 1
            start = next_start
    if not chunks:
        raise DocumentParseError("文档解析成功，但未产生可索引的文本分块")
    return chunks
