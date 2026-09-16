import hashlib
from pathlib import Path

import pymupdf
import pytest

from app.services.documents import (
    DocumentParseError,
    PageText,
    ParsedDocument,
    chunk_document,
    extract_document,
)


def test_extract_txt_normalizes_text_and_assigns_page_one(tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_bytes("Annual bene-\r\nfit\tpolicy\r\n\r\n\r\n年假制度".encode("utf-8"))

    parsed = extract_document(path)

    assert parsed.page_count == 1
    assert [page.page_number for page in parsed.pages] == [1]
    assert parsed.pages[0].text == "Annual benefit policy\n\n年假制度"


def test_chunk_document_respects_page_boundaries_overlap_and_stable_ids() -> None:
    page_one = "甲" * 82 + "。" + "乙" * 77 + "！" + "丙" * 45
    page_two = "PageTwo " * 30 + "End."
    parsed = ParsedDocument(
        pages=[PageText(1, page_one), PageText(2, page_two)],
        page_count=2,
    )
    kwargs = {
        "document_id": "document-1",
        "knowledge_base_id": "kb-1",
        "document_name": "handbook.txt",
        "chunk_size": 100,
        "overlap": 20,
    }

    chunks = chunk_document(parsed, **kwargs)
    repeated = chunk_document(parsed, **kwargs)

    assert len(chunks) > 2
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in repeated]
    assert [chunk.page_number for chunk in chunks] == sorted(
        chunk.page_number for chunk in chunks
    )
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert all(
        chunk.content_hash == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        for chunk in chunks
    )
    assert all("PageTwo" not in chunk.text for chunk in chunks if chunk.page_number == 1)
    assert all("甲" not in chunk.text for chunk in chunks if chunk.page_number == 2)

    first_page_chunks = [chunk for chunk in chunks if chunk.page_number == 1]
    assert first_page_chunks[0].text.endswith("。")
    assert first_page_chunks[0].text[-20:] in first_page_chunks[1].text


def test_extract_pdf_and_chunking_preserve_physical_page_numbers(tmp_path: Path) -> None:
    path = tmp_path / "handbook.pdf"
    pdf = pymupdf.open()
    expected_markers = ["PAGE_ONE_POLICY", "PAGE_TWO_SECURITY"]
    for index, marker in enumerate(expected_markers, start=1):
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            f"{marker} page {index}. " + ("Enterprise policy content. " * 10),
        )
    pdf.save(path)
    pdf.close()

    parsed = extract_document(path)
    chunks = chunk_document(
        parsed,
        document_id="pdf-document",
        knowledge_base_id="kb-pdf",
        document_name=path.name,
        chunk_size=120,
        overlap=20,
    )

    assert parsed.page_count == 2
    assert [page.page_number for page in parsed.pages] == [1, 2]
    assert expected_markers[0] in parsed.pages[0].text
    assert expected_markers[1] in parsed.pages[1].text
    assert any(chunk.page_number == 1 and expected_markers[0] in chunk.text for chunk in chunks)
    assert any(chunk.page_number == 2 and expected_markers[1] in chunk.text for chunk in chunks)
    assert not any(chunk.page_number == 1 and expected_markers[1] in chunk.text for chunk in chunks)


def test_chunk_document_rejects_an_unsafe_small_chunk_size() -> None:
    parsed = ParsedDocument(pages=[PageText(1, "content")], page_count=1)

    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document(
            parsed,
            document_id="doc",
            knowledge_base_id="kb",
            document_name="doc.txt",
            chunk_size=99,
            overlap=0,
        )


def test_extract_pdf_rejects_page_count_above_configured_limit(tmp_path: Path) -> None:
    path = tmp_path / "too-many-pages.pdf"
    pdf = pymupdf.open()
    for page_number in range(2):
        page = pdf.new_page()
        page.insert_text((72, 72), f"Policy page {page_number + 1} with searchable text.")
    pdf.save(path)
    pdf.close()

    with pytest.raises(DocumentParseError, match="PDF 页数超过限制.*1 页"):
        extract_document(path, max_pages=1)


def test_extract_pdf_rejects_extracted_text_above_character_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "too-much-text.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    for line_number in range(5):
        page.insert_text(
            (72, 72 + line_number * 18),
            f"LIMIT_LINE_{line_number} " + ("x" * 45),
        )
    pdf.save(path)
    pdf.close()

    with pytest.raises(DocumentParseError, match="PDF 文本超过处理上限.*60"):
        extract_document(path, max_characters=60)

    # The parser must close the PDF even when it aborts at a limit on Windows.
    reopened = pymupdf.open(path)
    assert reopened.page_count == 1
    reopened.close()


def test_chunk_limit_allows_exact_limit_and_rejects_one_extra_chunk() -> None:
    parsed = ParsedDocument(pages=[PageText(1, "x" * 350)], page_count=1)
    kwargs = {
        "document_id": "limited-document",
        "knowledge_base_id": "limited-kb",
        "document_name": "large.txt",
        "chunk_size": 100,
        "overlap": 0,
    }

    chunks = chunk_document(parsed, max_chunks=4, **kwargs)

    assert len(chunks) == 4
    with pytest.raises(DocumentParseError, match="分块超过处理上限.*3"):
        chunk_document(parsed, max_chunks=3, **kwargs)
