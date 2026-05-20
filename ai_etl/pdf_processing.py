from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import fitz
import requests


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str

_NULL_CHAR_PATTERN = re.compile(r"\x00")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(?:[-–—]\s*)?(?:page|trang)?\s*\d+(?:\s*/\s*\d+)?(?:\s*[-–—])?\s*$",
    re.IGNORECASE,
)
_HEADER_FOOTER_REPEAT_THRESHOLD = 2


def sanitize_text(text: str) -> str:
    """Remove PostgreSQL-breaking null chars and normalize noisy spacing."""
    text = _NULL_CHAR_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def _normalize_layout_line(line: str) -> str:
    line = _NULL_CHAR_PATTERN.sub("", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _line_signature(line: str) -> str:
    signature = _compact_whitespace(line.lower())
    signature = re.sub(r"\d+", "#", signature)
    return signature.strip(" -–—|")


def _compact_whitespace(value: str) -> str:
    return " ".join(value.split())


def _clean_pdf_pages(page_texts: list[str]) -> str:
    """Remove repeated page furniture before downstream chunking."""
    line_counts: dict[str, int] = {}
    page_lines: list[list[str]] = []

    for page_text in page_texts:
        lines = [_normalize_layout_line(line) for line in page_text.splitlines()]
        lines = [line for line in lines if line]
        page_lines.append(lines)

        candidates = lines[:3] + lines[-3:]
        page_signatures: set[str] = set()
        for line in candidates:
            if _PAGE_NUMBER_PATTERN.match(line):
                continue
            signature = _line_signature(line)
            if signature and signature not in page_signatures:
                line_counts[signature] = line_counts.get(signature, 0) + 1
                page_signatures.add(signature)

    min_repeats = min(
        max(_HEADER_FOOTER_REPEAT_THRESHOLD, len(page_texts) // 3),
        max(_HEADER_FOOTER_REPEAT_THRESHOLD, len(page_texts)),
    )
    repeated_furniture = {
        signature for signature, count in line_counts.items()
        if count >= min_repeats
    }

    cleaned_parts: list[str] = []
    for lines in page_lines:
        kept: list[str] = []
        for line in lines:
            if _PAGE_NUMBER_PATTERN.match(line):
                continue
            if _line_signature(line) in repeated_furniture:
                continue
            kept.append(line)
        cleaned_parts.append(" ".join(kept))

    text = "\n".join(cleaned_parts)
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    return sanitize_text(text)


def extract_pdf_text(pdf_source: Union[Path, str]) -> str:
    """Extract text from PDF file or URL.

    Args:
        pdf_source: Either a Path to local file or a URL string
    """
    if isinstance(pdf_source, str) and (pdf_source.startswith('http://') or pdf_source.startswith('https://')):
        # Download PDF from URL
        response = requests.get(pdf_source, timeout=30)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        try:
            doc = fitz.open(tmp_path)
        finally:
            tmp_path_obj = Path(tmp_path)
    else:
        # Local file
        doc = fitz.open(str(pdf_source))

    try:
        parts = [page.get_text("text") or "" for page in doc]
    finally:
        doc.close()
        if isinstance(pdf_source, str) and (pdf_source.startswith('http://') or pdf_source.startswith('https://')):
            tmp_path_obj.unlink(missing_ok=True)

    return _clean_pdf_pages(parts)


def sliding_window_chunk(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []

    start = 0
    index = 1
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(chunk_id=f"c{index:05d}", text=chunk_text))
            index += 1

        if end >= len(text):
            break
        start += step

    return chunks
