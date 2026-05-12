from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import requests
from pypdf import PdfReader


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str

_NULL_CHAR_PATTERN = re.compile(r"\x00")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def sanitize_text(text: str) -> str:
    """Remove PostgreSQL-breaking null chars and normalize noisy spacing."""
    text = _NULL_CHAR_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


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
            reader = PdfReader(tmp_path)
        finally:
            Path(tmp_path).unlink()
    else:
        # Local file
        reader = PdfReader(str(pdf_source))

    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        parts.append(page_text)
    return sanitize_text(" ".join(parts))


def sliding_window_chunk(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
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
