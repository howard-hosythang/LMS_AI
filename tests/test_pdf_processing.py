from __future__ import annotations

import pytest

from ai_etl.pdf_processing import _clean_pdf_pages, sanitize_text, sliding_window_chunk


def test_clean_pdf_pages_removes_repeated_page_furniture_and_noise() -> None:
    pages = [
        "Smart Library PDF\n1\nMachine learn-\ning models infer patterns from labelled examples.\nConfidential footer",
        "Smart Library PDF\n2\nRegression and classification workflows evaluate unseen data.\nConfidential footer",
        "Smart Library PDF\n3\nFeature engineering improves model quality and validation.\nConfidential footer",
    ]

    cleaned = _clean_pdf_pages(pages)

    assert "Smart Library PDF" not in cleaned
    assert "Confidential footer" not in cleaned
    assert "\x00" not in cleaned
    assert "Machine learning models" in cleaned
    assert "Regression and classification" in cleaned


def test_sanitize_text_removes_nulls_and_collapses_whitespace() -> None:
    assert sanitize_text("  A\x00I\n\n  service\tmetadata  ") == "AI service metadata"


def test_sliding_window_chunk_creates_stable_ids_and_overlap() -> None:
    chunks = sliding_window_chunk("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=3)

    assert [chunk.chunk_id for chunk in chunks] == ["c00001", "c00002", "c00003", "c00004"]
    assert [chunk.text for chunk in chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]


def test_sliding_window_chunk_rejects_invalid_window_parameters() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller"):
        sliding_window_chunk("abc", chunk_size=3, overlap=3)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        sliding_window_chunk("abc", chunk_size=0, overlap=0)
