from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_etl.pdf_processing import Chunk
from ai_etl.pipeline import AIEtlPipeline


class FakeDatabase:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def upsert_publication_translation(self, *args, **kwargs) -> None:
        self.calls.append(("upsert_publication_translation", args, kwargs))

    def upsert_tag_translations(self, *args, **kwargs) -> None:
        self.calls.append(("upsert_tag_translations", args, kwargs))


class FakeLlm:
    def __init__(self, result=None, translation=None, should_fail: bool = False) -> None:
        self.result = result
        self.translation = translation
        self.should_fail = should_fail

    def reduce_publication(self, reduce_inputs, publication_context):
        if self.should_fail:
            raise RuntimeError("LLM unavailable")
        return self.result

    def translate_catalog_metadata(self, **kwargs):
        if self.should_fail:
            raise RuntimeError("translation unavailable")
        return self.translation


def _pipeline(settings=None, llm=None, database=None) -> AIEtlPipeline:
    pipeline = AIEtlPipeline.__new__(AIEtlPipeline)
    pipeline.settings = settings or SimpleNamespace(
        reduce_max_chunks=8,
        reduce_max_chars_per_chunk=20,
        reduce_total_chars_budget=42,
    )
    pipeline.llm_client = llm or FakeLlm()
    pipeline.database = database or FakeDatabase()
    return pipeline


def test_select_representative_chunks_keeps_opening_context_and_diverse_centroid() -> None:
    pipeline = _pipeline()
    chunks = [
        Chunk("c00001", "intro"),
        Chunk("c00002", "database normalization"),
        Chunk("c00003", "sql indexing"),
        Chunk("c00004", "deep learning"),
        Chunk("c00005", "transaction recovery"),
    ]
    vectors = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 1.0, 0.0],
        [0.7, 0.3, 0.0],
    ]

    selected = pipeline._select_representative_chunks(chunks, vectors, max_chunks=3)

    assert selected[0] == "intro"
    assert "database normalization" in selected
    assert len(selected) == 3


def test_apply_reduce_budget_cleans_and_stops_before_total_budget() -> None:
    pipeline = _pipeline(SimpleNamespace(reduce_max_chars_per_chunk=15, reduce_total_chars_budget=32))

    selected = pipeline._apply_reduce_budget([
        "Title: noisy catalog field. Real content about database normalization and SQL.",
        "Second sentence about indexing and transactions.",
        "Third sentence should not fit in the reduce budget.",
    ])

    assert selected == ["Real content ab", "Second sentence"]


def test_reduce_with_fallback_uses_llm_result_when_available() -> None:
    expected = {
        "master_summary": "summary",
        "audience": "KHOA_KHOA_HOC_VA_KY_THUAT_MAY_TINH",
        "tags": ["SQL", "Chuẩn Hóa Dữ Liệu", "Chỉ Mục Dữ Liệu", "Truy Vấn Dữ Liệu", "Giao Dịch Dữ Liệu"],
        "tags_en": ["SQL", "Database Normalization", "Database Indexing", "Database Queries", "Data Transactions"],
    }
    pipeline = _pipeline(llm=FakeLlm(result=expected))

    assert pipeline._reduce_with_fallback(["content"], "") == expected


def test_reduce_with_fallback_generates_quality_tags_when_llm_fails() -> None:
    pipeline = _pipeline(llm=FakeLlm(should_fail=True))

    result = pipeline._reduce_with_fallback(
        [
            "Supervised learning studies labelled examples, regression, classification, feature engineering, "
            "model validation, overfitting control and error analysis for prediction workflows."
        ],
        "Title: Machine Learning cơ bản\nCategories: Công Nghệ Thông Tin",
    )

    assert "regression" in result["master_summary"] or "classification" in result["master_summary"]
    assert len(result["tags"]) == 5
    assert result["tags_en"]
    assert result["audience"] == "KHOA_KHOA_HOC_VA_KY_THUAT_MAY_TINH"


def test_try_upsert_english_metadata_writes_optional_translation_and_tag_aliases() -> None:
    database = FakeDatabase()
    llm = FakeLlm(translation={
        "title": "Database Systems",
        "subtitle": None,
        "description": "Relational database design.",
        "ai_summary": "The document covers normalization and transactions.",
        "tags": ["SQL", "Database Normalization"],
    })
    pipeline = _pipeline(llm=llm, database=database)

    result = pipeline._try_upsert_english_metadata(
        conn=object(),
        publication_id=7,
        publication_context="Title: Hệ Cơ Sở Dữ Liệu",
        ai_summary_vi="Tài liệu trình bày chuẩn hóa dữ liệu.",
        tags=["SQL", "Chuẩn Hóa Dữ Liệu"],
        tag_ids=[10, 11],
    )

    assert result is True
    assert [call[0] for call in database.calls] == [
        "upsert_publication_translation",
        "upsert_tag_translations",
    ]
    assert database.calls[1][2]["translated_names"] == ["SQL", "Database Normalization"]


def test_try_upsert_english_metadata_is_best_effort() -> None:
    pipeline = _pipeline(llm=FakeLlm(should_fail=True), database=FakeDatabase())

    assert pipeline._try_upsert_english_metadata(object(), 1, "", "", [], []) is False


def test_calculate_file_hash_uses_content_for_files_and_url_string_for_remote_sources(tmp_path: Path) -> None:
    file_path = tmp_path / "book.pdf"
    file_path.write_bytes(b"same file content")

    assert AIEtlPipeline._calculate_file_hash(file_path) == AIEtlPipeline._calculate_file_hash(file_path)
    assert (
        AIEtlPipeline._calculate_file_hash("https://example.test/books/1.pdf")
        != AIEtlPipeline._calculate_file_hash("https://example.test/books/2.pdf")
    )
