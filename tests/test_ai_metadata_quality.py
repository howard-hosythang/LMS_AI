from __future__ import annotations

import sys
import types

sys.modules.setdefault("fitz", types.SimpleNamespace())

from ai_etl.pipeline import AIEtlPipeline
from ai_etl.llm_client import GeminiClient
from ai_etl.tag_quality import align_english_tags, dedupe_quality_tags, is_low_quality_tag


def test_fallback_metadata_prefers_specific_catalog_tags() -> None:
    pipeline = AIEtlPipeline.__new__(AIEtlPipeline)

    context = {
        "title": "Database System Concepts",
        "description": "SQL, relational design, normalization, indexing, transactions, concurrency control and recovery.",
        "categories": "Cơ Sở Dữ Liệu, Công Nghệ Thông Tin",
    }

    tags = pipeline._extract_fallback_tags(
        "relational database sql transaction normalization indexing query processing recovery",
        context,
        "Title: Database System Concepts\nCategories: Cơ Sở Dữ Liệu, Công Nghệ Thông Tin",
    )

    assert "SQL" in tags
    assert "Chuẩn Hóa Dữ Liệu" in tags
    assert "Giao Dịch Dữ Liệu" in tags
    assert "Cơ Sở Dữ Liệu" not in tags
    assert "Công Nghệ" not in tags
    assert "Database System Concepts" not in tags
    assert len(tags) == 5


def test_llm_tag_padding_keeps_specific_tags() -> None:
    tags = GeminiClient._fallback_tags_from_text(
        "Tài liệu trình bày machine learning, deep learning, neural network và dữ liệu."
    )

    assert "Học Máy" in tags
    assert "Học Sâu" in tags
    assert "Khoa Học Dữ Liệu" in tags
    assert len(tags) <= 5


def test_ai_tag_quality_filters_title_category_and_generic_terms() -> None:
    context = "Title: Machine Learning cơ bản\nAuthors: Vũ Khắc Tiệp\nCategories: Công Nghệ Thông Tin, Lập Trình"

    tags = dedupe_quality_tags(
        [
            "Machine Learning cơ bản",
            "Công Nghệ Thông Tin",
            "Publication",
            "Học Có Giám Sát",
            "Kỹ Thuật Đặc Trưng",
            "Đánh Giá Mô Hình",
        ],
        context,
        limit=5,
    )

    assert tags == ["Học Có Giám Sát", "Kỹ Thuật Đặc Trưng", "Đánh Giá Mô Hình"]
    assert is_low_quality_tag("Publication")


def test_fallback_summary_uses_pdf_content_not_catalog_fields() -> None:
    pipeline = AIEtlPipeline.__new__(AIEtlPipeline)

    result = pipeline._fallback_reduce_publication(
        [
            "Supervised learning studies how models infer a mapping from labelled examples and evaluate "
            "generalization on unseen data. The material explains regression, classification, feature "
            "engineering, model validation, overfitting control, and error analysis for practical prediction tasks. "
            "Later sections compare decision trees, linear models, support vector machines, and ensemble methods "
            "through training workflows and experimental evaluation."
        ],
        "Title: Machine Learning cơ bản\nAuthors: Vũ Khắc Tiệp\nCategories: Công Nghệ Thông Tin",
    )

    summary = str(result["master_summary"])

    assert "Machine Learning cơ bản của Vũ Khắc Tiệp" not in summary
    assert "thuộc nhóm" not in summary
    assert "lập chỉ mục" not in summary.lower()
    assert "regression" in summary or "classification" in summary
    assert len(result["tags"]) == 5
    assert result["tags_en"] == align_english_tags(result["tags"])


def test_tag_translations_keep_search_terms_in_english() -> None:
    assert align_english_tags(["Lập Trình Web", "Học Máy"]) == [
        "Web Development",
        "Machine Learning",
    ]
    assert align_english_tags(["Chuẩn Hóa Dữ Liệu"], ["Database Normalization"]) == [
        "Database Normalization",
    ]
