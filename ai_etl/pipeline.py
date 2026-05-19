from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from pathlib import Path
from typing import Union

import numpy as np
from psycopg2.extensions import connection as PGConnection

from .config import Settings
from .db import Database
from .embedding_client import EmbeddingClient
from .llm_client import GeminiClient
from .pdf_processing import Chunk, extract_pdf_text, sliding_window_chunk
from .tag_quality import align_english_tags, dedupe_quality_tags, suggest_tags_from_text


class AIEtlPipeline:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        llm_client: GeminiClient,
        embedding_client: EmbeddingClient,
    ) -> None:
        self.settings = settings
        self.database = database
        self.llm_client = llm_client
        self.embedding_client = embedding_client

    def run(
        self,
        publication_id: int,
        pdf_path: Union[Path, str],
        conn: PGConnection | None = None,
        force_reprocess: bool = False,
    ) -> dict[str, object]:
        file_hash = self._calculate_file_hash(pdf_path)
        if conn is not None:
            return self._run_with_connection(conn, publication_id, pdf_path, file_hash, force_reprocess)

        with self.database.connect() as conn:
            return self._run_with_connection(conn, publication_id, pdf_path, file_hash, force_reprocess)

    def _run_with_connection(
        self,
        conn: PGConnection,
        publication_id: int,
        pdf_path: Union[Path, str],
        file_hash: str,
        force_reprocess: bool = False,
    ) -> dict[str, object]:
        self.database.ensure_etl_metadata_table(conn)
        self.database.mark_publication_etl_running(conn, publication_id, file_hash)
        conn.commit()

        try:
            full_text = extract_pdf_text(pdf_path)
            chunks = sliding_window_chunk(
                text=full_text,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            )
            if not chunks:
                raise ValueError("No valid text chunks extracted from PDF")

            if self.settings.enable_hash_skip and not force_reprocess:
                existing_hash = self.database.get_publication_file_hash(conn, publication_id)
                has_outputs = self.database.publication_has_ai_outputs(conn, publication_id)
                if existing_hash == file_hash and has_outputs:
                    self.database.upsert_publication_file_hash(
                        conn,
                        publication_id,
                        file_hash,
                        chunks_count=len(chunks),
                        vectors_count=0,
                    )
                    conn.commit()
                    return {
                        "publication_id": publication_id,
                        "chunks": len(chunks),
                        "vectors": 0,
                        "skipped": True,
                        "skip_reason": "file_hash_unchanged",
                    }
        except Exception as exc:
            conn.rollback()
            self.database.mark_publication_etl_failed(conn, publication_id, file_hash, str(exc))
            conn.commit()
            raise

        self.database.clear_previous_ai_data(conn, publication_id)
        conn.commit()

        try:
            chunk_texts = [chunk.text for chunk in chunks]
            vectors = self.embedding_client.encode(chunk_texts)

            vector_rows = list(zip(chunk_texts, vectors))
            for batch_rows in self._iter_batches(vector_rows, self.settings.vector_insert_batch_size):
                self.database.bulk_insert_vectors(conn, publication_id, batch_rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        representative_chunks = self._select_representative_chunks(
            chunks=chunks,
            vectors=vectors,
            max_chunks=self.settings.reduce_max_chunks,
        )
        reduce_inputs = self._apply_reduce_budget(representative_chunks)
        if not reduce_inputs:
            raise ValueError("No representative chunks available for reduce phase")

        publication_context = self.database.get_publication_context(conn, publication_id)
        reduce_result = self._reduce_with_fallback(reduce_inputs, publication_context)

        try:
            self.database.update_publication_summary(
                conn,
                publication_id,
                reduce_result["master_summary"],
                reduce_result["audience"],
            )

            tag_ids = [
                self.database.find_or_create_tag_id(conn, tag_name)
                for tag_name in reduce_result["tags"]
            ]
            self.database.insert_publication_tag_links(conn, publication_id, tag_ids)
            self.database.upsert_tag_translations(
                conn,
                tag_ids=tag_ids,
                translated_names=align_english_tags(
                    [str(tag) for tag in reduce_result["tags"]],
                    [str(tag) for tag in reduce_result.get("tags_en", [])],
                ),
                language_code="en",
            )
            translation_generated = False
            if self.settings.enable_english_translation:
                translation_generated = self._try_upsert_english_metadata(
                    conn,
                    publication_id,
                    publication_context,
                    str(reduce_result["master_summary"]),
                    [str(tag) for tag in reduce_result["tags"]],
                    tag_ids,
                )
            self.database.upsert_publication_file_hash(
                conn,
                publication_id,
                file_hash,
                chunks_count=len(chunks),
                vectors_count=len(vectors),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        return {
            "publication_id": publication_id,
            "chunks": len(chunks),
            "vectors": len(vectors),
            "reduce_inputs": len(reduce_inputs),
            "skipped": False,
            "master_summary": reduce_result["master_summary"],
            "audience": reduce_result["audience"],
            "tags": reduce_result["tags"],
            "translation_generated": translation_generated,
        }

    def _try_upsert_english_metadata(
        self,
        conn: PGConnection,
        publication_id: int,
        publication_context: str,
        ai_summary_vi: str,
        tags: list[str],
        tag_ids: list[int],
    ) -> bool:
        try:
            translation = self.llm_client.translate_catalog_metadata(
                publication_context=publication_context,
                ai_summary_vi=ai_summary_vi,
                tags=tags,
            )
            self.database.upsert_publication_translation(
                conn,
                publication_id=publication_id,
                language_code="en",
                title=translation.get("title"),
                subtitle=translation.get("subtitle"),
                description=translation.get("description"),
                ai_summary=translation.get("ai_summary"),
            )
            self.database.upsert_tag_translations(
                conn,
                tag_ids=tag_ids,
                translated_names=translation.get("tags", []),
                language_code="en",
            )
            return True
        except Exception:
            return False

    def _reduce_with_fallback(
        self,
        reduce_inputs: list[str],
        publication_context: str = "",
    ) -> dict[str, object]:
        try:
            return self.llm_client.reduce_publication(reduce_inputs, publication_context)
        except Exception:
            return self._fallback_reduce_publication(reduce_inputs, publication_context)

    def _fallback_reduce_publication(
        self,
        reduce_inputs: list[str],
        publication_context: str = "",
    ) -> dict[str, object]:
        context = self._parse_publication_context(publication_context)
        joined_text = self._clean_summary_source(" ".join(reduce_inputs))
        fallback_tags = self._extract_fallback_tags(joined_text, context, publication_context)
        summary = self._extractive_content_summary(joined_text)

        if not summary:
            summary = (
                "Chưa đủ nội dung trích xuất đáng tin cậy để tạo tóm tắt học thuật. "
                "Cần kiểm tra lại file PDF hoặc chạy lại xử lý với bản tài liệu có văn bản đọc được."
            )

        return {
            "master_summary": summary,
            "audience": self._infer_faculty(" ".join([joined_text, publication_context])),
            "tags": fallback_tags,
            "tags_en": align_english_tags(fallback_tags),
        }

    @staticmethod
    def _clean_summary_source(text: str) -> str:
        text = re.sub(r"(?i)\b(title|subtitle|description|authors|categories):", " ", text)
        text = re.sub(r"\.{4,}", " ", text)
        text = re.sub(r"\b\d+\s*/\s*\d+\b", " ", text)
        text = re.sub(r"\bpage\s+\d+\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        useful = []
        noise_terms = ("contents", "table of contents", "references", "bibliography", "copyright")
        for sentence in sentences:
            lowered = sentence.lower()
            symbol_ratio = sum(1 for ch in sentence if ch in "=∑∫√≤≥<>|{}[]") / max(len(sentence), 1)
            if len(sentence) < 40 or any(term in lowered for term in noise_terms):
                continue
            if symbol_ratio > 0.06:
                continue
            useful.append(sentence)
        return " ".join(useful[:8]) or text

    @staticmethod
    def _extractive_content_summary(text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        blocked_terms = (
            "title:",
            "subtitle:",
            "description:",
            "authors:",
            "categories:",
            "copyright",
            "references",
            "bibliography",
            "table of contents",
            "contents",
        )
        candidates: list[str] = []
        for sentence in sentences:
            clean = re.sub(r"\s+", " ", sentence).strip()
            lowered = clean.lower()
            if len(clean) < 70 or len(clean) > 360:
                continue
            if any(term in lowered for term in blocked_terms):
                continue
            alpha_ratio = sum(ch.isalpha() for ch in clean) / max(len(clean), 1)
            symbol_ratio = sum(1 for ch in clean if ch in "=∑∫√≤≥<>|{}[]") / max(len(clean), 1)
            if alpha_ratio < 0.55 or symbol_ratio > 0.04:
                continue
            candidates.append(clean)
            if len(candidates) >= 5:
                break

        if not candidates:
            return ""

        summary = " ".join(candidates)
        if len(summary) > 950:
            summary = summary[:950].rsplit(" ", 1)[0].rstrip(" .,;:") + "."
        return summary

    @staticmethod
    def _parse_publication_context(text: str) -> dict[str, str]:
        context: dict[str, str] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if value:
                context[key] = value
        return context

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = unicodedata.normalize("NFD", text.lower())
        without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        return without_marks.replace("đ", "d")

    def _infer_faculty(self, text: str) -> str:
        normalized = self._normalize_text(text)
        keyword_map = [
            ("KHOA_KHOA_HOC_VA_KY_THUAT_MAY_TINH", ("computer", "software", "algorithm", "data", "program", "code", "lap trinh", "thuat toan", "co so du lieu")),
            ("KHOA_DIEN_DIEN_TU", ("dien", "dien tu", "circuit", "electronics", "signal")),
            ("KHOA_CO_KHI", ("co khi", "mechanical", "manufacturing")),
            ("KHOA_KY_THUAT_HOA_HOC", ("hoa hoc", "chemical", "chemistry")),
            ("KHOA_KY_THUAT_XAY_DUNG", ("xay dung", "construction", "civil")),
            ("KHOA_KY_THUAT_GIAO_THONG", ("giao thong", "transport", "traffic")),
            ("KHOA_QUAN_LY_CONG_NGHIEP", ("quan ly", "management", "industrial")),
            ("KHOA_MOI_TRUONG_VA_TAI_NGUYEN", ("moi truong", "environment", "resource")),
            ("KHOA_CONG_NGHE_VAT_LIEU", ("vat lieu", "material")),
            ("KHOA_KHOA_HOC_UNG_DUNG", ("toan", "physics", "applied", "ung dung")),
            ("KHOA_KY_THUAT_DIA_CHAT_VA_DAU_KHI", ("dia chat", "dau khi", "geology", "petroleum")),
        ]
        for faculty, keywords in keyword_map:
            if any(keyword in normalized for keyword in keywords):
                return faculty
        return "KHOA_KHOA_HOC_VA_KY_THUAT_MAY_TINH"

    def _extract_fallback_tags(
        self,
        text: str,
        context: dict[str, str] | None = None,
        publication_context: str = "",
    ) -> list[str]:
        context = context or {}
        source = " ".join([
            context.get("title", ""),
            context.get("subtitle", ""),
            context.get("description", ""),
            context.get("categories", ""),
            text,
        ])
        tags = suggest_tags_from_text(source, publication_context, limit=5)
        if len(tags) >= 5:
            return tags[:5]

        backfill = dedupe_quality_tags(
            [
                "Nội Dung Chuyên Ngành",
                "Kiến Thức Nền Tảng",
                "Phương Pháp Thực Hành",
                "Ứng Dụng Thực Tế",
                "Tư Duy Kỹ Thuật",
            ],
            publication_context,
            limit=5,
        )
        for tag in backfill:
            if len(tags) >= 5:
                break
            if tag.lower() not in {existing.lower() for existing in tags}:
                tags.append(tag)
        return tags[:5]

    @staticmethod
    def _calculate_file_hash(pdf_source: Union[Path, str]) -> str:
        """Calculate hash of PDF file or URL content.

        For URLs, hash the URL string itself for efficiency (don't download twice).
        For local files, hash the file content.
        """
        digest = hashlib.sha256()

        if isinstance(pdf_source, str) and (pdf_source.startswith('http://') or pdf_source.startswith('https://')):
            # Hash the URL string for remote files (efficient, avoids re-downloading)
            digest.update(pdf_source.encode('utf-8'))
        else:
            # Hash file content for local files
            pdf_path = Path(pdf_source) if isinstance(pdf_source, str) else pdf_source
            with pdf_path.open("rb") as file_obj:
                for block in iter(lambda: file_obj.read(1024 * 1024), b""):
                    digest.update(block)

        return digest.hexdigest()

    def _select_representative_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        max_chunks: int,
    ) -> list[str]:
        if not chunks:
            return []

        embeddings = np.array(vectors, dtype=np.float32)
        total_chunks = len(chunks)
        target = min(max_chunks, max(8, int(math.sqrt(total_chunks) * 2)), total_chunks)

        centroid = embeddings.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            return [chunk.text for chunk in chunks[:target]]
        centroid = centroid / centroid_norm

        similarities = embeddings @ centroid
        selected: list[int] = []
        for seed_idx in range(min(2, total_chunks)):
            if seed_idx not in selected:
                selected.append(seed_idx)

        centroid_idx = int(np.argmax(similarities))
        if centroid_idx not in selected:
            selected.append(centroid_idx)
        candidate_indices = set(range(total_chunks))
        candidate_indices.difference_update(selected)

        lambda_weight = 0.65
        while len(selected) < target and candidate_indices:
            best_idx = None
            best_score = float("-inf")

            for idx in candidate_indices:
                relevance = float(similarities[idx])
                diversity_penalty = max(float(embeddings[idx] @ embeddings[s]) for s in selected)
                score = lambda_weight * relevance - (1.0 - lambda_weight) * diversity_penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            selected.append(best_idx)
            candidate_indices.remove(best_idx)

        selected.sort()
        return [chunks[idx].text for idx in selected]

    def _apply_reduce_budget(self, texts: list[str]) -> list[str]:
        selected: list[str] = []
        total_chars = 0

        for text in texts:
            clipped = self._clean_summary_source(text)[: self.settings.reduce_max_chars_per_chunk].strip()
            if not clipped:
                continue

            if total_chars + len(clipped) > self.settings.reduce_total_chars_budget:
                break

            selected.append(clipped)
            total_chars += len(clipped)

        return selected

    @staticmethod
    def _iter_batches(items: list, batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]
