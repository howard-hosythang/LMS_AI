from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import Any

import requests

from .tag_quality import (
    align_english_tags,
    clean_ai_tag,
    dedupe_quality_tags,
    is_low_quality_tag,
    normalize_key,
    suggest_tags_from_text,
)


class LLMResponseError(RuntimeError):
    pass


FACULTY_TARGETS = {
    "KHOA_KHOA_HOC_VA_KY_THUAT_MAY_TINH",
    "KHOA_DIEN_DIEN_TU",
    "KHOA_CO_KHI",
    "KHOA_KY_THUAT_HOA_HOC",
    "KHOA_KY_THUAT_XAY_DUNG",
    "KHOA_KY_THUAT_GIAO_THONG",
    "KHOA_QUAN_LY_CONG_NGHIEP",
    "KHOA_MOI_TRUONG_VA_TAI_NGUYEN",
    "KHOA_CONG_NGHE_VAT_LIEU",
    "KHOA_KHOA_HOC_UNG_DUNG",
    "KHOA_KY_THUAT_DIA_CHAT_VA_DAU_KHI",
}


class GeminiClient:
    _throttle_lock = threading.Lock()
    _last_request_started_at = 0.0

    def __init__(
        self,
        api_key: str,
        cheap_model: str,
        premium_model: str,
        timeout_seconds: int,
        retry_attempts: int,
        min_request_interval_seconds: float = 1.2,
    ) -> None:
        self.api_key = api_key
        self.cheap_model = cheap_model
        self.premium_model = premium_model
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.min_request_interval_seconds = max(0.0, float(min_request_interval_seconds))

    def _throttle(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return

        with self._throttle_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_started_at
            wait_seconds = self.min_request_interval_seconds - elapsed
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_request_started_at = time.monotonic()

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            return True
        if isinstance(exc, requests.HTTPError):
            return getattr(getattr(exc, "response", None), "status_code", None) == 429
        return False

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        if response is None:
            return None

        retry_after = response.headers.get("Retry-After")
        if not retry_after:
            return None

        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            return None

    def _call(self, prompt: str, model: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                self._throttle()
                response = requests.post(url, json=payload, timeout=self.timeout_seconds)
                response.raise_for_status()
                body = response.json()
                return self._extract_text(body)
            except Exception as exc:
                last_error = exc
                if attempt == self.retry_attempts:
                    break
                # Exponential backoff with jitter to reduce thundering-herd retries.
                backoff = min(60.0, 2 ** (attempt - 1))
                if self._is_rate_limit_error(exc):
                    retry_after = self._retry_after_seconds(exc)
                    if retry_after is not None:
                        backoff = max(backoff, retry_after)
                    else:
                        backoff = max(backoff, 5.0)
                time.sleep(backoff + random.uniform(0.0, 0.5))

        raise LLMResponseError(f"LLM API failed after {self.retry_attempts} attempts: {last_error}")

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        candidates = body.get("candidates") or []
        if not candidates:
            raise LLMResponseError("LLM response has no candidates")

        parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
        if not parts:
            raise LLMResponseError("LLM response has no content parts")

        text = parts[0].get("text")
        if not text:
            raise LLMResponseError("LLM response has empty text")
        return text

    @staticmethod
    def extract_json_payload(raw_text: str) -> Any:
        """Recover JSON if model returns wrapped markdown or extra text."""
        stripped = raw_text.strip()

        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?", "", stripped)
            stripped = re.sub(r"```$", "", stripped).strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
            if not match:
                raise LLMResponseError("Model output is not valid JSON")
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                raise LLMResponseError("Model output JSON is malformed") from exc

    def reduce_publication(
        self,
        representative_chunks: list[str],
        publication_context: str = "",
    ) -> dict[str, Any]:
        models_to_try = [self.cheap_model]
        if self.premium_model and self.premium_model != self.cheap_model:
            models_to_try.append(self.premium_model)

        last_error: Exception | None = None
        for model in models_to_try:
            try:
                return self._reduce_once(model, representative_chunks, publication_context)
            except Exception as exc:
                last_error = exc

        raise LLMResponseError(f"Reduce failed on all model routes: {last_error}")

    def translate_catalog_metadata(
        self,
        publication_context: str,
        ai_summary_vi: str,
        tags: list[str],
    ) -> dict[str, Any]:
        models_to_try = [self.cheap_model]
        if self.premium_model and self.premium_model != self.cheap_model:
            models_to_try.append(self.premium_model)

        last_error: Exception | None = None
        for model in models_to_try:
            try:
                return self._translate_catalog_metadata_once(
                    model,
                    publication_context,
                    ai_summary_vi,
                    tags,
                )
            except Exception as exc:
                last_error = exc

        raise LLMResponseError(f"Translation failed on all model routes: {last_error}")

    def translate_terms(self, terms: list[str], target_language: str = "English") -> list[str]:
        clean_terms = [str(term).strip() for term in terms if str(term).strip()]
        if not clean_terms:
            return []

        prompt = (
            "You are translating academic library category/tag names.\n"
            f"Translate each term to {target_language}. Preserve technical acronyms and proper nouns. "
            "Return exactly one JSON array with the same length and order as the input. "
            "No markdown, no explanation.\n\n"
            f"TERMS:\n{json.dumps(clean_terms, ensure_ascii=False)}"
        )
        raw = self._call(prompt, model=self.cheap_model)
        payload = self.extract_json_payload(raw)
        if not isinstance(payload, list):
            raise LLMResponseError("Term translation response is not a JSON array")
        translated = [str(item).strip()[:150] for item in payload]
        if len(translated) != len(clean_terms):
            raise LLMResponseError("Term translation response length mismatch")
        return translated

    def _translate_catalog_metadata_once(
        self,
        model: str,
        publication_context: str,
        ai_summary_vi: str,
        tags: list[str],
    ) -> dict[str, Any]:
        prompt = (
            "You are a professional academic library metadata translator.\n"
            "Translate the catalogue metadata to English for an English UI. Preserve proper nouns, "
            "book titles that are already English, author names, ISBN-like tokens, technical terms, "
            "and code/library names. Do not add facts.\n"
            "Return exactly one JSON object with these fields:\n"
            "- title: English title or original title if already English.\n"
            "- subtitle: English subtitle, null if unavailable.\n"
            "- description: fluent English translation of the description, null if unavailable.\n"
            "- ai_summary: fluent English translation of the AI content summary.\n"
            "- tags: array with exactly the same number of tags as input, translated to concise English search tags.\n"
            "No markdown, no explanation.\n\n"
            "CATALOG_CONTEXT:\n"
            f"{publication_context or '(none)'}\n\n"
            "AI_SUMMARY_VI:\n"
            f"{ai_summary_vi}\n\n"
            "TAGS:\n"
            f"{json.dumps(tags, ensure_ascii=False)}"
        )

        raw = self._call(prompt, model=model)
        payload = self.extract_json_payload(raw)
        if not isinstance(payload, dict):
            raise LLMResponseError("Translation response is not a JSON object")

        translated_tags = payload.get("tags", [])
        if not isinstance(translated_tags, list):
            raise LLMResponseError("Translation response tags must be an array")

        return {
            "title": self._optional_string(payload.get("title")),
            "subtitle": self._optional_string(payload.get("subtitle")),
            "description": self._optional_string(payload.get("description")),
            "ai_summary": self._optional_string(payload.get("ai_summary")),
            "tags": [str(tag).strip()[:100] for tag in translated_tags if str(tag).strip()],
        }

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _reduce_once(
        self,
        model: str,
        representative_chunks: list[str],
        publication_context: str = "",
    ) -> dict[str, Any]:
        faculty_values = ", ".join(sorted(FACULTY_TARGETS))
        prompt = (
            "Bạn là thủ thư đại học và chuyên viên biên mục học thuật. Nhiệm vụ là đọc các "
            "đoạn nội dung trích từ PDF và tạo metadata có thể dùng thật trong thư viện.\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. master_summary chỉ được dựa trên CONTENT_EXCERPTS. Không được ghép Title, Authors, "
            "Categories hoặc Description thành summary. CATALOG_CONTEXT chỉ dùng để khử nhập nhằng "
            "và chọn audience/tag khi nội dung PDF đã hỗ trợ nhận định đó.\n"
            "2. Summary phải nói về nội dung học thuật bên trong tài liệu: khái niệm, phương pháp, "
            "phạm vi kiến thức, kỹ năng người đọc học được. Không mô tả hệ thống AI, lập chỉ mục, "
            "metadata, thư viện, file PDF hoặc quy trình xử lý dữ liệu.\n"
            "3. Nếu excerpt không đủ bằng chứng cho một ý, không đưa ý đó vào summary.\n"
            "4. Category là nhóm bao phủ rộng; tag là nhãn chủ đề/kỹ năng/công nghệ cụ thể hơn category. "
            "Không dùng tên khoa, tên category rộng, tên sách, tên tác giả hoặc publisher làm tag.\n"
            "5. Bỏ qua nhiễu OCR/PDF như số trang, header/footer, mục lục, tài liệu tham khảo, "
            "copyright, công thức/code rời rạc không đủ ngữ cảnh.\n"
            "6. Tag bắt buộc viết bằng tiếng Việt, là cụm danh từ tìm kiếm thực tế, 2-4 từ nếu có thể, "
            "tối đa 50 ký tự, không quá chung như Sách, Giáo Trình, Công Nghệ, Kỹ Thuật, "
            "Science, Result, Numbers.\n"
            "7. tags_en là bản tiếng Anh tương ứng từng tag tiếng Việt, cùng thứ tự, dùng thuật ngữ "
            "người học hay tìm kiếm như Web Development, Machine Learning, Database Normalization. "
            "Không dịch máy từng chữ nếu thuật ngữ chuyên ngành tiếng Anh đã có cách gọi chuẩn.\n"
            "Hãy tổng hợp thành metadata toàn cục và xuất đúng 1 JSON object duy nhất.\n"
            "JSON bắt buộc có đúng các trường:\n"
            "- master_summary: tiếng Việt tự nhiên, 120-180 từ, 4-6 câu, nêu đúng nội dung chính, "
            "phạm vi kiến thức, cách tiếp cận và giá trị học tập; không mở đầu bằng tên sách/tác giả, "
            "không chép nguyên văn excerpt dài.\n"
            "- audience: đúng 1 mã khoa phù hợp nhất trong danh sách sau, không được mô tả tự do: "
            f"{faculty_values}\n"
            "- tags: mảng chính xác 5 tag tiếng Việt tốt nhất, không trùng nhau; ưu tiên thuật ngữ "
            "chuyên môn người dùng có khả năng bấm/tìm kiếm, ví dụ Học Máy, Chuẩn Hóa Dữ Liệu, "
            "Lập Trình Python, Mạch Điện Tử, Quản Lý Dự Án.\n"
            "- tags_en: mảng chính xác 5 tag tiếng Anh tương ứng 1-1 với tags, không trùng nhau.\n"
            "Không thêm trường khác, không markdown, không giải thích.\n\n"
            "CATALOG_CONTEXT:\n"
            f"{publication_context or '(không có)'}\n\n"
            "CONTENT_EXCERPTS:\n"
            f"{json.dumps(representative_chunks, ensure_ascii=False)}"
        )

        raw = self._call(prompt, model=model)
        payload = self.extract_json_payload(raw)
        if not isinstance(payload, dict):
            raise LLMResponseError("Reduce response is not a JSON object")

        master_summary = str(payload.get("master_summary", "")).strip()
        audience = self._coerce_audience(str(payload.get("audience", "")).strip(), master_summary)
        tags = payload.get("tags", [])
        tags_en = payload.get("tags_en", [])

        if not master_summary or not audience or not isinstance(tags, list):
            raise LLMResponseError("Reduce response missing required fields")

        raw_tags = [str(tag) for tag in tags if str(tag).strip()]
        raw_tags_en = [str(tag) for tag in tags_en if str(tag).strip()] if isinstance(tags_en, list) else []
        clean_tags: list[str] = []
        clean_tags_en_candidates: list[str] = []
        seen_tag_keys: set[str] = set()
        for index, raw_tag in enumerate(raw_tags):
            clean_tag = clean_ai_tag(raw_tag)
            tag_key = normalize_key(clean_tag)
            if (
                not clean_tag
                or tag_key in seen_tag_keys
                or is_low_quality_tag(clean_tag, publication_context)
            ):
                continue
            clean_tags.append(clean_tag)
            clean_tags_en_candidates.append(raw_tags_en[index] if index < len(raw_tags_en) else "")
            seen_tag_keys.add(tag_key)
            if len(clean_tags) >= 5:
                break

        for fallback_tag in self._fallback_tags_from_text(
            " ".join([master_summary, *representative_chunks]),
            publication_context,
        ):
            if len(clean_tags) >= 5:
                break
            if fallback_tag.lower() not in {tag.lower() for tag in clean_tags}:
                clean_tags.append(fallback_tag)
                clean_tags_en_candidates.append("")

        for fallback_tag in dedupe_quality_tags(
            [
                "Kiến Thức Nền Tảng",
                "Phương Pháp Thực Hành",
                "Ứng Dụng Thực Tế",
                "Bài Toán Chuyên Ngành",
                "Tư Duy Phân Tích",
            ],
            publication_context,
            limit=5,
        ):
            if len(clean_tags) >= 5:
                break
            if fallback_tag.lower() not in {tag.lower() for tag in clean_tags}:
                clean_tags.append(fallback_tag)
                clean_tags_en_candidates.append("")

        if len(clean_tags) != 5:
            raise LLMResponseError("Reduce response must contain exactly 5 useful unique tags")

        if self._looks_like_catalog_summary(master_summary):
            raise LLMResponseError("Reduce summary appears to be catalog metadata, not content")

        clean_tags_en = align_english_tags(
            clean_tags,
            clean_tags_en_candidates,
        )

        return {
            "master_summary": master_summary,
            "audience": audience,
            "tags": clean_tags[:5],
            "tags_en": clean_tags_en,
        }

    @staticmethod
    def _fallback_tags_from_text(text: str, publication_context: str = "") -> list[str]:
        return suggest_tags_from_text(text, publication_context, limit=5)

    @staticmethod
    def _looks_like_catalog_summary(summary: str) -> bool:
        lowered = summary.lower()
        blocked_phrases = (
            "publication id:",
            "language:",
            "publication year:",
            "authors:",
            "categories:",
            "tags:",
            "ai summary:",
            "hệ thống ai",
            "lập chỉ mục",
            "metadata",
            "thuộc nhóm",
            "các chủ đề nổi bật gồm",
            "file pdf",
            "thư viện",
        )
        return any(phrase in lowered for phrase in blocked_phrases)

    @staticmethod
    def _coerce_audience(raw_audience: str, summary: str) -> str:
        if raw_audience in FACULTY_TARGETS:
            return raw_audience

        text = f"{raw_audience} {summary}".lower()
        keyword_map = [
            ("KHOA_KHOA_HOC_VA_KY_THUAT_MAY_TINH", ("computer", "software", "algorithm", "data", "ai", "machine learning", "lập trình", "thuật toán", "cơ sở dữ liệu", "máy tính")),
            ("KHOA_DIEN_DIEN_TU", ("điện", "điện tử", "mạch", "signal", "circuit", "electronics")),
            ("KHOA_CO_KHI", ("cơ khí", "mechanical", "machine design", "manufacturing")),
            ("KHOA_KY_THUAT_HOA_HOC", ("hóa", "chemical", "chemistry")),
            ("KHOA_KY_THUAT_XAY_DUNG", ("xây dựng", "construction", "civil")),
            ("KHOA_KY_THUAT_GIAO_THONG", ("giao thông", "transport", "traffic")),
            ("KHOA_QUAN_LY_CONG_NGHIEP", ("quản lý", "management", "industrial")),
            ("KHOA_MOI_TRUONG_VA_TAI_NGUYEN", ("môi trường", "environment", "resource")),
            ("KHOA_CONG_NGHE_VAT_LIEU", ("vật liệu", "material")),
            ("KHOA_KHOA_HOC_UNG_DUNG", ("toán", "physics", "applied", "ứng dụng")),
            ("KHOA_KY_THUAT_DIA_CHAT_VA_DAU_KHI", ("địa chất", "dầu khí", "geology", "petroleum")),
        ]
        for faculty, keywords in keyword_map:
            if any(keyword in text for keyword in keywords):
                return faculty
        return "KHOA_KHOA_HOC_VA_KY_THUAT_MAY_TINH"
