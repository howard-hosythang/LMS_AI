from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import Any

import requests


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

    def reduce_publication(self, representative_chunks: list[str]) -> dict[str, Any]:
        models_to_try = [self.cheap_model]
        if self.premium_model and self.premium_model != self.cheap_model:
            models_to_try.append(self.premium_model)

        last_error: Exception | None = None
        for model in models_to_try:
            try:
                return self._reduce_once(model, representative_chunks)
            except Exception as exc:
                last_error = exc

        raise LLMResponseError(f"Reduce failed on all model routes: {last_error}")

    def _reduce_once(self, model: str, representative_chunks: list[str]) -> dict[str, Any]:
        faculty_values = ", ".join(sorted(FACULTY_TARGETS))
        prompt = (
            "Bạn là một thủ thư đại học chuyên phân loại sách.\n"
            "Bạn nhận được metadata và các đoạn văn đại diện đã trích xuất từ PDF.\n"
            "Hãy bỏ qua nhiễu OCR/PDF như số trang, header/footer, mục lục, công thức rời rạc, "
            "bảng tài liệu tham khảo, đoạn code lẻ, ký hiệu toán học không đủ ngữ cảnh.\n"
            "Hãy tổng hợp thành metadata toàn cục và xuất đúng 1 JSON object duy nhất.\n"
            "JSON bắt buộc có đúng các trường:\n"
            "- master_summary: tiếng Việt tự nhiên, 100-160 từ, 3-5 câu, nêu đúng chủ đề chính, "
            "không chép nguyên văn đoạn PDF, không mở đầu bằng Title/Description/Authors/Categories, "
            "không đưa công thức hoặc ký hiệu toán rời rạc.\n"
            "- audience: đúng 1 mã khoa phù hợp nhất trong danh sách sau, không được mô tả tự do: "
            f"{faculty_values}\n"
            "- tags: mảng chính xác 5 tag phân loại người dùng nhìn thấy được, mỗi tag 2-4 từ nếu có thể, "
            "không trùng nhau, không dùng mã khoa, không dùng dấu gạch dưới, không dùng tag quá chung như "
            "Sách, Giáo trình, Chương.\n"
            "Không thêm trường khác, không markdown, không giải thích.\n\n"
            "Danh sách đoạn văn đại diện:\n"
            f"{json.dumps(representative_chunks, ensure_ascii=False)}"
        )

        raw = self._call(prompt, model=model)
        payload = self.extract_json_payload(raw)
        if not isinstance(payload, dict):
            raise LLMResponseError("Reduce response is not a JSON object")

        master_summary = str(payload.get("master_summary", "")).strip()
        audience = self._coerce_audience(str(payload.get("audience", "")).strip(), master_summary)
        tags = payload.get("tags", [])

        if not master_summary or not audience or not isinstance(tags, list):
            raise LLMResponseError("Reduce response missing required fields")

        clean_tags = [self._clean_tag(str(tag)) for tag in tags if str(tag).strip()]
        clean_tags = [tag for tag in clean_tags if tag]
        clean_tags = list(dict.fromkeys(clean_tags))

        if len(clean_tags) != 5:
            raise LLMResponseError("Reduce response must contain exactly 5 unique tags")

        return {
            "master_summary": master_summary,
            "audience": audience,
            "tags": clean_tags,
        }

    @staticmethod
    def _clean_tag(tag: str) -> str:
        tag = re.sub(r"[_\s]+", " ", tag).strip()
        tag = re.sub(r"[^\wÀ-ỹ+#.\- ]+", "", tag, flags=re.UNICODE)
        tag = re.sub(r"\s+", " ", tag).strip(" -_.")
        if not tag:
            return ""
        if tag.isupper() or tag.islower():
            tag = tag.title()
        return tag[:50]

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
