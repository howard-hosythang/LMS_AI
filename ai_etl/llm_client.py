from __future__ import annotations

import json
import random
import re
import time
from typing import Any

import requests


class LLMResponseError(RuntimeError):
    pass


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        cheap_model: str,
        premium_model: str,
        timeout_seconds: int,
        retry_attempts: int,
    ) -> None:
        self.api_key = api_key
        self.cheap_model = cheap_model
        self.premium_model = premium_model
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts

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
                response = requests.post(url, json=payload, timeout=self.timeout_seconds)
                response.raise_for_status()
                body = response.json()
                return self._extract_text(body)
            except Exception as exc:
                last_error = exc
                if attempt == self.retry_attempts:
                    break
                # Exponential backoff with jitter to reduce thundering-herd retries.
                backoff = 2 ** (attempt - 1)
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
        prompt = (
            "Bạn là một thủ thư đại học chuyên phân loại sách.\n"
            "Bạn nhận được các đoạn văn đại diện của một cuốn sách.\n"
            "Hãy tổng hợp thành metadata toàn cục và xuất đúng 1 JSON object duy nhất.\n"
            "JSON bắt buộc có đúng các trường:\n"
            "- master_summary: tóm tắt tổng thể khoảng 150 từ\n"
            "- audience: mô tả nhóm độc giả mục tiêu\n"
            "- tags: mảng chính xác 5 từ khóa phân loại (không trùng nhau)\n"
            "Không thêm trường khác, không markdown, không giải thích.\n\n"
            "Danh sách đoạn văn đại diện:\n"
            f"{json.dumps(representative_chunks, ensure_ascii=False)}"
        )

        raw = self._call(prompt, model=model)
        payload = self.extract_json_payload(raw)
        if not isinstance(payload, dict):
            raise LLMResponseError("Reduce response is not a JSON object")

        master_summary = str(payload.get("master_summary", "")).strip()
        audience = str(payload.get("audience", "")).strip()
        tags = payload.get("tags", [])

        if not master_summary or not audience or not isinstance(tags, list):
            raise LLMResponseError("Reduce response missing required fields")

        clean_tags = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
        clean_tags = list(dict.fromkeys(clean_tags))

        if len(clean_tags) != 5:
            raise LLMResponseError("Reduce response must contain exactly 5 unique tags")

        return {
            "master_summary": master_summary,
            "audience": audience,
            "tags": clean_tags,
        }
