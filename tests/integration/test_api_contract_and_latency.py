from __future__ import annotations

import os
import time

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
API_LATENCY_THRESHOLD_MS = float(os.getenv("API_LATENCY_THRESHOLD_MS", "200"))


def _post(path: str, payload: dict):
    started = time.perf_counter()
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=10)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return response, elapsed_ms


def test_semantic_search_contract_and_latency():
    response, elapsed_ms = _post(
        "/api/v1/semantic-search",
        {
            "query_text": "Trí tuệ nhân tạo và học máy ứng dụng",
            "limit": 5,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "publication_ids" in body
    assert isinstance(body["publication_ids"], list)
    assert elapsed_ms < API_LATENCY_THRESHOLD_MS, f"Latency too high: {elapsed_ms:.2f}ms"


def test_recommendation_contract_and_latency():
    response, elapsed_ms = _post(
        "/api/v1/recommendations",
        {
            "user_id": int(os.getenv("TEST_USER_ID", "1")),
            "limit": 5,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "publication_ids" in body
    assert "strategy" in body
    assert isinstance(body["publication_ids"], list)
    assert isinstance(body["strategy"], str)
    assert elapsed_ms < API_LATENCY_THRESHOLD_MS, f"Latency too high: {elapsed_ms:.2f}ms"
