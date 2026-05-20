from __future__ import annotations

import asyncio
from types import SimpleNamespace

import api_service


class FakeEmbeddingModel:
    def encode(self, query_text: str, normalize_embeddings: bool = True):
        assert normalize_embeddings is True
        return SimpleNamespace(tolist=lambda: [0.1, 0.2, 0.3])


class FakeRecommender:
    async def recommend_for_user(self, user_id: int, limit: int) -> list[int]:
        assert user_id == 99
        assert limit == 5
        return [30, 31, 32]


def _install_app_state(settings=None) -> SimpleNamespace:
    state = SimpleNamespace(
        settings=settings or SimpleNamespace(
            default_search_limit=10,
            max_search_limit=50,
            default_recommend_limit=10,
            max_recommend_limit=50,
        ),
        embedding_model=FakeEmbeddingModel(),
        recommender=FakeRecommender(),
    )
    api_service.app.state.ctx = state
    return state


def test_vector_to_pg_literal_uses_fixed_precision_for_pgvector_cast() -> None:
    assert api_service._vector_to_pg_literal([1, 0.125, -2.5]) == "[1.0000000000,0.1250000000,-2.5000000000]"


def test_query_tokenization_is_accent_insensitive_and_filters_short_terms() -> None:
    assert api_service._meaningful_query_terms("Lập trình AI, C++ và Web") == ["lap", "trinh", "web"]
    assert api_service._prefer_keyword_first("web development") is True
    assert api_service._prefer_keyword_first("advanced neural information retrieval systems") is False


def test_merge_ranked_unique_preserves_rank_and_limit() -> None:
    assert api_service._merge_ranked_unique([1, 2, 3], [3, 4, 5], limit=4) == [1, 2, 3, 4]


def test_semantic_search_prefers_lexical_results_for_short_queries(monkeypatch) -> None:
    _install_app_state()
    log_calls: list[tuple[str, list[int]]] = []

    monkeypatch.setattr(api_service, "_has_publication_vectors", lambda _state: True)
    monkeypatch.setattr(api_service, "_semantic_search_publication_ids", lambda _state, _vector, _limit: [8, 9])
    monkeypatch.setattr(api_service, "_keyword_fallback_search", lambda _state, _query, _limit: [1, 2, 8])
    monkeypatch.setattr(api_service, "_chunk_keyword_search", lambda _state, _query, _limit: [3, 2])
    monkeypatch.setattr(api_service, "_log_search", lambda _state, query, ids: log_calls.append((query, ids)))

    response = api_service.semantic_search(api_service.SemanticSearchRequest(query_text="web development", limit=5))

    assert response.publication_ids == [1, 2, 8, 3, 9]
    assert log_calls == [("web development", [1, 2, 8, 3, 9])]


def test_semantic_search_prefers_semantic_chunk_order_for_long_queries(monkeypatch) -> None:
    _install_app_state()

    monkeypatch.setattr(api_service, "_has_publication_vectors", lambda _state: True)
    monkeypatch.setattr(api_service, "_semantic_search_publication_ids", lambda _state, _vector, _limit: [40, 41])
    monkeypatch.setattr(api_service, "_keyword_fallback_search", lambda _state, _query, _limit: [1, 40])
    monkeypatch.setattr(api_service, "_chunk_keyword_search", lambda _state, _query, _limit: [30, 31])
    monkeypatch.setattr(api_service, "_log_search", lambda *_args: None)

    response = api_service.semantic_search(
        api_service.SemanticSearchRequest(query_text="advanced neural information retrieval systems", limit=5)
    )

    assert response.publication_ids == [30, 31, 40, 41, 1]


def test_recommendations_use_trending_fallback_for_cold_start_users(monkeypatch) -> None:
    _install_app_state()
    persisted: list[tuple[int, list[int], str]] = []

    monkeypatch.setattr(api_service, "_has_user_interactions", lambda _state, user_id: False)
    monkeypatch.setattr(api_service, "_get_trending_books", lambda _state, limit: [10, 11, 12][:limit])
    monkeypatch.setattr(
        api_service,
        "_persist_recommendations",
        lambda _state, user_id, ids, strategy: persisted.append((user_id, ids, strategy)),
    )

    response = asyncio.run(api_service.recommendations(api_service.RecommendationRequest(user_id=99, limit=3)))

    assert response.publication_ids == [10, 11, 12]
    assert response.strategy == "TRENDING_FALLBACK"
    assert persisted == [(99, [10, 11, 12], "TRENDING_FALLBACK")]


def test_recommendations_merge_cbf_als_and_trending_for_warm_users(monkeypatch) -> None:
    _install_app_state()
    persisted: list[tuple[int, list[int], str]] = []

    monkeypatch.setattr(api_service, "_has_user_interactions", lambda _state, user_id: True)
    monkeypatch.setattr(api_service, "_content_based_recommendations", lambda _state, user_id, limit: [20, 21])
    monkeypatch.setattr(api_service, "_filter_available_books", lambda _state, ids, limit: [30, 31])
    monkeypatch.setattr(api_service, "_get_trending_books", lambda _state, limit: [21, 40, 41, 42])
    monkeypatch.setattr(
        api_service,
        "_persist_recommendations",
        lambda _state, user_id, ids, strategy: persisted.append((user_id, ids, strategy)),
    )

    response = asyncio.run(api_service.recommendations(api_service.RecommendationRequest(user_id=99, limit=5)))

    assert response.publication_ids == [20, 21, 30, 31, 40]
    assert response.strategy == "REALTIME_CBF_ALS_HYBRID"
    assert persisted == [(99, [20, 21, 30, 31, 40], "REALTIME_CBF_ALS_HYBRID")]
