from __future__ import annotations

import asyncio
import re
import unicodedata
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

import numpy as np
from ai_etl.config import Settings as EtlSettings
from ai_etl.config import load_settings as load_etl_settings
from ai_etl.db import Database
from ai_etl.db import DatabaseConnectionPool
from ai_etl.embedding_client import EmbeddingClient
from ai_etl.llm_client import GeminiClient
from ai_etl.pipeline import AIEtlPipeline
from ai_gateway.config import ApiSettings, load_api_settings
from ai_gateway.recommender import RecommenderEngine
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from psycopg2.extras import Json
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


@dataclass
class AppState:
    settings: ApiSettings
    etl_settings: EtlSettings
    db_pool: DatabaseConnectionPool
    etl_database: Database
    etl_pipeline: AIEtlPipeline
    embedding_model: SentenceTransformer
    recommender: RecommenderEngine
    retrain_task: asyncio.Task | None = None


class SemanticSearchRequest(BaseModel):
    query_text: str = Field(min_length=2, max_length=5000)
    limit: int | None = Field(default=None, ge=1, le=100)


class SemanticSearchResponse(BaseModel):
    publication_ids: list[int]


class SimilarPublicationsResponse(BaseModel):
    publication_ids: list[int]


class RecommendationRequest(BaseModel):
    user_id: int = Field(gt=0)
    limit: int | None = Field(default=None, ge=1, le=100)


class RecommendationResponse(BaseModel):
    publication_ids: list[int]
    strategy: str


class ProcessPublicationRequest(BaseModel):
    publication_id: int = Field(gt=0)
    pdf_url: str | None = Field(default=None, min_length=1, max_length=5000)
    force_reprocess: bool = False


class ProcessPublicationResponse(BaseModel):
    publication_id: int
    status: str
    skipped: bool
    chunks: int
    vectors: int
    summary_generated: bool
    tags: list[str]
    ai_target_audience: str | None = None
    error: str | None = None


class RefreshRecommendationRequest(BaseModel):
    user_id: int = Field(gt=0)
    limit: int | None = Field(default=None, ge=1, le=100)


def _vector_to_pg_literal(vector: list[float]) -> str:
    # Parameterized query + explicit vector cast avoids SQL injection risk.
    return "[" + ",".join(f"{x:.10f}" for x in vector) + "]"


@contextmanager
def _pooled_connection(app_state: AppState):
    with app_state.db_pool.connection() as conn:
        yield conn


class _SharedEmbeddingClient(EmbeddingClient):
    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model


def _semantic_search_publication_ids(
    app_state: AppState,
    query_vector: list[float],
    limit: int,
) -> list[int]:
    vector_literal = _vector_to_pg_literal(query_vector)
    candidate_limit = max(limit * 20, 100)
    max_distance = 1.0 - app_state.settings.search_min_similarity

    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                WITH nearest_chunks AS (
                    SELECT publication_id, embedding <=> %s::vector AS distance
                    FROM ai_engine.publication_vectors
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                )
                SELECT publication_id, MIN(distance) AS min_distance
                FROM nearest_chunks
                GROUP BY publication_id
                HAVING MIN(distance) <= %s
                ORDER BY min_distance ASC
                LIMIT %s
                """,
                (vector_literal, vector_literal, candidate_limit, max_distance, limit),
            )
            return [int(row[0]) for row in cur.fetchall()]


def _chunk_keyword_search(app_state: AppState, query_text: str, limit: int) -> list[int]:
    query_text = " ".join(query_text.strip().split())
    query_terms = _meaningful_query_terms(query_text)
    if not query_text or not query_terms:
        return []

    phrase_like = f"%{query_text}%"
    term_likes = [f"%{term}%" for term in query_terms]
    term_conditions = " AND ".join(["pv.chunk_text ILIKE %s" for _term in query_terms])

    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                f"""
                SELECT
                    pv.publication_id,
                    COUNT(*) FILTER (WHERE pv.chunk_text ILIKE %s) AS phrase_hits,
                    COUNT(*) AS chunk_hits,
                    MIN(pv.id) AS first_hit_id
                FROM ai_engine.publication_vectors pv
                WHERE pv.chunk_text ILIKE %s
                   OR ({term_conditions})
                GROUP BY pv.publication_id
                ORDER BY
                    phrase_hits DESC,
                    chunk_hits DESC,
                    first_hit_id ASC
                LIMIT %s
                """,
                (phrase_like, phrase_like, *term_likes, limit),
            )
            return [int(row[0]) for row in cur.fetchall()]


def _similar_publication_ids(
    app_state: AppState,
    publication_id: int,
    limit: int,
) -> list[int]:
    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                SELECT embedding
                FROM ai_engine.publication_vectors
                WHERE publication_id = %s
                ORDER BY id ASC
                LIMIT 80
                """,
                (publication_id,),
            )
            rows = cur.fetchall()

    if not rows:
        return []

    vectors = np.array([np.asarray(row[0], dtype=np.float32) for row in rows], dtype=np.float32)
    profile = vectors.mean(axis=0)
    norm = np.linalg.norm(profile)
    if norm == 0:
        return []

    profile = profile / norm
    vector_literal = _vector_to_pg_literal(profile.tolist())
    candidate_limit = max(limit * 30, 150)

    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                WITH nearest_chunks AS (
                    SELECT pv.publication_id, pv.embedding <=> %s::vector AS distance
                    FROM ai_engine.publication_vectors pv
                    WHERE pv.publication_id <> %s
                      AND EXISTS (
                          SELECT 1
                          FROM public.items i
                          WHERE i.publication_id = pv.publication_id
                            AND i.status = 'AVAILABLE'
                      )
                    ORDER BY pv.embedding <=> %s::vector
                    LIMIT %s
                )
                SELECT publication_id, MIN(distance) AS min_distance
                FROM nearest_chunks
                GROUP BY publication_id
                ORDER BY min_distance ASC
                LIMIT %s
                """,
                (vector_literal, publication_id, vector_literal, candidate_limit, limit),
            )
            return [int(row[0]) for row in cur.fetchall()]


def _has_publication_vectors(app_state: AppState) -> bool:
    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute("SELECT EXISTS(SELECT 1 FROM ai_engine.publication_vectors LIMIT 1)")
            row = cur.fetchone()
            return bool(row and row[0])


def _has_user_interactions(app_state: AppState, user_id: int) -> bool:
    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM public.user_interactions
                    WHERE user_id = %s
                )
                """,
                (user_id,),
            )
            row = cur.fetchone()
            return bool(row and row[0])


def _content_based_recommendations(
    app_state: AppState,
    user_id: int,
    limit: int,
    exclude_publication_ids: list[int] | None = None,
) -> list[int]:
    excluded_ids = set(exclude_publication_ids or [])

    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                SELECT DISTINCT publication_id
                FROM public.user_interactions
                WHERE user_id = %s
                """,
                (user_id,),
            )
            interacted_ids = {int(row[0]) for row in cur.fetchall()}
            excluded_ids.update(interacted_ids)

            cur.execute(
                """
                SELECT
                    pv.embedding,
                    CASE ui.type
                        WHEN 'WATCH' THEN 1.0
                        WHEN 'WISHLIST' THEN 5.0
                        WHEN 'BORROWED' THEN 10.0
                        ELSE 1.0
                    END AS weight
                FROM public.user_interactions ui
                JOIN ai_engine.publication_vectors pv ON pv.publication_id = ui.publication_id
                WHERE ui.user_id = %s
                ORDER BY ui.created_at DESC NULLS LAST, ui.id DESC
                LIMIT 300
                """,
                (user_id,),
            )
            profile_rows = cur.fetchall()

    if not profile_rows:
        return []

    vectors = np.array([np.asarray(row[0], dtype=np.float32) for row in profile_rows], dtype=np.float32)
    weights = np.array([float(row[1]) for row in profile_rows], dtype=np.float32)
    profile = np.average(vectors, axis=0, weights=weights)
    norm = np.linalg.norm(profile)
    if norm == 0:
        return []
    profile = profile / norm

    vector_literal = _vector_to_pg_literal(profile.tolist())
    candidate_limit = max(limit * 30, 150)
    excluded_list = list(excluded_ids) or [-1]

    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                WITH nearest_chunks AS (
                    SELECT pv.publication_id, pv.embedding <=> %s::vector AS distance
                    FROM ai_engine.publication_vectors pv
                    WHERE NOT (pv.publication_id = ANY(%s))
                      AND EXISTS (
                          SELECT 1
                          FROM public.items i
                          WHERE i.publication_id = pv.publication_id
                            AND i.status = 'AVAILABLE'
                      )
                    ORDER BY pv.embedding <=> %s::vector
                    LIMIT %s
                )
                SELECT publication_id, MIN(distance) AS min_distance
                FROM nearest_chunks
                GROUP BY publication_id
                ORDER BY min_distance ASC
                LIMIT %s
                """,
                (vector_literal, excluded_list, vector_literal, candidate_limit, limit),
            )
            return [int(row[0]) for row in cur.fetchall()]


def _merge_ranked_unique(primary: list[int], secondary: list[int], limit: int) -> list[int]:
    merged: list[int] = []
    seen: set[int] = set()
    for publication_id in [*primary, *secondary]:
        if publication_id in seen:
            continue
        seen.add(publication_id)
        merged.append(publication_id)
        if len(merged) >= limit:
            break
    return merged


def _filter_available_books(app_state: AppState, publication_ids: list[int], limit: int) -> list[int]:
    if not publication_ids:
        return []

    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                SELECT p.id
                FROM public.publications p
                WHERE p.id = ANY(%s)
                  AND EXISTS (
                      SELECT 1
                      FROM public.items i
                      WHERE i.publication_id = p.id
                        AND i.status = 'AVAILABLE'
                  )
                """,
                (publication_ids,),
            )
            available_ids = {int(row[0]) for row in cur.fetchall()}

    ranked = [pid for pid in publication_ids if pid in available_ids]
    return ranked[:limit]


def _get_trending_books(app_state: AppState, limit: int) -> list[int]:
    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                WITH interaction_scores AS (
                    SELECT
                        publication_id,
                        SUM(
                            CASE type
                                WHEN 'WATCH' THEN 1
                                WHEN 'WISHLIST' THEN 5
                                WHEN 'BORROWED' THEN 10
                                ELSE 0
                            END
                        ) AS score
                    FROM public.user_interactions
                    GROUP BY publication_id
                ),
                borrow_scores AS (
                    SELECT i.publication_id, COUNT(*) * 8 AS score
                    FROM public.borrowing_transactions bt
                    JOIN public.items i ON i.id = bt.item_id
                    GROUP BY i.publication_id
                ),
                rating_scores AS (
                    SELECT publication_id, AVG(star) * 3 + COUNT(*) AS score
                    FROM public.ratings
                    GROUP BY publication_id
                )
                SELECT p.id
                FROM public.publications p
                LEFT JOIN interaction_scores ix ON ix.publication_id = p.id
                LEFT JOIN borrow_scores bs ON bs.publication_id = p.id
                LEFT JOIN rating_scores rs ON rs.publication_id = p.id
                WHERE EXISTS (
                    SELECT 1
                    FROM public.items i
                    WHERE i.publication_id = p.id
                      AND i.status = 'AVAILABLE'
                )
                ORDER BY
                    COALESCE(ix.score, 0)
                    + COALESCE(bs.score, 0)
                    + COALESCE(rs.score, 0) DESC,
                    p.created_at DESC NULLS LAST,
                    p.id ASC
                LIMIT %s
                """,
                (limit,),
            )
            return [int(row[0]) for row in cur.fetchall()]


def _normalize_search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower())
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return " ".join(without_marks.replace("đ", "d").split())


def _tokenize_search_text(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_search_text(value))


def _meaningful_query_terms(query_text: str) -> list[str]:
    return [
        term for term in _tokenize_search_text(query_text)
        if len(term) >= 3
    ]


def _keyword_fallback_search(app_state: AppState, query_text: str, limit: int) -> list[int]:
    like_query = f"%{query_text.strip()}%"
    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                SELECT p.id
                FROM public.publications p
                LEFT JOIN public.publication_authors pa ON pa.publication_id = p.id
                LEFT JOIN public.authors a ON a.id = pa.author_id
                LEFT JOIN public.publication_categories pc ON pc.publication_id = p.id
                LEFT JOIN public.categories c ON c.id = pc.category_id
                WHERE p.title ILIKE %s
                   OR p.subtitle ILIKE %s
                   OR p.description ILIKE %s
                   OR p.ai_summary ILIKE %s
                   OR a.name ILIKE %s
                   OR c.name ILIKE %s
                GROUP BY p.id
                ORDER BY
                    CASE
                        WHEN p.title ILIKE %s THEN 0
                        WHEN p.subtitle ILIKE %s THEN 1
                        WHEN p.description ILIKE %s OR p.ai_summary ILIKE %s THEN 2
                        WHEN BOOL_OR(a.name ILIKE %s) OR BOOL_OR(c.name ILIKE %s) THEN 3
                        ELSE 4
                    END,
                    p.created_at DESC NULLS LAST,
                    p.id ASC
                LIMIT %s
                """,
                (
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    limit,
                ),
            )
            rows = cur.fetchall()
            if rows:
                return [int(row[0]) for row in rows]

            cur.execute(
                """
                SELECT
                    p.id,
                    CONCAT_WS(
                        ' ',
                        p.title,
                        p.subtitle,
                        p.description,
                        p.ai_summary,
                        STRING_AGG(DISTINCT a.name, ' '),
                        STRING_AGG(DISTINCT c.name, ' ')
                    ) AS searchable_text
                FROM public.publications p
                LEFT JOIN public.publication_authors pa ON pa.publication_id = p.id
                LEFT JOIN public.authors a ON a.id = pa.author_id
                LEFT JOIN public.publication_categories pc ON pc.publication_id = p.id
                LEFT JOIN public.categories c ON c.id = pc.category_id
                GROUP BY p.id
                ORDER BY p.created_at DESC NULLS LAST, p.id ASC
                LIMIT 500
                """
            )
            candidates = [(int(row[0]), str(row[1] or "")) for row in cur.fetchall()]

    query_terms = _meaningful_query_terms(query_text)
    if not query_terms:
        return []

    scored: list[tuple[int, int]] = []
    for publication_id, searchable_text in candidates:
        searchable_terms = set(_tokenize_search_text(searchable_text))
        score = sum(1 for term in query_terms if term in searchable_terms)
        if score == len(query_terms):
            scored.append((publication_id, score))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return [publication_id for publication_id, _score in scored[:limit]]


def _prefer_keyword_first(query_text: str) -> bool:
    terms = _normalize_search_text(query_text).split()
    return len(terms) <= 4



def _persist_recommendations(
    app_state: AppState,
    user_id: int,
    publication_ids: list[int],
    strategy: str,
) -> None:
    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                INSERT INTO ai_engine.recommendation_cache (user_id, publication_ids, strategy, computed_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (user_id)
                DO UPDATE SET
                    publication_ids = EXCLUDED.publication_ids,
                    strategy = EXCLUDED.strategy,
                    computed_at = NOW()
                """,
                (user_id, Json(publication_ids), strategy),
            )
            cur.execute(
                """
                INSERT INTO public.ai_recommendations (user_id, pub_ids, strategy, computed_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (user_id)
                DO UPDATE SET
                    pub_ids = EXCLUDED.pub_ids,
                    strategy = EXCLUDED.strategy,
                    computed_at = NOW()
                """,
                (user_id, Json(publication_ids), strategy[:30]),
            )
        conn.commit()


def _log_search(app_state: AppState, query_text: str, publication_ids: list[int]) -> None:
    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                INSERT INTO ai_engine.search_logs (query_text, publication_ids)
                VALUES (%s, %s)
                """,
                (query_text, Json(publication_ids)),
            )
        conn.commit()


def _resolve_pdf_source(app_state: AppState, request: ProcessPublicationRequest) -> str:
    if request.pdf_url:
        return request.pdf_url

    with _pooled_connection(app_state) as conn:
        return app_state.etl_database.get_publication_file_url(conn, request.publication_id)


def _mark_process_failed(app_state: AppState, publication_id: int, pdf_source: str, exc: Exception) -> None:
    file_hash = AIEtlPipeline._calculate_file_hash(pdf_source)
    with _pooled_connection(app_state) as conn:
        app_state.etl_database.ensure_etl_metadata_table(conn)
        app_state.etl_database.mark_publication_etl_failed(conn, publication_id, file_hash, str(exc))
        conn.commit()


async def _retrain_loop(app_state: AppState) -> None:
    while True:
        try:
            await app_state.recommender.train_and_persist(
                lambda: _pooled_connection(app_state),
            )
        except Exception:
            # Retry in next cycle to keep API available even if retrain fails.
            pass

        await asyncio.sleep(app_state.settings.als_retrain_interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_api_settings()
    etl_settings = load_etl_settings()
    db_pool = DatabaseConnectionPool(
        database_url=settings.database_url,
        minconn=settings.db_pool_minconn,
        maxconn=settings.db_pool_maxconn,
    )
    embedding_model = SentenceTransformer(settings.embedding_model_name)
    recommender = RecommenderEngine(settings)
    etl_database = Database(etl_settings.database_url)
    llm_client = GeminiClient(
        api_key=etl_settings.gemini_api_key,
        cheap_model=etl_settings.gemini_model_cheap,
        premium_model=etl_settings.gemini_model_premium,
        timeout_seconds=etl_settings.llm_timeout_seconds,
        retry_attempts=etl_settings.llm_retry_attempts,
        min_request_interval_seconds=etl_settings.gemini_min_request_interval_seconds,
    )
    etl_pipeline = AIEtlPipeline(
        settings=etl_settings,
        database=etl_database,
        llm_client=llm_client,
        embedding_client=_SharedEmbeddingClient(embedding_model),
    )

    app_state = AppState(
        settings=settings,
        etl_settings=etl_settings,
        db_pool=db_pool,
        etl_database=etl_database,
        etl_pipeline=etl_pipeline,
        embedding_model=embedding_model,
        recommender=recommender,
    )

    await recommender.load_or_train(lambda: _pooled_connection(app_state))
    app_state.retrain_task = asyncio.create_task(_retrain_loop(app_state))

    app.state.ctx = app_state

    try:
        yield
    finally:
        if app_state.retrain_task is not None:
            app_state.retrain_task.cancel()
            try:
                await app_state.retrain_task
            except asyncio.CancelledError:
                pass
        db_pool.close()


app = FastAPI(title="LMS AI Gateway", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/v1/publications/process",
    response_model=ProcessPublicationResponse,
    response_class=JSONResponse,
)
def process_publication(request: ProcessPublicationRequest) -> ProcessPublicationResponse:
    app_state: AppState = app.state.ctx
    pdf_source = _resolve_pdf_source(app_state, request)

    try:
        result = app_state.etl_pipeline.run(
            publication_id=request.publication_id,
            pdf_path=pdf_source,
            force_reprocess=request.force_reprocess,
        )
        return ProcessPublicationResponse(
            publication_id=request.publication_id,
            status="SUCCESS",
            skipped=bool(result.get("skipped", False)),
            chunks=int(result.get("chunks", 0)),
            vectors=int(result.get("vectors", 0)),
            summary_generated=bool(result.get("master_summary")),
            tags=[str(tag) for tag in result.get("tags", [])],
            ai_target_audience=str(result.get("audience")) if result.get("audience") else None,
        )
    except Exception as exc:
        try:
            _mark_process_failed(app_state, request.publication_id, pdf_source, exc)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Publication processing failed: {exc}",
        ) from exc


@app.post("/api/v1/semantic-search", response_model=SemanticSearchResponse)
def semantic_search(request: SemanticSearchRequest) -> SemanticSearchResponse:
    app_state: AppState = app.state.ctx
    limit = min(
        request.limit or app_state.settings.default_search_limit,
        app_state.settings.max_search_limit,
    )

    try:
        if _has_publication_vectors(app_state):
            vector = app_state.embedding_model.encode(request.query_text, normalize_embeddings=True)
            semantic_ids = _semantic_search_publication_ids(app_state, vector.tolist(), limit)
        else:
            semantic_ids = []

        keyword_ids = _keyword_fallback_search(app_state, request.query_text, limit)
        chunk_keyword_ids = _chunk_keyword_search(app_state, request.query_text, limit)
        if _prefer_keyword_first(request.query_text):
            publication_ids = _merge_ranked_unique(
                _merge_ranked_unique(keyword_ids, chunk_keyword_ids, limit),
                semantic_ids,
                limit,
            )
        else:
            publication_ids = _merge_ranked_unique(
                _merge_ranked_unique(chunk_keyword_ids, semantic_ids, limit),
                keyword_ids,
                limit,
            )

        _log_search(app_state, request.query_text, publication_ids)
        return SemanticSearchResponse(publication_ids=publication_ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {exc}") from exc


@app.get("/api/v1/publications/{publication_id}/similar", response_model=SimilarPublicationsResponse)
def similar_publications(publication_id: int, limit: int = 6) -> SimilarPublicationsResponse:
    app_state: AppState = app.state.ctx
    safe_limit = min(max(limit, 1), app_state.settings.max_recommend_limit)

    try:
        publication_ids = _similar_publication_ids(app_state, publication_id, safe_limit)
        if len(publication_ids) < safe_limit:
            trending_ids = _get_trending_books(app_state, safe_limit)
            publication_ids = _merge_ranked_unique(publication_ids, trending_ids, safe_limit)
        return SimilarPublicationsResponse(publication_ids=publication_ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Similar publication search failed: {exc}") from exc


@app.post("/api/v1/recommendations", response_model=RecommendationResponse)
async def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    app_state: AppState = app.state.ctx
    limit = min(
        request.limit or app_state.settings.default_recommend_limit,
        app_state.settings.max_recommend_limit,
    )

    try:
        has_interactions = _has_user_interactions(app_state, request.user_id)
        if not has_interactions:
            publication_ids = _get_trending_books(app_state, limit)
            _persist_recommendations(app_state, request.user_id, publication_ids, "TRENDING_FALLBACK")
            return RecommendationResponse(publication_ids=publication_ids, strategy="TRENDING_FALLBACK")

        cbf_ids = _content_based_recommendations(app_state, request.user_id, limit)
        try:
            raw_recommendations = await app_state.recommender.recommend_for_user(
                user_id=request.user_id,
                limit=limit,
            )
        except Exception:
            raw_recommendations = []
        als_ids = _filter_available_books(app_state, raw_recommendations, limit) if raw_recommendations else []
        publication_ids = _merge_ranked_unique(cbf_ids, als_ids, limit)

        if len(publication_ids) < limit:
            trending_ids = _get_trending_books(app_state, limit)
            publication_ids = _merge_ranked_unique(publication_ids, trending_ids, limit)

        if not publication_ids:
            _persist_recommendations(app_state, request.user_id, [], "TRENDING_FALLBACK")
            return RecommendationResponse(publication_ids=[], strategy="TRENDING_FALLBACK")

        if cbf_ids and als_ids:
            strategy = "REALTIME_CBF_ALS_HYBRID"
        elif cbf_ids:
            strategy = "REALTIME_CONTENT_BASED"
        elif als_ids:
            strategy = "ALS_HYBRID"
        else:
            strategy = "TRENDING_FALLBACK"

        _persist_recommendations(app_state, request.user_id, publication_ids, strategy)
        return RecommendationResponse(publication_ids=publication_ids, strategy=strategy)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc


@app.post("/api/v1/recommendations/refresh", response_model=RecommendationResponse)
async def refresh_recommendations(request: RefreshRecommendationRequest) -> RecommendationResponse:
    return await recommendations(RecommendationRequest(user_id=request.user_id, limit=request.limit))
