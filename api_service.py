from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Iterator

from ai_etl.db import DatabaseConnectionPool
from ai_gateway.config import ApiSettings, load_api_settings
from ai_gateway.recommender import RecommenderEngine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


@dataclass
class AppState:
    settings: ApiSettings
    db_pool: DatabaseConnectionPool
    embedding_model: SentenceTransformer
    recommender: RecommenderEngine
    retrain_task: asyncio.Task | None = None


class SemanticSearchRequest(BaseModel):
    query_text: str = Field(min_length=2, max_length=5000)
    limit: int | None = Field(default=None, ge=1, le=100)


class SemanticSearchResponse(BaseModel):
    publication_ids: list[int]


class RecommendationRequest(BaseModel):
    user_id: int = Field(gt=0)
    limit: int | None = Field(default=None, ge=1, le=100)


class RecommendationResponse(BaseModel):
    publication_ids: list[int]
    strategy: str


def _vector_to_pg_literal(vector: list[float]) -> str:
    # Parameterized query + explicit vector cast avoids SQL injection risk.
    return "[" + ",".join(f"{x:.10f}" for x in vector) + "]"


@contextmanager
def _pooled_connection(app_state: AppState):
    with app_state.db_pool.connection() as conn:
        yield conn


def _semantic_search_publication_ids(
    app_state: AppState,
    query_vector: list[float],
    limit: int,
) -> list[int]:
    vector_literal = _vector_to_pg_literal(query_vector)

    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                SELECT
                    publication_id,
                    MIN(embedding <=> %s::vector) AS min_distance
                FROM ai_engine.publication_vectors
                GROUP BY publication_id
                ORDER BY min_distance ASC
                LIMIT %s
                """,
                (vector_literal, limit),
            )
            return [int(row[0]) for row in cur.fetchall()]


def _has_user_interactions(app_state: AppState, user_id: int) -> bool:
    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM library_core.user_interactions
                    WHERE user_id = %s
                )
                """,
                (user_id,),
            )
            row = cur.fetchone()
            return bool(row and row[0])


def _filter_available_books(app_state: AppState, publication_ids: list[int], limit: int) -> list[int]:
    if not publication_ids:
        return []

    with _pooled_connection(app_state) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (app_state.settings.statement_timeout_ms,))
            cur.execute(
                """
                SELECT id
                FROM library_core.publications
                WHERE id = ANY(%s)
                  AND available_copies > 0
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
                SELECT id
                FROM library_core.publications
                WHERE available_copies > 0
                ORDER BY total_borrows DESC, id ASC
                LIMIT %s
                """,
                (limit,),
            )
            return [int(row[0]) for row in cur.fetchall()]


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
    db_pool = DatabaseConnectionPool(
        database_url=settings.database_url,
        minconn=settings.db_pool_minconn,
        maxconn=settings.db_pool_maxconn,
    )
    embedding_model = SentenceTransformer(settings.embedding_model_name)
    recommender = RecommenderEngine(settings)

    app_state = AppState(
        settings=settings,
        db_pool=db_pool,
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


@app.post("/api/v1/semantic-search", response_model=SemanticSearchResponse)
def semantic_search(request: SemanticSearchRequest) -> SemanticSearchResponse:
    app_state: AppState = app.state.ctx
    limit = min(
        request.limit or app_state.settings.default_search_limit,
        app_state.settings.max_search_limit,
    )

    try:
        vector = app_state.embedding_model.encode(request.query_text, normalize_embeddings=True)
        publication_ids = _semantic_search_publication_ids(app_state, vector.tolist(), limit)
        return SemanticSearchResponse(publication_ids=publication_ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {exc}") from exc


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
            return RecommendationResponse(
                publication_ids=_get_trending_books(app_state, limit),
                strategy="TRENDING_FALLBACK",
            )

        raw_recommendations = await app_state.recommender.recommend_for_user(
            user_id=request.user_id,
            limit=limit,
        )

        if not raw_recommendations:
            return RecommendationResponse(
                publication_ids=_get_trending_books(app_state, limit),
                strategy="TRENDING_FALLBACK",
            )

        filtered_ids = _filter_available_books(app_state, raw_recommendations, limit)
        if not filtered_ids:
            return RecommendationResponse(
                publication_ids=_get_trending_books(app_state, limit),
                strategy="TRENDING_FALLBACK",
            )

        return RecommendationResponse(
            publication_ids=filtered_ids,
            strategy="ALS_HYBRID",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc
