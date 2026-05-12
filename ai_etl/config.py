from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str
    rabbitmq_url: str
    gemini_api_key: str
    gemini_model_cheap: str
    gemini_model_premium: str
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    vector_insert_batch_size: int
    reduce_max_chunks: int
    reduce_max_chars_per_chunk: int
    reduce_total_chars_budget: int
    enable_hash_skip: bool
    llm_timeout_seconds: int
    llm_retry_attempts: int
    webhook_timeout_seconds: int
    worker_db_pool_minconn: int
    worker_db_pool_maxconn: int


def _parse_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_settings() -> Settings:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError("DATABASE_URL is required")

    rabbitmq_url = os.getenv("RABBITMQ_URL", "").strip()
    if not rabbitmq_url:
        raise ValueError("RABBITMQ_URL is required")

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required")

    chunk_size = int(os.getenv("AI_ETL_CHUNK_SIZE", "1800"))
    chunk_overlap = int(os.getenv("AI_ETL_CHUNK_OVERLAP", "150"))

    if chunk_overlap >= chunk_size:
        raise ValueError("AI_ETL_CHUNK_OVERLAP must be smaller than AI_ETL_CHUNK_SIZE")

    return Settings(
        database_url=database_url,
        rabbitmq_url=rabbitmq_url,
        gemini_api_key=gemini_api_key,
        gemini_model_cheap=os.getenv("AI_ETL_GEMINI_MODEL_CHEAP", "gemini-2.5-flash"),
        gemini_model_premium=os.getenv("AI_ETL_GEMINI_MODEL_PREMIUM", "gemini-2.5-pro"),
        embedding_model_name=os.getenv(
            "AI_ETL_EMBEDDING_MODEL",
            "bkai-foundation-models/vietnamese-bi-encoder",
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        vector_insert_batch_size=int(os.getenv("AI_ETL_VECTOR_INSERT_BATCH_SIZE", "256")),
        reduce_max_chunks=int(os.getenv("AI_ETL_REDUCE_MAX_CHUNKS", "30")),
        reduce_max_chars_per_chunk=int(os.getenv("AI_ETL_REDUCE_MAX_CHARS_PER_CHUNK", "1200")),
        reduce_total_chars_budget=int(os.getenv("AI_ETL_REDUCE_TOTAL_CHARS_BUDGET", "36000")),
        enable_hash_skip=_parse_bool(os.getenv("AI_ETL_ENABLE_HASH_SKIP", "true"), True),
        llm_timeout_seconds=int(os.getenv("AI_ETL_LLM_TIMEOUT_SECONDS", "60")),
        llm_retry_attempts=int(os.getenv("AI_ETL_LLM_RETRY_ATTEMPTS", "3")),
        webhook_timeout_seconds=int(os.getenv("AI_ETL_WEBHOOK_TIMEOUT_SECONDS", "8")),
        worker_db_pool_minconn=int(os.getenv("AI_ETL_DB_POOL_MINCONN", "1")),
        worker_db_pool_maxconn=int(os.getenv("AI_ETL_DB_POOL_MAXCONN", "6")),
    )
