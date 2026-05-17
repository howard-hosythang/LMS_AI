from __future__ import annotations

from contextlib import contextmanager
import random
import re
import time
from typing import Iterator

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extensions import connection as PGConnection
from psycopg2.extras import execute_values
from psycopg2.pool import SimpleConnectionPool


class DatabaseConnectionPool:
    def __init__(self, database_url: str, minconn: int, maxconn: int) -> None:
        self.pool = SimpleConnectionPool(minconn=minconn, maxconn=maxconn, dsn=database_url)

    @contextmanager
    def connection(self) -> Iterator[PGConnection]:
        conn = self.pool.getconn()
        try:
            register_vector(conn)
            yield conn
        finally:
            try:
                if not conn.closed:
                    conn.rollback()
            except Exception:
                pass
            # Always return connection to pool to prevent zombie/leaked connections.
            self.pool.putconn(conn)

    def close(self) -> None:
        self.pool.closeall()


class Database:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    @contextmanager
    def connect(self) -> Iterator[PGConnection]:
        conn = psycopg2.connect(self.database_url)
        register_vector(conn)
        try:
            yield conn
        finally:
            conn.close()

    def clear_previous_ai_data(self, conn: PGConnection, publication_id: int) -> None:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM ai_engine.publication_vectors WHERE publication_id = %s",
                (publication_id,),
            )
            cur.execute(
                "DELETE FROM ai_engine.publication_tags WHERE publication_id = %s",
                (publication_id,),
            )
            cur.execute(
                "DELETE FROM public.publication_tags WHERE publication_id = %s",
                (publication_id,),
            )

    def ensure_etl_metadata_table(self, conn: PGConnection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_engine.publication_etl_runs (
                    publication_id BIGINT PRIMARY KEY REFERENCES public.publications(id) ON DELETE CASCADE,
                    file_hash VARCHAR(64) NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'SUCCESS',
                    error_message TEXT NULL,
                    chunks_count INT NOT NULL DEFAULT 0,
                    vectors_count INT NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("ALTER TABLE ai_engine.publication_etl_runs ALTER COLUMN file_hash DROP NOT NULL")
            cur.execute("ALTER TABLE ai_engine.publication_etl_runs ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'SUCCESS'")
            cur.execute("ALTER TABLE ai_engine.publication_etl_runs ADD COLUMN IF NOT EXISTS error_message TEXT NULL")
            cur.execute("ALTER TABLE ai_engine.publication_etl_runs ADD COLUMN IF NOT EXISTS chunks_count INT NOT NULL DEFAULT 0")
            cur.execute("ALTER TABLE ai_engine.publication_etl_runs ADD COLUMN IF NOT EXISTS vectors_count INT NOT NULL DEFAULT 0")

    def get_publication_file_hash(self, conn: PGConnection, publication_id: int) -> str | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT file_hash
                FROM ai_engine.publication_etl_runs
                WHERE publication_id = %s
                """,
                (publication_id,),
            )
            row = cur.fetchone()
            return str(row[0]) if row else None

    def upsert_publication_file_hash(
        self,
        conn: PGConnection,
        publication_id: int,
        file_hash: str,
        chunks_count: int = 0,
        vectors_count: int = 0,
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_engine.publication_etl_runs (
                    publication_id,
                    file_hash,
                    status,
                    error_message,
                    chunks_count,
                    vectors_count,
                    updated_at
                )
                VALUES (%s, %s, 'SUCCESS', NULL, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (publication_id)
                DO UPDATE SET
                    file_hash = EXCLUDED.file_hash,
                    status = 'SUCCESS',
                    error_message = NULL,
                    chunks_count = EXCLUDED.chunks_count,
                    vectors_count = EXCLUDED.vectors_count,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (publication_id, file_hash, chunks_count, vectors_count),
            )

    def mark_publication_etl_running(
        self,
        conn: PGConnection,
        publication_id: int,
        file_hash: str | None = None,
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_engine.publication_etl_runs (
                    publication_id,
                    file_hash,
                    status,
                    error_message,
                    chunks_count,
                    vectors_count,
                    updated_at
                )
                VALUES (%s, %s, 'RUNNING', NULL, 0, 0, CURRENT_TIMESTAMP)
                ON CONFLICT (publication_id)
                DO UPDATE SET
                    file_hash = COALESCE(EXCLUDED.file_hash, ai_engine.publication_etl_runs.file_hash),
                    status = 'RUNNING',
                    error_message = NULL,
                    chunks_count = 0,
                    vectors_count = 0,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (publication_id, file_hash),
            )

    def mark_publication_etl_failed(
        self,
        conn: PGConnection,
        publication_id: int,
        file_hash: str | None,
        error_message: str,
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_engine.publication_etl_runs (
                    publication_id,
                    file_hash,
                    status,
                    error_message,
                    updated_at
                )
                VALUES (%s, %s, 'FAILED', %s, CURRENT_TIMESTAMP)
                ON CONFLICT (publication_id)
                DO UPDATE SET
                    file_hash = COALESCE(EXCLUDED.file_hash, ai_engine.publication_etl_runs.file_hash),
                    status = 'FAILED',
                    error_message = EXCLUDED.error_message,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (publication_id, file_hash, error_message[:2000]),
            )

    def get_publication_file_url(self, conn: PGConnection, publication_id: int) -> str:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT file_url FROM public.publications WHERE id = %s",
                (publication_id,),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Publication {publication_id} not found in public.publications")
            if not row[0]:
                raise ValueError(f"Publication {publication_id} has no file_url")
            return str(row[0])

    def get_publication_context(self, conn: PGConnection, publication_id: int) -> str:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    p.title,
                    p.subtitle,
                    p.description,
                    STRING_AGG(DISTINCT a.name, ', ') AS authors,
                    STRING_AGG(DISTINCT c.name, ', ') AS categories
                FROM public.publications p
                LEFT JOIN public.publication_authors pa ON pa.publication_id = p.id
                LEFT JOIN public.authors a ON a.id = pa.author_id
                LEFT JOIN public.publication_categories pc ON pc.publication_id = p.id
                LEFT JOIN public.categories c ON c.id = pc.category_id
                WHERE p.id = %s
                GROUP BY p.id
                """,
                (publication_id,),
            )
            row = cur.fetchone()
            if not row:
                return ""

        labels = ("Title", "Subtitle", "Description", "Authors", "Categories")
        parts = [f"{label}: {value}" for label, value in zip(labels, row) if value]
        return "\n".join(parts)

    def publication_has_vectors(self, conn: PGConnection, publication_id: int) -> bool:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM ai_engine.publication_vectors
                    WHERE publication_id = %s
                )
                """,
                (publication_id,),
            )
            row = cur.fetchone()
            return bool(row and row[0])

    def publication_has_ai_outputs(self, conn: PGConnection, publication_id: int) -> bool:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    EXISTS(
                        SELECT 1
                        FROM ai_engine.publication_vectors
                        WHERE publication_id = %s
                    ) AS has_vectors,
                    EXISTS(
                        SELECT 1
                        FROM public.publication_tags
                        WHERE publication_id = %s
                    ) AS has_tags,
                    EXISTS(
                        SELECT 1
                        FROM public.publications
                        WHERE id = %s
                          AND ai_summary IS NOT NULL
                          AND ai_target_audience IS NOT NULL
                    ) AS has_summary
                """,
                (publication_id, publication_id, publication_id),
            )
            row = cur.fetchone()
            return bool(row and row[0] and row[1] and row[2])

    def bulk_insert_vectors(
        self,
        conn: PGConnection,
        publication_id: int,
        rows: list[tuple[str, list[float]]],
    ) -> None:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO ai_engine.publication_vectors (publication_id, chunk_text, embedding)
                VALUES %s
                """,
                [(publication_id, chunk_text, embedding) for chunk_text, embedding in rows],
                template="(%s, %s, %s)",
            )

    def update_publication_summary(
        self,
        conn: PGConnection,
        publication_id: int,
        master_summary: str,
        audience: str,
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.publications
                SET ai_summary = %s,
                    ai_target_audience = %s
                WHERE id = %s
                """,
                (master_summary, audience, publication_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"Publication {publication_id} not found in public.publications")

    def find_or_create_tag_id(self, conn: PGConnection, tag_name: str) -> int:
        normalized_tag = self._normalize_public_tag_name(tag_name)
        if not normalized_tag:
            raise ValueError("Tag name is empty after normalization")

        with conn.cursor() as cur:
            cur.execute(
                """
                WITH inserted AS (
                    INSERT INTO public.tags (id, name, created_at, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (name) DO NOTHING
                    RETURNING id
                )
                SELECT id FROM inserted
                UNION ALL
                SELECT id FROM public.tags WHERE name = %s
                LIMIT 1
                """,
                (self._generate_bigint_id(), normalized_tag, normalized_tag),
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError(f"Failed to resolve tag id for tag: {normalized_tag}")
            return int(row[0])

    def insert_publication_tag_links(
        self,
        conn: PGConnection,
        publication_id: int,
        tag_ids: list[int],
    ) -> None:
        unique_tag_ids = list(dict.fromkeys(tag_ids))
        if not unique_tag_ids:
            return

        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO public.publication_tags (id, publication_id, tag_id)
                SELECT v.id, v.publication_id, v.tag_id
                FROM (VALUES %s) AS v(id, publication_id, tag_id)
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM public.publication_tags pt
                    WHERE pt.publication_id = v.publication_id
                      AND pt.tag_id = v.tag_id
                )
                ON CONFLICT (id) DO NOTHING
                """,
                [
                    (self._generate_bigint_id(), publication_id, tag_id)
                    for tag_id in unique_tag_ids
                ],
                template="(%s, %s, %s)",
            )

    @staticmethod
    def _generate_bigint_id() -> int:
        millis = int(time.time() * 1000) & ((1 << 42) - 1)
        suffix = random.getrandbits(20)
        return (millis << 20) | suffix

    @staticmethod
    def _normalize_public_tag_name(tag_name: str) -> str:
        tag = re.sub(r"[_\s]+", " ", str(tag_name or "")).strip()
        tag = re.sub(r"[^\wÀ-ỹ+#.\- ]+", "", tag, flags=re.UNICODE)
        tag = re.sub(r"\s+", " ", tag).strip(" -_.")
        if not tag:
            return ""
        if tag.isupper() or tag.islower():
            tag = tag.title()
        return tag[:50]
