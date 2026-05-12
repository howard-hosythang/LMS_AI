from __future__ import annotations

from contextlib import contextmanager
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

    def ensure_etl_metadata_table(self, conn: PGConnection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_engine.publication_etl_runs (
                    publication_id INT PRIMARY KEY REFERENCES library_core.publications(id) ON DELETE CASCADE,
                    file_hash VARCHAR(64) NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

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
    ) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_engine.publication_etl_runs (publication_id, file_hash, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (publication_id)
                DO UPDATE SET
                    file_hash = EXCLUDED.file_hash,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (publication_id, file_hash),
            )

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
                UPDATE library_core.publications
                SET ai_summary = %s,
                    ai_target_audience = %s
                WHERE id = %s
                """,
                (master_summary, audience, publication_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"Publication {publication_id} not found in library_core.publications")

    def find_or_create_tag_id(self, conn: PGConnection, tag_name: str) -> int:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH inserted AS (
                    INSERT INTO ai_engine.tags (name)
                    VALUES (%s)
                    ON CONFLICT (name) DO NOTHING
                    RETURNING id
                )
                SELECT id FROM inserted
                UNION ALL
                SELECT id FROM ai_engine.tags WHERE name = %s
                LIMIT 1
                """,
                (tag_name, tag_name),
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError(f"Failed to resolve tag id for tag: {tag_name}")
            return int(row[0])

    def insert_publication_tag_links(
        self,
        conn: PGConnection,
        publication_id: int,
        tag_ids: list[int],
    ) -> None:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO ai_engine.publication_tags (publication_id, tag_id)
                VALUES %s
                ON CONFLICT (publication_id, tag_id) DO NOTHING
                """,
                [(publication_id, tag_id) for tag_id in tag_ids],
                template="(%s, %s)",
            )
