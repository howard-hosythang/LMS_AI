from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Union

import numpy as np
import requests
from psycopg2.extensions import connection as PGConnection

from .config import Settings
from .db import Database
from .embedding_client import EmbeddingClient
from .llm_client import GeminiClient
from .pdf_processing import Chunk, extract_pdf_text, sliding_window_chunk


class AIEtlPipeline:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        llm_client: GeminiClient,
        embedding_client: EmbeddingClient,
    ) -> None:
        self.settings = settings
        self.database = database
        self.llm_client = llm_client
        self.embedding_client = embedding_client

    def run(
        self,
        publication_id: int,
        pdf_path: Union[Path, str],
        conn: PGConnection | None = None,
    ) -> dict[str, object]:
        full_text = extract_pdf_text(pdf_path)
        file_hash = self._calculate_file_hash(pdf_path)
        chunks = sliding_window_chunk(
            text=full_text,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No valid text chunks extracted from PDF")

        if conn is not None:
            return self._run_with_connection(conn, publication_id, chunks, file_hash)

        with self.database.connect() as conn:
            return self._run_with_connection(conn, publication_id, chunks, file_hash)

    def _run_with_connection(
        self,
        conn: PGConnection,
        publication_id: int,
        chunks: list[Chunk],
        file_hash: str,
    ) -> dict[str, object]:
        self.database.ensure_etl_metadata_table(conn)

        if self.settings.enable_hash_skip:
            existing_hash = self.database.get_publication_file_hash(conn, publication_id)
            has_vectors = self.database.publication_has_vectors(conn, publication_id)
            if existing_hash == file_hash and has_vectors:
                return {
                    "publication_id": publication_id,
                    "chunks": len(chunks),
                    "vectors": 0,
                    "skipped": True,
                    "skip_reason": "file_hash_unchanged",
                }

        self.database.clear_previous_ai_data(conn, publication_id)
        conn.commit()

        try:
            chunk_texts = [chunk.text for chunk in chunks]
            vectors = self.embedding_client.encode(chunk_texts)

            vector_rows = list(zip(chunk_texts, vectors))
            for batch_rows in self._iter_batches(vector_rows, self.settings.vector_insert_batch_size):
                self.database.bulk_insert_vectors(conn, publication_id, batch_rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        representative_chunks = self._select_representative_chunks(
            chunks=chunks,
            vectors=vectors,
            max_chunks=self.settings.reduce_max_chunks,
        )
        reduce_inputs = self._apply_reduce_budget(representative_chunks)
        if not reduce_inputs:
            raise ValueError("No representative chunks available for reduce phase")

        try:
            reduce_result = self.llm_client.reduce_publication(reduce_inputs)
            self.database.update_publication_summary(
                conn,
                publication_id,
                reduce_result["master_summary"],
                reduce_result["audience"],
            )

            tag_ids = [
                self.database.find_or_create_tag_id(conn, tag_name)
                for tag_name in reduce_result["tags"]
            ]
            self.database.insert_publication_tag_links(conn, publication_id, tag_ids)
            self.database.upsert_publication_file_hash(conn, publication_id, file_hash)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        return {
            "publication_id": publication_id,
            "chunks": len(chunks),
            "vectors": len(vectors),
            "reduce_inputs": len(reduce_inputs),
            "skipped": False,
            "master_summary": reduce_result["master_summary"],
            "audience": reduce_result["audience"],
            "tags": reduce_result["tags"],
        }

    @staticmethod
    def _calculate_file_hash(pdf_source: Union[Path, str]) -> str:
        """Calculate hash of PDF file or URL content.

        For URLs, hash the URL string itself for efficiency (don't download twice).
        For local files, hash the file content.
        """
        digest = hashlib.sha256()

        if isinstance(pdf_source, str) and (pdf_source.startswith('http://') or pdf_source.startswith('https://')):
            # Hash the URL string for remote files (efficient, avoids re-downloading)
            digest.update(pdf_source.encode('utf-8'))
        else:
            # Hash file content for local files
            pdf_path = Path(pdf_source) if isinstance(pdf_source, str) else pdf_source
            with pdf_path.open("rb") as file_obj:
                for block in iter(lambda: file_obj.read(1024 * 1024), b""):
                    digest.update(block)

        return digest.hexdigest()

    def _select_representative_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        max_chunks: int,
    ) -> list[str]:
        if not chunks:
            return []

        embeddings = np.array(vectors, dtype=np.float32)
        total_chunks = len(chunks)
        target = min(max_chunks, max(8, int(math.sqrt(total_chunks) * 2)), total_chunks)

        centroid = embeddings.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            return [chunk.text for chunk in chunks[:target]]
        centroid = centroid / centroid_norm

        similarities = embeddings @ centroid
        selected: list[int] = [int(np.argmax(similarities))]
        candidate_indices = set(range(total_chunks))
        candidate_indices.remove(selected[0])

        lambda_weight = 0.65
        while len(selected) < target and candidate_indices:
            best_idx = None
            best_score = float("-inf")

            for idx in candidate_indices:
                relevance = float(similarities[idx])
                diversity_penalty = max(float(embeddings[idx] @ embeddings[s]) for s in selected)
                score = lambda_weight * relevance - (1.0 - lambda_weight) * diversity_penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            selected.append(best_idx)
            candidate_indices.remove(best_idx)

        selected.sort()
        return [chunks[idx].text for idx in selected]

    def _apply_reduce_budget(self, texts: list[str]) -> list[str]:
        selected: list[str] = []
        total_chars = 0

        for text in texts:
            clipped = text[: self.settings.reduce_max_chars_per_chunk].strip()
            if not clipped:
                continue

            if total_chars + len(clipped) > self.settings.reduce_total_chars_budget:
                break

            selected.append(clipped)
            total_chars += len(clipped)

        return selected

    @staticmethod
    def _iter_batches(items: list, batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]
