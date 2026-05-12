from __future__ import annotations

import asyncio
import pickle
from dataclasses import dataclass
from pathlib import Path

import implicit
import numpy as np
from psycopg2.extensions import connection as PGConnection
from scipy.sparse import csr_matrix

from .config import ApiSettings


@dataclass
class RecommenderBundle:
    model: implicit.als.AlternatingLeastSquares
    user_id_to_idx: dict[int, int]
    idx_to_user_id: list[int]
    item_id_to_idx: dict[int, int]
    idx_to_item_id: list[int]
    user_items_csr: csr_matrix


class RecommenderEngine:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self._bundle: RecommenderBundle | None = None
        self._lock = asyncio.Lock()

    async def load_or_train(self, conn_factory) -> None:
        async with self._lock:
            if self.settings.als_model_path.exists():
                self._bundle = self._load_bundle(self.settings.als_model_path)
                return

        await self.train_and_persist(conn_factory)

    async def train_and_persist(self, conn_factory) -> None:
        async with self._lock:
            bundle = self._train_from_db(conn_factory)
            bundle_path = self.settings.als_model_path
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            with bundle_path.open("wb") as f:
                pickle.dump(bundle, f)
            self._bundle = bundle

    async def recommend_for_user(
        self,
        user_id: int,
        limit: int,
    ) -> list[int]:
        async with self._lock:
            if self._bundle is None:
                if self.settings.als_model_path.exists():
                    self._bundle = self._load_bundle(self.settings.als_model_path)
                else:
                    return []

            bundle = self._bundle

        user_idx = bundle.user_id_to_idx.get(user_id)
        if user_idx is None:
            return []

        item_ids_idx, _scores = bundle.model.recommend(
            userid=user_idx,
            user_items=bundle.user_items_csr,
            N=limit * 3,
            filter_already_liked_items=True,
            recalculate_user=True,
        )

        return [bundle.idx_to_item_id[idx] for idx in item_ids_idx[: limit * 3]]

    def _train_from_db(self, conn_factory) -> RecommenderBundle:
        with conn_factory() as conn:
            rows = self._fetch_weighted_interactions(conn)

        if not rows:
            return RecommenderBundle(
                model=implicit.als.AlternatingLeastSquares(
                    factors=self.settings.als_factors,
                    regularization=self.settings.als_regularization,
                    iterations=self.settings.als_iterations,
                    random_state=42,
                ),
                user_id_to_idx={},
                idx_to_user_id=[],
                item_id_to_idx={},
                idx_to_item_id=[],
                user_items_csr=csr_matrix((0, 0), dtype=np.float32),
            )

        user_ids = sorted({int(row[0]) for row in rows})
        item_ids = sorted({int(row[1]) for row in rows})
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

        data: list[float] = []
        row_idx: list[int] = []
        col_idx: list[int] = []

        for user_id, publication_id, score in rows:
            row_idx.append(user_id_to_idx[int(user_id)])
            col_idx.append(item_id_to_idx[int(publication_id)])
            data.append(float(score))

        user_items = csr_matrix(
            (data, (row_idx, col_idx)),
            shape=(len(user_ids), len(item_ids)),
            dtype=np.float32,
        )

        model = implicit.als.AlternatingLeastSquares(
            factors=self.settings.als_factors,
            regularization=self.settings.als_regularization,
            iterations=self.settings.als_iterations,
            random_state=42,
        )
        model.fit(user_items.T)

        return RecommenderBundle(
            model=model,
            user_id_to_idx=user_id_to_idx,
            idx_to_user_id=user_ids,
            item_id_to_idx=item_id_to_idx,
            idx_to_item_id=item_ids,
            user_items_csr=user_items,
        )

    def _fetch_weighted_interactions(self, conn: PGConnection) -> list[tuple[int, int, float]]:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (self.settings.statement_timeout_ms,))
            cur.execute(
                """
                SELECT
                    user_id,
                    publication_id,
                    SUM(
                        CASE action_type
                            WHEN 'VIEW' THEN 1
                            WHEN 'WISHLIST' THEN 3
                            WHEN 'BORROW' THEN 5
                            ELSE 0
                        END
                    )::float AS score
                FROM library_core.user_interactions
                GROUP BY user_id, publication_id
                HAVING SUM(
                    CASE action_type
                        WHEN 'VIEW' THEN 1
                        WHEN 'WISHLIST' THEN 3
                        WHEN 'BORROW' THEN 5
                        ELSE 0
                    END
                ) > 0
                """
            )
            return cur.fetchall()

    @staticmethod
    def _load_bundle(path: Path) -> RecommenderBundle:
        with path.open("rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, RecommenderBundle):
            raise ValueError("Invalid ALS bundle file format")
        return loaded
