import sqlite3
from pathlib import Path

import faiss
import numpy as np

from config import DB_PATH, EMBED_DIM, INDEX_PATH, THUMBNAILS_DIR


class IndexStore:
    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self._conn: sqlite3.Connection | None = None

    def _ensure_dirs(self):
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._ensure_dirs()
            self._conn = sqlite3.connect(DB_PATH)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paintings (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    title TEXT,
                    artist TEXT
                )
                """
            )
            self._conn.commit()
        return self._conn

    def create_index(self):
        self._ensure_dirs()
        self.index = faiss.IndexFlatIP(EMBED_DIM)

    def load(self) -> bool:
        self._ensure_dirs()
        conn = self._connect()
        if not INDEX_PATH.exists():
            return False
        self.index = faiss.read_index(str(INDEX_PATH))
        # SQLite is committed per batch, while FAISS is checkpointed periodically.
        # After an interruption, discard metadata newer than the last FAISS checkpoint.
        conn.execute("DELETE FROM paintings WHERE id >= ?", (self.index.ntotal,))
        conn.commit()
        metadata_count = conn.execute("SELECT COUNT(*) FROM paintings").fetchone()[0]
        if metadata_count != self.index.ntotal:
            raise RuntimeError(
                f"Index has {self.index.ntotal} vectors but metadata has "
                f"{metadata_count} rows"
            )
        return True

    def save(self):
        if self.index is None:
            raise RuntimeError("No index to save")
        temporary_path = INDEX_PATH.with_suffix(f"{INDEX_PATH.suffix}.tmp")
        faiss.write_index(self.index, str(temporary_path))
        temporary_path.replace(INDEX_PATH)

    def add(self, vector: np.ndarray, path: str, title: str | None = None, artist: str | None = None) -> int:
        return self.add_batch(
            vector.reshape(1, -1),
            [(path, title, artist)],
        )[0]

    def add_batch(
        self,
        vectors: np.ndarray,
        metadata: list[tuple[str, str | None, str | None]],
    ) -> list[int]:
        if self.index is None:
            self.create_index()
        if len(vectors) != len(metadata):
            raise ValueError("Vector and metadata batch sizes differ")
        if not metadata:
            return []

        first_id = self.index.ntotal
        row_ids = list(range(first_id, first_id + len(metadata)))
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

        conn = self._connect()
        try:
            conn.executemany(
                "INSERT INTO paintings (id, path, title, artist) VALUES (?, ?, ?, ?)",
                [
                    (row_id, path, title, artist)
                    for row_id, (path, title, artist) in zip(row_ids, metadata)
                ],
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return row_ids

    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(vector.reshape(1, -1), k)
        results = []
        for score, row_id in zip(scores[0], ids[0]):
            if row_id >= 0:
                results.append((int(row_id), float(score)))
        return results

    def get_by_id(self, row_id: int) -> dict | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT id, path, title, artist FROM paintings WHERE id = ?",
            (row_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def count(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
