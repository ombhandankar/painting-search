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
        self._connect()
        if not INDEX_PATH.exists():
            return False
        self.index = faiss.read_index(str(INDEX_PATH))
        return True

    def save(self):
        if self.index is None:
            raise RuntimeError("No index to save")
        faiss.write_index(self.index, str(INDEX_PATH))

    def add(self, vector: np.ndarray, path: str, title: str | None = None, artist: str | None = None) -> int:
        if self.index is None:
            self.create_index()

        row_id = self.index.ntotal
        self.index.add(vector.reshape(1, -1))

        conn = self._connect()
        conn.execute(
            "INSERT INTO paintings (id, path, title, artist) VALUES (?, ?, ?, ?)",
            (row_id, path, title, artist),
        )
        conn.commit()
        return row_id

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
