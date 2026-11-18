"""Unified retrieval: BM25 chunk-level + document-level dense + RRF.

BM25 scores chunks and is collapsed to documents; BGE-M3 and Qwen3 score whole
documents through 1-2 windows.  The three ranked document lists are fused with
Reciprocal Rank Fusion (k=60).  This keeps the three required retrieval
categories while cutting dense encoding from ~1M chunks to ~100k-200k vectors.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import config
from index.embedding import load_embedder
from index.text_utils import make_snippet, tokenize_text

DENSE_METHODS = ("bge", "qwen3")


class RetrievalError(RuntimeError):
    """No usable retrieval index."""


@dataclass
class SearchResult:
    doc_id: str
    score: float
    snippet: str
    url: str = ""
    title: str = ""
    date: str = ""
    sources: list[str] = field(default_factory=list)
    raw_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": round(float(self.score), 6),
            "snippet": self.snippet,
            "url": self.url,
            "title": self.title,
            "date": self.date,
            "sources": self.sources,
        }


class UnifiedRetriever:
    def __init__(
        self,
        db_path: Path | str | None = None,
        index_dir: Path | str | None = None,
        device: str | None = None,
        nprobe: int | None = None,
    ):
        self.db_path = Path(db_path or config.CORPUS_DB).expanduser().resolve()
        self.index_dir = Path(index_dir or config.INDEX_DIR).expanduser().resolve()
        self.device = device
        self.nprobe = nprobe

        self._lock = threading.RLock()
        self._bm25 = None
        self._bm25_chunk_ids: np.ndarray | None = None
        self._bm25_vocab: dict[str, int] | None = None
        self._dense: dict[str, dict[str, Any]] = {}

        self.bm25_dir = self.index_dir / "bm25"
        self.dense_dirs = {
            "bge": self.index_dir / "dense_bge",
            "qwen3": self.index_dir / "dense_qwen3",
        }

    # ------------------------------------------------------------------
    # Availability / loading
    # ------------------------------------------------------------------
    def _bm25_available(self) -> bool:
        needed = [
            "data.csc.index.npy",
            "indices.csc.index.npy",
            "indptr.csc.index.npy",
            "vocab.index.json",
            "params.index.json",
            "chunk_ids.npy",
        ]
        return all((self.bm25_dir / name).exists() for name in needed)

    def _dense_available(self, name: str) -> bool:
        directory = self.dense_dirs.get(name)
        if directory is None:
            return False
        return all(
            (directory / filename).exists()
            for filename in ("index.faiss", "doc_ids.json", "meta.json")
        )

    def available_methods(self) -> list[str]:
        methods = []
        if self._bm25_available():
            methods.append("bm25")
        for name in DENSE_METHODS:
            if self._dense_available(name):
                methods.append(name)
        return methods

    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        with self._lock:
            if self._bm25 is not None:
                return
            try:
                import bm25s
            except ImportError as exc:  # pragma: no cover
                raise RetrievalError("bm25s is missing. `pip install bm25s`.") from exc
            if not self._bm25_available():
                raise RetrievalError(f"BM25 index not found under {self.bm25_dir}.")
            print(f"Loading BM25 index from {self.bm25_dir} ...")
            self._bm25 = bm25s.BM25.load(str(self.bm25_dir), mmap=True, load_corpus=False)
            self._bm25_chunk_ids = np.load(self.bm25_dir / "chunk_ids.npy", mmap_mode="r")
            self._bm25_vocab = dict(self._bm25.vocab_dict)

    def _ensure_dense(self, name: str):
        if name in self._dense:
            return self._dense[name]
        with self._lock:
            if name in self._dense:
                return self._dense[name]
            directory = self.dense_dirs.get(name)
            if directory is None or not self._dense_available(name):
                raise RetrievalError(f"Dense index '{name}' not found under {directory}.")
            try:
                import faiss
            except ImportError as exc:  # pragma: no cover
                raise RetrievalError("faiss is missing. `pip install faiss-cpu`.") from exc

            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            print(f"Loading {name} doc index from {directory} ...")
            index = faiss.read_index(str(directory / "index.faiss"))
            if meta.get("index_type") == "ivfpq":
                nlist = max(1, int(meta.get("nlist", 4096)))
                index.nprobe = self.nprobe or max(8, min(32, nlist // 32))

            embedder = load_embedder(
                meta.get("backend", name),
                device=self.device,
                model_name=meta.get("model_name"),
            )
            embedder.query_prefix = meta.get("query_prefix") or embedder.query_prefix

            doc_ids = json.loads((directory / "doc_ids.json").read_text(encoding="utf-8"))
            self._dense[name] = {
                "index": index,
                "doc_ids": doc_ids,
                "meta": meta,
                "embedder": embedder,
                "lock": threading.Lock(),
            }
            return self._dense[name]

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise RetrievalError(f"Corpus database not found: {self.db_path}")
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _fetch_chunks(self, chunk_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        chunk_ids = [int(cid) for cid in chunk_ids]
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        connection = self._connect()
        try:
            rows = connection.execute(
                f"SELECT chunk_id, docid, text FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            return {
                int(row["chunk_id"]): {
                    "doc_id": str(row["docid"]),
                    "text": row["text"] or "",
                }
                for row in rows
            }
        finally:
            connection.close()

    def _fetch_docs(self, doc_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        doc_ids = [str(docid) for docid in doc_ids]
        if not doc_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_ids)
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT docid, url, domain, title, date, text FROM docs "
                f"WHERE docid IN ({placeholders})",
                doc_ids,
            )
            return {str(row["docid"]): dict(row) for row in rows}
        finally:
            connection.close()

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT docid, url, domain, title, date, text, n_chars, n_words "
                "FROM docs WHERE docid = ?",
                (str(doc_id),),
            ).fetchone()
            return dict(row) if row else None
        finally:
            connection.close()

    # ------------------------------------------------------------------
    # Individual retrieval methods: both return ranked DOCUMENTS
    # ------------------------------------------------------------------
    def _bm25_docs(self, query: str, pool: int) -> list[dict[str, Any]]:
        self._ensure_bm25()
        assert self._bm25 is not None and self._bm25_chunk_ids is not None
        tokens = tokenize_text(query)
        if not tokens or self._bm25_vocab is None:
            return []
        token_ids = [self._bm25_vocab[t] for t in tokens if t in self._bm25_vocab]
        if not token_ids:
            return []

        scores = np.asarray(self._bm25.get_scores_from_ids(token_ids), dtype=np.float32)
        n = len(scores)
        if n == 0:
            return []
        pool = min(max(pool, 30), n)
        top = np.argpartition(scores, -pool)[-pool:]
        top = top[np.argsort(scores[top])[::-1]]

        candidate_chunk_ids = [
            int(self._bm25_chunk_ids[position]) for position in top if scores[position] > 0
        ]
        chunks = self._fetch_chunks(candidate_chunk_ids)
        docs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for position in top:
            score = float(scores[position])
            if score <= 0:
                continue
            chunk_id = int(self._bm25_chunk_ids[position])
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            doc_id = chunk["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            docs.append(
                {
                    "doc_id": doc_id,
                    "raw_score": score,
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                }
            )
        return docs

    def _dense_docs(self, name: str, query: str, pool: int) -> list[dict[str, Any]]:
        state = self._ensure_dense(name)
        index = state["index"]
        meta = state["meta"]
        doc_ids: list[str] = state["doc_ids"]
        embedder = state["embedder"]
        with state["lock"]:
            query_vector = np.asarray(embedder.encode_query(query), dtype=np.float32).reshape(1, -1)
            pool = min(pool, int(index.ntotal))
            if pool <= 0:
                return []
            distances, indices = index.search(query_vector, pool)

        docs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for distance, position in zip(distances[0], indices[0]):
            position = int(position)
            if position < 0 or position >= len(doc_ids):
                continue
            if meta.get("index_type") == "ivfpq":
                score = 1.0 - float(distance) / 2.0
            else:
                score = float(distance)
            doc_id = str(doc_ids[position])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            docs.append({"doc_id": doc_id, "raw_score": score})
        return docs

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = config.DEFAULT_RETRIEVAL_K,
        method: str = "hybrid",
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        k = max(1, min(int(k), config.MAX_RETRIEVAL_K))
        available = self.available_methods()
        if not available:
            raise RetrievalError("No retrieval index found. Build BM25 and/or dense indexes first.")

        method = (method or "hybrid").lower()
        methods = available if method == "hybrid" or method not in available else [method]
        pool = min(
            max(k * config.RETRIEVAL_POOL_FACTOR, config.RETRIEVAL_POOL_MIN),
            config.RETRIEVAL_POOL_MAX,
        )

        ranked_lists: dict[str, list[dict[str, Any]]] = {}
        failures: dict[str, str] = {}
        for name in methods:
            try:
                if name == "bm25":
                    ranked_lists[name] = self._bm25_docs(query, pool)
                else:
                    ranked_lists[name] = self._dense_docs(name, query, pool)
            except Exception as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"
                ranked_lists[name] = []
                print(f"WARNING: retrieval method '{name}' failed: {exc}")
        if not any(ranked_lists.values()) and failures:
            raise RetrievalError(
                "All requested retrieval methods failed: "
                + "; ".join(f"{name}: {message}" for name, message in failures.items())
            )

        fused: dict[str, dict[str, Any]] = {}
        for name, docs in ranked_lists.items():
            for rank, entry in enumerate(docs, start=1):
                doc_id = entry["doc_id"]
                slot = fused.setdefault(
                    doc_id,
                    {
                        "score": 0.0,
                        "sources": [],
                        "best_chunk_id": None,
                        "bm25_text": None,
                        "raw_scores": {},
                    },
                )
                slot["score"] += 1.0 / (config.RRF_K + rank)
                slot["sources"].append(name)
                slot["raw_scores"][name] = float(entry["raw_score"])
                if name == "bm25" and slot["best_chunk_id"] is None:
                    slot["best_chunk_id"] = entry.get("chunk_id")
                    slot["bm25_text"] = entry.get("text")

        ordered = sorted(fused.items(), key=lambda item: -item[1]["score"])
        top_docs = [doc_id for doc_id, _ in ordered[:k]]
        doc_rows = self._fetch_docs(top_docs)

        results: list[SearchResult] = []
        for doc_id in top_docs:
            slot = fused[doc_id]
            doc = doc_rows.get(doc_id)
            if doc is None:
                continue
            if slot.get("bm25_text"):
                snippet = make_snippet(slot["bm25_text"], query=query, max_chars=config.MAX_SNIPPET_CHARS)
            else:
                snippet = make_snippet(doc.get("text", ""), query=query, max_chars=config.MAX_SNIPPET_CHARS)
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=slot["score"],
                    snippet=snippet,
                    url=str(doc.get("url") or ""),
                    title=str(doc.get("title") or ""),
                    date=str(doc.get("date") or ""),
                    sources=list(slot["sources"]),
                    raw_scores=dict(slot["raw_scores"]),
                )
            )
        return [result.to_dict() for result in results]

    def stats(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "database": str(self.db_path),
            "database_exists": self.db_path.exists(),
            "available_methods": self.available_methods(),
            "index_dir": str(self.index_dir),
        }
        if self.db_path.exists():
            connection = self._connect()
            try:
                info["documents"] = connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
                info["chunks"] = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            finally:
                connection.close()
        for name in DENSE_METHODS:
            if self._dense_available(name):
                try:
                    meta = json.loads((self.dense_dirs[name] / "meta.json").read_text(encoding="utf-8"))
                    info[f"{name}_vectors"] = meta.get("n_vectors")
                    info[f"{name}_model"] = meta.get("model_name")
                except Exception:
                    pass
        return info


_RETRIEVER: UnifiedRetriever | None = None
_RETRIEVER_LOCK = threading.Lock()


def get_retriever(
    db_path: Path | str | None = None,
    index_dir: Path | str | None = None,
) -> UnifiedRetriever:
    global _RETRIEVER
    if _RETRIEVER is None or db_path is not None or index_dir is not None:
        with _RETRIEVER_LOCK:
            if _RETRIEVER is None or db_path is not None or index_dir is not None:
                _RETRIEVER = UnifiedRetriever(db_path=db_path, index_dir=index_dir)
    return _RETRIEVER
