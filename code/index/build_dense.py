"""Build a document-level FAISS index for BGE-M3 or Qwen3-Embedding.

Why document level?
-------------------
Encoding one million passage chunks is the main runtime bottleneck.  This
builder instead creates 1-2 windows per document (``--doc-windows auto``), so a
full corpus needs roughly 100k-200k vectors rather than 1M.  BM25 remains
chunk-level and supplies fine-grained lexical matching.

Output (under INDEX_DIR/dense_bge or dense_qwen3):
    index.faiss
    doc_ids.json          # one entry per vector (documents may repeat)
    doc_windows.npy       # windows per document
    offsets.npy           # cumulative vector offsets
    meta.json
    progress.json         # removed after a successful build
    embeddings.f32        # removed unless --keep-embeddings
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).absolute().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import config  # noqa: E402
from index.embedding import backend_directory, configure_hf_environment, load_embedder  # noqa: E402

WORD_RE = re.compile(r"\S+")


def _default_model(backend: str) -> str:
    if backend == "bge":
        local = config.MODELS_DIR / "bge-m3"
        return str(local) if local.exists() else config.BGE_MODEL_NAME
    if backend == "qwen3":
        local = config.MODELS_DIR / "qwen3-embedding-0.6b"
        return str(local) if local.exists() else config.QWEN3_MODEL_NAME
    return config.BGE_FALLBACK_MODEL_NAME


def _count_docs(db_path: Path, limit: int) -> int:
    connection = sqlite3.connect(str(db_path))
    try:
        total = connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    finally:
        connection.close()
    return min(total, limit) if limit else total


def _iter_docs(db_path: Path, n_docs: int):
    connection = sqlite3.connect(str(db_path))
    cursor = connection.execute(
        "SELECT docid, title, text, n_words FROM docs ORDER BY rowid LIMIT ?",
        (n_docs,),
    )
    try:
        while True:
            rows = cursor.fetchmany(200)
            if not rows:
                break
            for docid, title, text, n_words in rows:
                yield str(docid), title or "", text or "", int(n_words or 0)
    finally:
        connection.close()


def plan_windows(
    db_path: Path,
    n_docs: int,
    mode: str,
    window_words: int,
    auto_threshold: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """First pass: decide 1 or 2 windows per document without loading text."""
    doc_ids: list[str] = []
    windows = np.zeros(n_docs, dtype=np.int8)
    connection = sqlite3.connect(str(db_path))
    try:
        cursor = connection.execute(
            "SELECT docid, n_words FROM docs ORDER BY rowid LIMIT ?", (n_docs,)
        )
        for index, (docid, n_words) in enumerate(cursor):
            n_words = int(n_words or 0)
            if mode == "1":
                count = 1
            elif mode == "2":
                count = 2 if n_words > window_words else 1
            else:  # auto
                count = 2 if n_words > auto_threshold else 1
            doc_ids.append(str(docid))
            windows[index] = count
    finally:
        connection.close()

    offsets = np.zeros(n_docs + 1, dtype=np.int64)
    np.cumsum(windows, out=offsets[1:])
    return doc_ids, windows, offsets


def _plan_matches(plan_path: Path, n_docs: int, mode: str,
                  window_words: int, auto_threshold: int) -> bool:
    if not plan_path.exists():
        return False
    try:
        old = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        old.get("n_docs") == n_docs
        and old.get("doc_windows") == mode
        and old.get("window_words") == window_words
        and old.get("auto_threshold") == auto_threshold
    )


def build_window_texts(title: str, text: str, count: int, window_words: int) -> list[str]:
    spans = [match.span() for match in WORD_RE.finditer(text)]
    total = len(spans)
    if total == 0:
        return [title or ""]
    if total <= window_words or count <= 1:
        piece = text[spans[0][0] : spans[min(total, window_words) - 1][1]]
        return [(title + "\n" + piece).strip() if title else piece.strip()]

    starts = [0, max(0, int(total * 0.5) - window_words // 2)]
    starts[1] = min(starts[1], max(0, total - window_words))
    texts = []
    for start in starts[:count]:
        end = min(total, start + window_words)
        piece = text[spans[start][0] : spans[end - 1][1]]
        texts.append((title + "\n" + piece).strip() if title else piece.strip())
    return texts


def _is_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "out of memory" in message or "cuda" in message and "memory" in message
    )


def _encode_with_retry(embedder, texts: list[str], batch_size: int) -> np.ndarray:
    try:
        return embedder.encode_passages(texts, batch_size=batch_size)
    except BaseException as exc:
        if not _is_oom(exc):
            raise
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"  CUDA OOM; reducing batch (texts={len(texts)}, batch={batch_size})")
        if len(texts) > 1:
            middle = len(texts) // 2
            left = _encode_with_retry(embedder, texts[:middle], max(1, batch_size // 2))
            right = _encode_with_retry(embedder, texts[middle:], max(1, batch_size // 2))
            return np.concatenate([left, right], axis=0)
        if batch_size > 1:
            return _encode_with_retry(embedder, texts, max(1, batch_size // 2))
        raise


def encode_documents(
    db_path: Path,
    emb_path: Path,
    progress_path: Path,
    out_dir: Path,
    n_docs: int,
    doc_ids: list[str],
    windows: np.ndarray,
    offsets: np.ndarray,
    embedder,
    batch_size: int,
    window_words: int,
    resume: bool,
) -> np.ndarray:
    total_vectors = int(offsets[-1])
    if total_vectors == 0:
        raise SystemExit("No document windows to encode.")

    start_docs = 0
    dim = 0
    if resume and progress_path.exists() and emb_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("n_docs") == n_docs and progress.get("total_vectors") == total_vectors:
            start_docs = int(progress.get("docs_done", 0))
            dim = int(progress.get("dim", 0))
            print(f"Resuming at doc {start_docs:,}/{n_docs:,} "
                  f"(vector {int(offsets[start_docs]):,}/{total_vectors:,})")

    if start_docs and dim:
        matrix = np.memmap(emb_path, dtype=np.float32, mode="r+", shape=(total_vectors, dim))
    else:
        probe = embedder.encode_passages(["embedding dimension probe"], batch_size=1)
        dim = int(probe.shape[1])
        matrix = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(total_vectors, dim))
        matrix[:] = 0
        start_docs = 0
        progress_path.write_text(
            json.dumps({"docs_done": 0, "n_docs": n_docs, "total_vectors": total_vectors, "dim": dim}),
            encoding="utf-8",
        )

    print(f"Embedding {n_docs:,} docs -> {total_vectors:,} vectors (dim={dim})")

    started = time.time()
    batch_texts: list[str] = []
    batch_docs: list[int] = []
    vectors_done = int(offsets[start_docs])
    docs_done = start_docs

    def flush() -> None:
        nonlocal batch_texts, batch_docs, vectors_done, docs_done
        if not batch_texts:
            return
        vectors = _encode_with_retry(embedder, batch_texts, batch_size)
        if vectors.shape[1] != dim:
            raise RuntimeError(f"embedding dim changed: {vectors.shape[1]} != {dim}")
        expected = int(offsets[batch_docs[-1] + 1]) - int(offsets[batch_docs[0]])
        if len(vectors) != expected:
            raise RuntimeError(
                f"window count mismatch: generated {len(vectors)} vectors, expected {expected}"
            )
        end = vectors_done + len(vectors)
        matrix[vectors_done:end] = vectors
        matrix.flush()
        vectors_done = end
        docs_done = batch_docs[-1] + 1
        progress_path.write_text(
            json.dumps({"docs_done": docs_done, "n_docs": n_docs,
                        "total_vectors": total_vectors, "dim": dim}),
            encoding="utf-8",
        )
        elapsed = max(time.time() - started, 1e-6)
        rate = (docs_done - start_docs) / elapsed
        eta = (n_docs - docs_done) / rate / 60 if rate else 0
        print(f"  {docs_done:,}/{n_docs:,} docs | {vectors_done:,}/{total_vectors:,} vectors | "
              f"{rate:.1f} docs/s | ETA {eta:.1f} min", flush=True)
        batch_texts = []
        batch_docs = []

    for index, (docid, title, text, _n_words) in enumerate(_iter_docs(db_path, n_docs)):
        if index < start_docs:
            continue
        count = int(windows[index])
        texts = build_window_texts(title, text, count, window_words)
        if not texts:
            texts = [title or docid]
        batch_texts.extend(texts)
        batch_docs.append(index)
        if len(batch_docs) >= batch_size or len(batch_texts) >= batch_size * 3:
            flush()

    flush()
    if docs_done != n_docs:
        raise SystemExit(f"Only {docs_done}/{n_docs} docs encoded; re-run with --resume.")
    return matrix


def choose_pq_m(dim: int, requested: int) -> int:
    if requested and dim % requested == 0:
        return requested
    for candidate in range(min(requested or dim, dim), 0, -1):
        if dim % candidate == 0:
            return candidate
    return 1


def build_faiss(matrix: np.ndarray, n: int, dim: int, nlist: int, pq_m: int,
                flat_threshold: int):
    import faiss

    nlist_effective = max(1, min(nlist, n // 39)) if n >= 39 else 1
    use_ivf = n > flat_threshold and nlist_effective >= 2

    if not use_ivf:
        print(f"Building IndexFlatIP for {n:,} vectors")
        index = faiss.IndexFlatIP(dim)
        for start in range(0, n, 100_000):
            index.add(np.asarray(matrix[start : min(start + 100_000, n)], dtype=np.float32))
        return index, {"index_type": "flat", "nlist": 0, "pq_m": 0}

    train_n = min(n, max(nlist_effective * 50, 50_000), 300_000)
    print(f"Building IVF{nlist_effective}_PQ{pq_m} for {n:,} vectors "
          f"(train {train_n:,})")
    sample = np.linspace(0, n - 1, train_n, dtype=np.int64)
    train_vectors = np.asarray(matrix[sample], dtype=np.float32)
    index = faiss.IndexIVFPQ(
        faiss.IndexFlatL2(dim), dim, nlist_effective, pq_m, 8, faiss.METRIC_L2
    )
    index.train(train_vectors)
    del train_vectors
    for start in range(0, n, 100_000):
        index.add(np.asarray(matrix[start : min(start + 100_000, n)], dtype=np.float32))
    index.nprobe = max(8, min(32, nlist_effective // 32))
    return index, {"index_type": "ivfpq", "nlist": nlist_effective, "pq_m": pq_m}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["bge", "qwen3", "bge-small"], default="bge")
    parser.add_argument("--level", choices=["doc", "chunk"], default="doc")
    parser.add_argument("--db", type=Path, default=config.CORPUS_DB)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=config.EMBED_DEVICE)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--doc-windows", choices=["auto", "1", "2"], default=config.DENSE_DOC_WINDOWS)
    parser.add_argument("--window-words", type=int, default=config.DENSE_WINDOW_WORDS)
    parser.add_argument("--auto-threshold", type=int, default=config.DENSE_AUTO_SECOND_WINDOW_WORDS)
    parser.add_argument("--limit", type=int, default=0, help="Encode at most N documents.")
    parser.add_argument("--nlist", type=int, default=config.DENSE_NLIST)
    parser.add_argument("--pq", type=int, default=0)
    parser.add_argument("--flat-threshold", type=int, default=config.DENSE_FLAT_THRESHOLD)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-embeddings", action="store_true")
    args = parser.parse_args()

    if args.level != "doc":
        raise SystemExit("This lean build supports --level doc only. "
                         "Use the old chunk builder if you need chunk vectors.")
    configure_hf_environment()
    args.db = Path(args.db).expanduser().resolve()
    if not args.db.exists():
        raise SystemExit(f"Corpus database not found: {args.db}. Run data/preprocess.py first.")

    try:
        import faiss  # noqa: F401
    except ImportError as exc:
        raise SystemExit("faiss is not installed. `pip install faiss-cpu`.") from exc

    out_dir = Path(args.out).expanduser().resolve() if args.out else backend_directory(args.backend)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_docs = _count_docs(args.db, args.limit)
    if n_docs == 0:
        raise SystemExit("No documents found.")
    if args.batch_size <= 0:
        args.batch_size = config.DENSE_BATCH_QWEN3 if args.backend == "qwen3" else config.DENSE_BATCH_BGE

    print(f"Backend: {args.backend} | docs: {n_docs:,} | level: doc | windows: {args.doc_windows}")
    print(f"Output: {out_dir}")

    plan_path = out_dir / "plan.json"
    doc_ids_path = out_dir / "doc_ids.json"
    windows_path = out_dir / "doc_windows.npy"
    offsets_path = out_dir / "offsets.npy"
    if args.resume and _plan_matches(plan_path, n_docs, args.doc_windows,
                                      args.window_words, args.auto_threshold) \
            and doc_ids_path.exists() and windows_path.exists() and offsets_path.exists():
        doc_ids = json.loads(doc_ids_path.read_text(encoding="utf-8"))
        windows = np.load(windows_path)
        offsets = np.load(offsets_path)
        print("Reusing existing window plan.")
    else:
        print("Planning document windows ...")
        doc_ids, windows, offsets = plan_windows(
            args.db, n_docs, args.doc_windows, args.window_words, args.auto_threshold
        )
        (out_dir / "plan.json").write_text(
            json.dumps({"n_docs": n_docs, "doc_windows": args.doc_windows,
                        "window_words": args.window_words,
                        "auto_threshold": args.auto_threshold}),
            encoding="utf-8",
        )
        doc_ids_path.write_text(json.dumps(doc_ids), encoding="utf-8")
        np.save(windows_path, windows)
        np.save(offsets_path, offsets)

    total_vectors = int(offsets[-1])
    print(f"Window plan: {total_vectors:,} vectors for {n_docs:,} docs")

    model_name = args.model or _default_model(args.backend)
    embedder = load_embedder(args.backend, device=args.device, model_name=model_name)
    print(f"Model: {embedder.model_name} | device: {embedder.device} | batch: {args.batch_size}")

    emb_path = out_dir / "embeddings.f32"
    progress_path = out_dir / "progress.json"
    try:
        matrix = encode_documents(
            args.db, emb_path, progress_path, out_dir, n_docs, doc_ids, windows,
            offsets, embedder, args.batch_size, args.window_words, args.resume,
        )
    except KeyboardInterrupt:
        print("\nInterrupted; re-run with --resume.", file=sys.stderr)
        raise SystemExit(130)

    dim = int(matrix.shape[1])
    pq_m = args.pq or (128 if args.backend == "qwen3" else 64)
    pq_m = choose_pq_m(dim, pq_m)
    index, index_meta = build_faiss(matrix, total_vectors, dim, args.nlist, pq_m, args.flat_threshold)

    import faiss

    faiss.write_index(index, str(out_dir / "index.faiss"))
    meta = {
        "level": "doc",
        "backend": args.backend,
        "model_name": embedder.model_name,
        "dim": dim,
        "n_docs": n_docs,
        "n_vectors": total_vectors,
        "doc_windows": args.doc_windows,
        "window_words": args.window_words,
        "auto_threshold": args.auto_threshold,
        "normalized": True,
        "query_prefix": embedder.query_prefix,
        "index_type": index_meta["index_type"],
        "nlist": index_meta["nlist"],
        "pq_m": index_meta["pq_m"],
        "metric": "ip" if index_meta["index_type"] == "flat" else "l2-on-normalized",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database": str(args.db),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved FAISS index to {out_dir}")

    if not args.keep_embeddings:
        matrix.flush()
        del matrix
        emb_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        print("Removed intermediate embeddings to save disk.")

    print("Next: build the other backend or run validation.")


if __name__ == "__main__":
    main()
