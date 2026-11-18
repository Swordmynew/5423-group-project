"""Build the BM25S chunk index.

The corpus is large (~500M words), so the token stream is written to a disk
memmap instead of being kept as a Python list.  BM25S still performs the actual
index construction; it simply receives a disk-backed sequence of token IDs.

Example
-------
python index/build_bm25.py
python index/build_bm25.py --limit 20000 --out index/bm25_dev
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from array import array
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).absolute().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import config  # noqa: E402
from index.text_utils import tokenize_text  # noqa: E402


class DiskTokenIds:
    """Sequence of token-ID arrays backed by ``tokens.i32`` + ``offsets.npy``."""

    def __init__(self, tokens_path: Path, offsets_path: Path):
        self.tokens = np.memmap(tokens_path, dtype=np.int32, mode="r")
        self.offsets = np.load(offsets_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(len(self.offsets) - 1)

    def __getitem__(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return np.asarray(self.tokens[start:end], dtype=np.int32)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def count_chunks(db_path: Path, limit: int) -> int:
    connection = sqlite3.connect(str(db_path))
    try:
        if limit:
            return min(limit, connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        return connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        connection.close()


def write_token_stream(db_path: Path, tokens_path: Path, offsets_path: Path, limit: int):
    """Tokenise every chunk once, assigning integer IDs on the fly."""
    vocab: dict[str, int] = {}
    offsets = array("q", [0])
    buffer = array("i")
    chunk_ids = array("q")
    written_tokens = 0
    written_chunks = 0
    started = time.time()

    connection = sqlite3.connect(str(db_path))
    cursor = connection.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id")
    with tokens_path.open("wb") as token_file:
        while True:
            rows = cursor.fetchmany(2000)
            if not rows:
                break
            for chunk_id, text in rows:
                if limit and written_chunks >= limit:
                    break
                ids = []
                for token in tokenize_text(text):
                    token_id = vocab.get(token)
                    if token_id is None:
                        token_id = len(vocab)
                        vocab[token] = token_id
                    ids.append(token_id)
                if not ids:
                    # BM25S can score empty documents, but they are useless and
                    # make its empty-token bookkeeping noisier.
                    ids = []
                buffer.extend(ids)
                written_tokens += len(ids)
                offsets.append(written_tokens)
                chunk_ids.append(int(chunk_id))
                written_chunks += 1

                if len(buffer) >= 4_000_000:
                    buffer.tofile(token_file)
                    buffer = array("i")

                if written_chunks % 5000 == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        f"  tokenised {written_chunks:,} chunks | "
                        f"{written_tokens:,} tokens | {written_chunks / elapsed:.0f} chunks/s",
                        flush=True,
                    )
            if limit and written_chunks >= limit:
                break
        if buffer:
            buffer.tofile(token_file)

    connection.close()
    if written_tokens == 0:
        raise SystemExit("No tokens were produced from the chunks table. Check the corpus text.")
    np.save(offsets_path, np.asarray(offsets, dtype=np.int64))
    np.save(tokens_path.with_name("chunk_ids.npy"), np.asarray(chunk_ids, dtype=np.int64))

    with (tokens_path.parent / "vocab.json").open("w", encoding="utf-8") as handle:
        json.dump(vocab, handle, ensure_ascii=False)

    return vocab, written_chunks, written_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=config.CORPUS_DB)
    parser.add_argument("--out", type=Path, default=config.BM25_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Index at most N chunks (debug).")
    parser.add_argument("--keep-tokens", action="store_true", help="Keep the intermediate token stream.")
    parser.add_argument("--k1", type=float, default=config.BM25_K1)
    parser.add_argument("--b", type=float, default=config.BM25_B)
    args = parser.parse_args()

    try:
        import bm25s
        from bm25s.tokenization import Tokenized
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "bm25s is not installed. Install it with `pip install bm25s`."
        ) from exc

    args.db = Path(args.db).expanduser().resolve()
    args.out = Path(args.out).expanduser().resolve()
    if not args.db.exists():
        raise SystemExit(f"Corpus database not found: {args.db}. Run data/preprocess.py first.")
    args.out.mkdir(parents=True, exist_ok=True)

    n_chunks = count_chunks(args.db, args.limit)
    if n_chunks == 0:
        raise SystemExit("The chunks table is empty. Re-run data/preprocess.py.")

    tokens_path = args.out / "tokens.i32"
    offsets_path = args.out / "offsets.npy"
    print(f"Building BM25S index for {n_chunks:,} chunks ...")
    print(f"  token stream: {tokens_path}")
    vocab, n_chunks, n_tokens = write_token_stream(
        args.db, tokens_path, offsets_path, args.limit
    )
    print(f"Tokenisation complete: {n_chunks:,} chunks, {n_tokens:,} tokens, {len(vocab):,} terms")

    print("Building the BM25S sparse matrix (this is the memory-intensive step) ...")
    token_ids = DiskTokenIds(tokens_path, offsets_path)
    tokenized = Tokenized(ids=token_ids, vocab=vocab)
    try:
        retriever = bm25s.BM25(
            k1=args.k1,
            b=args.b,
            method="lucene",
            backend="numpy",
            csc_backend="scipy",
        )
    except (ImportError, ValueError, TypeError):
        retriever = bm25s.BM25(k1=args.k1, b=args.b, method="lucene", backend="numpy")
    build_started = time.time()
    retriever.index(tokenized, create_empty_token=True, show_progress=True)
    build_seconds = time.time() - build_started

    print(f"Saving index to {args.out} ...")
    retriever.save(str(args.out))

    meta = {
        "n_chunks": int(n_chunks),
        "n_tokens": int(n_tokens),
        "vocab_size": int(len(vocab)),
        "avg_tokens_per_chunk": round(n_tokens / max(n_chunks, 1), 2),
        "k1": args.k1,
        "b": args.b,
        "method": "lucene",
        "tokenizer": "lowercase alphanumeric, English stopwords removed",
        "chunk_ids_file": "chunk_ids.npy",
        "database": str(args.db),
        "build_seconds": round(build_seconds, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not args.keep_tokens:
        # Release the memmap before deleting the token stream (required on Windows).
        del token_ids
        del tokenized
        tokens_path.unlink(missing_ok=True)
        offsets_path.unlink(missing_ok=True)
        vocab_path = args.out / "vocab.json"
        vocab_path.unlink(missing_ok=True)
    print("Done.")
    print("Next step: build a dense index -> python index/build_dense.py --backend bge")


if __name__ == "__main__":
    main()
