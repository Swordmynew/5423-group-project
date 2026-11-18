"""Build the SQLite document/chunk store and FTS index.

The database contains ``docs``, ``chunks``, ``docs_fts`` and ``meta``.  Chunking
is word-based; the lean default is 1200 words with 150 words of overlap, which
keeps BM25 build memory and disk usage low while preserving passage-level
matching.

Examples
--------
python data/preprocess.py
python data/preprocess.py --rebuild
python data/preprocess.py --limit 5000 --db /tmp/dev.db
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

CODE_ROOT = Path(__file__).absolute().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import config  # noqa: E402

WORD_RE = re.compile(r"\S+")
TAG_RE = re.compile(r"<[^>]+>")
DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    docid   TEXT PRIMARY KEY,
    url     TEXT,
    domain  TEXT,
    title   TEXT,
    date    TEXT,
    text    TEXT,
    n_chars INTEGER,
    n_words INTEGER
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   INTEGER PRIMARY KEY,
    docid      TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    start_char INTEGER NOT NULL,
    end_char   INTEGER NOT NULL,
    text       TEXT NOT NULL,
    n_words    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_docid ON chunks(docid);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def clean_inline(text: str, max_chars: int = 200) -> str:
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars].strip()


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    closing = None
    for index in range(1, min(len(lines), 30)):
        if lines[index].strip() == "---":
            closing = index
            break
    if closing is None:
        return {}, text

    metadata: dict[str, str] = {}
    for line in lines[1:closing]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if key and value:
            metadata[key] = value
    body = "\n".join(lines[closing + 1 :]).lstrip("\n")
    return metadata, body


def infer_title(metadata: dict, text: str, url: str) -> str:
    for key in ("title", "headline", "name"):
        if metadata.get(key):
            return clean_inline(metadata[key])
    for line in text[:3000].splitlines():
        candidate = clean_inline(line)
        if 12 <= len(candidate) <= 200 and not candidate.lower().startswith("http"):
            if len(candidate) > 120:
                first_sentence = re.split(r"(?<=[.!?])\s+", candidate)[0].strip()
                candidate = first_sentence or candidate
            return candidate[:120].strip()
    parsed = urlparse(url)
    slug = parsed.path.rstrip("/").split("/")[-1]
    if slug:
        slug = re.sub(r"[-_]+", " ", slug)
        slug = re.sub(r"\.(html?|php|aspx?|jsp)$", "", slug, flags=re.I)
        candidate = clean_inline(slug)
        if candidate:
            return candidate
    return parsed.netloc or "Untitled document"


def infer_date(metadata: dict) -> str:
    value = metadata.get("date") or metadata.get("published") or metadata.get("published_date")
    if not value:
        return ""
    match = DATE_RE.search(value)
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else ""


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[tuple[int, int, int, str]]:
    spans = [match.span() for match in WORD_RE.finditer(text)]
    if not spans:
        return []
    total = len(spans)
    chunk_words = max(50, chunk_words)
    overlap_words = max(0, min(overlap_words, chunk_words // 2))
    step = max(1, chunk_words - overlap_words)
    min_tail = max(25, chunk_words // 4)

    chunks: list[tuple[int, int, int, str]] = []
    start_word = 0
    while start_word < total:
        end_word = min(start_word + chunk_words, total)
        if total - end_word < min_tail:
            end_word = total
        start_char = spans[start_word][0]
        end_char = spans[end_word - 1][1]
        chunks.append((start_char, end_char, end_word - start_word, text[start_char:end_char]))
        if end_word >= total:
            break
        start_word += step
    return chunks


def iter_records(patterns: list[str]):
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m).resolve() for m in matches)
        else:
            paths.append(Path(pattern).expanduser().resolve())
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            raise SystemExit(f"Corpus file not found: {path}")
        print(f"Reading {path} ...")
        if path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise SystemExit("pyarrow is missing. `pip install pyarrow`.") from exc
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=1000):
                yield from batch.to_pylist()
        elif path.suffix in {".jsonl", ".json", ".txt", ".gz"}:
            if path.suffix == ".gz":
                import gzip

                opener = gzip.open
            else:
                opener = open
            encoding = "utf-8" if path.suffix == ".gz" else "utf-8-sig"
            with opener(path, "r", encoding=encoding) as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        else:
            raise SystemExit(f"Unsupported corpus format: {path}")


def normalize_record(record: dict) -> tuple[str, str, str] | None:
    docid = record.get("docid") or record.get("doc_id") or record.get("id") or record.get("_id")
    text = record.get("text") or record.get("content") or record.get("body")
    url = record.get("url") or record.get("source_url") or record.get("link") or ""
    if docid is None or text is None:
        return None
    docid = str(docid).strip()
    text = str(text)
    if not docid or not text.strip():
        return None
    return docid, text, str(url)


def create_database(db_path: Path, rebuild: bool, use_fts: bool) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild and db_path.exists():
        print(f"Removing existing database {db_path}")
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            extra = Path(str(db_path) + suffix)
            if extra.exists():
                extra.unlink()

    connection = sqlite3.connect(str(db_path), timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-200000")
    connection.executescript(SCHEMA)

    if use_fts:
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5("
                "docid UNINDEXED, title, text, tokenize='unicode61 remove_diacritics 2')"
            )
        except sqlite3.OperationalError as exc:
            print(f"WARNING: FTS5 unavailable ({exc}); text_searcher will use LIKE.")
            use_fts = False
    connection.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        ("fts_enabled", "1" if use_fts else "0"),
    )
    connection.commit()
    return connection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        nargs="+",
        default=[str(config.RAW_DIR / "corpus.part-*.jsonl")],
        help="JSONL/parquet files or glob patterns.",
    )
    parser.add_argument("--db", type=Path, default=config.CORPUS_DB)
    parser.add_argument("--chunk-words", type=int, default=config.BM25_CHUNK_WORDS)
    parser.add_argument("--overlap-words", type=int, default=config.BM25_OVERLAP_WORDS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-fts", action="store_true")
    parser.add_argument("--batch-docs", type=int, default=100)
    args = parser.parse_args()

    config.ensure_directories()
    args.db = Path(args.db).expanduser().resolve()
    connection = create_database(args.db, args.rebuild, use_fts=not args.no_fts)

    existing: set[str] = set()
    if not args.rebuild:
        existing = {row[0] for row in connection.execute("SELECT docid FROM docs")}
        if existing:
            print(f"Resuming: {len(existing):,} documents already present.")

    fts_enabled = connection.execute(
        "SELECT value FROM meta WHERE key='fts_enabled'"
    ).fetchone() == ("1",)

    inserted_docs = 0
    inserted_chunks = 0
    skipped = 0
    started = time.time()
    pending_docs: list[tuple] = []
    pending_chunks: list[tuple] = []
    pending_fts: list[tuple] = []

    def flush() -> None:
        nonlocal pending_docs, pending_chunks, pending_fts
        if pending_docs:
            connection.executemany(
                "INSERT INTO docs(docid,url,domain,title,date,text,n_chars,n_words) "
                "VALUES (?,?,?,?,?,?,?,?)",
                pending_docs,
            )
        if pending_chunks:
            connection.executemany(
                "INSERT INTO chunks(docid,seq,start_char,end_char,text,n_words) "
                "VALUES (?,?,?,?,?,?)",
                pending_chunks,
            )
        if fts_enabled and pending_fts:
            connection.executemany(
                "INSERT INTO docs_fts(docid,title,text) VALUES (?,?,?)",
                pending_fts,
            )
        connection.commit()
        pending_docs, pending_chunks, pending_fts = [], [], []

    for record in iter_records(args.corpus):
        if args.limit and inserted_docs >= args.limit:
            break
        normalized = normalize_record(record)
        if normalized is None:
            skipped += 1
            continue
        docid, raw_text, url = normalized
        if docid in existing:
            skipped += 1
            continue

        metadata, body = parse_frontmatter(raw_text)
        if len(body.strip()) < 50 and len(raw_text.strip()) > len(body.strip()):
            body = raw_text
        title = infer_title(metadata, body, url)
        date = infer_date(metadata)
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        n_words = len(WORD_RE.findall(body))

        pending_docs.append((docid, url, domain, title, date, body, len(body), n_words))
        if fts_enabled:
            pending_fts.append((docid, title, body))
        for seq, (start_char, end_char, chunk_words, piece) in enumerate(
            chunk_text(body, args.chunk_words, args.overlap_words)
        ):
            pending_chunks.append((docid, seq, start_char, end_char, piece, chunk_words))
            inserted_chunks += 1

        existing.add(docid)
        inserted_docs += 1
        if len(pending_docs) >= args.batch_docs:
            flush()
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"  {inserted_docs:,} docs | {inserted_chunks:,} chunks | "
                f"{inserted_docs / elapsed:.1f} docs/s",
                flush=True,
            )

    flush()
    elapsed = time.time() - started
    connection.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", ("chunk_words", str(args.chunk_words)))
    connection.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", ("overlap_words", str(args.overlap_words)))
    connection.commit()
    connection.execute("ANALYZE")

    total_docs = connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    total_chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    avg_chunks = connection.execute(
        "SELECT AVG(c) FROM (SELECT COUNT(*) AS c FROM chunks GROUP BY docid)"
    ).fetchone()[0] or 0
    connection.close()

    stats = {
        "documents": int(total_docs),
        "chunks": int(total_chunks),
        "avg_chunks_per_document": round(float(avg_chunks), 2),
        "chunk_words": args.chunk_words,
        "overlap_words": args.overlap_words,
        "fts_enabled": bool(fts_enabled),
        "build_seconds": round(elapsed, 1),
    }
    (Path(args.db).parent / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\nDatabase ready:", args.db)
    print(json.dumps(stats, indent=2))
    print("\nNext: python index/build_bm25.py")


if __name__ == "__main__":
    main()
