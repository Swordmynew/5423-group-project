"""Download the BrowseComp-Plus corpus and the COMP5423 validation set.

Designed for flaky / proxied networks:
* ``--endpoint`` sets ``HF_ENDPOINT`` before huggingface_hub is imported.
* If the repository listing API is rate-limited (HTTP 429), the default corpus
  falls back to its seven known parquet shard names.
* ``--local-corpus`` / ``--local-validation`` import files already uploaded to
  the server, with no network access.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).absolute().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import config  # noqa: E402

DEFAULT_CORPUS_REPO = "Tevatron/browsecomp-plus-corpus"
DEFAULT_VAL_REPO = "HongruCai/COMP5423-BrowseComp-Plus-Val"
KNOWN_SHARDS = [
    f"data/train-{index:05d}-of-00007.parquet" for index in range(7)
]


def _hub():
    try:
        import huggingface_hub as hub
    except ImportError as exc:
        raise SystemExit("huggingface_hub is missing. `pip install huggingface_hub`.") from exc
    return hub


def _repo_files(repo_id: str, token: str | None) -> list[str]:
    hub = _hub()
    try:
        return hub.HfApi().list_repo_files(repo_id, repo_type="dataset", token=token)
    except Exception as exc:
        if repo_id == DEFAULT_CORPUS_REPO:
            print(f"WARNING: repository listing failed ({exc}); using the 7 known shard names.")
            return list(KNOWN_SHARDS)
        raise


def _download(repo_id: str, filename: str, cache_dir: Path, token: str | None) -> Path:
    hub = _hub()
    return Path(
        hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(cache_dir),
            token=token,
        )
    )


def _write_corpus_row(out, record: dict) -> bool:
    docid = record.get("docid") or record.get("doc_id") or record.get("id")
    text = record.get("text")
    url = record.get("url") or record.get("source_url") or ""
    if docid is None or text is None:
        return False
    out.write(
        json.dumps(
            {"docid": str(docid), "text": str(text), "url": str(url)},
            ensure_ascii=False,
        )
        + "\n"
    )
    return True


def _convert_parquet(path: Path, out_path: Path, limit: int, skip_existing: bool) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is missing. `pip install pyarrow`.") from exc

    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [skip] {out_path.name}")
        with out_path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    written = 0
    parquet_file = pq.ParquetFile(path)
    with tmp_path.open("w", encoding="utf-8") as out:
        for batch in parquet_file.iter_batches(batch_size=1000):
            for record in batch.to_pylist():
                if limit and written >= limit:
                    break
                if _write_corpus_row(out, record):
                    written += 1
            if limit and written >= limit:
                break
    tmp_path.replace(out_path)
    print(f"  wrote {written:,} documents -> {out_path.name}")
    return written


def _download_validation(args) -> Path:
    target = config.RAW_DIR / "validation.jsonl"
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if target.exists() and not args.overwrite:
        print(f"Validation set already present: {target}")
        return target

    if args.local_validation:
        source = Path(args.local_validation).expanduser().resolve()
        shutil.copyfile(source, target)
        print(f"Copied local validation set -> {target}")
        return target

    print(f"Downloading validation set from {args.val_repo} ...")
    path = _download(args.val_repo, "validation.jsonl", args.cache_dir, args.token)
    shutil.copyfile(path, target)
    print(f"Saved validation set -> {target}")
    return target


def _import_local_corpus(source: Path, args) -> int:
    if not source.exists():
        raise SystemExit(f"Local corpus path does not exist: {source}")
    files = (
        sorted([p for p in source.rglob("*") if p.suffix in {".jsonl", ".parquet"}])
        if source.is_dir()
        else [source]
    )
    if not files:
        raise SystemExit(f"No .jsonl or .parquet files under {source}")

    total = 0
    for index, path in enumerate(files):
        if args.limit and total >= args.limit:
            break
        print(f"[{index + 1}/{len(files)}] {path.name}")
        out_path = config.RAW_DIR / f"corpus.part-{index:05d}.jsonl"
        remaining = 0 if not args.limit else args.limit - total
        if path.suffix == ".parquet":
            total += _convert_parquet(
                path, out_path, remaining, skip_existing=not args.overwrite
            )
        else:
            if out_path.exists() and not args.overwrite:
                print(f"  [skip] {out_path.name}")
                total += sum(1 for _ in out_path.open("r", encoding="utf-8"))
                continue
            shutil.copyfile(path, out_path)
            count = sum(1 for _ in out_path.open("r", encoding="utf-8"))
            total += count
            print(f"  copied {count:,} documents -> {out_path.name}")
    print(f"Local corpus import complete: {total:,} documents")
    return total


def _download_corpus(args) -> int:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.local_corpus:
        return _import_local_corpus(Path(args.local_corpus).expanduser().resolve(), args)

    print(f"Listing files in {args.corpus_repo} ...")
    files = [
        name
        for name in _repo_files(args.corpus_repo, args.token)
        if name.endswith(".parquet") and Path(name).name.startswith("train-")
    ]
    files.sort()
    if not files:
        raise SystemExit(f"No parquet shards found in {args.corpus_repo}")

    print(f"Found {len(files)} parquet shard(s); cache: {args.cache_dir}")
    total = 0
    for index, filename in enumerate(files):
        if args.limit and total >= args.limit:
            break
        print(f"[{index + 1}/{len(files)}] {filename}")
        remaining = 0 if not args.limit else args.limit - total
        local_path = _download(args.corpus_repo, filename, args.cache_dir, args.token)
        total += _convert_parquet(
            local_path,
            config.RAW_DIR / f"corpus.part-{index:05d}.jsonl",
            remaining,
            skip_existing=not args.overwrite,
        )
        print(f"  running total: {total:,}")
    print(f"Corpus download complete: {total:,} documents -> {config.RAW_DIR}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-repo", default=DEFAULT_CORPUS_REPO)
    parser.add_argument("--val-repo", default=DEFAULT_VAL_REPO)
    parser.add_argument("--cache-dir", type=Path, default=config.RAW_DIR / "hf_cache")
    parser.add_argument("--local-corpus", default=None)
    parser.add_argument("--local-validation", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--endpoint", default=None, help="Hugging Face mirror, e.g. https://hf-mirror.com")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-corpus", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    if args.token:
        os.environ.setdefault("HF_TOKEN", args.token)

    config.ensure_directories()
    args.cache_dir = Path(args.cache_dir).expanduser().resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    if not args.skip_val:
        _download_validation(args)
    if not args.skip_corpus:
        _download_corpus(args)
    print(f"Done in {time.time() - started:.1f}s")
    print("Next: python data/preprocess.py")


if __name__ == "__main__":
    main()
