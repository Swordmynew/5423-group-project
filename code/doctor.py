"""Environment and artifact health check.

Run this before building indexes or running the agent:

    python doctor.py

It prints the effective paths, dependency status, GPU memory, free disk space,
and whether the corpus / BM25 / dense / model artifacts are present.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).absolute().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import config

PACKAGES = [
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "bm25s",
    "faiss",
    "FlagEmbedding",
    "sentence_transformers",
    "transformers",
    "safetensors",
    "torch",
    "gradio",
    "pypdf",
    "yaml",
    "pyarrow",
    "dotenv",
]


def _ok(label: str, detail: str = "") -> None:
    print(f"[ OK ] {label}" + (f"  {detail}" if detail else ""))


def _warn(label: str, detail: str = "") -> None:
    print(f"[WARN] {label}" + (f"  {detail}" if detail else ""))


def _missing(label: str, detail: str = "") -> None:
    print(f"[MISS] {label}" + (f"  {detail}" if detail else ""))


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def check_dependencies() -> dict[str, bool]:
    status = {}
    for name in PACKAGES:
        found = importlib.util.find_spec(name) is not None
        status[name] = found
        if found:
            _ok(f"package {name}")
        else:
            _missing(f"package {name}", "pip install -r requirements.txt")
    return status


def check_gpu() -> dict:
    info = {"available": False}
    if importlib.util.find_spec("torch") is None:
        _warn("GPU", "torch is not installed")
        return info
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["available"] = True
            info["device"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            info["free_gb"] = round(free / 1024**3, 2)
            info["total_gb"] = round(total / 1024**3, 2)
            _ok("GPU", f"{info['device']} | torch {torch.__version__} | "
                       f"{info['free_gb']:.1f}/{info['total_gb']:.1f} GB free")
        else:
            _warn("GPU", f"torch {torch.__version__} but CUDA is not available")
    except Exception as exc:
        _warn("GPU", f"{type(exc).__name__}: {exc}")
    return info


def check_disk() -> dict:
    info = {}
    for label, path in (
        ("data", config.DATA_DIR),
        ("index", config.INDEX_DIR),
        ("models", config.MODELS_DIR),
        ("project", config.PROJECT_ROOT),
    ):
        try:
            usage = shutil.disk_usage(path if path.exists() else path.parent)
            info[label] = {
                "path": str(path),
                "free_gb": round(usage.free / 1024**3, 1),
                "total_gb": round(usage.total / 1024**3, 1),
            }
            _ok(f"disk {label}", f"{path} | free {info[label]['free_gb']} GB / "
                                 f"{info[label]['total_gb']} GB")
        except Exception as exc:
            _warn(f"disk {label}", f"{path} | {type(exc).__name__}: {exc}")
    return info


def check_database() -> dict:
    info = {"exists": config.CORPUS_DB.exists()}
    if not info["exists"]:
        _missing("corpus.db", str(config.CORPUS_DB))
        return info
    try:
        import sqlite3

        connection = sqlite3.connect(str(config.CORPUS_DB))
        try:
            info["documents"] = connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            info["chunks"] = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            row = connection.execute(
                "SELECT value FROM meta WHERE key='chunk_words'"
            ).fetchone()
            info["chunk_words"] = int(row[0]) if row else None
            fts = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='docs_fts'"
            ).fetchone()
            info["fts"] = bool(fts)
        finally:
            connection.close()
        _ok("corpus.db", f"{info['documents']:,} docs / {info['chunks']:,} chunks / "
                          f"chunk_words={info['chunk_words']} / fts={info['fts']}")
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        _warn("corpus.db", info["error"])
    return info


def check_bm25() -> dict:
    needed = [
        "data.csc.index.npy",
        "indices.csc.index.npy",
        "indptr.csc.index.npy",
        "vocab.index.json",
        "params.index.json",
        "chunk_ids.npy",
    ]
    missing = [name for name in needed if not (config.BM25_DIR / name).exists()]
    info = {"exists": not missing, "missing": missing}
    if info["exists"]:
        _ok("BM25 index", str(config.BM25_DIR))
    else:
        _missing("BM25 index", f"{config.BM25_DIR} missing {missing}")
    return info


def check_dense(directory: Path, label: str) -> dict:
    needed = ["index.faiss", "doc_ids.json", "meta.json"]
    missing = [name for name in needed if not (directory / name).exists()]
    info = {"exists": not missing, "missing": missing}
    if info["exists"]:
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            info["meta"] = meta
            _ok(
                f"{label} index",
                f"{directory} | vectors={meta.get('n_vectors')} | "
                f"windows={meta.get('doc_windows')} | "
                f"{meta.get('index_type')} | model={meta.get('model_name')}",
            )
        except Exception as exc:
            _warn(f"{label} index", f"meta unreadable: {exc}")
    else:
        _missing(f"{label} index", f"{directory} missing {missing}")
    return info


def check_models() -> dict:
    info = {}
    bge = config.MODELS_DIR / "bge-m3"
    if bge.exists():
        weights = sorted(path.name for path in bge.glob("*.safetensors"))
        if (bge / "pytorch_model.bin").exists():
            weights.append("pytorch_model.bin")
        ok = (bge / "config.json").exists() and bool(weights)
        info["bge"] = {"path": str(bge), "ok": ok, "weights": weights}
        if ok:
            _ok("BGE-M3 model", f"{bge} | {weights}")
        else:
            _warn("BGE-M3 model", f"{bge} incomplete")
    else:
        info["bge"] = {"path": str(bge), "ok": False}
        _missing("BGE-M3 model", str(bge))

    qwen = config.MODELS_DIR / "qwen3-embedding-0.6b"
    if qwen.exists():
        weights = sorted(path.name for path in qwen.glob("*.safetensors"))
        if (qwen / "pytorch_model.bin").exists():
            weights.append("pytorch_model.bin")
        ok = (qwen / "config.json").exists() and bool(weights)
        info["qwen3"] = {"path": str(qwen), "ok": ok, "weights": weights}
        if ok:
            _ok("Qwen3 model", f"{qwen} | {weights}")
        else:
            _warn("Qwen3 model", f"{qwen} incomplete")
    else:
        info["qwen3"] = {"path": str(qwen), "ok": False}
        _missing("Qwen3 model", str(qwen))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if data/index is missing.")
    args = parser.parse_args()

    config.ensure_directories()
    print("=" * 78)
    print("COMP5423 search agent — environment check")
    print("=" * 78)
    print("paths:")
    for key, value in config.describe_paths().items():
        print(f"  {key:16s} {value}")
    print(f"  .env             {(config.CODE_ROOT / '.env').exists()}")
    print(f"  api key set      {bool(config.DEEPSEEK_API_KEY)}")
    print()
    print("dependencies:")
    deps = check_dependencies()
    print()
    gpu = check_gpu()
    print()
    disk = check_disk()
    print()
    db = check_database()
    bm25 = check_bm25()
    print()
    dense_bge = check_dense(config.DENSE_BGE_DIR, "BGE")
    dense_qwen = check_dense(config.DENSE_QWEN3_DIR, "Qwen3")
    print()
    models = check_models()
    print()

    next_steps = []
    if not db.get("exists"):
        next_steps.append("python data/download_data.py && python data/preprocess.py")
    if not bm25.get("exists"):
        next_steps.append("python index/build_bm25.py")
    if not all(model.get("ok") for model in models.values()):
        next_steps.append("python data/prepare_models.py")
    if not dense_bge.get("exists"):
        next_steps.append("python index/build_dense.py --backend bge")
    if not dense_qwen.get("exists"):
        next_steps.append("python index/build_dense.py --backend qwen3")
    if next_steps:
        print("suggested next steps:")
        for step in next_steps:
            print(f"  - {step}")
    else:
        print("all artifacts present.")

    if args.json:
        print(json.dumps(
            {
                "paths": config.describe_paths(),
                "dependencies": deps,
                "gpu": gpu,
                "disk": disk,
                "database": db,
                "bm25": bm25,
                "dense_bge": dense_bge,
                "dense_qwen3": dense_qwen,
                "models": models,
            },
            indent=2,
            default=str,
        ))

    if args.strict:
        critical_ok = (
            bool(config.DEEPSEEK_API_KEY)
            and db.get("exists")
            and bm25.get("exists")
            and dense_bge.get("exists")
            and dense_qwen.get("exists")
        )
        raise SystemExit(0 if critical_ok else 1)


if __name__ == "__main__":
    main()
