"""Prepare local embedding models for the dense indexes.

Downloads only the files needed by BGE-M3 / Qwen3-Embedding (avoiding image and
``.DS_Store`` files that mirrors sometimes reject), then converts BGE-M3's
``pytorch_model.bin`` to ``model.safetensors`` when the installed torch is older
than 2.6.

Examples
--------
python data/prepare_models.py
python data/prepare_models.py --only bge
python data/prepare_models.py --endpoint https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).absolute().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import config  # noqa: E402

ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.txt",
    "*.tiktoken",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.py",
    "1_Pooling/*",
]


def _download(repo_id: str, local_dir: Path, token: str | None, max_workers: int) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is missing. `pip install huggingface_hub`.") from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                allow_patterns=ALLOW_PATTERNS,
                token=token,
                max_workers=max_workers,
            )
            return
        except Exception as exc:
            last_error = exc
            wait = 15 * attempt
            print(f"  download attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt < 3:
                print(f"  retrying in {wait}s ...")
                time.sleep(wait)
    raise SystemExit(f"Could not download {repo_id}: {last_error}")


def convert_bge_bin(model_dir: Path) -> Path | None:
    """Convert ``pytorch_model.bin`` to ``model.safetensors`` if needed."""
    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        return safetensors_path
    if not bin_path.exists():
        return None

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        print(f"  cannot convert bin: {type(exc).__name__}: {exc}")
        return None

    print(f"  converting {bin_path.name} -> {safetensors_path.name} ...")
    try:
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(bin_path, map_location="cpu")
    except Exception as exc:
        print(f"  torch.load failed: {type(exc).__name__}: {exc}")
        return None

    state = {
        key: value.contiguous() if hasattr(value, "contiguous") else value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    save_file(state, str(safetensors_path))
    backup = bin_path.with_suffix(".bin.bak")
    if not backup.exists():
        bin_path.rename(backup)
    print(f"  wrote {safetensors_path} ({safetensors_path.stat().st_size / 1024**3:.1f} GB)")
    return safetensors_path


def check_model(model_dir: Path, required: list[str]) -> bool:
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        print(f"  MISSING in {model_dir}: {missing}")
        return False
    return True


def prepare_bge(args) -> bool:
    model_dir = config.MODELS_DIR / "bge-m3"
    print(f"BGE-M3 -> {model_dir}")
    weights_present = bool(list(model_dir.glob("*.safetensors"))) or (model_dir / "pytorch_model.bin").exists()
    if args.force or not (model_dir / "config.json").exists() or not weights_present:
        _download("BAAI/bge-m3", model_dir, args.token, args.max_workers)
    converted = convert_bge_bin(model_dir)
    required = ["config.json", "tokenizer.json", "sentencepiece.bpe.model"]
    ok = check_model(model_dir, required)
    weights = bool(list(model_dir.glob("*.safetensors"))) or (model_dir / "pytorch_model.bin").exists()
    if not weights:
        print(f"  MISSING weight file in {model_dir}")
        ok = False
    if converted:
        print(f"  safetensors: {converted}")
    print(f"  BGE-M3 ready: {ok}")
    return ok


def prepare_qwen3(args) -> bool:
    model_dir = config.MODELS_DIR / "qwen3-embedding-0.6b"
    print(f"Qwen3 -> {model_dir}")
    weights_present = bool(list(model_dir.glob("*.safetensors"))) or (model_dir / "pytorch_model.bin").exists()
    if args.force or not (model_dir / "config.json").exists() or not weights_present:
        _download("Qwen/Qwen3-Embedding-0.6B", model_dir, args.token, args.max_workers)
    required = ["config.json", "modules.json", "tokenizer_config.json"]
    ok = check_model(model_dir, required)
    weights = bool(list(model_dir.glob("*.safetensors"))) or (model_dir / "pytorch_model.bin").exists()
    if not weights:
        print(f"  MISSING weight file in {model_dir}")
        ok = False
    print(f"  Qwen3 ready: {ok}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["all", "bge", "qwen3"], default="all")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    if args.token:
        os.environ.setdefault("HF_TOKEN", args.token)

    config.ensure_directories()
    print(f"models dir: {config.MODELS_DIR}")
    print(f"HF_ENDPOINT: {os.getenv('HF_ENDPOINT', '(default)')}")
    print(f"HF_HOME: {os.getenv('HF_HOME', '(default)')}")

    ok = True
    if args.only in {"all", "bge"}:
        ok = prepare_bge(args) and ok
    if args.only in {"all", "qwen3"}:
        ok = prepare_qwen3(args) and ok
    if not ok:
        raise SystemExit(1)
    print("Models ready.")


if __name__ == "__main__":
    main()
