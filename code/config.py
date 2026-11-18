"""Central configuration for the COMP5423 skill-augmented search agent.

Design rules:
* ``CODE_ROOT`` is derived without resolving symlinks.  Code lives in ``code/``;
  data, indexes, and models are external directories selected by ``.env``.
* Every generated artifact is written under ``DATA_DIR``, ``INDEX_DIR``, or
  ``MODELS_DIR`` so the system disk is never filled.
"""

from __future__ import annotations

import os
from pathlib import Path

CODE_ROOT = Path(__file__).absolute().parent
PROJECT_ROOT = CODE_ROOT.parent

try:  # optional convenience; environment variables always win
    from dotenv import load_dotenv

    load_dotenv(CODE_ROOT / ".env", override=False)
except Exception:  # pragma: no cover
    pass


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _env_path("DATA_DIR", PROJECT_ROOT / "data")
INDEX_DIR = _env_path("INDEX_DIR", PROJECT_ROOT / "index")
MODELS_DIR = _env_path("MODELS_DIR", PROJECT_ROOT / "models")

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CORPUS_DB = _env_path("CORPUS_DB", PROCESSED_DIR / "corpus.db")
VALIDATION_JSONL = _env_path("VALIDATION_JSONL", RAW_DIR / "validation.jsonl")

BM25_DIR = INDEX_DIR / "bm25"
DENSE_BGE_DIR = INDEX_DIR / "dense_bge"
DENSE_QWEN3_DIR = INDEX_DIR / "dense_qwen3"

PREDICTION_PATH = _env_path(
    "PREDICTION_PATH", PROJECT_ROOT / "test_prediction.jsonl"
)
SKILLS_DIR = CODE_ROOT / "skills"

# Keep Hugging Face caches on the data disk when configured.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

EMBED_DEVICE = os.getenv("EMBED_DEVICE", "auto")
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
BGE_FALLBACK_MODEL_NAME = os.getenv("BGE_FALLBACK_MODEL_NAME", "BAAI/bge-small-en-v1.5")
QWEN3_MODEL_NAME = os.getenv("QWEN3_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")

BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))
BM25_CHUNK_WORDS = int(os.getenv("BM25_CHUNK_WORDS", "1200"))
BM25_OVERLAP_WORDS = int(os.getenv("BM25_OVERLAP_WORDS", "150"))

RRF_K = 60
DEFAULT_RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
MAX_RETRIEVAL_K = 20
RETRIEVAL_POOL_FACTOR = 6
RETRIEVAL_POOL_MIN = 60
RETRIEVAL_POOL_MAX = 120
MAX_SNIPPET_CHARS = 160

# Document-level dense windows.
DENSE_DOC_WINDOWS = os.getenv("DENSE_DOC_WINDOWS", "auto")  # auto | 1 | 2
DENSE_WINDOW_WORDS = int(os.getenv("DENSE_WINDOW_WORDS", "700"))
DENSE_MAX_SEQ_LEN = int(os.getenv("DENSE_MAX_SEQ_LEN", "1024"))
DENSE_AUTO_SECOND_WINDOW_WORDS = int(
    os.getenv("DENSE_AUTO_SECOND_WINDOW_WORDS", "1600")
)
DENSE_BATCH_BGE = int(os.getenv("DENSE_BATCH_BGE", "8"))
DENSE_BATCH_QWEN3 = int(os.getenv("DENSE_BATCH_QWEN3", "4"))
DENSE_FLAT_THRESHOLD = int(os.getenv("DENSE_FLAT_THRESHOLD", "30000"))
DENSE_NLIST = int(os.getenv("DENSE_NLIST", "4096"))

# ---------------------------------------------------------------------------
# Agent loop / context management (assignment values)
# ---------------------------------------------------------------------------

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "8"))
CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", "4000"))
KEEP_LAST_MESSAGES = int(os.getenv("KEEP_LAST_MESSAGES", "6"))
EVIDENCE_POOL_MAX = 5
EVIDENCE_POOL_PRUNE_THRESHOLD = 8

DOC_READER_DEFAULT_CHARS = int(os.getenv("DOC_READER_DEFAULT_CHARS", "1000"))
DOC_READER_MAX_CHARS = 4000
TEXT_SEARCH_MAX_RESULTS = 4
TEXT_SEARCH_CONTEXT_CHARS = 80
TRAJECTORY_OBSERVATION_MAX_CHARS = 4000

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}


def require_api_key() -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Put it in code/.env or export it "
            "before running."
        )
    return DEEPSEEK_API_KEY


def ensure_directories() -> None:
    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, INDEX_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def describe_paths() -> dict[str, str]:
    return {
        "code_root": str(CODE_ROOT),
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "corpus_db": str(CORPUS_DB),
        "index_dir": str(INDEX_DIR),
        "models_dir": str(MODELS_DIR),
        "prediction_path": str(PREDICTION_PATH),
    }
