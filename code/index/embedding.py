"""Embedding backends shared by the dense index builder and the retriever.

Two dense categories are supported, matching the project specification:

* ``bge``    BGE-M3 through FlagEmbedding (encoder-based dense retrieval)
* ``qwen3``  Qwen3-Embedding-0.6B through sentence-transformers
             (instruction-aware / LLM-level dense retrieval)
* ``bge-small`` is a lightweight fallback for machines that cannot run BGE-M3.

All passage and query vectors are L2-normalised, so inner product equals cosine
similarity.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

import config

QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def resolve_device(device: str | None = None) -> str:
    device = (device or config.EMBED_DEVICE or "auto").lower()
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _prepare_local_bge(model_name: str) -> None:
    """Convert a local BGE .bin to safetensors, or relax the torch<2.6 check."""
    path = Path(model_name).expanduser()
    if not path.exists() or not path.is_dir():
        return
    bin_path = path / "pytorch_model.bin"
    safe_path = path / "model.safetensors"
    if safe_path.exists() or not bin_path.exists():
        return

    try:
        from data.prepare_models import convert_bge_bin

        convert_bge_bin(path)
    except Exception as exc:
        print(f"WARNING: could not convert {bin_path} to safetensors: {exc}")

    if not safe_path.exists():
        # The model is a trusted, locally downloaded artifact.  Old torch
        # versions are blocked by transformers' conservative safety check even
        # for weights_only=True; bypass that check only for this local model.
        try:
            from transformers.utils import import_utils

            if hasattr(import_utils, "check_torch_load_is_safe"):
                import_utils.check_torch_load_is_safe = lambda: None
                print("WARNING: using pytorch_model.bin with a compatibility shim.")
        except Exception:
            pass


class BaseEmbedder:
    backend = "base"
    query_prefix = ""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            probe = self.encode_passages(["dimension probe"])
            self._dim = int(probe.shape[1])
        return self._dim

    def _load(self):
        raise NotImplementedError

    def encode_passages(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_passages([self.query_prefix + query], batch_size=1)[0]


class BGEM3Embedder(BaseEmbedder):
    backend = "bge"
    query_prefix = ""

    def _load(self):
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "FlagEmbedding is required for the BGE-M3 index. "
                    "Install it with `pip install FlagEmbedding`."
                ) from exc
            _prepare_local_bge(self.model_name)
            kwargs = {"use_fp16": self.device.startswith("cuda")}
            try:
                self._model = BGEM3FlagModel(self.model_name, device=self.device, **kwargs)
            except TypeError:  # older FlagEmbedding versions
                self._model = BGEM3FlagModel(self.model_name, **kwargs)
        return self._model

    def encode_passages(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        model = self._load()
        output = model.encode(
            list(texts),
            batch_size=max(1, batch_size),
            max_length=config.DENSE_MAX_SEQ_LEN,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vectors = output["dense_vecs"] if isinstance(output, dict) else output
        return normalize_rows(np.asarray(vectors, dtype=np.float32))


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, device: str, backend: str, query_prefix: str = ""):
        super().__init__(model_name, device)
        self.backend = backend
        self.query_prefix = query_prefix

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "sentence-transformers is required for this dense index. "
                    "Install it with `pip install sentence-transformers`."
                ) from exc
            device = self.device
            if device == "mps":
                device = "mps"
            self._model = SentenceTransformer(self.model_name, device=device)
            # Bound attention memory; doc windows are ~1000 tokens.
            self._model.max_seq_length = config.DENSE_MAX_SEQ_LEN
        return self._model

    def encode_passages(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=max(1, batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return normalize_rows(np.asarray(vectors, dtype=np.float32))


class BGESmallEmbedder(SentenceTransformerEmbedder):
    def __init__(self, model_name: str, device: str):
        super().__init__(model_name, device, backend="bge-small", query_prefix=BGE_QUERY_INSTRUCTION)


class Qwen3Embedder(SentenceTransformerEmbedder):
    def __init__(self, model_name: str, device: str):
        super().__init__(model_name, device, backend="qwen3", query_prefix=QWEN3_QUERY_INSTRUCTION)


def load_embedder(
    backend: str,
    device: str | None = None,
    model_name: str | None = None,
) -> BaseEmbedder:
    """Instantiate an embedder without loading model weights yet."""
    device = resolve_device(device)
    backend = backend.lower()
    if backend == "bge":
        return BGEM3Embedder(model_name or config.BGE_MODEL_NAME, device)
    if backend == "bge-small":
        return BGESmallEmbedder(model_name or config.BGE_FALLBACK_MODEL_NAME, device)
    if backend == "qwen3":
        return Qwen3Embedder(model_name or config.QWEN3_MODEL_NAME, device)
    raise ValueError(f"Unknown embedding backend: {backend}")


def backend_directory(backend: str):
    if backend == "qwen3":
        return config.DENSE_QWEN3_DIR
    return config.DENSE_BGE_DIR


def configure_hf_environment() -> None:
    """Reduce tokeniser warnings / avoid accidental telemetry."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
