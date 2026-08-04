"""Embeddings via bge-small-zh-v1.5 ONNX (512-dim, Chinese-optimized).

ONNX model exported locally, zero PyTorch at runtime. Falls back to subprocess.
"""

from __future__ import annotations

import json, logging, os, subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.path.expanduser("~/.cache/hermes/onnx/bge-small-zh-v1.5"))
_MODEL_PATH = _CACHE_DIR / "model.onnx"
_TOKENIZER_PATH = _CACHE_DIR / "tokenizer.json"
_DIM = 512


class OnnxEmbedder:
    def __init__(self):
        self._session = None
        self._tokenizer = None

    @property
    def available(self) -> bool:
        return _MODEL_PATH.exists() and _TOKENIZER_PATH.exists()

    def _ensure_loaded(self):
        if self._session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self._session = ort.InferenceSession(
            str(_MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))

    def embed(self, text: str):
        import numpy as np
        self._ensure_loaded()
        enc = self._tokenizer.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        out = self._session.run(None, {
            "input_ids": ids, "attention_mask": mask,
            "token_type_ids": np.zeros_like(ids),
        })
        # CLS pooling + L2 normalize
        vec = out[0][0, 0, :].astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed_batch(self, texts: list[str]):
        import numpy as np
        self._ensure_loaded()
        encs = [self._tokenizer.encode(t) for t in texts]
        max_len = min(max(len(e.ids) for e in encs), 512)
        n = len(texts)
        ids = np.zeros((n, max_len), dtype=np.int64)
        mask = np.zeros((n, max_len), dtype=np.int64)
        for i, enc in enumerate(encs):
            l = min(len(enc.ids), max_len)
            ids[i, :l] = enc.ids[:l]
            mask[i, :l] = enc.attention_mask[:l]
        out = self._session.run(None, {
            "input_ids": ids, "attention_mask": mask,
            "token_type_ids": np.zeros_like(ids),
        })
        vecs = out[0][:, 0, :].astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    @property
    def dim(self) -> int:
        return _DIM


_embedder: OnnxEmbedder | None = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = OnnxEmbedder()
    return _embedder

def embed(text: str):
    return get_embedder().embed(text)

def embed_batch(texts: list[str]):
    return get_embedder().embed_batch(texts)

def cosine(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b))
