"""Holographic Reduced Representations (HRR) with phase encoding.

HRRs are a vector symbolic architecture for encoding compositional structure
into fixed-width distributed representations. This module uses *phase vectors*:
each concept is a vector of angles in [0, 2π). The algebraic operations are:

  bind   — circular convolution (phase addition)  — associates two concepts
  unbind — circular correlation (phase subtraction) — retrieves a bound value
  bundle — superposition (circular mean)           — merges multiple concepts

Phase encoding is numerically stable, avoids the magnitude collapse of
traditional complex-number HRRs, and maps cleanly to cosine similarity.

Atoms are generated deterministically from SHA-256 so representations are
identical across processes, machines, and language versions.

References:
  Plate (1995) — Holographic Reduced Representations
  Gayler (2004) — Vector Symbolic Architectures answer Jackendoff's challenges
"""

import hashlib
import logging
import struct
import math

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise RuntimeError("numpy is required for holographic operations")


def encode_atom(word: str, dim: int = 1024) -> "np.ndarray":
    """Deterministic phase vector via SHA-256 counter blocks.

    Uses hashlib (not numpy RNG) for cross-platform reproducibility.

    Algorithm:
    - Generate enough SHA-256 blocks by hashing f"{word}:{i}" for i=0,1,2,...
    - Concatenate digests, interpret as uint16 values via struct.unpack
    - Scale to [0, 2π): phases = values * (2π / 65536)
    - Truncate to dim elements
    - Returns np.float64 array of shape (dim,)
    """
    _require_numpy()

    # Each SHA-256 digest is 32 bytes = 16 uint16 values.
    values_per_block = 16
    blocks_needed = math.ceil(dim / values_per_block)

    uint16_values: list[int] = []
    for i in range(blocks_needed):
        digest = hashlib.sha256(f"{word}:{i}".encode()).digest()
        uint16_values.extend(struct.unpack("<16H", digest))

    phases = np.array(uint16_values[:dim], dtype=np.float64) * (_TWO_PI / 65536.0)
    return phases


def bind(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """Circular convolution = element-wise phase addition.

    Binding associates two concepts into a single composite vector.
    The result is dissimilar to both inputs (quasi-orthogonal).

    Raises ValueError when vectors have different dimensions — this is a
    programming error at the call site, not a recoverable runtime mismatch.
    """
    _require_numpy()
    if a.shape != b.shape:
        raise ValueError(
            f"Vector dimension mismatch in bind(): {a.shape[0]} vs {b.shape[0]}. "
            f"Both vectors must have the same hrr_dim."
        )
    return (a + b) % _TWO_PI


def unbind(memory: "np.ndarray", key: "np.ndarray") -> "np.ndarray":
    """Circular correlation = element-wise phase subtraction.

    Unbinding retrieves the value associated with a key from a memory vector.
    unbind(bind(a, b), a) ≈ b  (up to superposition noise)

    Raises ValueError when vectors have different dimensions — this is a
    programming error at the call site, not a recoverable runtime mismatch.
    """
    _require_numpy()
    if memory.shape != key.shape:
        raise ValueError(
            f"Vector dimension mismatch in unbind(): {memory.shape[0]} vs {key.shape[0]}. "
            f"Both vectors must have the same hrr_dim."
        )
    return (memory - key) % _TWO_PI


def bundle(*vectors: "np.ndarray") -> "np.ndarray":
    """Superposition via circular mean of complex exponentials.

    Bundling merges multiple vectors into one that is similar to each input.
    The result can hold O(sqrt(dim)) items before similarity degrades.

    Returns a zero-phase vector when no vectors are provided.

    Raises ValueError when vectors have different dimensions — all bundled
    vectors must share the same hrr_dim.
    """
    _require_numpy()
    if not vectors:
        return np.zeros(1, dtype=np.float64)
    dim = vectors[0].shape[0]
    for i, v in enumerate(vectors):
        if v.shape[0] != dim:
            raise ValueError(
                f"Vector dimension mismatch in bundle(): vector[0]={dim} vs "
                f"vector[{i}]={v.shape[0]}. All vectors must have the same hrr_dim."
            )
    complex_sum = np.sum([np.exp(1j * v) for v in vectors], axis=0)
    return np.angle(complex_sum) % _TWO_PI


def similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    """Phase cosine similarity. Range [-1, 1].

    Returns 1.0 for identical vectors, near 0.0 for random (unrelated) vectors,
    and -1.0 for perfectly anti-correlated vectors.

    Returns 0.0 when vectors have different dimensions (graceful degradation
    instead of crashing — dimension mismatch is not a semantic match).
    """
    _require_numpy()
    if a.shape != b.shape:
        return 0.0
    return float(np.mean(np.cos(a - b)))


def encode_text(text: str, dim: int = 1024) -> "np.ndarray":
    """Bag-of-words: bundle of atom vectors for each token.

    Tokenizes by lowercasing, splitting on whitespace, and stripping
    leading/trailing punctuation from each token.

    Returns bundle of all token atom vectors.
    If text is empty or produces no tokens, returns encode_atom("__hrr_empty__", dim).
    """
    _require_numpy()

    tokens = [
        token.strip(".,!?;:\"'()[]{}")
        for token in text.lower().split()
    ]
    tokens = [t for t in tokens if t]

    if not tokens:
        return encode_atom("__hrr_empty__", dim)

    atom_vectors = [encode_atom(token, dim) for token in tokens]
    return bundle(*atom_vectors)


def encode_fact(content: str, entities: list[str], dim: int = 1024) -> "np.ndarray":
    """Structured encoding: content bound to ROLE_CONTENT, each entity bound to ROLE_ENTITY, all bundled.

    Role vectors are reserved atoms: "__hrr_role_content__", "__hrr_role_entity__"

    Components:
    1. bind(encode_text(content, dim), encode_atom("__hrr_role_content__", dim))
    2. For each entity: bind(encode_atom(entity.lower(), dim), encode_atom("__hrr_role_entity__", dim))
    3. bundle all components together

    This enables algebraic extraction:
        unbind(fact, bind(entity, ROLE_ENTITY)) ≈ content_vector
    """
    _require_numpy()

    role_content = encode_atom("__hrr_role_content__", dim)
    role_entity = encode_atom("__hrr_role_entity__", dim)

    components: list[np.ndarray] = [
        bind(encode_text(content, dim), role_content)
    ]

    for entity in entities:
        components.append(bind(encode_atom(entity.lower(), dim), role_entity))

    return bundle(*components)


def phases_to_bytes(phases: "np.ndarray") -> bytes:
    """Serialize phase vector to bytes. float64 tobytes — 8 KB at dim=1024."""
    _require_numpy()
    return phases.tobytes()


def bytes_to_phases(data: bytes) -> "np.ndarray":
    """Deserialize bytes back to phase vector. Inverse of phases_to_bytes.

    The .copy() call is required because frombuffer returns a read-only view
    backed by the bytes object; callers expect a mutable array.
    """
    _require_numpy()
    return np.frombuffer(data, dtype=np.float64).copy()


def safe_vec(data: bytes, expected_dim: int):
    """Unpack a stored HRR vector and verify its dimension matches.

    Returns the numpy array on success, ``None`` when the stored dimension
    differs — the fact was encoded with a different ``hrr_dim`` and its
    vector is not comparable to the current one.  Also returns ``None``
    on deserialization errors (corrupt blob).

    Callers should count skipped facts and log a warning suggesting
    ``MemoryStore.rebuild_all_vectors()`` when mismatches are found.
    """
    try:
        vec = bytes_to_phases(data)
        if vec.shape[0] != expected_dim:
            return None
        return vec
    except Exception:
        return None


def snr_estimate(dim: int, n_items: int) -> float:
    """Signal-to-noise ratio estimate for holographic storage.

    SNR = sqrt(dim / n_items) when n_items > 0, else inf.

    The SNR falls below 2.0 when n_items > dim / 4, meaning retrieval
    errors become likely. Logs a warning when this threshold is crossed.
    """
    _require_numpy()

    if n_items <= 0:
        return float("inf")

    snr = math.sqrt(dim / n_items)

    if snr < 2.0:
        logger.warning(
            "HRR storage near capacity: SNR=%.2f (dim=%d, n_items=%d). "
            "Retrieval accuracy may degrade. Consider increasing dim or reducing stored items.",
            snr,
            dim,
            n_items,
        )

    return snr
