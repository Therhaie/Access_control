import numpy as np

# -- Rotation Access control ----------------------

def make_rotation_matrix(dim: int, seed: int = 42) -> np.ndarray:
    """
    Return a deterministic (dim × dim) orthogonal rotation matrix via
    QR decomposition of a random Gaussian matrix.
 
    Using a fixed seed means every call produces the SAME rotation, which
    is critical: queries and stored documents must be rotated identically.
    """
    rng = np.random.default_rng(seed)
    H   = rng.standard_normal((dim, dim)).astype(np.float32)
    Q, _ = np.linalg.qr(H)   # Q is orthogonal
    return Q                  # shape: (dim, dim)


def rotate_vectors(vectors: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Apply rotation R to each row vector.
    vectors: (N, dim)  →  output: (N, dim)
    """
    return (vectors @ R.T).astype(np.float32)



# -- Adding dimension -----------------
"""
Should the vector be normalized after adding dimension, perhaps good things 
that it's not in order to give extra weight to the extra security-dimensions. 
"""


def append_extra_dimensions(
    vectors: np.ndarray,
    extra_dims: int = 4,
    mode: str = "zeros",
    seed: int = 0,
) -> np.ndarray:
    """
    Append extra_dims columns to each vector.
 
    mode options:
      "zeros"  — pad with zeros (safe default, preserves cosine similarity)
      "random" — small random noise (changes similarity, useful for experiments)
      "norm"   — each appended value = L2-norm of the original vector
                 (encodes magnitude as a feature; useful with "ip" space)
    """
    N, dim = vectors.shape
    if mode == "zeros":
        extra = np.zeros((N, extra_dims), dtype=np.float32)
    elif mode == "random":
        rng   = np.random.default_rng(seed)
        extra = (rng.standard_normal((N, extra_dims)) * 0.01).astype(np.float32)
    elif mode == "norm":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)  # (N,1)
        extra = np.tile(norms, (1, extra_dims)).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return np.concatenate([vectors, extra], axis=1)
 
 
def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit-normalise each row (required for cosine space to work correctly)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
    return (vectors / norms).astype(np.float32)