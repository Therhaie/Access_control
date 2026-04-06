"""
dim_experiment.py
=================
Access-Control Embedding Experiment  —  three complementary methods.

═══════════════════════════════════════════════════════════════════════════════
METHOD 1  (baseline)   — no access control, plain cosine retrieval
METHOD 2  (extra-dim)  — append extra dimensions encoding accessibility
METHOD 3  (metadata)   — Chroma where-filter on triplet_index at query time
═══════════════════════════════════════════════════════════════════════════════

Extra-Dimension encoding  (Method 2)
--------------------------------------
Every stored vector gets `extra_dims` dimensions appended:

  Restricted chunk   →  append  [large_value, large_value, …]   (len = extra_dims)
  Open chunk         →  append  [0, 0, …]                        (not targeted by any query)

Query side (symmetric):
  Authorised query   →  append  [0, 0, …]                        (aligns with open chunks)
  Unauthorised query →  append  [large_value, large_value, …]    (aligns with restricted, pushes away open)

Wait — that is *inverted* from what we want.  The correct semantic is:
  "I have access"  means I can retrieve restricted chunks  →  my extra dims must ALIGN with theirs.
  "I have no access" must DIVERGE from restricted chunks.

  Restricted chunk   →  [large_value, …]
  Authorised query   →  [large_value, …]   (same → high cosine on extra dims)
  Unauthorised query →  [0, …]             (orthogonal → cosine penalty on extra dims)
  Open chunk         →  [0, …]
  Any query vs open  →  always aligned on the zero dims → no penalty

This is the encoding used below.  `large_value` is configurable (default 1e6).
Normalisation (before/after appending) is optional.

Metadata Filtering  (Method 3)
--------------------------------
Chunks are stored with a boolean metadata field  `restricted = True / False`.
At query time, an authorised query filters  where={"restricted": {"$eq": False}}
is NOT applied — it retrieves everything.  An unauthorised query adds
  where={"restricted": {"$eq": False}}
so restricted chunks are excluded by the database before similarity ranking.

Timing decorator
-----------------
`@timed(label)` wraps any function and appends a structured record to a
module-level `TIMING_LOG` list.  After the experiment the log is written to
  results/dim_timing.json

Pipeline mirrors rotation_experiment.py so both can be compared directly.

Usage
-----
python dim_experiment.py
python dim_experiment.py --gt results/ground_truth_retrievals.json
python dim_experiment.py --extra-dims 8 --large-value 1e9 --normalize-after
python dim_experiment.py --weights 1e4 1e6 1e8          # sweep large_value
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from ingestion_pipeline import get_embedding_model
from query_pipeline import BGE_QUERY_PREFIX
from config import COLLECTION

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR       = Path("results")
GT_FILE           = Path("documents_RAGBench/merged_id_triplets_with_metadata2.json")
DIM_RESULTS_FILE  = RESULTS_DIR / "dim_results.json"
DIM_TIMING_FILE   = RESULTS_DIR / "dim_timing.json"
DIM_REGISTRY_FILE = RESULTS_DIR / "dim_registry.json"

ORIGINAL_CHROMA     = os.path.join(os.getcwd(), "./chroma_db")
ORIGINAL_COLLECTION = COLLECTION
DIM_CHROMA_BASE     = os.path.join(os.getcwd(), "./chroma_dim_db")
META_CHROMA_BASE    = os.path.join(os.getcwd(), "./chroma_meta_db")

DEFAULT_TOP_K      = 20
DEFAULT_EXTRA_DIMS = 4
DEFAULT_LARGE_VAL  = 1e6


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Timing infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level log — every @timed call appends here
TIMING_LOG: list[dict] = []


def timed(label: str):
    """
    Decorator factory.  Wraps a function and records wall-clock duration.

    Usage::

        @timed("embed_query")
        def my_func(…): …

    Each call appends::

        {
            "label":      "embed_query",
            "start_iso":  "2025-…",
            "duration_s": 0.0312,
            "args_repr":  "…",          # first 120 chars of positional args
        }

    to the module-level TIMING_LOG.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Build a lightweight repr so we know which query / config was running
            args_repr = str(args[:2])[:120]  # first two args, truncated
            start     = time.perf_counter()
            start_iso = datetime.now(timezone.utc).isoformat()
            try:
                result = fn(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                TIMING_LOG.append({
                    "label":      label,
                    "start_iso":  start_iso,
                    "duration_s": round(duration, 6),
                    "args_repr":  args_repr,
                })
            return result
        return wrapper
    return decorator


def save_timing_log(path: Path = DIM_TIMING_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(TIMING_LOG, fh, indent=2)
    print(f"⏱  Timing log saved to {path}  ({len(TIMING_LOG)} entries)")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtraDimConfig:
    """
    All knobs for one extra-dimension experimental condition.

    large_value       : scalar appended to restricted chunks / unauthorised queries.
    normalize_before  : L2-normalise the base vector BEFORE appending.
    normalize_after   : L2-normalise the full (base + extra) vector AFTER appending.
    """
    extra_dims:       int
    large_value:      float = DEFAULT_LARGE_VAL
    normalize_before: bool  = False
    normalize_after:  bool  = False

    @property
    def config_id(self) -> str:
        lv  = f"{self.large_value:.0e}".replace("+", "")
        nb  = "nb1" if self.normalize_before else "nb0"
        na  = "na1" if self.normalize_after  else "na0"
        return f"d{self.extra_dims}_lv{lv}_{nb}_{na}"

    def to_dict(self) -> dict:
        return {
            "extra_dims":       self.extra_dims,
            "large_value":      self.large_value,
            "normalize_before": self.normalize_before,
            "normalize_after":  self.normalize_after,
            "config_id":        self.config_id,
        }


def _default_configs() -> list[ExtraDimConfig]:
    """Representative sweep if no CLI config is provided."""
    return [
        ExtraDimConfig(extra_dims=4, large_value=1e4,  normalize_before=False, normalize_after=False),
        ExtraDimConfig(extra_dims=4, large_value=1e6,  normalize_before=False, normalize_after=False),
        ExtraDimConfig(extra_dims=4, large_value=1e8,  normalize_before=False, normalize_after=False),
        ExtraDimConfig(extra_dims=4, large_value=1e6,  normalize_before=True,  normalize_after=False),
        ExtraDimConfig(extra_dims=4, large_value=1e6,  normalize_before=False, normalize_after=True),
        ExtraDimConfig(extra_dims=4, large_value=1e6,  normalize_before=True,  normalize_after=True),
        ExtraDimConfig(extra_dims=8, large_value=1e6,  normalize_before=False, normalize_after=False),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Vector augmentation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def augment_vector(
    base_vec:      np.ndarray,
    cfg:           ExtraDimConfig,
    is_restricted: bool,
) -> np.ndarray:
    """
    Append extra dimensions to `base_vec`.

    Encoding:
      restricted=True  (restricted chunk OR authorised query) → append [large_value, …]
      restricted=False (open chunk OR unauthorised query)     → append [0, …]

    This makes authorised queries and restricted chunks ALIGN on the extra dims,
    while unauthorised queries DIVERGE from restricted chunks (cosine penalty).
    Open chunks always have zeros, so any query is neutral toward them.
    """
    v = base_vec.astype(np.float32)
    if cfg.normalize_before:
        v = _l2_norm(v)

    extra = (
        np.full(cfg.extra_dims, cfg.large_value, dtype=np.float32)
        if is_restricted
        else np.zeros(cfg.extra_dims, dtype=np.float32)
    )
    augmented = np.concatenate([v, extra])

    if cfg.normalize_after:
        augmented = _l2_norm(augmented)

    return augmented


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ChromaDB helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_client(path: str) -> chromadb.PersistentClient:
    os.makedirs(path, exist_ok=True)
    return chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_original_collection(path: str, name: str):
    return _get_client(path).get_collection(name=name)


def _get_dim_collection(cfg_id: str):
    """Each config gets its own Chroma collection under DIM_CHROMA_BASE."""
    client = _get_client(DIM_CHROMA_BASE)
    return client.get_or_create_collection(
        name=f"dim_{cfg_id}",
        metadata={"hnsw:space": "cosine"},
    )


def _get_meta_collection():
    """Single shared collection for Method 3 (metadata filtering)."""
    client = _get_client(META_CHROMA_BASE)
    return client.get_or_create_collection(
        name="meta_access_control",
        metadata={"hnsw:space": "cosine"},
    )


@timed("fetch_original_vector")
def _fetch_original_vector(
    collection,
    triplet_index: str,
    document_id:   str,
    phrase_seq:    str,
) -> np.ndarray | None:
    """Retrieve stored embedding from the original ChromaDB collection."""
    try:
        result = collection.get(
            where={"$and": [
                {"triplet_index": {"$eq": triplet_index}},
                {"document_id":   {"$eq": document_id}},
                {"phrase_seq":    {"$eq": phrase_seq}},
            ]},
            include=["embeddings", "documents"],
        )
        embs = result.get("embeddings", [])
        if embs and len(embs) > 0 and len(embs[0]) > 0:
            return np.array(embs[0], dtype=np.float32), result["documents"][0]
    except Exception as e:
        warnings.warn(f"_fetch_original_vector failed: {e}")
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Result data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DimChunkSimilarities:
    """Similarity measurements for one chunk under one config."""
    chunk_key:     str
    config_id:     str
    is_restricted: bool

    # (a) authorised query   ↔ restricted chunk  (both carry large_value → align)
    sim_auth_restricted:   float
    # (b) unauthorised query ↔ restricted chunk  (zeros vs large_value → diverge)
    sim_unauth_restricted: float
    # (c) authorised query   ↔ open chunk        (large_value vs zeros)
    sim_auth_open:         float
    # (d) unauthorised query ↔ open chunk        (both zeros → neutral)
    sim_unauth_open:       float
    # baseline: no augmentation
    sim_baseline:          float

    @property
    def security_delta(self) -> float:
        """
        Similarity drop for unauthorised on a restricted chunk.
        Negative = restriction working (unauthorised sees lower similarity).
        """
        return self.sim_unauth_restricted - self.sim_auth_restricted

    @property
    def collateral_delta(self) -> float:
        """
        Change for open chunks when authorised query is used vs baseline.
        Should be close to 0 — open chunks should not be affected.
        """
        return self.sim_auth_open - self.sim_baseline


@dataclass
class TopKResult:
    """Top-K retrieval result for one query under one method."""
    method:                  str   # "extradim" | "metadata"
    config_id:               str   # config_id or "metadata_filter"
    topk_auth_ids:           list[str]  = field(default_factory=list)
    topk_unauth_ids:         list[str]  = field(default_factory=list)
    restricted_in_auth:      int        = 0
    restricted_in_unauth:    int        = 0
    retrieval_time_auth_s:   float      = 0.0
    retrieval_time_unauth_s: float      = 0.0


@dataclass
class DimQueryResult:
    query_id:      str
    question:      str
    triplet_index: str
    n_restricted:  int
    n_open:        int

    # Method 2 results (one per config)
    chunk_sims:    list[DimChunkSimilarities] = field(default_factory=list)
    topk_results:  list[TopKResult]           = field(default_factory=list)

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Method 2 — Extra-dimension access control
# ═══════════════════════════════════════════════════════════════════════════════

@timed("method2_embed_and_upsert_chunks")
def _m2_upsert_chunks(
    stable_chunks: list[str],
    triplet_index: str,
    cfg:           ExtraDimConfig,
    embedder,
    orig_collection,
    dim_collection,
    restricted_keys: set[str],
    written_ids:     set[str],
) -> dict[str, np.ndarray]:
    """
    Fetch/embed every chunk, augment it, upsert to `dim_collection`.
    Returns mapping  chunk_id → raw (un-augmented) numpy vector.
    """
    raw_vecs: dict[str, np.ndarray] = {}

    for chunk_id in stable_chunks:
        tid  = chunk_id.split("|")[0]
        did  = chunk_id[-2]
        pseq = chunk_id[-1]
        cid  = f"{tid}_{did}_{pseq}"
        ckey = chunk_id

        raw_vec, content = _fetch_original_vector(orig_collection, tid, did, pseq)

        if raw_vec is None:
            warnings.warn(f"  ⚠  Vector not found for {ckey}, skipping.")
            continue

        raw_vecs[ckey] = raw_vec
        is_restr = ckey in restricted_keys

        aug_vec = augment_vector(raw_vec, cfg, is_restricted=is_restr)

        if cid not in written_ids:
            dim_collection.upsert(
                ids        = [cid],
                embeddings = [aug_vec.tolist()],
                documents  = [content if content else ""],
                metadatas  = [{
                    "triplet_index": triplet_index,
                    "document_id":   did,
                    "phrase_seq":    pseq,
                    "restricted":    is_restr,
                    "config_id":     cfg.config_id,
                }],
            )
            written_ids.add(cid)

    return raw_vecs


@timed("method2_compute_similarities")
def _m2_compute_sims(
    raw_vecs:        dict[str, np.ndarray],
    raw_query_vec:   np.ndarray,
    cfg:             ExtraDimConfig,
    restricted_keys: set[str],
) -> list[DimChunkSimilarities]:
    """Compute all four similarity pairs for every chunk."""
    auth_q   = augment_vector(raw_query_vec, cfg, is_restricted=True)   # authorised
    unauth_q = augment_vector(raw_query_vec, cfg, is_restricted=False)  # unauthorised
    sims: list[DimChunkSimilarities] = []

    for ckey, raw_vec in raw_vecs.items():
        is_restr = ckey in restricted_keys
        aug_restr = augment_vector(raw_vec, cfg, is_restricted=True)
        aug_open  = augment_vector(raw_vec, cfg, is_restricted=False)

        sims.append(DimChunkSimilarities(
            chunk_key             = ckey,
            config_id             = cfg.config_id,
            is_restricted         = is_restr,
            sim_auth_restricted   = cosine_similarity(auth_q,   aug_restr),
            sim_unauth_restricted = cosine_similarity(unauth_q, aug_restr),
            sim_auth_open         = cosine_similarity(auth_q,   aug_open),
            sim_unauth_open       = cosine_similarity(unauth_q, aug_open),
            sim_baseline          = cosine_similarity(raw_query_vec, raw_vec),
        ))

    return sims


@timed("method2_topk_retrieval")
def _m2_topk(
    dim_collection,
    auth_q_vec:      np.ndarray,
    unauth_q_vec:    np.ndarray,
    restricted_keys: set[str],
    top_k:           int,
    cfg:             ExtraDimConfig,
) -> TopKResult:
    """Run top-K retrieval for both authorised and unauthorised queries."""
    n_in = max(1, dim_collection.count())

    t0 = time.perf_counter()
    auth_res = dim_collection.query(
        query_embeddings=[auth_q_vec.tolist()],
        n_results=min(top_k, n_in),
        include=["metadatas"],
    )
    t_auth = time.perf_counter() - t0

    t0 = time.perf_counter()
    unauth_res = dim_collection.query(
        query_embeddings=[unauth_q_vec.tolist()],
        n_results=min(top_k, n_in),
        include=["metadatas"],
    )
    t_unauth = time.perf_counter() - t0

    def _id(m: dict) -> str:
        return f"{m.get('triplet_index','?')}|{m.get('document_id','?')}|{m.get('phrase_seq','?')}"

    topk_auth   = [_id(m) for m in auth_res["metadatas"][0]]
    topk_unauth = [_id(m) for m in unauth_res["metadatas"][0]]

    return TopKResult(
        method                  = "extradim",
        config_id               = cfg.config_id,
        topk_auth_ids           = topk_auth,
        topk_unauth_ids         = topk_unauth,
        restricted_in_auth      = len(set(topk_auth)   & restricted_keys),
        restricted_in_unauth    = len(set(topk_unauth) & restricted_keys),
        retrieval_time_auth_s   = round(t_auth,   6),
        retrieval_time_unauth_s = round(t_unauth, 6),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Method 3 — Metadata filtering
# ═══════════════════════════════════════════════════════════════════════════════

@timed("method3_upsert_chunks")
def _m3_upsert_chunks(
    stable_chunks:   list[str],
    triplet_index:   str,
    embedder,
    orig_collection,
    meta_collection,
    restricted_keys: set[str],
    written_ids:     set[str],
) -> None:
    """
    Upsert plain (un-augmented) vectors into the metadata collection.
    Each chunk gets a boolean  `restricted` metadata field.
    """
    for chunk_id in stable_chunks:
        tid  = chunk_id.split("|")[0]
        did  = chunk_id[-2]
        pseq = chunk_id[-1]
        cid  = f"{tid}_{did}_{pseq}"

        if cid in written_ids:
            continue

        raw_vec, content = _fetch_original_vector(orig_collection, tid, did, pseq)
        if raw_vec is None:
            warnings.warn(f"  ⚠  Method3: vector not found for {chunk_id}, skipping.")
            continue

        ckey     = chunk_id
        is_restr = ckey in restricted_keys

        meta_collection.upsert(
            ids        = [cid],
            embeddings = [raw_vec.tolist()],
            documents  = [content if content else ""],
            metadatas  = [{
                "triplet_index": triplet_index,
                "document_id":   did,
                "phrase_seq":    pseq,
                "restricted":    is_restr,
            }],
        )
        written_ids.add(cid)


@timed("method3_topk_retrieval")
def _m3_topk(
    meta_collection,
    raw_query_vec:   np.ndarray,
    restricted_keys: set[str],
    top_k:           int,
) -> TopKResult:
    """
    Authorised   → no filter (retrieves everything including restricted chunks).
    Unauthorised → filter where restricted == False  (excludes restricted chunks).
    """
    n_in = max(1, meta_collection.count())

    # Authorised: no filter
    t0 = time.perf_counter()
    auth_res = meta_collection.query(
        query_embeddings=[raw_query_vec.tolist()],
        n_results=min(top_k, n_in),
        include=["metadatas"],
    )
    t_auth = time.perf_counter() - t0

    # Unauthorised: exclude restricted chunks via metadata filter
    t0 = time.perf_counter()
    try:
        unauth_res = meta_collection.query(
            query_embeddings=[raw_query_vec.tolist()],
            n_results=min(top_k, n_in),
            where={"restricted": {"$eq": False}},
            include=["metadatas"],
        )
    except Exception as e:
        warnings.warn(f"Method3 unauth query failed: {e}. Falling back to no filter.")
        unauth_res = auth_res
    t_unauth = time.perf_counter() - t0

    def _id(m: dict) -> str:
        return f"{m.get('triplet_index','?')}|{m.get('document_id','?')}|{m.get('phrase_seq','?')}"

    topk_auth   = [_id(m) for m in auth_res["metadatas"][0]]
    topk_unauth = [_id(m) for m in unauth_res["metadatas"][0]]

    return TopKResult(
        method                  = "metadata",
        config_id               = "metadata_filter",
        topk_auth_ids           = topk_auth,
        topk_unauth_ids         = topk_unauth,
        restricted_in_auth      = len(set(topk_auth)   & restricted_keys),
        restricted_in_unauth    = len(set(topk_unauth) & restricted_keys),
        retrieval_time_auth_s   = round(t_auth,   6),
        retrieval_time_unauth_s = round(t_unauth, 6),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Per-query orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

@timed("run_query_dim_experiment")
def run_query_dim_experiment(
    gt_record:       dict,
    configs:         list[ExtraDimConfig],
    embedder,
    orig_collection,
    dim_collections: dict[str, object],   # config_id → chroma collection
    meta_collection,
    top_k:           int  = DEFAULT_TOP_K,
    verbose:         bool = True,
) -> DimQueryResult:
    """
    Full access-control experiment for one query:
    runs Method 2 for every config AND Method 3.
    """
    question      = gt_record["question"]
    query_id      = f'triplet_{gt_record["id_triplets"]}'
    triplet_index = gt_record["id_triplets"]
    stable_chunks = gt_record["targeted_chunk"]

    if verbose:
        print(f"\n  Query: {question[:70]}…")
        print(f"  Targeted chunks: {len(stable_chunks)}")

    # Chunks explicitly targeted by this query are "restricted"
    restricted_keys: set[str] = set(stable_chunks)

    # ── Embed the raw query ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    raw_query_vec = np.array(
        embedder.embed_query(BGE_QUERY_PREFIX + question), dtype=np.float32
    )
    TIMING_LOG.append({
        "label":      "embed_query",
        "start_iso":  datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.perf_counter() - t0, 6),
        "args_repr":  query_id,
    })

    all_chunk_sims: list[DimChunkSimilarities] = []
    all_topk:       list[TopKResult]           = []

    # ── Method 2: Extra-dimension (one pass per config) ────────────────────────
    written_dim_ids: dict[str, set[str]] = {cfg.config_id: set() for cfg in configs}

    for cfg in configs:
        dim_coll = dim_collections[cfg.config_id]

        raw_vecs = _m2_upsert_chunks(
            stable_chunks   = stable_chunks,
            triplet_index   = triplet_index,
            cfg             = cfg,
            embedder        = embedder,
            orig_collection = orig_collection,
            dim_collection  = dim_coll,
            restricted_keys = restricted_keys,
            written_ids     = written_dim_ids[cfg.config_id],
        )

        sims = _m2_compute_sims(raw_vecs, raw_query_vec, cfg, restricted_keys)
        all_chunk_sims.extend(sims)

        auth_q_vec   = augment_vector(raw_query_vec, cfg, is_restricted=True)
        unauth_q_vec = augment_vector(raw_query_vec, cfg, is_restricted=False)

        topk_m2 = _m2_topk(dim_coll, auth_q_vec, unauth_q_vec, restricted_keys, top_k, cfg)
        all_topk.append(topk_m2)

        if verbose:
            avg_sec_delta = (
                sum(c.security_delta for c in sims if c.is_restricted)
                / max(sum(1 for c in sims if c.is_restricted), 1)
            )
            print(
                f"    [M2 cfg={cfg.config_id}]  "
                f"avg_security_delta={avg_sec_delta:+.4f}  "
                f"restricted_in_auth={topk_m2.restricted_in_auth}  "
                f"restricted_in_unauth={topk_m2.restricted_in_unauth}  "
                f"t_auth={topk_m2.retrieval_time_auth_s:.4f}s  "
                f"t_unauth={topk_m2.retrieval_time_unauth_s:.4f}s"
            )

    # ── Method 3: Metadata filtering ──────────────────────────────────────────
    written_meta_ids: set[str] = set()
    _m3_upsert_chunks(
        stable_chunks   = stable_chunks,
        triplet_index   = triplet_index,
        embedder        = embedder,
        orig_collection = orig_collection,
        meta_collection = meta_collection,
        restricted_keys = restricted_keys,
        written_ids     = written_meta_ids,
    )

    topk_m3 = _m3_topk(meta_collection, raw_query_vec, restricted_keys, top_k)
    all_topk.append(topk_m3)

    if verbose:
        print(
            f"    [M3 metadata-filter]  "
            f"restricted_in_auth={topk_m3.restricted_in_auth}  "
            f"restricted_in_unauth={topk_m3.restricted_in_unauth}  "
            f"t_auth={topk_m3.retrieval_time_auth_s:.4f}s  "
            f"t_unauth={topk_m3.retrieval_time_unauth_s:.4f}s"
        )

    return DimQueryResult(
        query_id      = query_id,
        question      = question,
        triplet_index = triplet_index,
        n_restricted  = len(restricted_keys),
        n_open        = 0,
        chunk_sims    = all_chunk_sims,
        topk_results  = all_topk,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Full experiment runner
# ═══════════════════════════════════════════════════════════════════════════════

@timed("run_dim_experiment")
def run_dim_experiment(
    gt_path:  str                       = str(GT_FILE),
    configs:  list[ExtraDimConfig]|None = None,
    top_k:    int                       = DEFAULT_TOP_K,
    verbose:  bool                      = True,
) -> list[dict]:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(gt_path, encoding="utf-8") as fh:
        gt_records: list[dict] = json.load(fh)

    if configs is None:
        configs = _default_configs()

    print(f"\n{'═'*66}")
    print(f"  Access-Control Embedding Experiment")
    print(f"  Ground truth: {gt_path}  ({len(gt_records)} queries)")
    print(f"  Configs (Method 2): {len(configs)}")
    print(f"  Top-K: {top_k}")
    print(f"{'═'*66}")

    # ── Initialise models and collections ─────────────────────────────────────
    t0 = time.perf_counter()
    embedder    = get_embedding_model()
    TIMING_LOG.append({
        "label":      "init_embedding_model",
        "duration_s": round(time.perf_counter() - t0, 6),
        "args_repr":  "",
    })

    orig_coll     = _get_original_collection(ORIGINAL_CHROMA, ORIGINAL_COLLECTION)
    dim_colls     = {cfg.config_id: _get_dim_collection(cfg.config_id) for cfg in configs}
    meta_coll     = _get_meta_collection()

    all_results:  list[dict] = []
    registry:     list[dict] = []

    for i, record in enumerate(gt_records, 1):
        print(f"\n[{i}/{len(gt_records)}] triplet_{record.get('id_triplets', '?')}")

        if not record.get("id_triplets") or not record.get("targeted_chunk"):
            print("  ⚠  No targeted chunks — skipping.")
            continue

        result = run_query_dim_experiment(
            gt_record       = record,
            configs         = configs,
            embedder        = embedder,
            orig_collection = orig_coll,
            dim_collections = dim_colls,
            meta_collection = meta_coll,
            top_k           = top_k,
            verbose         = verbose,
        )
        all_results.append(asdict(result))
        registry.append({
            "query_id":     result.query_id,
            "n_restricted": result.n_restricted,
            "configs":      [cfg.to_dict() for cfg in configs],
        })

    # ── Serialise results ──────────────────────────────────────────────────────
    with open(DIM_RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print(f"\n✅  Results saved to {DIM_RESULTS_FILE}")

    with open(DIM_REGISTRY_FILE, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)
    print(f"✅  Registry saved to {DIM_REGISTRY_FILE}")

    save_timing_log()

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Access-control embedding experiment (extra-dim + metadata filter)"
    )
    parser.add_argument("--gt",               default=str(GT_FILE),
                        help="Path to merged_id_triplets_with_metadata2.json")
    parser.add_argument("--top-k",            type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--extra-dims",       type=int,   default=DEFAULT_EXTRA_DIMS,
                        help="Number of extra dimensions to append  (default %(default)s)")
    parser.add_argument("--large-value",      type=float, default=None,
                        help="Single large_value scalar to use (sweeps default configs if omitted)")
    parser.add_argument("--large-values",     type=float, nargs="+", default=None,
                        help="Sweep over multiple large_value scalars  e.g. 1e4 1e6 1e8")
    parser.add_argument("--normalize-before", action="store_true",
                        help="L2-normalise base vector BEFORE appending extra dims")
    parser.add_argument("--normalize-after",  action="store_true",
                        help="L2-normalise augmented vector AFTER appending extra dims")
    parser.add_argument("--quiet", "-q",      action="store_true")
    args = parser.parse_args()

    # ── Build configs ──────────────────────────────────────────────────────────
    if args.large_values is not None:
        configs = [
            ExtraDimConfig(
                extra_dims       = args.extra_dims,
                large_value      = lv,
                normalize_before = args.normalize_before,
                normalize_after  = args.normalize_after,
            )
            for lv in args.large_values
        ]
    elif args.large_value is not None:
        configs = [
            ExtraDimConfig(
                extra_dims       = args.extra_dims,
                large_value      = args.large_value,
                normalize_before = args.normalize_before,
                normalize_after  = args.normalize_after,
            )
        ]
    else:
        configs = _default_configs()

    run_dim_experiment(
        gt_path = args.gt,
        configs = configs,
        top_k   = args.top_k,
        verbose = not args.quiet,
    )