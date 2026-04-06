"""
dim_experiment.py
=================
Access-Control Embedding Experiment — two methods, three phases.

═══════════════════════════════════════════════════════════════════════════════
METHOD 2  (extra-dim)   — append N extra dimensions (one per query in GT file)
METHOD 3  (metadata)    — Chroma where-filter on `restricted` at query time
═══════════════════════════════════════════════════════════════════════════════

Extra-Dimension encoding  (Method 2)
--------------------------------------
The GT file contains N queries.  Every stored vector gets exactly N extra
dimensions appended — one slot per query.

  chunk targeted by query i  →  slot i = large_value,  all other slots = 0
  chunk NOT targeted by any  →  all slots = 0

Query side (symmetric, one-hot):
  query i (authorised)       →  slot i = large_value,  all other slots = 0
  query j≠i (unauthorised)   →  slot j = large_value   (diverges from slot i)

Cosine effect:
  authorised query i ↔ its chunk  →  extra dims align → higher cosine ✓
  unauthorised query j ↔ chunk i  →  extra dims are orthogonal → lower cosine ✓
  non-targeted chunk ↔ any query  →  extra dims are all-zero → neutral ✓

Optional post-append L2-normalisation is configurable per-ExtraDimConfig.

Metadata Filtering  (Method 3)
--------------------------------
Plain embeddings stored with boolean field `restricted`.
Authorised query → no filter (retrieves everything).
Unauthorised query → where={"restricted": {"$eq": False}}
The filter is applied by Chroma before similarity ranking.

Three phases
------------
Phase 1 — build_dim_db / build_meta_db
    Full pass over GT records.  Augments every chunk vector and upserts it.
    No queries executed here.

Phase 2 — run_query_phase
    Full pass over GT records.  For each query:
      • embed the query vector
      • build its augmented version  (Method 2)  or leave plain  (Method 3)
      • retrieve from the appropriate Chroma collection
    Precise per-step timings recorded.  No metric computation.

Phase 3 — evaluate_results
    Takes raw retrieval data from Phase 2 and computes all metrics.
    Writes results/dim_results.json.

Timing decorator
    @timed("label") records wall-clock duration of every major call and
    appends structured entries to TIMING_LOG → logs/dim_timing.json.

Usage
-----
python dim_experiment.py
python dim_experiment.py --gt documents_RAGBench/merged_id_triplets_with_metadata2.json
python dim_experiment.py --large-value 1e8 --normalize-after
python dim_experiment.py --large-values 1e4 1e6 1e8
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

from config import COLLECTION
from ingestion_pipeline import get_embedding_model
from query_pipeline import BGE_QUERY_PREFIX

# ── Paths ──────────────────────────────────────────────────────────────────────
LOGS_DIR          = Path("logs")
RESULTS_DIR       = Path("results")
GT_FILE           = Path("documents_RAGBench/merged_id_triplets_with_metadata2.json")
DIM_RESULTS_FILE  = RESULTS_DIR / "dim_results.json"
DIM_REGISTRY_FILE = RESULTS_DIR / "dim_registry.json"
TIMING_FILE       = LOGS_DIR    / "dim_timing.json"

ORIGINAL_CHROMA     = os.path.join(os.getcwd(), "./chroma_db")
ORIGINAL_COLLECTION = COLLECTION
DIM_CHROMA_BASE     = os.path.join(os.getcwd(), "./chroma_dim_db")
META_CHROMA_BASE    = os.path.join(os.getcwd(), "./chroma_meta_db")

DEFAULT_TOP_K     = 20
DEFAULT_LARGE_VAL = 1e6

os.mkdir(LOGS_DIR, exist_ok=True)
os.mkdir(RESULTS_DIR, exist_ok=True)
os.mkdir(DIM_CHROMA_BASE, exist_ok=True)
os.mkdir(META_CHROMA_BASE, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Timing infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

TIMING_LOG: list[dict] = []


def timed(label: str):
    """
    Decorator factory — records wall-clock duration of every call.

    Appends to module-level TIMING_LOG::

        {
            "label":      "build_dim_db",
            "start_iso":  "2025-…",
            "duration_s": 4.123456,
            "args_repr":  "…"
        }
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            args_repr = str(args[:2])[:120]
            start     = time.perf_counter()
            start_iso = datetime.now(timezone.utc).isoformat()
            try:
                result = fn(*args, **kwargs)
            finally:
                TIMING_LOG.append({
                    "label":      label,
                    "start_iso":  start_iso,
                    "duration_s": round(time.perf_counter() - start, 6),
                    "args_repr":  args_repr,
                })
            return result
        return wrapper
    return decorator


def save_timing_log() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TIMING_FILE, "w", encoding="utf-8") as fh:
        json.dump(TIMING_LOG, fh, indent=2)
    print(f"⏱  Timing log → {TIMING_FILE}  ({len(TIMING_LOG)} entries)")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtraDimConfig:
    """
    Knobs for one extra-dimension experimental condition.

    extra_dims is derived automatically from the number of queries in the GT
    file (one slot per query), so it is set after loading the GT file.
    The user-facing parameters are large_value and normalisation flags.
    """
    large_value:      float = DEFAULT_LARGE_VAL
    normalize_after:  bool  = False

    # Set by the experiment runner once the GT file is loaded
    n_queries: int = 0

    @property
    def config_id(self) -> str:
        lv = f"{self.large_value:.0e}".replace("+", "")
        na = "na1" if self.normalize_after else "na0"
        return f"q{self.n_queries}_lv{lv}_{na}"

    def to_dict(self) -> dict:
        return {
            "large_value":     self.large_value,
            "normalize_after": self.normalize_after,
            "n_queries":       self.n_queries,
            "config_id":       self.config_id,
        }


def _default_configs() -> list[ExtraDimConfig]:
    """Representative sweep; n_queries is filled in by the runner."""
    return [
        ExtraDimConfig(large_value=1e4,  normalize_after=False),
        ExtraDimConfig(large_value=1e6,  normalize_after=False),
        ExtraDimConfig(large_value=1e8,  normalize_after=False),
        ExtraDimConfig(large_value=1e6,  normalize_after=True),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Augmentation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def augment_chunk_vector(
    base_vec:    np.ndarray,
    cfg:         ExtraDimConfig,
    query_index: int | None,
) -> np.ndarray:
    """
    Append N extra dimensions to `base_vec` (N = cfg.n_queries).

    query_index : the slot index for the query that owns this chunk.
                  If None the chunk is non-targeted → all-zero extra dims.
    """
    v     = base_vec.astype(np.float32)
    extra = np.zeros(cfg.n_queries, dtype=np.float32)
    if query_index is not None:
        extra[query_index] = cfg.large_value

    augmented = np.concatenate([v, extra])
    if cfg.normalize_after:
        augmented = _l2_norm(augmented)
    return augmented


def augment_query_vector(
    base_vec:    np.ndarray,
    cfg:         ExtraDimConfig,
    query_index: int,
    authorised:  bool,
) -> np.ndarray:
    """
    Append N extra dimensions to a query vector.

    authorised=True  → slot[query_index] = large_value  (aligns with own chunks)
    authorised=False → slot[query_index] = 0             (diverges from own chunks)

    All other slots are always 0 for a query vector — a query only ever
    carries signal in its own slot.
    """
    v     = base_vec.astype(np.float32)
    extra = np.zeros(cfg.n_queries, dtype=np.float32)
    if authorised:
        extra[query_index] = cfg.large_value

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
        path=path, settings=Settings(anonymized_telemetry=False)
    )


def _get_original_collection(path: str, name: str):
    return _get_client(path).get_collection(name=name)


def _get_dim_collection(cfg_id: str):
    """Each config gets its own Chroma collection."""
    client = _get_client(DIM_CHROMA_BASE)
    return client.get_or_create_collection(
        name=f"dim_{cfg_id}", metadata={"hnsw:space": "cosine"}
    )


def _get_meta_collection():
    """Single shared collection for Method 3."""
    return _get_client(META_CHROMA_BASE).get_or_create_collection(
        name="meta_access_control", metadata={"hnsw:space": "cosine"}
    )


@timed("fetch_chunk_from_original_db")
def fetch_chunk(
    collection,
    triplet_index: str,
    document_id:   str,
    phrase_seq:    str,
) -> tuple[np.ndarray | None, str | None]:
    """Return (embedding, document_text) or (None, None) if not found."""
    try:
        result = collection.get(
            where={"$and": [
                {"triplet_index": {"$eq": triplet_index}},
                {"document_id":   {"$eq": document_id}},
                {"phrase_seq":    {"$eq": phrase_seq}},
            ]},
            include=["embeddings", "documents"],
        )
        embs = result.get("embeddings", [[]])
        docs = result.get("documents", [[]])
        if embs and len(embs[0]) > 0:
            return np.array(embs[0], dtype=np.float32), (docs[0] if docs else None)
    except Exception as e:
        warnings.warn(f"fetch_chunk failed for {triplet_index}|{document_id}|{phrase_seq}: {e}")
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Result data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DimChunkSimilarities:
    chunk_key:             str
    config_id:             str
    query_index:           int     # which slot belongs to this chunk's query
    is_targeted:           bool

    # (a) authorised query ↔ chunk (both carry large_value in slot[query_index])
    sim_auth:              float
    # (b) unauthorised query ↔ chunk (query has 0 in slot[query_index])
    sim_unauth:            float
    # (c) no augmentation at all — pure baseline
    sim_baseline:          float

    @property
    def security_delta(self) -> float:
        """sim_unauth − sim_auth.  Negative = restriction working."""
        return self.sim_unauth - self.sim_auth


@dataclass
class MethodTopK:
    method:                  str    # "extradim" | "metadata"
    config_id:               str
    topk_auth_ids:           list[str]  = field(default_factory=list)
    topk_unauth_ids:         list[str]  = field(default_factory=list)
    targeted_in_auth:        int        = 0
    targeted_in_unauth:      int        = 0
    t_augment_auth_s:        float      = 0.0   # time to augment the auth query vector
    t_augment_unauth_s:      float      = 0.0
    t_retrieval_auth_s:      float      = 0.0   # time for the Chroma query call
    t_retrieval_unauth_s:    float      = 0.0


@dataclass
class RawQueryRetrieval:
    """Output of Phase 2 for one query — raw data, no computed metrics."""
    query_id:            str
    question:            str
    triplet_index:       str
    query_index:         int           # position in GT list (0-based)
    targeted_chunk_ids:  list[str]

    # Raw query vector (un-augmented)
    query_vec:           list[float]

    # Per-chunk raw vectors: chunk_key → {"base": list[float]}
    chunk_vectors:       dict          = field(default_factory=dict)

    # Top-K results, one entry per method/config
    topk_results:        list[MethodTopK] = field(default_factory=list)

    # Phase 2 step timings
    t_embed_query_s:     float = 0.0

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class QueryEvalResult:
    """Output of Phase 3 — fully evaluated metrics for one query."""
    query_id:          str
    question:          str
    triplet_index:     str
    query_index:       int
    n_targeted_chunks: int

    chunk_similarities: list[DimChunkSimilarities] = field(default_factory=list)
    topk_results:       list[MethodTopK]            = field(default_factory=list)

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Phase 1 — Build databases
# ═══════════════════════════════════════════════════════════════════════════════

@timed("build_dim_db_single_record")
def _build_dim_record(
    record:         dict,
    query_index:    int,
    cfg:            ExtraDimConfig,
    orig_collection,
    dim_collection,
    written_ids:    set[str],
) -> int:
    """Augment and upsert all targeted chunks for one GT record (Method 2)."""
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    n_ok = 0

    for chunk_id in stable_chunks:
        tid  = chunk_id.split("|")[0]
        did  = chunk_id[-2]
        pseq = chunk_id[-1]
        cid  = f"{cfg.config_id}_{triplet_index}_{did}_{pseq}"

        if cid in written_ids:
            n_ok += 1
            continue

        base_vec, content = fetch_chunk(orig_collection, tid, did, pseq)
        if base_vec is None:
            warnings.warn(f"  ⚠  Build(dim): chunk not found {chunk_id} — skipping.")
            continue

        aug_vec = augment_chunk_vector(base_vec, cfg, query_index=query_index)

        dim_collection.upsert(
            ids        = [cid],
            embeddings = [aug_vec.tolist()],
            documents  = [content or ""],
            metadatas  = [{
                "triplet_index": triplet_index,
                "document_id":   did,
                "phrase_seq":    pseq,
                "query_index":   query_index,
                "config_id":     cfg.config_id,
                "restricted":    True,
            }],
        )
        written_ids.add(cid)
        n_ok += 1

    return n_ok


@timed("build_dim_db")
def build_dim_db(
    gt_records:     list[dict],
    configs:        list[ExtraDimConfig],
    orig_collection,
    dim_collections: dict[str, object],
    verbose:        bool = True,
) -> None:
    """
    Phase 1 (Method 2) — full pass over GT records.
    Each chunk is augmented with its query's slot set to large_value.
    """
    print(f"\n{'─'*60}")
    print(f"  Phase 1a — Building extra-dim database ({len(gt_records)} records)")
    print(f"{'─'*60}")

    for cfg in configs:
        written_ids: set[str] = set()
        dim_coll = dim_collections[cfg.config_id]
        print(f"\n  Config: {cfg.config_id}")

        for i, record in enumerate(gt_records):
            triplet_index = record.get("id_triplets")
            if not triplet_index or not record.get("targeted_chunk"):
                continue

            n = _build_dim_record(
                record, i, cfg, orig_collection, dim_coll, written_ids
            )
            if verbose:
                print(
                    f"    [{i+1}/{len(gt_records)}] triplet_{triplet_index}"
                    f"  chunks_upserted={n}"
                )

    print(f"\n  ✅ Extra-dim build complete")


@timed("build_meta_db_single_record")
def _build_meta_record(
    record:         dict,
    orig_collection,
    meta_collection,
    written_ids:    set[str],
) -> int:
    """Upsert plain vectors with restricted=True metadata for one GT record (Method 3)."""
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    n_ok = 0

    for chunk_id in stable_chunks:
        tid  = chunk_id.split("|")[0]
        did  = chunk_id[-2]
        pseq = chunk_id[-1]
        cid  = f"meta_{triplet_index}_{did}_{pseq}"

        if cid in written_ids:
            n_ok += 1
            continue

        base_vec, content = fetch_chunk(orig_collection, tid, did, pseq)
        if base_vec is None:
            warnings.warn(f"  ⚠  Build(meta): chunk not found {chunk_id} — skipping.")
            continue

        meta_collection.upsert(
            ids        = [cid],
            embeddings = [base_vec.tolist()],
            documents  = [content or ""],
            metadatas  = [{
                "triplet_index": triplet_index,
                "document_id":   did,
                "phrase_seq":    pseq,
                "restricted":    True,
            }],
        )
        written_ids.add(cid)
        n_ok += 1

    return n_ok


@timed("build_meta_db")
def build_meta_db(
    gt_records:     list[dict],
    orig_collection,
    meta_collection,
    verbose:        bool = True,
) -> None:
    """
    Phase 1 (Method 3) — full pass over GT records.
    Stores plain embeddings with boolean `restricted` metadata field.
    """
    print(f"\n{'─'*60}")
    print(f"  Phase 1b — Building metadata-filter database ({len(gt_records)} records)")
    print(f"{'─'*60}")

    written_ids: set[str] = set()

    for i, record in enumerate(gt_records):
        triplet_index = record.get("id_triplets")
        if not triplet_index or not record.get("targeted_chunk"):
            continue

        n = _build_meta_record(record, orig_collection, meta_collection, written_ids)
        if verbose:
            print(
                f"  [{i+1}/{len(gt_records)}] triplet_{triplet_index}"
                f"  chunks_upserted={n}"
            )

    print(f"\n  ✅ Metadata build complete")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Phase 2 — Query phase
# ═══════════════════════════════════════════════════════════════════════════════

def _query_dim_single(
    raw_query_vec:   np.ndarray,
    query_index:     int,
    cfg:             ExtraDimConfig,
    dim_collection,
    targeted_keys:   set[str],
    top_k:           int,
    authorised:      bool,
) -> tuple[list[str], float, float]:
    """
    Augment one query vector and retrieve from the dim collection.
    Returns (topk_ids, t_augment_s, t_retrieval_s).
    """
    t0 = time.perf_counter()
    aug_q = augment_query_vector(raw_query_vec, cfg, query_index, authorised=authorised)
    t_aug = time.perf_counter() - t0

    n_in = max(1, dim_collection.count())
    t0   = time.perf_counter()
    res  = dim_collection.query(
        query_embeddings=[aug_q.tolist()],
        n_results=min(top_k, n_in),
        include=["metadatas"],
    )
    t_ret = time.perf_counter() - t0

    ids = [
        f"{m.get('triplet_index','?')}|{m.get('document_id','?')}|{m.get('phrase_seq','?')}"
        for m in res["metadatas"][0]
    ]
    return ids, round(t_aug, 6), round(t_ret, 6)


def _query_meta_single(
    raw_query_vec:  np.ndarray,
    meta_collection,
    top_k:          int,
    authorised:     bool,
) -> tuple[list[str], float]:
    """
    Query the metadata collection with or without the restricted filter.
    Returns (topk_ids, t_retrieval_s).
    No augmentation — timing covers only the Chroma call.
    """
    n_in = max(1, meta_collection.count())
    t0   = time.perf_counter()

    if authorised:
        # No filter — can see everything
        res = meta_collection.query(
            query_embeddings=[raw_query_vec.tolist()],
            n_results=min(top_k, n_in),
            include=["metadatas"],
        )
    else:
        # Exclude restricted chunks via metadata filter
        try:
            res = meta_collection.query(
                query_embeddings=[raw_query_vec.tolist()],
                n_results=min(top_k, n_in),
                where={"restricted": {"$eq": False}},
                include=["metadatas"],
            )
        except Exception as e:
            warnings.warn(f"Metadata filter query failed: {e}. Falling back.")
            res = meta_collection.query(
                query_embeddings=[raw_query_vec.tolist()],
                n_results=min(top_k, n_in),
                include=["metadatas"],
            )

    t_ret = time.perf_counter() - t0
    ids = [
        f"{m.get('triplet_index','?')}|{m.get('document_id','?')}|{m.get('phrase_seq','?')}"
        for m in res["metadatas"][0]
    ]
    return ids, round(t_ret, 6)


@timed("query_phase_single_record")
def _query_record(
    record:          dict,
    query_index:     int,
    configs:         list[ExtraDimConfig],
    embedder,
    orig_collection,
    dim_collections: dict[str, object],
    meta_collection,
    top_k:           int,
) -> RawQueryRetrieval:
    """
    Full query pipeline for one GT record — both methods, both auth states.
    Precise per-step timings are stored in MethodTopK entries.
    """
    question      = record["question"]
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    query_id      = f"triplet_{triplet_index}"
    targeted_keys = set(stable_chunks)

    # ── Embed raw query ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    raw_q = np.array(
        embedder.embed_query(BGE_QUERY_PREFIX + question), dtype=np.float32
    )
    t_embed = time.perf_counter() - t0
    TIMING_LOG.append({
        "label":      "embed_query",
        "start_iso":  datetime.now(timezone.utc).isoformat(),
        "duration_s": round(t_embed, 6),
        "args_repr":  query_id,
    })

    topk_results: list[MethodTopK] = []

    # ── Method 2: Extra-dim (one pass per config) ──────────────────────────────
    for cfg in configs:
        dim_coll = dim_collections[cfg.config_id]

        auth_ids, t_aug_a, t_ret_a = _query_dim_single(
            raw_q, query_index, cfg, dim_coll, targeted_keys, top_k, authorised=True
        )
        unauth_ids, t_aug_u, t_ret_u = _query_dim_single(
            raw_q, query_index, cfg, dim_coll, targeted_keys, top_k, authorised=False
        )

        topk_results.append(MethodTopK(
            method               = "extradim",
            config_id            = cfg.config_id,
            topk_auth_ids        = auth_ids,
            topk_unauth_ids      = unauth_ids,
            targeted_in_auth     = sum(1 for cid in auth_ids   if cid in targeted_keys),
            targeted_in_unauth   = sum(1 for cid in unauth_ids if cid in targeted_keys),
            t_augment_auth_s     = t_aug_a,
            t_augment_unauth_s   = t_aug_u,
            t_retrieval_auth_s   = t_ret_a,
            t_retrieval_unauth_s = t_ret_u,
        ))

    # ── Method 3: Metadata filter ──────────────────────────────────────────────
    auth_ids_m3,   t_ret_a3 = _query_meta_single(raw_q, meta_collection, top_k, authorised=True)
    unauth_ids_m3, t_ret_u3 = _query_meta_single(raw_q, meta_collection, top_k, authorised=False)

    topk_results.append(MethodTopK(
        method               = "metadata",
        config_id            = "metadata_filter",
        topk_auth_ids        = auth_ids_m3,
        topk_unauth_ids      = unauth_ids_m3,
        targeted_in_auth     = sum(1 for cid in auth_ids_m3   if cid in targeted_keys),
        targeted_in_unauth   = sum(1 for cid in unauth_ids_m3 if cid in targeted_keys),
        t_augment_auth_s     = 0.0,   # no augmentation for metadata method
        t_augment_unauth_s   = 0.0,
        t_retrieval_auth_s   = t_ret_a3,
        t_retrieval_unauth_s = t_ret_u3,
    ))

    # ── Collect raw chunk vectors for Phase 3 similarity computation ───────────
    chunk_vectors: dict[str, dict] = {}
    for chunk_id in stable_chunks:
        tid  = chunk_id.split("|")[0]
        did  = chunk_id[-2]
        pseq = chunk_id[-1]
        base_vec, _ = fetch_chunk(orig_collection, tid, did, pseq)
        if base_vec is not None:
            chunk_vectors[chunk_id] = {"base": base_vec.tolist()}

    return RawQueryRetrieval(
        query_id           = query_id,
        question           = question,
        triplet_index      = triplet_index,
        query_index        = query_index,
        targeted_chunk_ids = list(stable_chunks),
        query_vec          = raw_q.tolist(),
        chunk_vectors      = chunk_vectors,
        topk_results       = topk_results,
        t_embed_query_s    = round(t_embed, 6),
    )


@timed("run_query_phase")
def run_query_phase(
    gt_records:      list[dict],
    configs:         list[ExtraDimConfig],
    embedder,
    orig_collection,
    dim_collections: dict[str, object],
    meta_collection,
    top_k:           int  = DEFAULT_TOP_K,
    verbose:         bool = True,
) -> list[RawQueryRetrieval]:
    """
    Phase 2 — full pass over GT records.
    Returns raw retrieval data for all queries, both methods, both auth states.
    """
    print(f"\n{'─'*60}")
    print(f"  Phase 2 — Query phase ({len(gt_records)} records, top-K={top_k})")
    print(f"{'─'*60}")

    raw_results: list[RawQueryRetrieval] = []

    for i, record in enumerate(gt_records):
        triplet_index = record.get("id_triplets")
        if not triplet_index or not record.get("targeted_chunk"):
            if verbose:
                print(f"  [{i+1}] ⚠  No targeted chunks — skipping.")
            continue

        raw = _query_record(
            record          = record,
            query_index     = i,
            configs         = configs,
            embedder        = embedder,
            orig_collection = orig_collection,
            dim_collections = dim_collections,
            meta_collection = meta_collection,
            top_k           = top_k,
        )
        raw_results.append(raw)

        if verbose:
            print(f"  [{i+1}/{len(gt_records)}] {raw.query_id}  t_embed={raw.t_embed_query_s:.4f}s")
            for tk in raw.topk_results:
                print(
                    f"    [{tk.method}/{tk.config_id}]"
                    f"  targeted_auth={tk.targeted_in_auth}/{len(raw.targeted_chunk_ids)}"
                    f"  targeted_unauth={tk.targeted_in_unauth}/{len(raw.targeted_chunk_ids)}"
                    f"  t_aug_auth={tk.t_augment_auth_s:.6f}s"
                    f"  t_ret_auth={tk.t_retrieval_auth_s:.4f}s"
                    f"  t_ret_unauth={tk.t_retrieval_unauth_s:.4f}s"
                )

    return raw_results


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Phase 3 — Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@timed("evaluate_single_record")
def _evaluate_record(
    raw:     RawQueryRetrieval,
    configs: list[ExtraDimConfig],
) -> QueryEvalResult:
    """Compute similarity metrics for one query from its raw retrieval data."""
    raw_q      = np.array(raw.query_vec, dtype=np.float32)
    chunk_sims: list[DimChunkSimilarities] = []

    for cfg in configs:
        for chunk_id, vecs in raw.chunk_vectors.items():
            base_vec = np.array(vecs["base"], dtype=np.float32)

            auth_q   = augment_query_vector(raw_q, cfg, raw.query_index, authorised=True)
            unauth_q = augment_query_vector(raw_q, cfg, raw.query_index, authorised=False)
            aug_chunk = augment_chunk_vector(base_vec, cfg, query_index=raw.query_index)

            chunk_sims.append(DimChunkSimilarities(
                chunk_key    = chunk_id,
                config_id    = cfg.config_id,
                query_index  = raw.query_index,
                is_targeted  = chunk_id in raw.targeted_chunk_ids,
                sim_auth     = cosine_similarity(auth_q,   aug_chunk),
                sim_unauth   = cosine_similarity(unauth_q, aug_chunk),
                sim_baseline = cosine_similarity(raw_q,    base_vec),
            ))

    return QueryEvalResult(
        query_id           = raw.query_id,
        question           = raw.question,
        triplet_index      = raw.triplet_index,
        query_index        = raw.query_index,
        n_targeted_chunks  = len(raw.targeted_chunk_ids),
        chunk_similarities = chunk_sims,
        topk_results       = raw.topk_results,
    )


@timed("evaluate_results")
def evaluate_results(
    raw_results: list[RawQueryRetrieval],
    configs:     list[ExtraDimConfig],
    verbose:     bool = True,
) -> list[QueryEvalResult]:
    """
    Phase 3 — compute all metrics and write results/dim_results.json.
    """
    print(f"\n{'─'*60}")
    print(f"  Phase 3 — Evaluation ({len(raw_results)} queries)")
    print(f"{'─'*60}")

    eval_results: list[QueryEvalResult] = []

    for raw in raw_results:
        ev = _evaluate_record(raw, configs)
        eval_results.append(ev)

        if verbose:
            for cfg in configs:
                cfg_sims = [c for c in ev.chunk_similarities if c.config_id == cfg.config_id]
                avg_delta = (
                    sum(c.security_delta for c in cfg_sims)
                    / max(len(cfg_sims), 1)
                )
                print(
                    f"  {ev.query_id}  cfg={cfg.config_id}"
                    f"  avg_security_delta={avg_delta:+.4f}"
                )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DIM_RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in eval_results], fh, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Results → {DIM_RESULTS_FILE}")

    return eval_results


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

@timed("run_experiment")
def run_experiment(
    gt_path:  str                       = str(GT_FILE),
    configs:  list[ExtraDimConfig]|None = None,
    top_k:    int                       = DEFAULT_TOP_K,
    verbose:  bool                      = True,
) -> list[QueryEvalResult]:
    """Runs all three phases for both methods and persists results + timing."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(gt_path, encoding="utf-8") as fh:
        gt_records: list[dict] = json.load(fh)

    n_queries = len(gt_records)

    if configs is None:
        configs = _default_configs()

    # Stamp every config with n_queries so the slot count is correct
    for cfg in configs:
        cfg.n_queries = n_queries

    print(f"\n{'═'*60}")
    print(f"  Extra-Dim + Metadata Access-Control Experiment")
    print(f"  GT file   : {gt_path}  ({n_queries} records)")
    print(f"  Extra dims: {n_queries}  (one per query)")
    print(f"  Configs   : {len(configs)}")
    print(f"  Top-K     : {top_k}")
    print(f"{'═'*60}")

    # ── Init models and collections ────────────────────────────────────────────
    t0       = time.perf_counter()
    embedder = get_embedding_model()
    TIMING_LOG.append({
        "label":      "init_embedding_model",
        "start_iso":  datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.perf_counter() - t0, 6),
        "args_repr":  "",
    })

    orig_coll    = _get_original_collection(ORIGINAL_CHROMA, ORIGINAL_COLLECTION)
    dim_colls    = {cfg.config_id: _get_dim_collection(cfg.config_id) for cfg in configs}
    meta_coll    = _get_meta_collection()

    # Persist registry / config info
    registry_data = {
        "n_queries":  n_queries,
        "gt_path":    gt_path,
        "configs":    [cfg.to_dict() for cfg in configs],
        "query_index_map": {
            record["id_triplets"]: i
            for i, record in enumerate(gt_records)
            if record.get("id_triplets")
        },
    }
    with open(DIM_REGISTRY_FILE, "w", encoding="utf-8") as fh:
        json.dump(registry_data, fh, indent=2)
    print(f"\n  Registry → {DIM_REGISTRY_FILE}")

    # ── Phase 1: Build ─────────────────────────────────────────────────────────
    build_dim_db(gt_records, configs, orig_coll, dim_colls, verbose=verbose)
    build_meta_db(gt_records, orig_coll, meta_coll, verbose=verbose)

    # ── Phase 2: Query ─────────────────────────────────────────────────────────
    raw_results = run_query_phase(
        gt_records, configs, embedder, orig_coll, dim_colls, meta_coll,
        top_k=top_k, verbose=verbose,
    )

    # ── Phase 3: Evaluate ──────────────────────────────────────────────────────
    eval_results = evaluate_results(raw_results, configs, verbose=verbose)

    save_timing_log()
    return eval_results


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Access-control embedding experiment (extra-dim + metadata filter)"
    )
    parser.add_argument("--gt",               default=str(GT_FILE))
    parser.add_argument("--top-k",            type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--large-value",      type=float, default=None,
                        help="Single large_value scalar (uses default sweep if omitted)")
    parser.add_argument("--large-values",     type=float, nargs="+", default=None,
                        help="Sweep over multiple large_value scalars e.g. 1e4 1e6 1e8")
    parser.add_argument("--normalize-after",  action="store_true",
                        help="L2-normalise the augmented vector after appending extra dims")
    parser.add_argument("--quiet", "-q",      action="store_true")
    args = parser.parse_args()

    if args.large_values is not None:
        user_configs = [
            ExtraDimConfig(large_value=lv, normalize_after=args.normalize_after)
            for lv in args.large_values
        ]
    elif args.large_value is not None:
        user_configs = [
            ExtraDimConfig(large_value=args.large_value, normalize_after=args.normalize_after)
        ]
    else:
        user_configs = _default_configs()

    run_experiment(
        gt_path = args.gt,
        configs = user_configs,
        top_k   = args.top_k,
        verbose = not args.quiet,
    )
