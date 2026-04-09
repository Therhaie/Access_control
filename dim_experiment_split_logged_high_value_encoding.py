"""
dim_experiment.py
=================
Extra-Dimension + Metadata-Filter Access-Control Experiment — three phases.

═══════════════════════════════════════════════════════════════════════════════
METHOD 2  (extra-dim)   — append N extra dimensions, one slot per GT query
METHOD 3  (metadata)    — Chroma where-filter on `restricted` at query time
═══════════════════════════════════════════════════════════════════════════════

Extra-Dimension encoding  (Method 2)
--------------------------------------
The GT file contains N queries.  Every stored vector gets exactly N extra
dimensions appended (one slot per query):

  chunk targeted by query i  →  slot i = large_value,  all other slots = 0
  chunk NOT targeted by any  →  all slots = 0

Query side (one-hot, symmetric with chunks):
  query i — authorised       →  slot i = large_value,  all other slots = 0
  query i — unauthorised     →  all slots = 0  (diverges from its own chunks)

Cosine effect:
  auth query ↔ its aug chunk       →  extra dims align  → higher cosine ✓
  unauth query ↔ aug chunk         →  extra dims are zero vs large_value → lower ✓
  any query ↔ non-targeted chunk   →  extra dims all-zero → neutral ✓

Symmetrical experiment (mirrors rotation_experiment.py):
  targeted_in_auth_query_aug_db   — auth (augmented) query    → augmented DB
  targeted_in_unauth_query_aug_db — unauth (plain zeros) query → augmented DB
  sim_auth_query_aug_chunk        — (a) both sides augmented   → should be high
  sim_unauth_query_aug_chunk      — (b) unauth query, aug chunk → should drop
  sim_auth_query_plain_chunk      — (c) auth query, plain chunk → collateral
  sim_plain_query_plain_chunk     — (d) no augmentation        → baseline

Metadata Filtering  (Method 3)
--------------------------------
Plain embeddings stored with boolean `restricted` metadata.
Authorised   → no filter (retrieves everything).
Unauthorised → where={"restricted": {"$eq": False}} (excludes restricted chunks).
targeted_in_auth_meta / targeted_in_unauth_meta track the same metric.

Three phases
------------
Phase 1 — build_aug_db + build_meta_db
    Full pass over GT records.  Augments every chunk vector and upserts it
    into the augmented collection, and stores plain vectors + metadata in
    the metadata collection.  No queries here.

Phase 2 — run_query_phase
    Full pass over GT records.  For each query:
      • embed the raw query vector                                  [timed]
      • build augmented (auth) and plain (unauth) query vectors     [timed]
      • retrieve from augmented DB with both query types            [timed]
      • retrieve from metadata DB with/without filter               [timed]
    Returns RawQueryRetrieval.  No metric computation.

Phase 3 — evaluate_results
    Computes all metrics from raw data.  Writes results/dim_results.json.

Timing log  →  logs/dim_timing.json
    "entries"         — one dict per @timed call
    "summary"         — per-label {mean_s, std_s, min_s, max_s, n}
    "retrieval_stats" — mean / std of targeted_in_* across all queries

Usage
-----
python dim_experiment.py
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
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import chromadb
import numpy as np
from chromadb.config import Settings

from config import COLLECTION
from ingestion_pipeline import get_embedding_model
from query_pipeline import BGE_QUERY_PREFIX

#helpers
from plot_PCA import get_all_chunk_ids, get_list_id_targeted_chunk

DEFAULT_TOP_K     = 20
DEFAULT_LARGE_VAL = 1e9 # 1e12 #1000000 # 1e6
DEFAULT_LARGE_VAL_UNTARGETED = 100 # DEFAULT_LARGE_VAL   #* 0.001
EXTRA_DIM_UNTARGETED_AS_LARGE_VALUE = True # use a 101 dimension for thhe non-targeted chunks
NORMALIZE_BEFORE_ADDING_EXTRA_DIMS = False 
DISTANCE_METRIC = "l2"  # "cosine" or "l2" or "ip"
NUMBER_OF_EXTRA_DIMS = 4  # number of dimension to represent the untargeted chunks

# ── Paths ──────────────────────────────────────────────────────────────────────
LOGS_DIR          = Path("logs")
RESULTS_DIR       = Path("results")
GT_FILE           = Path("documents_RAGBench/merged_id_triplets_with_metadata2.json")
DIM_RESULTS_FILE  = RESULTS_DIR / "dim_results.json"
DIM_REGISTRY_FILE = RESULTS_DIR / "dim_registry.json"
TIMING_FILE       = LOGS_DIR    / f"dim_timing_largeval_{DEFAULT_LARGE_VAL}_topk_{DEFAULT_TOP_K}_extra_dim_untargeted.json"

ORIGINAL_CHROMA     = os.path.join(os.getcwd(), "./chroma_db")
ORIGINAL_COLLECTION = COLLECTION
AUG_CHROMA_BASE     = os.path.join(os.getcwd(), "./chroma_aug_db")
AUGMENTED_NAME     = "augmented_db_norm_high_value_encoding_extra_dim_untargeted_nb_extra_dims_" + str(NUMBER_OF_EXTRA_DIMS)  # collection name prefix for augmented DB (one per config)
META_CHROMA_BASE    = os.path.join(os.getcwd(), "./chroma_meta_db")
META_NAME           = "meta_access_control"




# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Timing infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

TIMING_LOG: list[dict] = []


def timed(label: str):
    """
    Decorator factory — records wall-clock duration of every call.

    Each call appends to module-level TIMING_LOG:
        {
            "label":      "build_aug_db_single_record",
            "start_iso":  "2025-…",
            "duration_s": 0.123456,
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


def _compute_timing_summary() -> dict:
    """Group log entries by label, return per-label mean ± std / min / max / n."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for entry in TIMING_LOG:
        buckets[entry["label"]].append(entry["duration_s"])

    summary = {}
    for label, durations in sorted(buckets.items()):
        arr = np.array(durations)
        summary[label] = {
            "n":      int(len(arr)),
            "mean_s": round(float(arr.mean()), 6),
            "std_s":  round(float(arr.std()),  6),
            "min_s":  round(float(arr.min()),  6),
            "max_s":  round(float(arr.max()),  6),
        }
    return summary


def _compute_retrieval_stats(eval_results: "list[QueryEvalResult]") -> dict:
    """
    Mean / std of every targeted_in_* counter across all evaluated queries,
    both as raw counts and as fractions of n_targeted_chunks.
    """
    fields = [
        "targeted_in_auth_query_aug_db",
        "targeted_in_unauth_query_aug_db",
        "targeted_in_auth_meta",
        "targeted_in_unauth_meta",
    ]
    totals = np.array([r.n_targeted_chunks for r in eval_results], dtype=float)

    stats = {}
    for f in fields:
        counts = np.array([getattr(r, f) for r in eval_results], dtype=float)
        fracs  = np.where(totals > 0, counts / totals, 0.0)
        stats[f] = {
            "mean_count": round(float(counts.mean()), 4),
            "std_count":  round(float(counts.std()),  4),
            "min_count":  round(float(counts.min()),  4),
            "max_count":  round(float(counts.max()),  4),
            "mean_frac":  round(float(fracs.mean()),  4),
            "std_frac":   round(float(fracs.std()),   4),
            "n_queries":  int(len(counts)),
        }
    return stats


def save_timing_log(
    eval_results: "list[QueryEvalResult] | None" = None,
) -> None:
    """
    Flush TIMING_LOG to logs/dim_timing.json.

    Structure:
        {
          "entries":         [ …per-call records… ],
          "summary":         { label: {n, mean_s, std_s, min_s, max_s} },
          "retrieval_stats": { metric: {mean_count, std_count, mean_frac, …} }
        }
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    output: dict = {
        "entries": TIMING_LOG,
        "summary": _compute_timing_summary(),
    }
    if eval_results:
        output["retrieval_stats"] = _compute_retrieval_stats(eval_results)

    with open(TIMING_FILE, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print(f"\n⏱  Timing log → {TIMING_FILE}  ({len(TIMING_LOG)} entries)")
    print("\n  ── Timing summary ─────────────────────────────────────────────")
    for label, s in output["summary"].items():
        print(
            f"  {label:<50}"
            f"  n={s['n']:<4}"
            f"  mean={s['mean_s']:.4f}s"
            f"  std={s['std_s']:.4f}s"
            f"  [{s['min_s']:.4f}s … {s['max_s']:.4f}s]"
        )
    if "retrieval_stats" in output:
        print("\n  ── Retrieval stats ─────────────────────────────────────────────")
        for key, s in output["retrieval_stats"].items():
            print(
                f"  {key:<50}"
                f"  mean={s['mean_count']:.2f} ± {s['std_count']:.2f} chunks"
                f"  ({s['mean_frac']*100:.1f}% ± {s['std_frac']*100:.1f}%)"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtraDimConfig:
    """
    Parameters for the extra-dimension experiment.

    n_queries is set automatically by the runner after the GT file is loaded
    (one slot per query in the file).  Do not set it manually.

    large_value:     scalar written into the active slot of restricted chunks
                     and authorised query vectors.
    normalize_after: if True, L2-normalise the full augmented vector.
    """
    large_value:      float = DEFAULT_LARGE_VAL
    normalize_after:  bool  = True
    n_queries:        int   = 0   # set by runner

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


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Augmentation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def augment_chunk(
    base_vec:    np.ndarray,
    cfg:         ExtraDimConfig,
    query_index: int | None,
    untargeted:   bool = False,
) -> np.ndarray:
    """
    Append N extra dims to a chunk vector.
    query_index : the owning query's slot — set to large_value.
    None        : non-targeted chunk — all extra dims are zero.
    """
    extra = np.zeros(cfg.n_queries, dtype=np.float32)
    # if EXTRA_DIM_UNTARGETED_AS_LARGE_VALUE:
    #     untargeted_dimensions = np.zeros(1, dtype=np.float32) 
    #     untargeted_dimensions[0]= DEFAULT_LARGE_VAL_UNTARGETED
    #     extra = np.concatenate([base_vec.astype(np.float32), untargeted_dimensions])
    # normalize the base vec before adding the extra dims
    if NORMALIZE_BEFORE_ADDING_EXTRA_DIMS:
        base_vec = _l2_norm(base_vec)
    if untargeted:
        untargeted_dimensions = np.zeros(NUMBER_OF_EXTRA_DIMS, dtype=np.float32) 
        untargeted_dimensions[NUMBER_OF_EXTRA_DIMS:]= DEFAULT_LARGE_VAL_UNTARGETED

        extra = np.concatenate([extra.astype(np.float32), untargeted_dimensions])
        # all-zero extra dims → non-targeted chunk
        concat = np.concatenate([base_vec.astype(np.float32), extra]) 
        return concat
        # norm = np.linalg.norm(concat)
        # return concat / norm if norm > 1e-10 else concat
    
        # return _l2_norm(np.concatenate([base_vec.astype(np.float32), extra])) if cfg.normalize_after else np.concatenate([base_vec.astype(np.float32), extra])

    untargeted_dimensions = np.zeros(NUMBER_OF_EXTRA_DIMS, dtype=np.float32) 
    # untargeted_dimensions[0]= 0
    # untargeted_dimensions[1]= 0
    extra = np.concatenate([extra.astype(np.float32), untargeted_dimensions])
    if query_index is not None:
        # extra[query_index] = cfg.large_value
        extra[query_index] = DEFAULT_LARGE_VAL

    aug = np.concatenate([base_vec.astype(np.float32), extra])
    # norm = np.linalg.norm(aug)
    # return _l2_norm(aug) if cfg.normalize_after else aug
    # return aug / norm if norm > 1e-10 else aug
    return aug


def augment_query(
    base_vec:    np.ndarray,
    cfg:         ExtraDimConfig,
    query_index: int,
    authorised:  bool,
) -> np.ndarray:
    """
    Append N extra dims to a query vector.
    authorised=True  → slot[query_index] = large_value  (aligns with own chunks)
    authorised=False → all-zero extra dims               (diverges from own chunks)
    """
    extra = np.zeros(cfg.n_queries, dtype=np.float32)
    # if EXTRA_DIM_UNTARGETED_AS_LARGE_VALUE:
    #     untargeted_dimensions = np.zeros(1, dtype=np.float32) 
    #     untargeted_dimensions[0]= DEFAULT_LARGE_VAL_UNTARGETED
    #     extra = np.concatenate([base_vec.astype(np.float32), untargeted_dimensions])
    # extra.fill(DEFAULT_LARGE_VAL)
    if NORMALIZE_BEFORE_ADDING_EXTRA_DIMS:
        base_vec = _l2_norm(base_vec)
    if authorised:
        # extra[query_index] = cfg.large_value
        untargeted_dimensions = np.zeros(NUMBER_OF_EXTRA_DIMS, dtype=np.float32) 
        untargeted_dimensions[NUMBER_OF_EXTRA_DIMS:]= 0
        extra = np.concatenate([extra.astype(np.float32), untargeted_dimensions])
        extra[query_index] = DEFAULT_LARGE_VAL
    else:
        untargeted_dimensions = np.zeros(NUMBER_OF_EXTRA_DIMS, dtype=np.float32) 
        untargeted_dimensions[NUMBER_OF_EXTRA_DIMS:]= DEFAULT_LARGE_VAL_UNTARGETED
        extra = np.concatenate([extra.astype(np.float32), untargeted_dimensions])

    aug = np.concatenate([base_vec.astype(np.float32), extra])
    # norm = np.linalg.norm(aug)
    return aug
    # return _l2_norm(aug) if cfg.normalize_after else aug
    # return aug / norm if norm > 1e-10 else aug

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


def _get_aug_collection(cfg_id: str, name: str=AUGMENTED_NAME):
    return _get_client(AUG_CHROMA_BASE).get_or_create_collection(
        name=f"{name}_{DEFAULT_LARGE_VAL}", metadata={"hnsw:space": f"{DISTANCE_METRIC}"}
    )


def _get_meta_collection():
    return _get_client(META_CHROMA_BASE).get_or_create_collection(
        name="meta_access_control", metadata={"hnsw:space": f"{DISTANCE_METRIC}"}
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
        if len(embs[0]) > 0:
            return np.array(embs[0], dtype=np.float32), (docs[0] if docs else None)
    except Exception as e:
        warnings.warn(f"fetch_chunk failed for {triplet_index}|{document_id}|{phrase_seq}: {e}")
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Result data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkSimilarities:
    """
    Four cosine-similarity measurements for one chunk, mirroring the
    rotation experiment's four pairs.

    (a) auth query (augmented)   ↔ aug chunk      → both sides carry large_value → high
    (b) unauth query (plain)     ↔ aug chunk      → zeros vs large_value → lower
    (c) auth query (augmented)   ↔ plain chunk    → large_value vs zeros → collateral
    (d) plain query (no aug)     ↔ plain chunk    → baseline, no extra dims
    """
    chunk_key:                  str
    query_index:                int
    config_id:                  str
    sim_auth_query_aug_chunk:   float   # (a)
    sim_unauth_query_aug_chunk: float   # (b)
    sim_auth_query_plain_chunk: float   # (c)
    sim_plain_query_plain_chunk: float  # (d) baseline

    @property
    def security_delta(self) -> float:
        """(b) − (a): how much similarity drops for an unauthorised query. Negative = good."""
        return self.sim_unauth_query_aug_chunk - self.sim_auth_query_aug_chunk

    @property
    def collateral_delta(self) -> float:
        """(c) − (d): change on plain chunks when auth query is used. Should be ~0."""
        return self.sim_auth_query_plain_chunk - self.sim_plain_query_plain_chunk


@dataclass
class RawQueryRetrieval:
    """Output of Phase 2 — raw retrieval data, no metrics yet."""
    query_id:           str
    question:           str
    triplet_index:      str
    query_index:        int           # 0-based position in GT list
    targeted_chunk_ids: list[str]

    query_vec:          list[float]   # un-augmented base embedding

    # Per-chunk raw base vectors: chunk_key → {"base": list[float]}
    chunk_vectors: dict = field(default_factory=dict)

    # ── Method 2: augmented DB results ────────────────────────────────────────
    # auth (augmented) query → augmented DB
    aug_topk_auth_ids:   list[str] = field(default_factory=list)
    # unauth (plain/zero) query → augmented DB
    aug_topk_unauth_ids: list[str] = field(default_factory=list)

    # ── Method 3: metadata filter results ─────────────────────────────────────
    meta_topk_auth_ids:   list[str] = field(default_factory=list)   # no filter
    meta_topk_unauth_ids: list[str] = field(default_factory=list)   # filtered

    # Precise per-step timings (Method 2)
    t_embed_query_s:        float = 0.0
    t_augment_auth_s:       float = 0.0
    t_augment_unauth_s:     float = 0.0
    t_query_aug_auth_s:     float = 0.0
    t_query_aug_unauth_s:   float = 0.0
    # Precise per-step timings (Method 3)
    t_query_meta_auth_s:    float = 0.0
    t_query_meta_unauth_s:  float = 0.0

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
    config_id:         str

    chunk_similarities: list[ChunkSimilarities] = field(default_factory=list)

    # Method 2 retrieval counts
    targeted_in_auth_query_aug_db:   int = 0
    targeted_in_unauth_query_aug_db: int = 0

    # Method 3 retrieval counts
    targeted_in_auth_meta:   int = 0
    targeted_in_unauth_meta: int = 0

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Phase 1 — Build databases
# ═══════════════════════════════════════════════════════════════════════════════

@timed("build_aug_db_all_untargeted_chunks")
def _build_aug_db_all_untargeted_chunks(
    gt_records:     list[dict],
    cfg:            ExtraDimConfig,
    orig_collection,
    aug_collection,
    written_ids:    set[str],
    verbose:        bool = True,
) -> int:
    """
    Augment and upsert all untargeted chunks for all GT records (Method 2).
    This is a fallback to ensure that every chunk in the original DB is present
    in the augmented DB with some augmentation state (targeted or non-targeted).
    """
    n_ok = 0

    # get the id of all chunks
    untargeted_chunks = []
    list_of_chunk_ids = get_all_chunk_ids(gt_records)
    list_of_targeted_chunk_ids = get_list_id_targeted_chunk(gt_records)
    for chunk_id in list_of_chunk_ids:
        if chunk_id not in list_of_targeted_chunk_ids:
            untargeted_chunks.append(chunk_id)

    for chunk_id in untargeted_chunks:
        tid  = chunk_id.split("|")[0]
        did  = chunk_id.split("|")[1]
        pseq = chunk_id.split("|")[2]
        cid  = f"{cfg.config_id}_{chunk_id}"


        # if cid in written_ids:
        #     n_ok += 1
        #     continue

        base_vec, content = fetch_chunk(orig_collection, tid, did, pseq)
        if base_vec is None:
            warnings.warn(f"  ⚠  Build(aug): chunk not found {chunk_id} — skipping.")
            continue

        aug_vec = augment_chunk(base_vec, cfg, query_index=None, untargeted=True)  # query_index = None -> non-targeted
        aug_collection.upsert(
            ids        = [cid],
            embeddings = [aug_vec.tolist()],
            documents  = [content or ""],
            metadatas  = [{
                "triplet_index": tid,
                "document_id":   did,
                "phrase_seq":    pseq,
                "query_index":   None,
                "config_id":     cfg.config_id,
                "restricted":    False,
            }],
        )
        written_ids.add(cid)
        n_ok += 1

        if verbose:
            print(f"  Augmented untargeted chunk {chunk_id} → {cid}")

    return n_ok


@timed("build_aug_db_single_record")
def _build_aug_record(
    record:         dict,
    query_index:    int,
    cfg:            ExtraDimConfig,
    orig_collection,
    aug_collection,
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
            warnings.warn(f"  ⚠  Build(aug): chunk not found {chunk_id} — skipping.")
            continue

        aug_vec = augment_chunk(base_vec, cfg, query_index=query_index)
        aug_collection.upsert(
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


@timed("build_aug_db")
def build_aug_db(
    gt_records:     list[dict],
    cfg:            ExtraDimConfig,
    orig_collection,
    aug_collection,
    verbose:        bool = True,
) -> None:
    """Phase 1 (Method 2) — full pass over GT records to build the augmented DB."""
    print(f"\n{'─'*60}")
    print(f"  Phase 1a — Building augmented DB  [{cfg.config_id}]")
    print(f"  Phase 1a — Building augmented DB  [large_val: {DEFAULT_LARGE_VAL}, top_k: {DEFAULT_TOP_K}]")

    print(f"{'─'*60}")

    written_ids: set[str] = set()

    for i, record in enumerate(gt_records):
        triplet_index = record.get("id_triplets")
        if not triplet_index or not record.get("targeted_chunk"):
            continue

        if True:
            with open(DIM_REGISTRY_FILE, "r", encoding="utf-8") as fh:
                file = json.load(fh)  
                dict_key_dim = file.get("query_index_map", {})
        
            if dict_key_dim.get(str(triplet_index)) != i:
                print("error in the mapping")

        n = _build_aug_record(record, i, cfg, orig_collection, aug_collection, written_ids)
        if verbose:
            print(f"  [{i+1}/{len(gt_records)}] triplet_{triplet_index}  chunks_upserted={n}")

    print(f"\n  ✅ Augmented DB build complete")

@timed("build_meta_db_add_untargeted_chunks")
def build_meta_db_add_untargeted_chunks(
    gt_records:         list[dict],
    orig_collection,
    meta_collection,
    written_ids:    set[str],
) -> int:
    """Upsert plain vectors with restricted=True metadata (Method 3)."""
    # triplet_index = gt_records["id_triplets"]
    # stable_chunks = gt_records["targeted_chunk"]
    n_ok = 0

    untargeted_chunks = []
    list_of_chunk_ids = get_all_chunk_ids(gt_records)
    list_of_targeted_chunk_ids = get_list_id_targeted_chunk(gt_records)
    for chunk_id in list_of_chunk_ids:
        if chunk_id not in list_of_targeted_chunk_ids:
            untargeted_chunks.append(chunk_id)

    for chunk_id in untargeted_chunks:
        tid  = chunk_id.split("|")[0]
        did  = chunk_id.split("|")[1]
        pseq = chunk_id.split("|")[2]
        cid  = f"{tid}_{did}_{pseq}"

        if cid in written_ids:
            n_ok += 1
            continue

        base_vec, content = fetch_chunk(orig_collection, tid, did, pseq)
        if base_vec is None:
            warnings.warn(f"  ⚠  Build(aug): chunk not found {chunk_id} — skipping.")
            continue

        meta_collection.upsert(
            ids        = [cid],
            embeddings = [base_vec.tolist()],
            documents  = [content or ""],
            metadatas  = [{
                "triplet_index": tid,
                "document_id":   did,
                "phrase_seq":    pseq,
                "restricted":    False,
            }],
        )
        written_ids.add(cid)
        n_ok += 1

    return n_ok

@timed("build_meta_db_single_record")
def _build_meta_record(
    record:         dict,
    orig_collection,
    meta_collection,
    written_ids:    set[str],
) -> int:
    """Upsert plain vectors with restricted=True metadata (Method 3)."""
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
                "restricted":    f"{triplet_index}_True",
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
    """Phase 1 (Method 3) — full pass over GT records to build the metadata DB."""
    print(f"\n{'─'*60}")
    print(f"  Phase 1b — Building metadata DB")
    print(f"{'─'*60}")

    written_ids: set[str] = set()

    for i, record in enumerate(gt_records):
        triplet_index = record.get("id_triplets")
        if not triplet_index or not record.get("targeted_chunk"):
            continue

        n = _build_meta_record(record, orig_collection, meta_collection, written_ids)
        if verbose:
            print(f"  [{i+1}/{len(gt_records)}] triplet_{triplet_index}  chunks_upserted={n}")

    print(f"\n  ✅ Metadata DB build complete")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Phase 2 — Query phase
# ═══════════════════════════════════════════════════════════════════════════════

@timed("query_phase_single_record")
def _query_record(
    record:         dict,
    query_index:    int,
    cfg:            ExtraDimConfig,
    embedder,
    orig_collection,
    aug_collection,
    meta_collection,
    top_k:          int,
) -> RawQueryRetrieval:
    """
    Full query pipeline for one GT record — both methods, both auth states.
    Every sub-step is timed individually with perf_counter so that embed /
    augment / retrieve costs are separately attributable.
    """
    question      = record["question"]
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    query_id      = f"triplet_{triplet_index}"
    targeted_keys = set(stable_chunks)

    # Step 1 — embed raw query
    t0 = time.perf_counter()
    raw_q = np.array(
        embedder.embed_query(BGE_QUERY_PREFIX + question), dtype=np.float32
    )
    t_embed = time.perf_counter() - t0

    # Step 2 — augment query vectors
    t0 = time.perf_counter()
    auth_q   = augment_query(raw_q, cfg, query_index, authorised=True)
    t_aug_a  = time.perf_counter() - t0

    t0 = time.perf_counter()
    unauth_q = augment_query(raw_q, cfg, query_index, authorised=False)
    t_aug_u  = time.perf_counter() - t0

    # Step 3 — retrieve from augmented DB
    n_aug = max(1, aug_collection.count())

    t0 = time.perf_counter()
#     aug_res_auth = aug_collection.query(
#     query_embeddings=[auth_q.tolist()],
#     n_results=min(top_k, n_aug, len(stable_chunks)),
#     include=["metadatas"],
# )
    aug_res_auth = aug_collection.query(
    query_embeddings=[auth_q.tolist()],
    n_results=min(top_k, n_aug),
    include=["metadatas"],
    )
    
    if True: # code to check the overlap between retrieved chunks and targeted chunks -> proof that it's working
        out = aug_collection.query(
            query_embeddings=[auth_q.tolist()],
            n_results=min(top_k, n_aug, len(stable_chunks)),
            include=["metadatas", "distances", "embeddings"],
        )
        aug_res_auth_, distances, embeddings = out["metadatas"], out.get("distances", [[]])[0], out.get("embeddings", [[]])[0]
        # take out["metadatas"] instead of out["metadatas"][0] because rest of the code
        dist = distances
        list_cosine = []
        for emb in embeddings:
            a = 1 -cosine_similarity(np.array(auth_q, dtype=np.float64), np.array(emb, dtype=np.float64))
            list_cosine.append(a)
        list_chunk_returned = []
        list_stable_chunk_id = []
        for chunk in aug_res_auth_[0]:
            chunk_id = f"{chunk['triplet_index']}|{chunk['document_id']}|{chunk['phrase_seq']}"
            list_chunk_returned.append(chunk_id)
        
        for chunk in stable_chunks:
            chunk_id = f"{chunk.split('|')[0]}|{chunk[-2]}|{chunk[-1]}"
            list_stable_chunk_id.append(chunk_id)
        
        # overlap = set(list_chunk_returned) & set(list_stable_chunk_id)
        # print(f"overlap between stable chunks and retrieved chunks: {overlap}")

        sum_overlap = sum ([ 1 for chunk in list_chunk_returned if chunk in list_stable_chunk_id])
        if sum_overlap < len(stable_chunks):
            print(f"⚠  Only {sum_overlap} out of {len(stable_chunks)} targeted chunks were retrieved in the augmented DB top-{top_k}. Consider increasing top_k or checking the augmentation process.")
        # print(f"sum of overlap: {sum(sum_overlap)} out of {len(stable_chunks)} targeted chunks")



    # aug_res_auth = aug_collection.query(
    #     query_embeddings=[auth_q.tolist()],
    #     n_results=n_aug*2,
    #     include=["metadatas"],
    # )
    t_q_aug_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    aug_res_unauth = aug_collection.query(
        query_embeddings=[unauth_q.tolist()],
        n_results=min(top_k, n_aug),
        include=["metadatas"],
    )
    # aug_res_unauth = aug_collection.query(
    #     query_embeddings=[unauth_q.tolist()],
    #     n_results=n_aug*2,
    #     include=["metadatas"],
    # )
    t_q_aug_u = time.perf_counter() - t0

    def _ids(metas: list[dict]) -> list[str]:
        return [
            f"{m.get('triplet_index','?')}|{m.get('document_id','?')}|{m.get('phrase_seq','?')}"
            for m in metas
        ]

    aug_topk_auth   = _ids(aug_res_auth["metadatas"][0])
    aug_topk_unauth = _ids(aug_res_unauth["metadatas"][0])

    # Step 4 — retrieve from metadata DB
    n_meta = max(1, meta_collection.count())

    t0 = time.perf_counter()
    meta_res_auth = meta_collection.query(
        query_embeddings=[raw_q.tolist()],
        n_results=min(top_k, n_meta),
        where={"restricted": {"$eq": f"{triplet_index}_True"}},
        include=["metadatas"],
    )
    t_q_meta_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        meta_res_unauth = meta_collection.query(
            query_embeddings=[raw_q.tolist()],
            n_results=min(top_k, n_meta),
            where={"restricted": {"$ne": f"{triplet_index}_True"}},
            include=["metadatas"],
        )
    except Exception as e:
        warnings.warn(f"Metadata filter query failed: {e}. Falling back to no filter.")
        meta_res_unauth = meta_res_auth
    t_q_meta_u = time.perf_counter() - t0

    meta_topk_auth   = _ids(meta_res_auth["metadatas"][0])
    meta_topk_unauth = _ids(meta_res_unauth["metadatas"][0])

    # Collect per-chunk base vectors for Phase 3 similarity computation
    chunk_vectors: dict[str, dict] = {}
    for chunk_id in stable_chunks:
        tid  = chunk_id.split("|")[0]
        did  = chunk_id[-2]
        pseq = chunk_id[-1]
        base_vec, _ = fetch_chunk(orig_collection, tid, did, pseq)
        if base_vec is not None:
            chunk_vectors[chunk_id] = {"base": base_vec.tolist()}

    return RawQueryRetrieval(
        query_id             = query_id,
        question             = question,
        triplet_index        = triplet_index,
        query_index          = query_index,
        targeted_chunk_ids   = list(stable_chunks),
        query_vec            = raw_q.tolist(),
        chunk_vectors        = chunk_vectors,
        aug_topk_auth_ids    = aug_topk_auth,
        aug_topk_unauth_ids  = aug_topk_unauth,
        meta_topk_auth_ids   = meta_topk_auth,
        meta_topk_unauth_ids = meta_topk_unauth,
        t_embed_query_s      = round(t_embed,    6),
        t_augment_auth_s     = round(t_aug_a,    6),
        t_augment_unauth_s   = round(t_aug_u,    6),
        t_query_aug_auth_s   = round(t_q_aug_a,  6),
        t_query_aug_unauth_s = round(t_q_aug_u,  6),
        t_query_meta_auth_s  = round(t_q_meta_a, 6),
        t_query_meta_unauth_s= round(t_q_meta_u, 6),
    )


@timed("run_query_phase")
def run_query_phase(
    gt_records:     list[dict],
    cfg:            ExtraDimConfig,
    embedder,
    orig_collection,
    aug_collection,
    meta_collection,
    top_k:          int  = DEFAULT_TOP_K,
    verbose:        bool = True,
) -> list[RawQueryRetrieval]:
    """Phase 2 — full pass over GT records. Returns raw retrieval data."""
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
        
        if True:
            with open(DIM_REGISTRY_FILE, "r", encoding="utf-8") as fh:
                file = json.load(fh)  
                dict_key_dim = file.get("query_index_map", {})
            
            if dict_key_dim.get(str(triplet_index)) != i:
                print("error in the mapping")



        raw = _query_record(
            record, i, cfg, embedder,
            orig_collection, aug_collection, meta_collection, top_k,
        )
        raw_results.append(raw)

        if verbose:
            n_t = len(raw.targeted_chunk_ids)
            targeted_chunk_ids_format = [f"{c.split('|')[0]}|{c[-2]}|{c[-1]}" for c in raw.targeted_chunk_ids]
            # print(
            #     f"  [{i+1}/{len(gt_records)}] {raw.query_id}"
            #     f"  t_embed={raw.t_embed_query_s:.4f}s"
            #     f"  t_aug_auth={raw.t_augment_auth_s:.6f}s"
            #     f"  t_q_aug_auth={raw.t_query_aug_auth_s:.4f}s"
            #     f"  t_q_aug_unauth={raw.t_query_aug_unauth_s:.4f}s"
            #     f"  t_q_meta_auth={raw.t_query_meta_auth_s:.4f}s"
            #     f"  t_q_meta_unauth={raw.t_query_meta_unauth_s:.4f}s"
            #     f"  targeted_aug_auth={sum(1 for c in raw.aug_topk_auth_ids if c in set(raw.targeted_chunk_ids))}/{n_t}"
            #     f"  targeted_aug_unauth={sum(1 for c in raw.aug_topk_unauth_ids if c in set(raw.targeted_chunk_ids))}/{n_t}"
            # )
            print(
                f"  [{i+1}/{len(gt_records)}] {raw.query_id}"
                f"  t_embed={raw.t_embed_query_s:.4f}s"
                f"  t_aug_auth={raw.t_augment_auth_s:.6f}s"
                f"  t_q_aug_auth={raw.t_query_aug_auth_s:.4f}s"
                f"  t_q_aug_unauth={raw.t_query_aug_unauth_s:.4f}s"
                f"  t_q_meta_auth={raw.t_query_meta_auth_s:.4f}s"
                f"  t_q_meta_unauth={raw.t_query_meta_unauth_s:.4f}s"
                f"  targeted_aug_auth={sum(1 for c in raw.aug_topk_auth_ids if c in set(targeted_chunk_ids_format))}/{n_t}"
                f"  targeted_aug_unauth={sum(1 for c in raw.aug_topk_unauth_ids if c in set(targeted_chunk_ids_format))}/{n_t}"
            )

    return raw_results


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Phase 3 — Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@timed("evaluate_single_record")
def _evaluate_record(
    raw: RawQueryRetrieval,
    cfg: ExtraDimConfig,
) -> QueryEvalResult:
    """Compute all metrics for one query from its raw retrieval data."""
    raw_q      = np.array(raw.query_vec, dtype=np.float32)
    targeted   = set(raw.targeted_chunk_ids)

    auth_q_aug   = augment_query(raw_q, cfg, raw.query_index, authorised=True)
    unauth_q_aug = augment_query(raw_q, cfg, raw.query_index, authorised=False)

    chunk_sims: list[ChunkSimilarities] = []
    for chunk_id, vecs in raw.chunk_vectors.items():
        base_vec   = np.array(vecs["base"], dtype=np.float32)
        aug_chunk  = augment_chunk(base_vec, cfg, query_index=raw.query_index)
        plain_chunk = augment_chunk(base_vec, cfg, query_index=None)  # all-zero extra dims

        chunk_sims.append(ChunkSimilarities(
            chunk_key                   = chunk_id,
            query_index                 = raw.query_index,
            config_id                   = cfg.config_id,
            sim_auth_query_aug_chunk    = cosine_similarity(auth_q_aug,   aug_chunk),    # (a)
            sim_unauth_query_aug_chunk  = cosine_similarity(unauth_q_aug, aug_chunk),    # (b)
            sim_auth_query_plain_chunk  = cosine_similarity(auth_q_aug,   plain_chunk),  # (c)
            sim_plain_query_plain_chunk = cosine_similarity(raw_q,        base_vec),     # (d)
        ))
    target_chunk_id = [f'{target.split("|")[0]}|{target[-2]}|{target[-1]}' for target in targeted]
    
    targeted_in_auth_query_aug_db = sum(1 for c in raw.aug_topk_auth_ids   if c in target_chunk_id)
    targeted_in_unauth_query_aug_db  = sum(1 for c in raw.aug_topk_unauth_ids if c in target_chunk_id)
    value = 0
    for e in raw.meta_topk_auth_ids:
        for c in target_chunk_id:
            if c == e:
                value += 1
    targeted_in_auth_meta            = sum(1 for c in raw.meta_topk_auth_ids   if c in target_chunk_id)
    targeted_in_unauth_meta          = sum(1 for c in raw.meta_topk_unauth_ids if c in target_chunk_id)

    return QueryEvalResult(
        query_id                         = raw.query_id,
        question                         = raw.question,
        triplet_index                    = raw.triplet_index,
        query_index                      = raw.query_index,
        n_targeted_chunks                = len(raw.targeted_chunk_ids),
        config_id                        = cfg.config_id,
        chunk_similarities               = chunk_sims,
        targeted_in_auth_query_aug_db    = sum(1 for c in raw.aug_topk_auth_ids   if c in target_chunk_id),
        targeted_in_unauth_query_aug_db  = sum(1 for c in raw.aug_topk_unauth_ids if c in target_chunk_id),
        targeted_in_auth_meta            = sum(1 for c in raw.meta_topk_auth_ids   if c in target_chunk_id),
        targeted_in_unauth_meta          = sum(1 for c in raw.meta_topk_unauth_ids if c in target_chunk_id),
    )
        # targeted_in_auth_query_aug_db    = sum(1 for c in raw.aug_topk_auth_ids   if c in targeted),
        # targeted_in_unauth_query_aug_db  = sum(1 for c in raw.aug_topk_unauth_ids if c in targeted),
        # targeted_in_auth_meta            = sum(1 for c in raw.meta_topk_auth_ids   if c in targeted),
        # targeted_in_unauth_meta          = sum(1 for c in raw.meta_topk_unauth_ids if c in targeted),


@timed("evaluate_results")
def evaluate_results(
    raw_results: list[RawQueryRetrieval],
    cfg:         ExtraDimConfig,
    verbose:     bool = True,
) -> list[QueryEvalResult]:
    """Phase 3 — compute metrics, write results/dim_results.json."""
    print(f"\n{'─'*60}")
    print(f"  Phase 3 — Evaluation ({len(raw_results)} queries)")
    print(f"{'─'*60}")

    eval_results: list[QueryEvalResult] = []

    for raw in raw_results:
        ev = _evaluate_record(raw, cfg)
        eval_results.append(ev)

        if verbose:
            avg_sec_delta = (
                sum(c.security_delta for c in ev.chunk_similarities)
                / max(len(ev.chunk_similarities), 1)
            )
            print(
                f"  {ev.query_id}"
                f"  avg_security_delta={avg_sec_delta:+.4f}"
                f"  targeted_aug_auth={ev.targeted_in_auth_query_aug_db}/{ev.n_targeted_chunks}"
                f"  targeted_aug_unauth={ev.targeted_in_unauth_query_aug_db}/{ev.n_targeted_chunks}"
                f"  targeted_meta_auth={ev.targeted_in_auth_meta}/{ev.n_targeted_chunks}"
                f"  targeted_meta_unauth={ev.targeted_in_unauth_meta}/{ev.n_targeted_chunks}"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DIM_RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in eval_results], fh, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Results → {DIM_RESULTS_FILE}")

    return eval_results

############## TESTING PART ############

def test_query_index_map_dim_database(gt_records, aug_collection):
    with open(DIM_REGISTRY_FILE, "r", encoding="utf-8") as fh:
        file = json.load(fh)  
        dict_key_dim = file.get("query_index_map", {})
    
    for record in gt_records:
        triplet_index = record.get("id_triplets")
        for target_chunk in record.get("targeted_chunk", []):
            tid  = target_chunk.split("|")[0]
            did  = target_chunk[-2]
            pseq = target_chunk[-1]
            # cid  = f"{cfg.config_id}_{triplet_index}_{did}_{pseq}"

            result = aug_collection.get(
                where={
                    "$and": [
                        {"triplet_index": {"$eq": tid}},
                        {"document_id":   {"$eq": did}},
                        {"phrase_seq":    {"$eq": pseq}},
                    ]
                },
                include=["embeddings"],
            )

            # check if the value of the 1024 + i dim is different from 0,

            results = result.get("embeddings", [])
            results = results[0] 
            id_from_triplet = dict_key_dim.get(str(triplet_index))
            modified_dim_index = 1024 + int(id_from_triplet)
            print(f"checking chunk with tid={tid}, did={did}, pseq={pseq}, modified_dim_index={modified_dim_index},\n value modified={results[modified_dim_index]}")
            if(results[modified_dim_index] == 0):
                print("error in the encoding of the chunk")

            
    #         if cid not in aug_collection.get(ids=[cid])["ids"]:
    #             print(f"Error: chunk {cid} not found in augmented collection.")
    #     if not triplet_index or not record.get("targeted_chunk"):
    #         continue
    
    # # Check if the mapping is correct for a few sample triplet indices
    # sample_triplet_indices = list(dict_key_dim.keys())[:5]  # take first 5 for testing
    # for triplet_index in sample_triplet_indices:
    #     query_index = dict_key_dim[triplet_index]
    #     print(f"Triplet index {triplet_index} is mapped to query index {query_index}")




# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

@timed("run_experiment")
def run_experiment(
    gt_path:  str   = str(GT_FILE),
    cfg:      ExtraDimConfig | None = None,
    top_k:    int   = DEFAULT_TOP_K,
    verbose:  bool  = True,
) -> list[QueryEvalResult]:
    """Runs all three phases for both methods and persists results + timing."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(gt_path, encoding="utf-8") as fh:
        gt_records: list[dict] = json.load(fh)

    n_queries = len(gt_records)

    if cfg is None:
        cfg = ExtraDimConfig()
    cfg.n_queries = n_queries   # always derived from GT file

    print(f"\n{'═'*60}")
    print(f"  Extra-Dim + Metadata Access-Control Experiment")
    print(f"  GT file    : {gt_path}  ({n_queries} records)")
    print(f"  Extra dims : {n_queries}  (one per query)")
    print(f"  Config     : {cfg.config_id}")
    print(f"  Top-K      : {top_k}")
    print(f"{'═'*60}")

    # Init model
    t0 = time.perf_counter()
    embedder = get_embedding_model()
    TIMING_LOG.append({
        "label":      "init_embedding_model",
        "start_iso":  datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.perf_counter() - t0, 6),
        "args_repr":  "",
    })

    orig_coll = _get_original_collection(ORIGINAL_CHROMA, ORIGINAL_COLLECTION)
    aug_coll  = _get_aug_collection(cfg.config_id)
    meta_coll = _get_meta_collection()



    # Persist registry
    with open(DIM_REGISTRY_FILE, "w", encoding="utf-8") as fh:
        json.dump({
            "n_queries":  n_queries,
            "gt_path":    gt_path,
            "config":     cfg.to_dict(),
            "query_index_map": {
                record["id_triplets"]: i
                for i, record in enumerate(gt_records)
                if record.get("id_triplets")
            },
        }, fh, indent=2)
    print(f"  Registry → {DIM_REGISTRY_FILE}")

    # Phase 1
    build_aug_db(gt_records, cfg, orig_coll, aug_coll, verbose=verbose)
    build_meta_db(gt_records, orig_coll, meta_coll, verbose=verbose)
    
    # test_query_index_map_dim_database(gt_records, aug_coll)

    # build_aug_db(gt_records, cfg, orig_coll, aug_coll, verbose=False)
    # build_meta_db(gt_records, orig_coll, meta_coll, verbose=False)

    # add the untargeted chunks to the augmented DB (Method 2)
    written_ids: set[str] = set()
    _build_aug_db_all_untargeted_chunks(gt_records, cfg, orig_coll, aug_coll, written_ids)
    # written_ids: set[str] = set()
    # build_meta_db_add_untargeted_chunks(gt_records, orig_coll, meta_coll, written_ids)

    # Phase 2
    raw_results = run_query_phase(
        gt_records, cfg, embedder, orig_coll, aug_coll, meta_coll,
        top_k=top_k, verbose=verbose,
    )
    # raw_results = run_query_phase(
    #     gt_records, cfg, embedder, orig_coll, aug_coll, meta_coll,
    #     top_k=top_k, verbose=False,
    # )

    # Phase 3
    eval_results = evaluate_results(raw_results, cfg, verbose=verbose)

    save_timing_log(eval_results=eval_results)
    return eval_results




# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Access-control embedding experiment (extra-dim + metadata filter)"
    )
    parser.add_argument("--gt",              default=str(GT_FILE))
    parser.add_argument("--top-k",           type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--large-value",     type=float, default=DEFAULT_LARGE_VAL,
                        help="Scalar written into the active extra-dim slot (default %(default)s)")
    parser.add_argument("--normalize-after", default=True,  action="store_true",
                        help="L2-normalise the augmented vector after appending extra dims")
    parser.add_argument("--quiet", "-q",     action="store_true")
    args = parser.parse_args()

    user_cfg = ExtraDimConfig(
        large_value     = args.large_value,
        normalize_after = args.normalize_after,
    )

    run_experiment(
        gt_path = args.gt,
        cfg     = user_cfg,
        top_k   = args.top_k,
        verbose = not args.quiet,
    )
