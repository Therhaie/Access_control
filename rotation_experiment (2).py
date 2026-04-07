"""
rotation_experiment.py
======================
Rotation-based access-control experiment — three clearly separated phases.

Phase 1 — build_rotated_db
    Full pass over GT records. Assigns one rotation matrix per triplet_index,
    embeds every targeted chunk, applies its rotation, upserts into the rotated
    Chroma collection.  No queries executed here.

Phase 2 — run_query_phase
    Full pass over GT records.  For each query:
      • embed the query vector                         [timed precisely]
      • apply the group rotation                       [timed precisely]
      • query the original Chroma collection           [timed precisely]
      • query the rotated Chroma collection            [timed precisely]
    Returns raw retrieval data (RawQueryRetrieval).  No metric computation.

Phase 3 — evaluate_results
    Takes raw retrieval data from Phase 2, computes all metrics:
      • four cosine-similarity measurements per chunk
      • top-K overlap between original and rotated collections
      • targeted_in_rot_query_rot_db   — how many targeted chunks the
            *rotated* query retrieves from the rotated DB
      • targeted_in_unrot_query_rot_db — same with the *unrotated* query
            (sanity / baseline comparison)
    Writes results/rotation_results.json.

Timing log  →  logs/rotation_timing.json
    Every @timed function appends one entry per call.
    At the end a "summary" section is appended with mean ± std per label,
    plus aggregate statistics for targeted_in_rot_query_rot_db and
    targeted_in_unrot_query_rot_db across all evaluated queries.

Usage
-----
python rotation_experiment.py
python rotation_experiment.py --gt documents_RAGBench/merged_id_triplets_with_metadata2.json
python rotation_experiment.py --top-k 15
"""

from __future__ import annotations

import argparse
import functools
import hashlib
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
from scipy.stats import ortho_group

from config import COLLECTION
from ingestion_pipeline import get_embedding_model
from query_pipeline import BGE_QUERY_PREFIX

from plot_PCA import get_all_chunk_ids, get_list_id_targeted_chunk

# ── Paths ──────────────────────────────────────────────────────────────────────
LOGS_DIR            = Path("logs")
RESULTS_DIR         = Path("results")
GT_FILE             = Path("documents_RAGBench/merged_id_triplets_with_metadata2.json")
ROTATION_RESULTS    = RESULTS_DIR / "rotation_results.json"
ROTATION_REGISTRY_F = RESULTS_DIR / "rotation_registry.json"
TIMING_FILE         = LOGS_DIR    / "rotation_timing.json"

ORIGINAL_CHROMA     = os.path.join(os.getcwd(), "./chroma_db")
ROTATED_CHROMA      = os.path.join(os.getcwd(), "./chroma_rotated_db_log")
ORIGINAL_COLLECTION = COLLECTION
ROTATED_COLLECTION  = "rotated_experiment"

DEFAULT_TOP_K = 20


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Timing infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

TIMING_LOG: list[dict] = []


def timed(label: str):
    """
    Decorator factory — records wall-clock duration of every call.

    Each call appends to module-level TIMING_LOG:
        {
            "label":      "build_rotated_db_single_record",
            "start_iso":  "2025-…",
            "duration_s": 0.123456,
            "args_repr":  "…"
        }
    Call save_timing_log() at the end to flush to disk with a summary.
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
    """Group log entries by label, return mean ± std / min / max / n per label."""
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
    Compute mean / std of targeted_in_rot_query_rot_db and
    targeted_in_unrot_query_rot_db across all evaluated queries,
    both as raw counts and as fractions of n_targeted_chunks.
    """
    rot_counts   = np.array([r.targeted_in_rot_query_rot_db   for r in eval_results], dtype=float)
    unrot_counts = np.array([r.targeted_in_unrot_query_rot_db for r in eval_results], dtype=float)
    totals       = np.array([r.n_targeted_chunks               for r in eval_results], dtype=float)
    fracs_rot    = np.where(totals > 0, rot_counts   / totals, 0.0)
    fracs_unrot  = np.where(totals > 0, unrot_counts / totals, 0.0)

    def _stats(counts: np.ndarray, fracs: np.ndarray) -> dict:
        return {
            "mean_count": round(float(counts.mean()), 4),
            "std_count":  round(float(counts.std()),  4),
            "min_count":  round(float(counts.min()),  4),
            "max_count":  round(float(counts.max()),  4),
            "mean_frac":  round(float(fracs.mean()),  4),
            "std_frac":   round(float(fracs.std()),   4),
            "n_queries":  int(len(counts)),
        }

    return {
        "targeted_in_rot_query_rot_db":   _stats(rot_counts,   fracs_rot),
        "targeted_in_unrot_query_rot_db": _stats(unrot_counts, fracs_unrot),
    }


def save_timing_log(
    eval_results: "list[QueryEvalResult] | None" = None,
) -> None:
    """
    Flush TIMING_LOG to logs/rotation_timing.json.

    Structure written:
        {
          "entries":         [ …one dict per @timed call… ],
          "summary":         { label: {n, mean_s, std_s, min_s, max_s} },
          "retrieval_stats": { metric: {mean_count, std_count, …} }   // if eval_results given
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
            f"  {label:<48}"
            f"  n={s['n']:<4}"
            f"  mean={s['mean_s']:.4f}s"
            f"  std={s['std_s']:.4f}s"
            f"  [{s['min_s']:.4f}s … {s['max_s']:.4f}s]"
        )
    if "retrieval_stats" in output:
        print("\n  ── Retrieval stats ─────────────────────────────────────────────")
        for key, s in output["retrieval_stats"].items():
            print(
                f"  {key:<48}"
                f"  mean={s['mean_count']:.2f} ± {s['std_count']:.2f} chunks"
                f"  ({s['mean_frac']*100:.1f}% ± {s['std_frac']*100:.1f}%)"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Rotation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _seed_from_key(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2 ** 31)


def make_rotation_matrix(dim: int, seed: int) -> np.ndarray:
    return ortho_group.rvs(dim=dim, random_state=seed).astype(np.float32)


def apply_rotation(vec: np.ndarray, R: np.ndarray) -> np.ndarray:
    if vec.ndim == 1:
        return R @ vec
    return (R @ vec.T).T


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  RotationRegistry
# ═══════════════════════════════════════════════════════════════════════════════

class RotationRegistry:
    """
    Maps each triplet_index to exactly one orthogonal rotation matrix.
    Once assigned the rotation never changes.
    """

    def __init__(self, dim: int):
        self.dim    = dim
        self._store: dict[str, dict] = {}

    def get_or_create(self, triplet_index: str) -> np.ndarray:
        if triplet_index not in self._store:
            seed = _seed_from_key(triplet_index)
            self._store[triplet_index] = {
                "seed":   seed,
                "matrix": make_rotation_matrix(self.dim, seed),
            }
        return self._store[triplet_index]["matrix"]

    def get(self, triplet_index: str) -> np.ndarray | None:
        entry = self._store.get(triplet_index)
        return entry["matrix"] if entry else None

    def seed_of(self, triplet_index: str) -> int | None:
        entry = self._store.get(triplet_index)
        return entry["seed"] if entry else None

    def all_keys(self) -> list[str]:
        return list(self._store.keys())

    def to_serialisable(self) -> dict:
        return {k: v["seed"] for k, v in self._store.items()}

    @classmethod
    def from_serialisable(cls, data: dict, dim: int) -> "RotationRegistry":
        reg = cls(dim=dim)
        for key, seed in data.items():
            reg._store[key] = {
                "seed":   seed,
                "matrix": make_rotation_matrix(dim, seed),
            }
        return reg


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


def _get_or_create_rotated_collection(path: str, name: str):
    return _get_client(path).get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
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
    chunk_key:                 str
    group_key:                 str
    rotation_seed:             int
    # (a) no rotation on either side — pure baseline
    sim_orig_query_orig_chunk: float
    # (b) both rotated — should equal (a) for an orthogonal matrix
    sim_rot_query_rot_chunk:   float
    # (c) only the chunk is rotated — rotation breaks alignment
    sim_orig_query_rot_chunk:  float
    # (d) only the query is rotated — mirror of (c)
    sim_rot_query_orig_chunk:  float

    @property
    def delta_c(self) -> float:
        return self.sim_orig_query_rot_chunk - self.sim_orig_query_orig_chunk

    @property
    def delta_d(self) -> float:
        return self.sim_rot_query_orig_chunk - self.sim_orig_query_orig_chunk


@dataclass
class RawQueryRetrieval:
    """Output of Phase 2 — raw retrieval data, no metrics yet."""
    query_id:           str
    question:           str
    triplet_index:      str
    targeted_chunk_ids: list[str]
    rotation_seed:      int

    query_vec:          list[float]   # unrotated
    rot_query_vec:      list[float]   # rotated

    # Top-K IDs from original collection  (unrotated query)
    original_topk_ids:            list[str] = field(default_factory=list)
    # Top-K IDs from rotated collection   (rotated query)   — main experiment
    rotated_topk_ids:             list[str] = field(default_factory=list)
    # Top-K IDs from rotated collection   (UNrotated query) — sanity baseline
    rotated_topk_ids_unrot_query: list[str] = field(default_factory=list)

    # Per-chunk raw vectors: chunk_key → {"orig": list[float], "rot": list[float]}
    chunk_vectors: dict = field(default_factory=dict)

    # Precise per-step timings (measured inside _query_record, not by @timed)
    t_embed_query_s:    float = 0.0
    t_apply_rotation_s: float = 0.0
    t_query_original_s: float = 0.0
    t_query_rotated_s:  float = 0.0

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class QueryEvalResult:
    """Output of Phase 3 — fully evaluated metrics for one query."""
    query_id:          str
    question:          str
    triplet_index:     str
    n_targeted_chunks: int
    rotation_seed:     int

    chunk_similarities: list[ChunkSimilarities] = field(default_factory=list)

    original_topk_ids:  list[str] = field(default_factory=list)
    rotated_topk_ids:   list[str] = field(default_factory=list)
    overlap_count:      int       = 0
    overlap_fraction:   float     = 0.0

    # (a) rotated query   → rotated DB   [main experiment]
    targeted_in_rot_query_rot_db:   int = 0
    # (b) unrotated query → rotated DB   [sanity baseline]
    targeted_in_unrot_query_rot_db: int = 0

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Phase 1 — Build the rotated database
# ═══════════════════════════════════════════════════════════════════════════════

# @timed("build_rotated_db_all_untargeted_chunks")
# def _build_rotated_db_all_untargeted_chunks(
#     gt_records:     list[dict],
#     cfg:            ExtraDimConfig,
#     orig_collection,
#     aug_collection,
#     written_ids:    set[str],
#     verbose:        bool = True,
# ) -> int:
#     """
#     Augment and upsert all untargeted chunks for all GT records (Method 2).
#     This is a fallback to ensure that every chunk in the original DB is present
#     in the augmented DB with some augmentation state (targeted or non-targeted).
#     """
#     n_ok = 0

#     # get the id of all chunks
#     untargeted_chunks = []
#     list_of_chunk_ids = get_all_chunk_ids(gt_records)
#     list_of_targeted_chunk_ids = get_list_id_targeted_chunk(gt_records)
#     for chunk_id in list_of_chunk_ids:
#         if chunk_id not in list_of_targeted_chunk_ids:
#             untargeted_chunks.append(chunk_id)

#     for chunk_id in untargeted_chunks:
#         tid  = chunk_id.split("|")[0]
#         did  = chunk_id.split("|")[1]
#         pseq = chunk_id.split("|")[2]
#         cid  = f"{cfg.config_id}_{chunk_id}"


#         # if cid in written_ids:
#         #     n_ok += 1
#         #     continue

#         base_vec, content = fetch_chunk(orig_collection, tid, did, pseq)
#         if base_vec is None:
#             warnings.warn(f"  ⚠  Build(aug): chunk not found {chunk_id} — skipping.")
#             continue

#         aug_vec = augment_chunk(base_vec, cfg, query_index=None, untargeted=True)  # query_index = None -> non-targeted
#         aug_collection.upsert(
#             ids        = [cid],
#             embeddings = [aug_vec.tolist()],
#             documents  = [content or ""],
#             metadatas  = [{
#                 "triplet_index": tid,
#                 "document_id":   did,
#                 "phrase_seq":    pseq,
#                 "query_index":   None,
#                 "config_id":     cfg.config_id,
#                 "restricted":    False,
#             }],
#         )
#         written_ids.add(cid)
#         n_ok += 1

#         if verbose:
#             print(f"  Augmented untargeted chunk {chunk_id} → {cid}")

#     return n_ok

@timed("build_rotated_db_untargeted_chunks")
def _build_rotated_db_untargeted_chunks(
    gt_records:         list[dict],
    registry:       RotationRegistry,
    orig_collection,
    rot_collection,
    written_ids:    set[str],
    verbose:        bool = True,
) -> int:
    """Rotate and upsert all targeted chunks for one GT record. Returns n upserted."""
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

        orig_vec, content = fetch_chunk(orig_collection, tid, did, pseq)

        rot_collection.upsert(
            ids        = [cid],
            embeddings = [orig_vec.tolist()],
            documents  = [content or ""],
            metadatas  = [{
                "triplet_index": tid,
                "document_id":   did,
                "phrase_seq":    pseq,
                "rotation_seed": None,
            }],
        )
        written_ids.add(cid)
        n_ok += 1
        if verbose:
            print(f"  Augmented untargeted chunk {chunk_id} → {cid}")
    return n_ok


@timed("build_rotated_db_single_record")
def _build_record(
    record:         dict,
    registry:       RotationRegistry,
    orig_collection,
    rot_collection,
    written_ids:    set[str],
) -> int:
    """Rotate and upsert all targeted chunks for one GT record. Returns n upserted."""
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    R             = registry.get_or_create(triplet_index)
    n_ok          = 0

    for chunk_id in stable_chunks:
        tid  = chunk_id.split("|")[0]
        did  = chunk_id[-2]
        pseq = chunk_id[-1]
        cid  = f"{triplet_index}_{did}_{pseq}"

        if cid in written_ids:
            n_ok += 1
            continue

        orig_vec, content = fetch_chunk(orig_collection, tid, did, pseq)
        if orig_vec is None:
            warnings.warn(f"  ⚠  Build: chunk not found {chunk_id} — skipping.")
            continue

        rot_vec = apply_rotation(orig_vec, R)
        rot_collection.upsert(
            ids        = [cid],
            embeddings = [rot_vec.tolist()],
            documents  = [content or ""],
            metadatas  = [{
                "triplet_index": triplet_index,
                "document_id":   did,
                "phrase_seq":    pseq,
                "rotation_seed": int(registry.seed_of(triplet_index)),
            }],
        )
        written_ids.add(cid)
        n_ok += 1

    return n_ok


@timed("build_rotated_db")
def build_rotated_db(
    gt_records:     list[dict],
    registry:       RotationRegistry,
    orig_collection,
    rot_collection,
    verbose:        bool = True,
) -> RotationRegistry:
    """Phase 1 — full pass over GT records. Returns the populated registry."""
    print(f"\n{'─'*60}")
    print(f"  Phase 1 — Building rotated database ({len(gt_records)} records)")
    print(f"{'─'*60}")

    written_ids:  set[str] = set()
    total_chunks: int      = 0

    for i, record in enumerate(gt_records, 1):
        triplet_index = record.get("id_triplets")
        if not triplet_index or not record.get("targeted_chunk"):
            if verbose:
                print(f"  [{i}] ⚠  No targeted chunks — skipping.")
            continue

        n = _build_record(record, registry, orig_collection, rot_collection, written_ids)
        total_chunks += n
        if verbose:
            print(f"  [{i}/{len(gt_records)}] triplet_{triplet_index}  chunks_upserted={n}")

    print(f"\n  ✅ Build complete — {total_chunks} chunk vectors in rotated DB")
    return registry


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Phase 2 — Query phase
# ═══════════════════════════════════════════════════════════════════════════════

@timed("query_phase_single_record")
def _query_record(
    record:         dict,
    registry:       RotationRegistry,
    embedder,
    orig_collection,
    rot_collection,
    top_k:          int,
) -> RawQueryRetrieval:
    """
    Embed, rotate, and retrieve for one GT record.
    Each step is timed precisely with perf_counter so that embed /
    rotate / retrieve costs are individually attributable.
    """
    question      = record["question"]
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    query_id      = f"triplet_{triplet_index}"
    R             = registry.get(triplet_index)
    seed          = registry.seed_of(triplet_index)

    if R is None:
        raise RuntimeError(
            f"No rotation matrix for {triplet_index}. Run build_rotated_db first."
        )

    # Step 1 — embed raw query
    t0 = time.perf_counter()
    query_vec = np.array(
        embedder.embed_query(BGE_QUERY_PREFIX + question), dtype=np.float32
    )
    t_embed = time.perf_counter() - t0

    # Step 2 — apply rotation
    t0 = time.perf_counter()
    rot_query_vec = apply_rotation(query_vec, R)
    t_rotate = time.perf_counter() - t0

    # Step 3 — retrieve from original DB (unrotated query)
    t0 = time.perf_counter()
    orig_res = orig_collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k, orig_collection.count()),
        include=["metadatas"],
    )
    t_query_orig = time.perf_counter() - t0
    original_topk_ids = [
        f"{m.get('triplet_index','?')}|{m.get('document_id','?')}|{m.get('phrase_seq','?')}"
        for m in orig_res["metadatas"][0]
    ]

    # Step 4 — retrieve from rotated DB (rotated query) — main experiment
    n_rot = max(1, rot_collection.count())
    t0 = time.perf_counter()
    rot_res = rot_collection.query(
        query_embeddings=[rot_query_vec.tolist()],
        n_results=min(top_k, n_rot),
        include=["metadatas"],
    )
    t_query_rot = time.perf_counter() - t0
    rotated_topk_ids = [
        f"{m.get('triplet_index','?')}|{m.get('document_id','?')}|{m.get('phrase_seq','?')}"
        for m in rot_res["metadatas"][0]
    ]

    # Step 5 — sanity: unrotated query vs rotated DB
    rot_res_unrot = rot_collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k, n_rot),
        include=["metadatas"],
    )
    rotated_topk_ids_unrot = [
        f"{m.get('triplet_index','?')}|{m.get('document_id','?')}|{m.get('phrase_seq','?')}"
        for m in rot_res_unrot["metadatas"][0]
    ]

    # Collect per-chunk vectors for Phase 3 similarity computation
    chunk_vectors: dict[str, dict] = {}
    for chunk_id in stable_chunks:
        tid  = chunk_id.split("|")[0]
        did  = chunk_id[-2]
        pseq = chunk_id[-1]
        orig_vec, _ = fetch_chunk(orig_collection, tid, did, pseq)
        if orig_vec is None:
            continue
        chunk_vectors[chunk_id] = {
            "orig": orig_vec.tolist(),
            "rot":  apply_rotation(orig_vec, R).tolist(),
        }

    return RawQueryRetrieval(
        query_id                      = query_id,
        question                      = question,
        triplet_index                 = triplet_index,
        targeted_chunk_ids            = list(stable_chunks),
        rotation_seed                 = int(seed),
        query_vec                     = query_vec.tolist(),
        rot_query_vec                 = rot_query_vec.tolist(),
        original_topk_ids             = original_topk_ids,
        rotated_topk_ids              = rotated_topk_ids,
        rotated_topk_ids_unrot_query  = rotated_topk_ids_unrot,
        chunk_vectors                 = chunk_vectors,
        t_embed_query_s               = round(t_embed,      6),
        t_apply_rotation_s            = round(t_rotate,     6),
        t_query_original_s            = round(t_query_orig, 6),
        t_query_rotated_s             = round(t_query_rot,  6),
    )


@timed("run_query_phase")
def run_query_phase(
    gt_records:     list[dict],
    registry:       RotationRegistry,
    embedder,
    orig_collection,
    rot_collection,
    top_k:          int  = DEFAULT_TOP_K,
    verbose:        bool = True,
) -> list[RawQueryRetrieval]:
    """Phase 2 — full pass over GT records. Returns raw retrieval data."""
    print(f"\n{'─'*60}")
    print(f"  Phase 2 — Query phase ({len(gt_records)} records, top-K={top_k})")
    print(f"{'─'*60}")

    raw_results: list[RawQueryRetrieval] = []

    for i, record in enumerate(gt_records, 1):
        triplet_index = record.get("id_triplets")
        if not triplet_index or not record.get("targeted_chunk"):
            if verbose:
                print(f"  [{i}] ⚠  No targeted chunks — skipping.")
            continue

        raw = _query_record(
            record, registry, embedder, orig_collection, rot_collection, top_k
        )
        raw_results.append(raw)

        if verbose:
            print(
                f"  [{i}/{len(gt_records)}] {raw.query_id}"
                f"  t_embed={raw.t_embed_query_s:.4f}s"
                f"  t_rotate={raw.t_apply_rotation_s:.6f}s"
                f"  t_query_orig={raw.t_query_original_s:.4f}s"
                f"  t_query_rot={raw.t_query_rotated_s:.4f}s"
            )

    return raw_results


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Phase 3 — Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@timed("evaluate_single_record")
def _evaluate_record(raw: RawQueryRetrieval) -> QueryEvalResult:
    """Compute all metrics for one query from its raw retrieval data."""
    query_vec     = np.array(raw.query_vec,     dtype=np.float32)
    rot_query_vec = np.array(raw.rot_query_vec, dtype=np.float32)
    targeted      = set(raw.targeted_chunk_ids)

    chunk_sims: list[ChunkSimilarities] = []
    for chunk_id, vecs in raw.chunk_vectors.items():
        orig_vec = np.array(vecs["orig"], dtype=np.float32)
        rot_vec  = np.array(vecs["rot"],  dtype=np.float32)
        chunk_sims.append(ChunkSimilarities(
            chunk_key                  = chunk_id,
            group_key                  = raw.triplet_index,
            rotation_seed              = raw.rotation_seed,
            sim_orig_query_orig_chunk  = cosine_similarity(query_vec,     orig_vec),
            sim_rot_query_rot_chunk    = cosine_similarity(rot_query_vec, rot_vec),
            sim_orig_query_rot_chunk   = cosine_similarity(query_vec,     rot_vec),
            sim_rot_query_orig_chunk   = cosine_similarity(rot_query_vec, orig_vec),
        ))

    overlap_set = set(raw.original_topk_ids) & set(raw.rotated_topk_ids)
    target_chunk_id = [f'{target.split("|")[0]}|{target[-2]}|{target[-1]}' for target in targeted]

    return QueryEvalResult(
        query_id                        = raw.query_id,
        question                        = raw.question,
        triplet_index                   = raw.triplet_index,
        n_targeted_chunks               = len(raw.targeted_chunk_ids),
        rotation_seed                   = raw.rotation_seed,
        chunk_similarities              = chunk_sims,
        original_topk_ids               = raw.original_topk_ids,
        rotated_topk_ids                = raw.rotated_topk_ids,
        overlap_count                   = len(overlap_set),
        overlap_fraction                = len(overlap_set) / max(len(raw.original_topk_ids), 1),
        targeted_in_rot_query_rot_db    = sum(1 for cid in raw.rotated_topk_ids             if cid in target_chunk_id),
        targeted_in_unrot_query_rot_db  = sum(1 for cid in raw.rotated_topk_ids_unrot_query if cid in target_chunk_id),
    )


@timed("evaluate_results")
def evaluate_results(
    raw_results: list[RawQueryRetrieval],
    verbose:     bool = True,
) -> list[QueryEvalResult]:
    """Phase 3 — compute metrics, write results/rotation_results.json."""
    print(f"\n{'─'*60}")
    print(f"  Phase 3 — Evaluation ({len(raw_results)} queries)")
    print(f"{'─'*60}")

    eval_results: list[QueryEvalResult] = []

    for raw in raw_results:
        ev = _evaluate_record(raw)
        eval_results.append(ev)

        if verbose:
            n        = max(len(ev.chunk_similarities), 1)
            avg_base = sum(c.sim_orig_query_orig_chunk for c in ev.chunk_similarities) / n
            avg_rot  = sum(c.sim_orig_query_rot_chunk  for c in ev.chunk_similarities) / n
            print(
                f"  {ev.query_id}"
                f"  avg_sim_orig={avg_base:.4f}"
                f"  avg_sim_rot={avg_rot:.4f}"
                f"  Δ={avg_rot - avg_base:+.4f}"
                f"  overlap={ev.overlap_count}/{len(ev.original_topk_ids)}"
                f"  targeted_rot_q_rot_db={ev.targeted_in_rot_query_rot_db}/{ev.n_targeted_chunks}"
                f"  targeted_unrot_q_rot_db={ev.targeted_in_unrot_query_rot_db}/{ev.n_targeted_chunks}"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ROTATION_RESULTS, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in eval_results], fh, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Results → {ROTATION_RESULTS}")

    return eval_results


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

@timed("run_experiment")
def run_experiment(
    gt_path: str  = str(GT_FILE),
    top_k:   int  = DEFAULT_TOP_K,
    verbose: bool = True,
) -> list[QueryEvalResult]:
    """Runs all three phases in sequence and persists results + timing log."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(gt_path, encoding="utf-8") as fh:
        gt_records: list[dict] = json.load(fh)

    print(f"\n{'═'*60}")
    print(f"  Rotation Access-Control Experiment")
    print(f"  GT file : {gt_path}  ({len(gt_records)} records)")
    print(f"  Top-K   : {top_k}")
    print(f"{'═'*60}")

    t0 = time.perf_counter()
    embedder = get_embedding_model()
    TIMING_LOG.append({
        "label":      "init_embedding_model",
        "start_iso":  datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.perf_counter() - t0, 6),
        "args_repr":  "",
    })

    dim      = len(embedder.embed_query("probe"))
    registry = RotationRegistry(dim=dim)

    orig_coll = _get_original_collection(ORIGINAL_CHROMA, ORIGINAL_COLLECTION)
    rot_coll  = _get_or_create_rotated_collection(ROTATED_CHROMA, ROTATED_COLLECTION)
    written_ids: set[str] = set()
    _build_rotated_db_untargeted_chunks(gt_records, registry, orig_coll, rot_coll, written_ids, verbose=verbose) # add the untargeted chunks

    # Phase 1
    build_rotated_db(gt_records, registry, orig_coll, rot_coll, verbose=verbose)
    with open(ROTATION_REGISTRY_F, "w", encoding="utf-8") as fh:
        json.dump(registry.to_serialisable(), fh, indent=2)
    print(f"  Registry → {ROTATION_REGISTRY_F}  ({len(registry.all_keys())} groups)")

    # Phase 2
    raw_results = run_query_phase(
        gt_records, registry, embedder, orig_coll, rot_coll,
        top_k=top_k, verbose=verbose,
    )

    # Phase 3
    eval_results = evaluate_results(raw_results, verbose=verbose)

    save_timing_log(eval_results=eval_results)
    return eval_results


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotation access-control experiment")
    parser.add_argument("--gt",    default=str(GT_FILE))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    run_experiment(gt_path=args.gt, top_k=args.top_k, verbose=not args.quiet)
