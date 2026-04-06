"""
rotation_experiment.py
======================
Rotation-based access-control experiment split into three clearly separated phases.

Phase 1 — build_rotated_db
    Iterates over every GT record, assigns one rotation matrix per
    (triplet_index) group via RotationRegistry, embeds every targeted chunk,
    applies its group rotation, and upserts into the rotated Chroma collection.
    Nothing is queried here.

Phase 2 — run_query_phase
    Iterates over every GT record a second time.  For each query:
      • embed the query vector
      • apply the group rotation that was assigned during the build phase
      • query BOTH the original and rotated Chroma collections
    Returns raw retrieval results (no metric computation).

Phase 3 — evaluate_results
    Takes the raw retrieval results from Phase 2 and computes all metrics:
      • four cosine-similarity measurements per chunk
      • top-K overlap between original and rotated collections
      • overlap of retrieved chunks against the targeted list
    Writes results/rotation_results.json.

Timing decorator
    @timed("label") wraps every major function and appends a structured entry
    to TIMING_LOG, which is flushed to logs/rotation_timing.json at the end.

Key invariant (RotationRegistry)
    A (triplet_index) group receives exactly ONE orthogonal rotation matrix,
    assigned the first time the group is encountered in Phase 1.
    Phase 2 reuses the same registry (loaded from memory or disk).

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

# ── Paths ──────────────────────────────────────────────────────────────────────
LOGS_DIR            = Path("logs")
RESULTS_DIR         = Path("results")
GT_FILE             = Path("documents_RAGBench/merged_id_triplets_with_metadata2.json")
ROTATION_RESULTS    = RESULTS_DIR / "rotation_results.json"
ROTATION_REGISTRY_F = RESULTS_DIR / "rotation_registry.json"
TIMING_FILE         = LOGS_DIR    / "rotation_timing.json"

ORIGINAL_CHROMA     = os.path.join(os.getcwd(), "./chroma_db")
ROTATED_CHROMA      = os.path.join(os.getcwd(), "./chroma_rotated_db")
ORIGINAL_COLLECTION = COLLECTION
ROTATED_COLLECTION  = "rotated_experiment_splited_code_with_logs" 

DEFAULT_TOP_K = 20

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Timing infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

TIMING_LOG: list[dict] = []


def timed(label: str):
    """
    Decorator factory — records wall-clock duration of every call.

    Appends to module-level TIMING_LOG::

        {
            "label":      "build_rotated_db",
            "start_iso":  "2025-…",
            "duration_s": 12.345678,
            "args_repr":  "…"          # first 120 chars of args, for context
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
# 1.  Rotation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _group_key(triplet_index: str) -> str:
    return triplet_index


def _seed_from_key(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2 ** 31)


def make_rotation_matrix(dim: int, seed: int) -> np.ndarray:
    """Random orthogonal matrix drawn from the Haar measure."""
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
        self._store: dict[str, dict] = {}   # key → {"seed": int, "matrix": ndarray}

    def get_or_create(self, triplet_index: str) -> np.ndarray:
        key = _group_key(triplet_index)
        if key not in self._store:
            seed = _seed_from_key(key)
            self._store[key] = {
                "seed":   seed,
                "matrix": make_rotation_matrix(self.dim, seed),
            }
        return self._store[key]["matrix"]

    def get(self, triplet_index: str) -> np.ndarray | None:
        key = _group_key(triplet_index)
        entry = self._store.get(key)
        return entry["matrix"] if entry else None

    def seed_of(self, triplet_index: str) -> int | None:
        key = _group_key(triplet_index)
        entry = self._store.get(key)
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
        # if embs and len(embs[0]) > 0:
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
    sim_orig_query_orig_chunk: float   # (a) baseline
    sim_rot_query_rot_chunk:   float   # (b) both rotated  — should equal (a)
    sim_orig_query_rot_chunk:  float   # (c) only chunk rotated
    sim_rot_query_orig_chunk:  float   # (d) only query rotated

    @property
    def delta_c(self) -> float:
        return self.sim_orig_query_rot_chunk - self.sim_orig_query_orig_chunk

    @property
    def delta_d(self) -> float:
        return self.sim_rot_query_orig_chunk - self.sim_orig_query_orig_chunk


@dataclass
class RawQueryRetrieval:
    """Output of Phase 2 — raw retrieval data, no metrics yet."""
    query_id:             str
    question:             str
    triplet_index:        str
    targeted_chunk_ids:   list[str]
    query_vec:            list[float]   # unrotated query vector
    rot_query_vec:        list[float]   # rotated query vector
    rotation_seed:        int

    # Top-K IDs from original collection (unrotated query)
    original_topk_ids:    list[str]     = field(default_factory=list)
    # Top-K IDs from rotated collection  (rotated query)
    rotated_topk_ids:     list[str]     = field(default_factory=list)
    # Top-K IDs from rotated collection  (UNrotated query — sanity check)
    rotated_topk_ids_unrot_query: list[str] = field(default_factory=list)

    # Per-chunk raw vectors stored during build (needed for sim computation)
    # chunk_key → {"orig": list[float], "rot": list[float]}
    chunk_vectors:        dict          = field(default_factory=dict)

    # Timing breakdown for this query's Phase 2 work
    t_embed_query_s:      float = 0.0
    t_apply_rotation_s:   float = 0.0
    t_query_original_s:   float = 0.0
    t_query_rotated_s:    float = 0.0

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

    chunk_similarities:  list[ChunkSimilarities] = field(default_factory=list)

    original_topk_ids:   list[str] = field(default_factory=list)
    rotated_topk_ids:    list[str] = field(default_factory=list)
    overlap_count:       int       = 0
    overlap_fraction:    float     = 0.0

    # Targeted chunks found in retrieval results
    targeted_in_rot_query_rot_db:   int = 0
    targeted_in_unrot_query_rot_db: int = 0

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Phase 1 — Build the rotated database
# ═══════════════════════════════════════════════════════════════════════════════

@timed("build_rotated_db_single_record")
def _build_record(
    record:         dict,
    registry:       RotationRegistry,
    orig_collection,
    rot_collection,
    written_ids:    set[str],
) -> int:
    """
    Process one GT record during the build phase.
    Returns the number of chunks successfully upserted.
    """
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]

    R    = registry.get_or_create(triplet_index)
    n_ok = 0

    for chunk_id in stable_chunks:
        tid  = chunk_id.split("|")[0] # id_triplets | triplet_index
        did  = chunk_id[-2] # document_id
        pseq = chunk_id[-1] # phrase_seq
        cid  = f"{triplet_index}_{did}_{pseq}" #chunk id in the Chroma collection

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
    """
    Phase 1 — full pass over GT records.
    Embeds, rotates, and upserts every targeted chunk.
    Returns the populated RotationRegistry.
    """
    print(f"\n{'─'*60}")
    print(f"  Phase 1 — Building rotated database ({len(gt_records)} records)")
    print(f"{'─'*60}")

    written_ids: set[str] = set()
    total_chunks = 0

    for i, record in enumerate(gt_records, 1): # verify that records is a chunk from the GT file
        triplet_index = record.get("id_triplets")
        if not triplet_index or not record.get("targeted_chunk"):
            if verbose:
                print(f"  [{i}] ⚠  No targeted chunks — skipping.")
            continue

        n = _build_record(record, registry, orig_collection, rot_collection, written_ids)
        total_chunks += n

        if verbose:
            print(
                f"  [{i}/{len(gt_records)}] triplet_{triplet_index}"
                f"  chunks_upserted={n}"
            )

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
    Process one GT record during the query phase.
    Measures embed, rotate, and retrieve timings precisely.
    """
    question      = record["question"]
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    query_id      = f"triplet_{triplet_index}"

    R    = registry.get(triplet_index)
    seed = registry.seed_of(triplet_index)

    if R is None:
        raise RuntimeError(
            f"No rotation matrix found for {triplet_index}. "
            f"Run build_rotated_db first."
        )

    # ── Embed query ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    query_vec = np.array(
        embedder.embed_query(BGE_QUERY_PREFIX + question), dtype=np.float32
    )
    t_embed = time.perf_counter() - t0

    # ── Apply rotation ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    rot_query_vec = apply_rotation(query_vec, R)
    t_rotate = time.perf_counter() - t0

    # ── Retrieve from original DB (unrotated query) ────────────────────────────
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

    # ── Retrieve from rotated DB (rotated query) ───────────────────────────────
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

    # ── Sanity check: unrotated query against rotated DB ──────────────────────
    rot_res_unrot = rot_collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k, n_rot),
        include=["metadatas"],
    )
    rotated_topk_ids_unrot = [
        f"{m.get('triplet_index','?')}|{m.get('document_id','?')}|{m.get('phrase_seq','?')}"
        for m in rot_res_unrot["metadatas"][0]
    ]

    # ── Collect per-chunk vectors for similarity evaluation (Phase 3) ─────────
    chunk_vectors: dict[str, dict] = {}
    for chunk_id in stable_chunks:
        tid  = chunk_id.split("|")[0]
        did  = chunk_id[-2]
        pseq = chunk_id[-1]

        orig_vec, _ = fetch_chunk(orig_collection, tid, did, pseq)
        if orig_vec is None:
            continue
        rot_vec = apply_rotation(orig_vec, R)
        chunk_vectors[chunk_id] = {
            "orig": orig_vec.tolist(),
            "rot":  rot_vec.tolist(),
        }

    return RawQueryRetrieval(
        query_id                      = query_id,
        question                      = question,
        triplet_index                 = triplet_index,
        targeted_chunk_ids            = list(stable_chunks),
        query_vec                     = query_vec.tolist(),
        rot_query_vec                 = rot_query_vec.tolist(),
        rotation_seed                 = int(seed),
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
    """
    Phase 2 — full pass over GT records.
    Embeds each query, applies rotation, retrieves from both collections.
    Returns raw retrieval data (no metric computation).
    """
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
            record          = record,
            registry        = registry,
            embedder        = embedder,
            orig_collection = orig_collection,
            rot_collection  = rot_collection,
            top_k           = top_k,
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
    gkey          = raw.triplet_index

    chunk_sims: list[ChunkSimilarities] = []
    for chunk_id, vecs in raw.chunk_vectors.items():
        orig_vec = np.array(vecs["orig"], dtype=np.float32)
        rot_vec  = np.array(vecs["rot"],  dtype=np.float32)

        chunk_sims.append(ChunkSimilarities(
            chunk_key                  = chunk_id,
            group_key                  = gkey,
            rotation_seed              = raw.rotation_seed,
            sim_orig_query_orig_chunk  = cosine_similarity(query_vec,     orig_vec),
            sim_rot_query_rot_chunk    = cosine_similarity(rot_query_vec, rot_vec),
            sim_orig_query_rot_chunk   = cosine_similarity(query_vec,     rot_vec),
            sim_rot_query_orig_chunk   = cosine_similarity(rot_query_vec, orig_vec),
        ))

    targeted = set(raw.targeted_chunk_ids)
    overlap_set = set(raw.original_topk_ids) & set(raw.rotated_topk_ids)

    targeted_in_rot_rot   = sum(1 for cid in raw.rotated_topk_ids              if cid in targeted)
    targeted_in_unrot_rot = sum(1 for cid in raw.rotated_topk_ids_unrot_query  if cid in targeted)

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
        targeted_in_rot_query_rot_db    = targeted_in_rot_rot,
        targeted_in_unrot_query_rot_db  = targeted_in_unrot_rot,
    )


@timed("evaluate_results")
def evaluate_results(
    raw_results: list[RawQueryRetrieval],
    verbose:     bool = True,
) -> list[QueryEvalResult]:
    """
    Phase 3 — compute metrics from raw retrieval data.
    Writes results/rotation_results.json.
    """
    print(f"\n{'─'*60}")
    print(f"  Phase 3 — Evaluation ({len(raw_results)} queries)")
    print(f"{'─'*60}")

    eval_results: list[QueryEvalResult] = []

    for raw in raw_results:
        ev = _evaluate_record(raw)
        eval_results.append(ev)

        if verbose:
            avg_base = (
                sum(c.sim_orig_query_orig_chunk for c in ev.chunk_similarities)
                / max(len(ev.chunk_similarities), 1)
            )
            avg_ab = (
                sum(c.sim_orig_query_rot_chunk for c in ev.chunk_similarities)
                / max(len(ev.chunk_similarities), 1)
            )
            print(
                f"  {ev.query_id}"
                f"  avg_sim_orig={avg_base:.4f}"
                f"  avg_sim_rot={avg_ab:.4f}"
                f"  Δ={avg_ab - avg_base:+.4f}"
                f"  overlap={ev.overlap_count}/{len(ev.original_topk_ids)}"
                f"  targeted_in_rot_db={ev.targeted_in_rot_query_rot_db}/{ev.n_targeted_chunks}"
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
    """
    Runs all three phases in sequence and persists results + timing log.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(gt_path, encoding="utf-8") as fh:
        gt_records: list[dict] = json.load(fh)

    print(f"\n{'═'*60}")
    print(f"  Rotation Access-Control Experiment")
    print(f"  GT file : {gt_path}  ({len(gt_records)} records)")
    print(f"  Top-K   : {top_k}")
    print(f"{'═'*60}")

    # Init models and collections
    t0       = time.perf_counter()
    embedder = get_embedding_model()
    TIMING_LOG.append({
        "label":      "init_embedding_model",
        "start_iso":  datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.perf_counter() - t0, 6),
        "args_repr":  "",
    })

    dim       = len(embedder.embed_query("probe"))
    registry  = RotationRegistry(dim=dim)

    orig_coll = _get_original_collection(ORIGINAL_CHROMA, ORIGINAL_COLLECTION)
    rot_coll  = _get_or_create_rotated_collection(ROTATED_CHROMA, ROTATED_COLLECTION)

    # ── Phase 1: Build ─────────────────────────────────────────────────────────
    build_rotated_db(gt_records, registry, orig_coll, rot_coll, verbose=verbose)

    # Persist registry after build
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ROTATION_REGISTRY_F, "w", encoding="utf-8") as fh:
        json.dump(registry.to_serialisable(), fh, indent=2)
    print(f"  Registry → {ROTATION_REGISTRY_F}  ({len(registry.all_keys())} groups)")

    # ── Phase 2: Query ─────────────────────────────────────────────────────────
    raw_results = run_query_phase(
        gt_records, registry, embedder, orig_coll, rot_coll, top_k=top_k, verbose=verbose
    )

    # ── Phase 3: Evaluate ──────────────────────────────────────────────────────
    eval_results = evaluate_results(raw_results, verbose=verbose)

    save_timing_log()
    return eval_results


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotation access-control experiment")
    parser.add_argument("--gt",    default=str(GT_FILE), help="Path to GT JSON file")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    run_experiment(gt_path=args.gt, top_k=args.top_k, verbose=not args.quiet)
