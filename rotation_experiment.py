"""
rotation_experiment.py
======================
Core rotation experiment.

Pipeline
--------
1.  Load ground-truth retrieval file produced by ground_truth_collector.py
2.  For every query, identify all (triplet_index, document_id) groups that
    appear in its stable chunk set.
3.  Assign ONE rotation matrix per group — using a global registry that
    guarantees no chunk is ever touched by two different rotations.
4.  Embed each stable chunk with the base embedder, then apply its group
    rotation → store in a SEPARATE Chroma collection ("rotated_experiment_db").
5.  Also embed each query vector, apply each group's rotation to it, then
    measure cosine similarity between:
        (a) original query   ↔ original chunk     [baseline]
        (b) rotated query    ↔ rotated chunk       [same rotation applied to both]
        (c) original query   ↔ rotated chunk       [rotation breaks alignment]
        (d) rotated query    ↔ original chunk      [mirror of (c)]
6.  Run top-K retrieval from BOTH Chroma collections for each query, record
    which chunk IDs appear in each result set → compute overlap.
7.  Cross-query experiment: apply the rotation learned from query A's chunks
    to query B's vector and measure whether previously-far chunks become closer.
8.  Persist everything to  results/rotation_results.json  and
                            results/rotation_registry.json

Key invariant (enforced by RotationRegistry)
--------------------------------------------
A chunk identified by (triplet_index, document_id) receives exactly ONE
rotation matrix, assigned the first time that group is encountered.
If a second query tries to assign a different rotation to the same group,
the registry reuses the existing matrix and logs a warning.  This guarantees
that every vector in the rotated Chroma collection was transformed by exactly
one deterministic orthogonal matrix.

Rotation matrices
-----------------
Each rotation is a random orthogonal matrix drawn from the Haar measure
(scipy.stats.ortho_group).  Different groups get different seeds derived
from hash(group_key) so they are reproducible but uncorrelated.

Usage
-----
python rotation_experiment.py
python rotation_experiment.py --gt results/ground_truth_retrievals.json
python rotation_experiment.py --top-k 15 --cross-queries 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings
from scipy.stats import ortho_group
from config import COLLECTION

from ingestion_pipeline import get_embedding_model   # base embedder (no rotation)

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR         = Path("results")
GT_FILE             = RESULTS_DIR / "ground_truth_retrievals.json"
ROTATION_RESULTS    = RESULTS_DIR / "rotation_results.json"
ROTATION_REGISTRY_F = RESULTS_DIR / "rotation_registry.json"

ORIGINAL_CHROMA     = os.path.join(os.getcwd(), "./chroma_db")
ROTATED_CHROMA      = os.path.join(os.getcwd(), "./chroma_rotated_db")

ORIGINAL_COLLECTION = COLLECTION         # name used during normal ingestion
ROTATED_COLLECTION  = "rotated_experiment"

DEFAULT_TOP_K        = 20
DEFAULT_CROSS_QUERIES = 3   # how many foreign queries to test per chunk group


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Rotation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _group_key(triplet_index: str, document_id: str) -> str:
    return f"{triplet_index}|{document_id}"


def _seed_from_key(key: str) -> int:
    """Deterministic integer seed derived from a group key string."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31)


def make_rotation_matrix(dim: int, seed: int) -> np.ndarray:
    """Draw a random orthogonal matrix from the Haar measure."""
    return ortho_group.rvs(dim=dim, random_state=seed).astype(np.float32)


def apply_rotation(vec: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply rotation matrix R to a single vector or a batch (N, dim)."""
    if vec.ndim == 1:
        return R @ vec
    return (R @ vec.T).T  # (dim, dim) @ (dim, N) → (dim, N) → (N, dim)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Rotation Registry  — enforces the one-rotation-per-group invariant
# ═══════════════════════════════════════════════════════════════════════════════

class RotationRegistry:
    """
    Keeps track of which (triplet_index, document_id) group has which rotation.

    Guarantees: once a group is registered, its rotation never changes.
    Any attempt to register the same group again simply returns the existing matrix.

    The registry can be serialised to JSON (seeds only — matrices are
    re-derived on load so we don't bloat the file with large arrays).
    """

    def __init__(self, dim: int):
        self.dim   = dim
        # group_key → {"seed": int, "matrix": np.ndarray}
        self._store: dict[str, dict] = {}

    def get_or_create(
        self,
        triplet_index: str,
        document_id: str,
        requested_seed: int | None = None,
    ) -> np.ndarray:
        """
        Return the rotation matrix for this group, creating it if needed.
        If `requested_seed` is None, a deterministic seed is derived from the key.
        """
        key = _group_key(triplet_index, document_id)
        if key not in self._store:
            seed = requested_seed if requested_seed is not None else _seed_from_key(key)
            self._store[key] = {
                "seed":   seed,
                "matrix": make_rotation_matrix(self.dim, seed),
            }
        return self._store[key]["matrix"]

    def has(self, triplet_index: str, document_id: str) -> bool:
        return _group_key(triplet_index, document_id) in self._store

    def all_keys(self) -> list[str]:
        return list(self._store.keys())

    def to_serialisable(self) -> dict:
        """Export registry as {group_key: seed} — matrices are re-derived on load."""
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
    return chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_original_collection(path: str, name: str):
    client = _get_client(path)
    return client.get_collection(name=name)


def _get_or_create_rotated_collection(path: str, name: str):
    client = _get_client(path)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def fetch_original_vector(
    collection,
    triplet_index: str,
    document_id: str,
    phrase_seq: str,
) -> np.ndarray | None:
    """
    Retrieve the stored embedding for a specific chunk from the original
    ChromaDB collection by metadata filters.
    Returns None if not found.
    """
    try:
        result = collection.get(
            where={
                "$and": [
                    {"triplet_index": {"$eq": triplet_index}},
                    {"document_id":   {"$eq": document_id}},
                    {"phrase_seq":    {"$eq": phrase_seq}},
                ]
            },
            include=["embeddings"],
        )
        if result["embeddings"] and len(result["embeddings"]) > 0:
            return np.array(result["embeddings"][0], dtype=np.float32)
    except Exception as e:
        warnings.warn(f"fetch_original_vector failed for {triplet_index}|{document_id}|{phrase_seq}: {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Per-query experiment data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkSimilarities:
    """Four cosine-similarity measurements for one chunk vs one query."""
    chunk_key: str                  # triplet|doc|phrase
    group_key: str                  # triplet|doc  (rotation group)
    rotation_seed: int

    # (a) baseline — no rotation applied to either side
    sim_orig_query_orig_chunk: float

    # (b) same rotation on both sides — should be identical to (a) for
    #     an orthogonal matrix since  (Rq)·(Rc) = q·R^T R c = q·c
    #     Any deviation indicates numerical error only.
    sim_rot_query_rot_chunk: float

    # (c) original query vs rotated chunk — rotation breaks alignment
    sim_orig_query_rot_chunk: float

    # (d) rotated query vs original chunk — mirror of (c)
    sim_rot_query_orig_chunk: float

    # Derived
    @property
    def delta_c(self) -> float:
        """How much the similarity drops when only the chunk is rotated."""
        return self.sim_orig_query_rot_chunk - self.sim_orig_query_orig_chunk

    @property
    def delta_d(self) -> float:
        """How much the similarity drops when only the query is rotated."""
        return self.sim_rot_query_orig_chunk - self.sim_orig_query_orig_chunk


@dataclass
class CrossQueryResult:
    """Similarity of a foreign query against chunks whose rotation was built for another query."""
    foreign_query_id: str
    foreign_question: str
    group_key: str
    rotation_seed: int
    chunk_key: str
    # Before applying the group's rotation to the foreign query
    sim_foreign_orig_vs_rot_chunk: float
    # After applying the group's rotation to the foreign query
    sim_foreign_rot_vs_rot_chunk: float

    @property
    def delta(self) -> float:
        return self.sim_foreign_rot_vs_rot_chunk - self.sim_foreign_orig_vs_rot_chunk


@dataclass
class QueryExperimentResult:
    query_id: str
    question: str
    triplet_index: str
    n_stable_chunks: int
    n_groups: int                   # distinct (triplet, doc_id) groups

    # Per-chunk similarity breakdown
    chunk_similarities: list[ChunkSimilarities] = field(default_factory=list)

    # Top-K retrieval overlap
    original_topk_ids: list[str]    = field(default_factory=list)
    rotated_topk_ids:  list[str]    = field(default_factory=list)
    overlap_count:     int          = 0
    overlap_fraction:  float        = 0.0

    # Cross-query results
    cross_query_results: list[CrossQueryResult] = field(default_factory=list)

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Core experiment for a single query
# ═══════════════════════════════════════════════════════════════════════════════

def run_query_experiment(
    gt_record: dict,
    all_gt_records: list[dict],
    registry: RotationRegistry,
    embedder,
    orig_collection,
    rot_collection,
    top_k: int = DEFAULT_TOP_K,
    n_cross_queries: int = DEFAULT_CROSS_QUERIES,
    verbose: bool = True,
) -> QueryExperimentResult:
    """
    Full rotation experiment for one query.

    Parameters
    ----------
    gt_record       : one entry from ground_truth_retrievals.json
    all_gt_records  : the full list (used to pick foreign queries)
    registry        : shared RotationRegistry (mutated in place)
    embedder        : base embedding model (no rotation)
    orig_collection : original ChromaDB collection
    rot_collection  : rotated ChromaDB collection (written to here)
    """
    question      = gt_record["question"]
    query_id      = gt_record["query_id"]
    triplet_index = gt_record["triplet_index"]
    stable_chunks = gt_record["stable_chunks"]

    if verbose:
        print(f"\n  Query: {question[:70]}…")
        print(f"  Stable chunks: {len(stable_chunks)}")

    # ── Embed the query ───────────────────────────────────────────────────────
    from query_pipeline import BGE_QUERY_PREFIX
    query_vec = np.array(
        embedder.embed_query(BGE_QUERY_PREFIX + question), dtype=np.float32
    )
    dim = query_vec.shape[0]

    # ── Discover groups and assign rotations ──────────────────────────────────
    groups_in_query: set[str] = set()
    for chunk in stable_chunks:
        tid = chunk["triplet_index"]
        did = chunk["document_id"]
        registry.get_or_create(tid, did)   # registers if new
        groups_in_query.add(_group_key(tid, did))

    if verbose:
        print(f"  Distinct rotation groups: {len(groups_in_query)}")

    # ── Process each chunk ────────────────────────────────────────────────────
    chunk_sims: list[ChunkSimilarities] = []
    rotated_chunk_ids_written: set[str] = set()

    for chunk in stable_chunks:
        tid       = chunk["triplet_index"]
        did       = chunk["document_id"]
        pseq      = chunk["phrase_seq"]
        ckey      = f"{tid}|{did}|{pseq}"
        gkey      = _group_key(tid, did)
        R         = registry.get_or_create(tid, did)
        seed      = registry._store[gkey]["seed"]

        # ── Get original embedding (from ChromaDB or re-embed) ────────────────
        orig_vec = fetch_original_vector(orig_collection, tid, did, pseq)
        if orig_vec is None:
            # Fall back to re-embedding the content
            orig_vec = np.array(
                embedder.embed_documents([chunk["content"]])[0], dtype=np.float32
            )

        # ── Apply rotation to chunk ───────────────────────────────────────────
        rot_vec = apply_rotation(orig_vec, R)

        # ── Write rotated chunk to rotated Chroma (once per chunk) ────────────
        chroma_id = f"{tid}_{did}_{pseq}"
        if chroma_id not in rotated_chunk_ids_written:
            rot_collection.upsert(
                ids        = [chroma_id],
                embeddings = [rot_vec.tolist()],
                documents  = [chunk["content"]],
                metadatas  = [{
                    "triplet_index": tid,
                    "document_id":   did,
                    "phrase_seq":    pseq,
                    "group_key":     gkey,
                    "rotation_seed": seed,
                }],
            )
            rotated_chunk_ids_written.add(chroma_id)

        # ── Rotate query with this group's rotation ───────────────────────────
        rot_query_vec = apply_rotation(query_vec, R)

        # ── Four similarity measurements ──────────────────────────────────────
        sim_aa = cosine_similarity(query_vec,     orig_vec)   # (a) baseline
        sim_bb = cosine_similarity(rot_query_vec, rot_vec)    # (b) both rotated
        sim_ab = cosine_similarity(query_vec,     rot_vec)    # (c) only chunk rotated
        sim_ba = cosine_similarity(rot_query_vec, orig_vec)   # (d) only query rotated

        chunk_sims.append(ChunkSimilarities(
            chunk_key                  = ckey,
            group_key                  = gkey,
            rotation_seed              = seed,
            sim_orig_query_orig_chunk  = sim_aa,
            sim_rot_query_rot_chunk    = sim_bb,
            sim_orig_query_rot_chunk   = sim_ab,
            sim_rot_query_orig_chunk   = sim_ba,
        ))

    # ── Top-K retrieval from both collections ─────────────────────────────────
    # Original collection: query with unrotated query vector
    orig_results = orig_collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k, orig_collection.count()),
        include=["metadatas", "distances"],
    )
    original_topk_ids = [
        f"{m.get('triplet_index','?')}|{m.get('document_id','?')}|{m.get('phrase_seq','?')}"
        for m in orig_results["metadatas"][0]
    ]

    # Rotated collection: we have one rotation per group — query each group
    # separately then merge, or use the mean rotation.  We choose mean rotation
    # as a single representative query vector for the rotated space.
    group_keys = list(groups_in_query)
    if group_keys:
        rot_query_vecs = np.stack([
            apply_rotation(query_vec, registry.get_or_create(*gk.split("|")))
            for gk in group_keys
        ])
        mean_rot_query = rot_query_vecs.mean(axis=0)
        mean_rot_query /= np.linalg.norm(mean_rot_query) + 1e-10

        rot_results = rot_collection.query(
            query_embeddings=[mean_rot_query.tolist()],
            n_results=min(top_k, max(1, rot_collection.count())),
            include=["metadatas", "distances"],
        )
        rotated_topk_ids = [
            f"{m.get('triplet_index','?')}|{m.get('document_id','?')}|{m.get('phrase_seq','?')}"
            for m in rot_results["metadatas"][0]
        ]
    else:
        rotated_topk_ids = []

    overlap_set      = set(original_topk_ids) & set(rotated_topk_ids)
    overlap_count    = len(overlap_set)
    overlap_fraction = overlap_count / max(len(original_topk_ids), 1)

    # ── Cross-query experiment ────────────────────────────────────────────────
    # Pick n_cross_queries foreign queries (different triplet_index)
    foreign_records = [
        r for r in all_gt_records
        if r["triplet_index"] != triplet_index and r["stable_chunks"]
    ][:n_cross_queries]

    cross_results: list[CrossQueryResult] = []
    for foreign in foreign_records:
        fq_vec = np.array(
            embedder.embed_query(BGE_QUERY_PREFIX + foreign["question"]),
            dtype=np.float32,
        )
        # For each group in the CURRENT query, apply its rotation to the foreign query
        # and measure how that changes similarity to the group's rotated chunks
        for gkey in groups_in_query:
            tid_g, did_g = gkey.split("|")
            R_g          = registry.get_or_create(tid_g, did_g)
            rot_fq_vec   = apply_rotation(fq_vec, R_g)

            # Find the chunks in this group that are in stable_chunks
            group_chunks = [
                c for c in stable_chunks
                if _group_key(c["triplet_index"], c["document_id"]) == gkey
            ]

            for chunk in group_chunks[:3]:   # limit per group to keep results manageable
                tid_c = chunk["triplet_index"]
                did_c = chunk["document_id"]
                pseq  = chunk["phrase_seq"]
                ckey  = f"{tid_c}|{did_c}|{pseq}"

                orig_c = fetch_original_vector(orig_collection, tid_c, did_c, pseq)
                if orig_c is None:
                    orig_c = np.array(
                        embedder.embed_documents([chunk["content"]])[0], dtype=np.float32
                    )
                rot_c = apply_rotation(orig_c, R_g)

                sim_before = cosine_similarity(fq_vec,     rot_c)
                sim_after  = cosine_similarity(rot_fq_vec, rot_c)

                cross_results.append(CrossQueryResult(
                    foreign_query_id              = foreign["query_id"],
                    foreign_question              = foreign["question"][:120],
                    group_key                     = gkey,
                    rotation_seed                 = registry._store[gkey]["seed"],
                    chunk_key                     = ckey,
                    sim_foreign_orig_vs_rot_chunk = sim_before,
                    sim_foreign_rot_vs_rot_chunk  = sim_after,
                ))

    if verbose:
        avg_base = sum(c.sim_orig_query_orig_chunk for c in chunk_sims) / max(len(chunk_sims), 1)
        avg_ab   = sum(c.sim_orig_query_rot_chunk  for c in chunk_sims) / max(len(chunk_sims), 1)
        print(f"  Avg similarity  orig↔orig={avg_base:.4f}  orig↔rot={avg_ab:.4f}  "
              f"Δ={avg_ab-avg_base:+.4f}")
        print(f"  Top-{top_k} overlap: {overlap_count}/{len(original_topk_ids)} "
              f"({overlap_fraction:.1%})")

    return QueryExperimentResult(
        query_id            = query_id,
        question            = question,
        triplet_index       = triplet_index,
        n_stable_chunks     = len(stable_chunks),
        n_groups            = len(groups_in_query),
        chunk_similarities  = chunk_sims,
        original_topk_ids   = original_topk_ids,
        rotated_topk_ids    = rotated_topk_ids,
        overlap_count       = overlap_count,
        overlap_fraction    = overlap_fraction,
        cross_query_results = cross_results,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Full experiment runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    gt_path: str           = str(GT_FILE),
    top_k: int             = DEFAULT_TOP_K,
    n_cross_queries: int   = DEFAULT_CROSS_QUERIES,
    verbose: bool          = True,
) -> list[QueryExperimentResult]:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(gt_path, encoding="utf-8") as fh:
        gt_records: list[dict] = json.load(fh)

    print(f"\n{'═'*62}")
    print(f"  Rotation Experiment")
    print(f"  Ground truth: {gt_path}  ({len(gt_records)} queries)")
    print(f"  Top-K: {top_k}   Cross-queries: {n_cross_queries}")
    print(f"{'═'*62}")

    # ── Initialise models and collections ─────────────────────────────────────
    embedder      = get_embedding_model()          # base, no rotation
    orig_coll     = _get_original_collection(ORIGINAL_CHROMA, ORIGINAL_COLLECTION)
    rot_coll      = _get_or_create_rotated_collection(ROTATED_CHROMA, ROTATED_COLLECTION)
    dim           = len(embedder.embed_query("probe"))
    registry      = RotationRegistry(dim=dim)

    all_results: list[QueryExperimentResult] = []

    for i, record in enumerate(gt_records, 1):
        print(f"\n[{i}/{len(gt_records)}] {record['query_id']}")
        if not record.get("stable_chunks"):
            print("  ⚠  No stable chunks — skipping.")
            continue

        result = run_query_experiment(
            gt_record       = record,
            all_gt_records  = gt_records,
            registry        = registry,
            embedder        = embedder,
            orig_collection = orig_coll,
            rot_collection  = rot_coll,
            top_k           = top_k,
            n_cross_queries = n_cross_queries,
            verbose         = verbose,
        )
        all_results.append(result)

    # ── Serialise results ─────────────────────────────────────────────────────
    def _serialise(obj):
        if isinstance(obj, QueryExperimentResult):
            d = asdict(obj)
            return d
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(ROTATION_RESULTS, "w", encoding="utf-8") as fh:
        json.dump(
            [asdict(r) for r in all_results],
            fh, indent=2, ensure_ascii=False,
        )
    print(f"\n✅  Results saved to {ROTATION_RESULTS}")

    # ── Serialise registry ────────────────────────────────────────────────────
    with open(ROTATION_REGISTRY_F, "w", encoding="utf-8") as fh:
        json.dump(registry.to_serialisable(), fh, indent=2)
    print(f"✅  Registry saved to {ROTATION_REGISTRY_F}")
    print(f"   {len(registry.all_keys())} unique rotation groups\n")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotation experiment for RAG embeddings")
    parser.add_argument("--gt",            default=str(GT_FILE),
                        help="Path to ground_truth_retrievals.json")
    parser.add_argument("--top-k",         type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cross-queries", type=int, default=DEFAULT_CROSS_QUERIES,
                        help="Number of foreign queries to test per query")
    parser.add_argument("--quiet", "-q",   action="store_true")
    args = parser.parse_args()

    run_experiment(
        gt_path         = args.gt,
        top_k           = args.top_k,
        n_cross_queries = args.cross_queries,
        verbose         = not args.quiet,
    )
