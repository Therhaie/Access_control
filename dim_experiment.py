"""
dim_experiment.py
=================
Extra-Dimension Security Experiment.

Mirrors rotation_experiment.py but instead of rotating vectors, we APPEND
extra dimensions whose values encode a security/access-control signal.

Security encoding
-----------------
Each chunk gets a "security vector" s ∈ ℝ^extra_dims appended to its base
embedding.  The value of s depends on whether the chunk is "restricted":

  restricted chunk  → s = weight_vector          (e.g. [w₁, w₂, …, wₙ])
  open chunk        → s = zero_vector             ([0,  0,  …, 0 ])

A query that "has access" gets  s_query = weight_vector  appended.
A query that "lacks access"    gets  s_query = zero_vector  appended.

When the query and chunk are on the SAME access level the extra dims contribute
positively to cosine similarity (they align).  When they differ the extra dims
push the vectors apart — a restricted chunk becomes harder to retrieve for an
unauthorised query.

Configurable parameters (all exposed via CLI and ExtraDimConfig)
----------------------------------------------------------------
  extra_dims      : int   — how many dimensions to append (default 4)
  weight          : float — scalar applied to every extra dimension (default 1.0)
  weight_vector   : list  — per-dimension weights; if None, uniform `weight` is used
  normalize_before: bool  — L2-normalise the base vector BEFORE appending
  normalize_after : bool  — L2-normalise the full (base + extra) vector AFTER appending

The combination (normalize_before=True, normalize_after=False) is the standard
cosine-space setup; (normalize_after=True) ensures the augmented vector also
lives on the unit hypersphere.

Experiment logic
----------------
For each ground-truth query:
1. Take the stable chunks (from ground_truth_retrievals.json).
2. Mark the stable chunks as "restricted" (they are the sensitive ones
   that we want to hide from unauthorised queries).
3. Embed + augment each chunk under multiple configs and store in separate
   Chroma collections (one per config, named  dim_exp_{config_id}).
4. For each config measure:
   (a) authorised query (same weight appended)   ↔  restricted chunk  [baseline match]
   (b) unauthorised query (zeros appended)        ↔  restricted chunk  [should be lower]
   (c) authorised query                           ↔  open chunk        [collateral]
   (d) unauthorised query                         ↔  open chunk        [should be unchanged]
5. Run top-K retrieval from each augmented collection with both query types,
   record which restricted chunks appear/disappear.
6. Cross-config experiment: does a weight_vector from config A accidentally
   hide or expose chunks when applied under config B?

Output
------
  results/dim_results.json      — full per-query, per-config results
  results/dim_registry.json     — which config was applied to which chunk

Usage
-----
python dim_experiment.py
python dim_experiment.py --gt results/ground_truth_retrievals.json
python dim_experiment.py --extra-dims 8 --weights 0.5 1.0 2.0 --normalize-after
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from ingestion_pipeline import get_embedding_model
from query_pipeline import BGE_QUERY_PREFIX

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR      = Path("results")
GT_FILE          = RESULTS_DIR / "ground_truth_retrievals.json"
DIM_RESULTS_FILE = RESULTS_DIR / "dim_results.json"
DIM_REGISTRY_FILE= RESULTS_DIR / "dim_registry.json"

ORIGINAL_CHROMA      = os.path.join(os.getcwd(), "./chroma_db")
ORIGINAL_COLLECTION  = "langchain"
DIM_CHROMA_BASE      = os.path.join(os.getcwd(), "./chroma_dim_db")

DEFAULT_TOP_K        = 20
DEFAULT_EXTRA_DIMS   = 4
DEFAULT_WEIGHT       = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtraDimConfig:
    """
    All knobs for one experimental condition.

    weight_vector : if None, a uniform vector of `weight` repeated `extra_dims` times
                    is used.  Otherwise must have length == extra_dims.
    """
    extra_dims:       int
    weight:           float              = DEFAULT_WEIGHT
    weight_vector:    Optional[list[float]] = None   # per-dim override
    normalize_before: bool               = False
    normalize_after:  bool               = False

    def __post_init__(self):
        if self.weight_vector is None:
            self.weight_vector = [self.weight] * self.extra_dims
        if len(self.weight_vector) != self.extra_dims:
            raise ValueError(
                f"weight_vector length {len(self.weight_vector)} "
                f"!= extra_dims {self.extra_dims}"
            )

    @property
    def config_id(self) -> str:
        wv_str  = "_".join(f"{w:.2f}" for w in self.weight_vector)
        nb      = "nb1" if self.normalize_before else "nb0"
        na      = "na1" if self.normalize_after  else "na0"
        return f"d{self.extra_dims}_w{wv_str}_{nb}_{na}"

    def to_dict(self) -> dict:
        return {
            "extra_dims":       self.extra_dims,
            "weight":           self.weight,
            "weight_vector":    self.weight_vector,
            "normalize_before": self.normalize_before,
            "normalize_after":  self.normalize_after,
            "config_id":        self.config_id,
        }


def _default_configs() -> list[ExtraDimConfig]:
    """
    A representative set of configs to sweep over if none are specified by the user.
    Covers: varying weight, normalisation on/off combinations.
    """
    return [
        ExtraDimConfig(extra_dims=4, weight=0.5,  normalize_before=False, normalize_after=False),
        ExtraDimConfig(extra_dims=4, weight=1.0,  normalize_before=False, normalize_after=False),
        ExtraDimConfig(extra_dims=4, weight=2.0,  normalize_before=False, normalize_after=False),
        ExtraDimConfig(extra_dims=4, weight=1.0,  normalize_before=True,  normalize_after=False),
        ExtraDimConfig(extra_dims=4, weight=1.0,  normalize_before=False, normalize_after=True),
        ExtraDimConfig(extra_dims=4, weight=1.0,  normalize_before=True,  normalize_after=True),
        ExtraDimConfig(extra_dims=8, weight=1.0,  normalize_before=False, normalize_after=False),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Augmentation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def augment_vector(
    base_vec: np.ndarray,
    cfg: ExtraDimConfig,
    is_restricted: bool,
) -> np.ndarray:
    """
    Append security extra-dims to `base_vec`.

    restricted=True  → append weight_vector
    restricted=False → append zeros
    """
    v = base_vec.astype(np.float32)
    if cfg.normalize_before:
        v = _l2_norm(v)

    extra = (
        np.array(cfg.weight_vector, dtype=np.float32)
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


def _fetch_original_vector(collection, triplet_index, document_id, phrase_seq) -> np.ndarray | None:
    try:
        result = collection.get(
            where={"$and": [
                {"triplet_index": {"$eq": triplet_index}},
                {"document_id":   {"$eq": document_id}},
                {"phrase_seq":    {"$eq": phrase_seq}},
            ]},
            include=["embeddings"],
        )
        if result["embeddings"] and len(result["embeddings"]) > 0:
            return np.array(result["embeddings"][0], dtype=np.float32)
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Per-chunk result structure
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DimChunkSimilarities:
    chunk_key:     str
    config_id:     str
    is_restricted: bool

    # (a) authorised query   ↔ restricted chunk   [both have weight_vector appended]
    sim_auth_restricted:   float
    # (b) unauthorised query ↔ restricted chunk   [query has zeros, chunk has weights]
    sim_unauth_restricted: float
    # (c) authorised query   ↔ open chunk         [query has weights, chunk has zeros]
    sim_auth_open:         float
    # (d) unauthorised query ↔ open chunk         [both have zeros appended]
    sim_unauth_open:       float
    # baseline: no augmentation at all
    sim_baseline:          float

    @property
    def security_delta(self) -> float:
        """
        Drop in similarity for an unauthorised query on a restricted chunk.
        Negative = the restriction is working (unauthorised sees lower similarity).
        """
        return self.sim_unauth_restricted - self.sim_auth_restricted

    @property
    def collateral_delta(self) -> float:
        """
        Change for open chunks when an authorised query is used.
        Should be close to 0 — open chunks should be unaffected.
        """
        return self.sim_auth_open - self.sim_baseline


@dataclass
class DimQueryResult:
    query_id:        str
    question:        str
    triplet_index:   str
    config_id:       str
    n_restricted:    int
    n_open:          int

    chunk_sims:      list[DimChunkSimilarities] = field(default_factory=list)

    # Top-K retrieval
    topk_auth_ids:   list[str] = field(default_factory=list)
    topk_unauth_ids: list[str] = field(default_factory=list)
    # how many restricted chunks appear in auth vs unauth retrieval
    restricted_in_auth:   int   = 0
    restricted_in_unauth: int   = 0

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Per-query experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_query_dim_experiment(
    gt_record: dict,
    cfg: ExtraDimConfig,
    embedder,
    orig_collection,
    dim_collection,
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = True,
) -> DimQueryResult:
    """Run the dimension experiment for one query under one config."""
    question      = gt_record["question"]
    query_id      = gt_record["query_id"]
    triplet_index = gt_record["triplet_index"]
    stable_chunks = gt_record["stable_chunks"]

    # stable_chunks are "restricted"; we also need a reference set of open chunks
    # We use all other chunks in the collection as "open" — but for similarity
    # measurement we only need a representative sample.  We'll embed on the fly.
    restricted_keys = {
        f"{c['triplet_index']}|{c['document_id']}|{c['phrase_seq']}"
        for c in stable_chunks
    }

    # ── Embed the raw query vector (no augmentation yet) ──────────────────────
    raw_query_vec = np.array(
        embedder.embed_query(BGE_QUERY_PREFIX + question), dtype=np.float32
    )

    # Build augmented query vectors
    auth_query_vec   = augment_vector(raw_query_vec, cfg, is_restricted=True)
    unauth_query_vec = augment_vector(raw_query_vec, cfg, is_restricted=False)

    chunk_sims: list[DimChunkSimilarities] = []
    written_ids: set[str] = set()

    # ── Process restricted chunks ─────────────────────────────────────────────
    for chunk in stable_chunks:
        tid  = chunk["triplet_index"]
        did  = chunk["document_id"]
        pseq = chunk["phrase_seq"]
        ckey = f"{tid}|{did}|{pseq}"
        cid  = f"{tid}_{did}_{pseq}"

        raw_vec = _fetch_original_vector(orig_collection, tid, did, pseq)
        if raw_vec is None:
            raw_vec = np.array(
                embedder.embed_documents([chunk["content"]])[0], dtype=np.float32
            )

        aug_restricted_vec = augment_vector(raw_vec, cfg, is_restricted=True)
        aug_open_vec       = augment_vector(raw_vec, cfg, is_restricted=False)

        # Upsert restricted version into dim collection
        if cid not in written_ids:
            dim_collection.upsert(
                ids        = [cid],
                embeddings = [aug_restricted_vec.tolist()],
                documents  = [chunk["content"]],
                metadatas  = [{
                    "triplet_index": tid,
                    "document_id":   did,
                    "phrase_seq":    pseq,
                    "restricted":    True,
                    "config_id":     cfg.config_id,
                }],
            )
            written_ids.add(cid)

        sim_baseline  = cosine_similarity(raw_query_vec,   raw_vec)
        sim_a         = cosine_similarity(auth_query_vec,   aug_restricted_vec)
        sim_b         = cosine_similarity(unauth_query_vec, aug_restricted_vec)
        sim_c         = cosine_similarity(auth_query_vec,   aug_open_vec)
        sim_d         = cosine_similarity(unauth_query_vec, aug_open_vec)

        chunk_sims.append(DimChunkSimilarities(
            chunk_key             = ckey,
            config_id             = cfg.config_id,
            is_restricted         = True,
            sim_auth_restricted   = sim_a,
            sim_unauth_restricted = sim_b,
            sim_auth_open         = sim_c,
            sim_unauth_open       = sim_d,
            sim_baseline          = sim_baseline,
        ))

    # ── Top-K retrieval ───────────────────────────────────────────────────────
    n_in_coll = max(1, dim_collection.count())

    auth_results = dim_collection.query(
        query_embeddings=[auth_query_vec.tolist()],
        n_results=min(top_k, n_in_coll),
        include=["metadatas"],
    )
    unauth_results = dim_collection.query(
        query_embeddings=[unauth_query_vec.tolist()],
        n_results=min(top_k, n_in_coll),
        include=["metadatas"],
    )

    def _make_id(m: dict) -> str:
        return f"{m.get('triplet_index','?')}|{m.get('document_id','?')}|{m.get('phrase_seq','?')}"

    topk_auth   = [_make_id(m) for m in auth_results["metadatas"][0]]
    topk_unauth = [_make_id(m) for m in unauth_results["metadatas"][0]]

    restricted_in_auth   = len(set(topk_auth)   & restricted_keys)
    restricted_in_unauth = len(set(topk_unauth) & restricted_keys)

    if verbose:
        avg_sec_delta = (
            sum(c.security_delta for c in chunk_sims) / max(len(chunk_sims), 1)
        )
        print(f"    cfg={cfg.config_id}  "
              f"avg_security_delta={avg_sec_delta:+.4f}  "
              f"restricted_in_auth={restricted_in_auth}  "
              f"restricted_in_unauth={restricted_in_unauth}")

    return DimQueryResult(
        query_id              = query_id,
        question              = question,
        triplet_index         = triplet_index,
        config_id             = cfg.config_id,
        n_restricted          = len(stable_chunks),
        n_open                = 0,
        chunk_sims            = chunk_sims,
        topk_auth_ids         = topk_auth,
        topk_unauth_ids       = topk_unauth,
        restricted_in_auth    = restricted_in_auth,
        restricted_in_unauth  = restricted_in_unauth,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Full experiment runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_dim_experiment(
    gt_path: str            = str(GT_FILE),
    configs: list[ExtraDimConfig] | None = None,
    top_k: int              = DEFAULT_TOP_K,
    verbose: bool           = True,
) -> list[dict]:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(gt_path, encoding="utf-8") as fh:
        gt_records: list[dict] = json.load(fh)

    if configs is None:
        configs = _default_configs()

    print(f"\n{'═'*62}")
    print(f"  Extra-Dimension Security Experiment")
    print(f"  Ground truth: {gt_path}  ({len(gt_records)} queries)")
    print(f"  Configs: {len(configs)}")
    print(f"{'═'*62}")

    embedder    = get_embedding_model()
    orig_coll   = _get_original_collection(ORIGINAL_CHROMA, ORIGINAL_COLLECTION)

    all_results: list[dict] = []
    registry: list[dict]    = []

    for i, record in enumerate(gt_records, 1):
        if not record.get("stable_chunks"):
            continue
        print(f"\n[{i}/{len(gt_records)}] {record['query_id']}")

        for cfg in configs:
            dim_coll = _get_dim_collection(cfg.config_id)
            result   = run_query_dim_experiment(
                gt_record      = record,
                cfg            = cfg,
                embedder       = embedder,
                orig_collection= orig_coll,
                dim_collection = dim_coll,
                top_k          = top_k,
                verbose        = verbose,
            )
            all_results.append(asdict(result))
            registry.append({
                "query_id":   record["query_id"],
                "config":     cfg.to_dict(),
                "n_restricted": len(record["stable_chunks"]),
            })

    with open(DIM_RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print(f"\n✅  Results saved to {DIM_RESULTS_FILE}")

    with open(DIM_REGISTRY_FILE, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)
    print(f"✅  Registry saved to {DIM_REGISTRY_FILE}\n")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extra-dimension security experiment for RAG embeddings"
    )
    parser.add_argument("--gt",              default=str(GT_FILE))
    parser.add_argument("--top-k",           type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--extra-dims",      type=int,   default=DEFAULT_EXTRA_DIMS,
                        help="Number of extra dimensions to append")
    parser.add_argument("--weights",         type=float, nargs="+", default=None,
                        help="One or more weight scalars to sweep (e.g. 0.5 1.0 2.0)")
    parser.add_argument("--weight-vector",   type=float, nargs="+", default=None,
                        help="Explicit per-dim weight vector (length must equal --extra-dims)")
    parser.add_argument("--normalize-before",action="store_true")
    parser.add_argument("--normalize-after", action="store_true")
    parser.add_argument("--quiet", "-q",     action="store_true")
    args = parser.parse_args()

    # Build configs from CLI
    if args.weight_vector is not None:
        configs = [ExtraDimConfig(
            extra_dims=args.extra_dims,
            weight_vector=args.weight_vector,
            normalize_before=args.normalize_before,
            normalize_after=args.normalize_after,
        )]
    elif args.weights is not None:
        configs = [
            ExtraDimConfig(
                extra_dims=args.extra_dims,
                weight=w,
                normalize_before=args.normalize_before,
                normalize_after=args.normalize_after,
            )
            for w in args.weights
        ]
    else:
        configs = _default_configs()

    run_dim_experiment(
        gt_path = args.gt,
        configs = configs,
        top_k   = args.top_k,
        verbose = not args.quiet,
    )
