"""
latency_benchmark.py
====================
Retrieval latency benchmark across four access-control strategies:

  1. baseline          — plain ANN search, no filtering
  2. rotation          — query rotated per group → queried against rotated collection
  3. extra_dim         — query augmented → queried against dim-augmented collection
  4. metadata_prefilter — Chroma where-filter (hard pre-retrieval block)
  5. metadata_postfilter— retrieve then discard restricted chunks in Python

Methodology
-----------
Uses the same "most-used chunks" approach as the rotation and dimension
experiments:
  1. Load ground_truth_retrievals.json (stable chunk sets per query).
  2. Mark those stable chunks as "restricted" by upserting a metadata tag
     { "restricted": True } back into the ORIGINAL Chroma collection.
  3. For each query, run each strategy N_REPEATS times and record wall-clock
     timings at three granularities:
       - t_embed_s    : time to embed the query vector
       - t_retrieve_s : time for the ANN / filter call (excluding embedding)
       - t_total_s    : end-to-end (embed + retrieve)
  4. Report mean ± std per strategy and produce plots.

The "mark restricted" step is idempotent — running the benchmark twice does
not double-tag chunks.

Output
------
  results/latency_benchmark.json   — full per-query, per-strategy timings
  results/latency_summary.json     — mean ± std per strategy

Usage
-----
python latency_benchmark.py
python latency_benchmark.py --gt results/ground_truth_retrievals.json --repeats 10
python latency_benchmark.py --strategies baseline rotation metadata_prefilter
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from config import COLLECTION
from ingestion_pipeline import get_embedding_model
from query_pipeline import BGE_QUERY_PREFIX
from rotation_experiment import (
    RotationRegistry, apply_rotation,
    ROTATED_CHROMA, ROTATED_COLLECTION,
    ROTATION_REGISTRY_F,
)
from dim_experiment import (
    ExtraDimConfig, augment_vector,
    DIM_CHROMA_BASE, DIM_REGISTRY_FILE,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR          = Path("results")
GT_FILE              = RESULTS_DIR / "ground_truth_retrievals.json"
LATENCY_RESULTS_FILE = RESULTS_DIR / "latency_benchmark.json"
LATENCY_SUMMARY_FILE = RESULTS_DIR / "latency_summary.json"

ORIGINAL_CHROMA      = os.path.join(os.getcwd(), "./chroma_db")
ORIGINAL_COLLECTION  = COLLECTION

DEFAULT_REPEATS      = 7
DEFAULT_TOP_K        = 20
DEFAULT_EXTRA_DIMS   = 4
DEFAULT_DIM_WEIGHT   = 1.0

ALL_STRATEGIES = [
    "baseline",
    "rotation",
    "extra_dim",
    "metadata_prefilter",
    "metadata_postfilter",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ChromaDB helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _client(path: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=path, settings=Settings(anonymized_telemetry=False)
    )


def _orig_coll():
    return _client(ORIGINAL_CHROMA).get_collection(ORIGINAL_COLLECTION)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Mark stable chunks as restricted in the original collection
# ═══════════════════════════════════════════════════════════════════════════════

def mark_restricted_chunks(gt_records: list[dict], verbose: bool = True) -> int:
    """
    Upsert { "restricted": True } metadata onto all stable chunks from the
    ground-truth file.  This makes them visible to metadata_prefilter and
    metadata_postfilter strategies.

    Returns the number of chunks tagged.
    """
    coll    = _orig_coll()
    tagged  = 0
    skipped = 0

    for record in gt_records:
        for chunk in record.get("stable_chunks", []):
            tid  = chunk["triplet_index"]
            did  = chunk["document_id"]
            pseq = chunk["phrase_seq"]

            # Look up the existing document by metadata
            try:
                existing = coll.get(
                    where={"$and": [
                        {"triplet_index": {"$eq": tid}},
                        {"document_id":   {"$eq": did}},
                        {"phrase_seq":    {"$eq": pseq}},
                    ]},
                    include=["embeddings", "documents", "metadatas"],
                )
            except Exception as e:
                skipped += 1
                continue

            if not existing["ids"]:
                skipped += 1
                continue

            doc_id   = existing["ids"][0]
            old_meta = existing["metadatas"][0]

            # Skip if already tagged
            if old_meta.get("restricted") is True:
                continue

            new_meta = {**old_meta, "restricted": True}
            coll.update(
                ids       = [doc_id],
                metadatas = [new_meta],
            )
            tagged += 1

    if verbose:
        print(f"  Tagged {tagged} chunks as restricted  ({skipped} not found / already tagged)")
    return tagged


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Per-strategy retrieval functions  (timing is measured by the caller)
# ═══════════════════════════════════════════════════════════════════════════════

def _do_baseline(query_vec: np.ndarray, top_k: int) -> list[dict]:
    coll = _orig_coll()
    res  = coll.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )
    return res["metadatas"][0]


def _do_rotation(
    query_vec: np.ndarray,
    group_keys: list[str],
    registry: RotationRegistry,
    top_k: int,
) -> list[dict]:
    try:
        coll = _client(ROTATED_CHROMA).get_collection(ROTATED_COLLECTION)
    except Exception:
        return []

    if group_keys:
        rot_vecs = np.stack([
            apply_rotation(query_vec, registry.get_or_create(*gk.split("|")))
            for gk in group_keys
        ])
        mean_rot = rot_vecs.mean(axis=0)
        mean_rot /= np.linalg.norm(mean_rot) + 1e-10
    else:
        mean_rot = query_vec

    res = coll.query(
        query_embeddings=[mean_rot.tolist()],
        n_results=min(top_k, max(1, coll.count())),
        include=["metadatas"],
    )
    return res["metadatas"][0]


def _do_extra_dim(
    query_vec: np.ndarray,
    cfg: ExtraDimConfig,
    is_authorised: bool,
    top_k: int,
) -> list[dict]:
    try:
        coll = _client(DIM_CHROMA_BASE).get_collection(f"dim_{cfg.config_id}")
    except Exception:
        return []

    aug_q = augment_vector(query_vec, cfg, is_restricted=is_authorised)
    res   = coll.query(
        query_embeddings=[aug_q.tolist()],
        n_results=min(top_k, max(1, coll.count())),
        include=["metadatas"],
    )
    return res["metadatas"][0]


def _do_metadata_prefilter(query_vec: np.ndarray, top_k: int) -> list[dict]:
    """Hard block via Chroma where-filter — restricted chunks never enter ANN."""
    coll = _orig_coll()
    res  = coll.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k, coll.count()),
        where={"restricted": {"$ne": True}},
        include=["metadatas"],
    )
    return res["metadatas"][0]


def _do_metadata_postfilter(query_vec: np.ndarray, top_k: int) -> list[dict]:
    """Retrieve over-fetch then discard restricted chunks in Python."""
    coll = _orig_coll()
    res  = coll.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k * 4, coll.count()),
        include=["metadatas"],
    )
    filtered = [m for m in res["metadatas"][0] if not m.get("restricted")]
    return filtered[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Timing data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyTiming:
    strategy:    str
    t_embed_s:   float
    t_retrieve_s:float
    t_total_s:   float
    n_returned:  int


@dataclass
class QueryBenchmarkResult:
    query_id:    str
    question:    str
    n_repeats:   int
    # strategy → list of StrategyTiming (one per repeat)
    timings:     dict[str, list[dict]] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Per-query benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_query(
    record: dict,
    strategies: list[str],
    embedder,
    registry: Optional[RotationRegistry],
    dim_cfg: Optional[ExtraDimConfig],
    n_repeats: int,
    top_k: int,
    verbose: bool,
) -> QueryBenchmarkResult:

    question  = record["question"]
    query_id  = record["query_id"]
    gt_chunks = record.get("stable_chunks", [])
    group_keys = list({
        f"{c['triplet_index']}|{c['document_id']}"
        for c in gt_chunks
    })

    result = QueryBenchmarkResult(
        query_id  = query_id,
        question  = question,
        n_repeats = n_repeats,
    )

    for strategy in strategies:
        timings_for_strategy: list[dict] = []

        for rep in range(n_repeats):

            # ── Time embedding ────────────────────────────────────────────────
            t_emb_start = time.perf_counter()
            raw_q_vec   = np.array(
                embedder.embed_query(BGE_QUERY_PREFIX + question), dtype=np.float32
            )
            t_embed = time.perf_counter() - t_emb_start

            # ── Time retrieval ────────────────────────────────────────────────
            t_ret_start = time.perf_counter()

            if strategy == "baseline":
                metas = _do_baseline(raw_q_vec, top_k)

            elif strategy == "rotation":
                if registry is None:
                    break
                metas = _do_rotation(raw_q_vec, group_keys, registry, top_k)

            elif strategy == "extra_dim":
                if dim_cfg is None:
                    break
                metas = _do_extra_dim(raw_q_vec, dim_cfg, is_authorised=True, top_k=top_k)

            elif strategy == "metadata_prefilter":
                metas = _do_metadata_prefilter(raw_q_vec, top_k)

            elif strategy == "metadata_postfilter":
                metas = _do_metadata_postfilter(raw_q_vec, top_k)

            else:
                break

            t_retrieve = time.perf_counter() - t_ret_start
            t_total    = t_embed + t_retrieve

            timings_for_strategy.append({
                "rep":         rep,
                "t_embed_s":   round(t_embed,    5),
                "t_retrieve_s":round(t_retrieve,  5),
                "t_total_s":   round(t_total,     5),
                "n_returned":  len(metas),
            })

        result.timings[strategy] = timings_for_strategy

    if verbose:
        for strategy, tlist in result.timings.items():
            if not tlist:
                continue
            mean_ret = statistics.mean(t["t_retrieve_s"] for t in tlist)
            std_ret  = statistics.stdev(t["t_retrieve_s"] for t in tlist) if len(tlist) > 1 else 0.0
            print(f"    {strategy:<22} "
                  f"ret={mean_ret*1000:.2f}ms ± {std_ret*1000:.2f}ms")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════

def _summarise(all_results: list[QueryBenchmarkResult], strategies: list[str]) -> dict:
    summary: dict[str, dict] = {}
    for strategy in strategies:
        all_ret, all_emb, all_tot = [], [], []
        for qr in all_results:
            for t in qr.timings.get(strategy, []):
                all_ret.append(t["t_retrieve_s"])
                all_emb.append(t["t_embed_s"])
                all_tot.append(t["t_total_s"])

        def _stats(lst):
            if not lst:
                return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
            return {
                "mean": round(statistics.mean(lst),   6),
                "std":  round(statistics.stdev(lst) if len(lst) > 1 else 0.0, 6),
                "min":  round(min(lst),               6),
                "max":  round(max(lst),               6),
                "n":    len(lst),
            }

        summary[strategy] = {
            "t_retrieve_s": _stats(all_ret),
            "t_embed_s":    _stats(all_emb),
            "t_total_s":    _stats(all_tot),
        }
    return summary


def _print_summary_table(summary: dict) -> None:
    bar = "═" * 72
    print(f"\n{bar}")
    print(f"  {'Strategy':<24} {'Embed ms':>9} {'Retrieve ms':>12} {'Total ms':>10}")
    print(f"  {'─'*66}")
    for strategy, s in summary.items():
        def _f(d, key):
            v = d.get(key, {})
            m = v.get("mean")
            sd= v.get("std")
            if m is None:
                return "    N/A"
            return f"{m*1000:>6.2f}±{sd*1000:.2f}"
        print(f"  {strategy:<24} "
              f"{_f(s, 't_embed_s'):>9}  "
              f"{_f(s, 't_retrieve_s'):>12}  "
              f"{_f(s, 't_total_s'):>10}")
    print(f"{bar}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Main runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    gt_path: str         = str(GT_FILE),
    strategies: list[str]= None,
    n_repeats: int       = DEFAULT_REPEATS,
    top_k: int           = DEFAULT_TOP_K,
    extra_dims: int      = DEFAULT_EXTRA_DIMS,
    dim_weight: float    = DEFAULT_DIM_WEIGHT,
    normalize_before: bool = False,
    normalize_after: bool  = False,
    verbose: bool        = True,
) -> dict:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    strategies = strategies or ALL_STRATEGIES

    with open(gt_path, encoding="utf-8") as fh:
        gt_records: list[dict] = json.load(fh)

    print(f"\n{'═'*62}")
    print(f"  Retrieval Latency Benchmark")
    print(f"  Ground truth: {gt_path}  ({len(gt_records)} queries)")
    print(f"  Strategies: {strategies}")
    print(f"  Repeats/query: {n_repeats}  Top-K: {top_k}")
    print(f"{'═'*62}\n")

    # ── Mark restricted chunks ─────────────────────────────────────────────────
    print("Marking stable chunks as restricted in original collection…")
    mark_restricted_chunks(gt_records, verbose=verbose)

    # ── Load rotation registry ────────────────────────────────────────────────
    registry: Optional[RotationRegistry] = None
    if "rotation" in strategies:
        embedder_probe = get_embedding_model()
        dim            = len(embedder_probe.embed_query("probe"))
        if ROTATION_REGISTRY_F.exists():
            with open(ROTATION_REGISTRY_F, encoding="utf-8") as fh:
                registry = RotationRegistry.from_serialisable(json.load(fh), dim=dim)
            print(f"  Rotation registry: {len(registry.all_keys())} groups loaded")
        else:
            print("  ⚠  Rotation registry not found; 'rotation' strategy will produce empty results")

    # ── Build dim config ──────────────────────────────────────────────────────
    dim_cfg: Optional[ExtraDimConfig] = None
    if "extra_dim" in strategies:
        dim_cfg = ExtraDimConfig(
            extra_dims=extra_dims,
            weight=dim_weight,
            normalize_before=normalize_before,
            normalize_after=normalize_after,
        )
        print(f"  Dim config: {dim_cfg.config_id}")

    embedder = get_embedding_model()

    # ── Run per-query benchmarks ──────────────────────────────────────────────
    all_results: list[QueryBenchmarkResult] = []

    for i, record in enumerate(gt_records, 1):
        if not record.get("stable_chunks"):
            continue
        print(f"\n[{i}/{len(gt_records)}] {record['query_id']}")
        qr = benchmark_query(
            record     = record,
            strategies = strategies,
            embedder   = embedder,
            registry   = registry,
            dim_cfg    = dim_cfg,
            n_repeats  = n_repeats,
            top_k      = top_k,
            verbose    = verbose,
        )
        all_results.append(qr)

    summary = _summarise(all_results, strategies)
    _print_summary_table(summary)

    # ── Persist ───────────────────────────────────────────────────────────────
    with open(LATENCY_RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in all_results], fh, indent=2, ensure_ascii=False)

    full_summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "gt_path":       gt_path,
        "strategies":    strategies,
        "n_queries":     len(all_results),
        "n_repeats":     n_repeats,
        "top_k":         top_k,
        "summary":       summary,
    }
    with open(LATENCY_SUMMARY_FILE, "w", encoding="utf-8") as fh:
        json.dump(full_summary, fh, indent=2, ensure_ascii=False)

    print(f"  Results → {LATENCY_RESULTS_FILE}")
    print(f"  Summary → {LATENCY_SUMMARY_FILE}\n")
    return full_summary


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval latency benchmark")
    parser.add_argument("--gt",               default=str(GT_FILE))
    parser.add_argument("--strategies",       nargs="+", choices=ALL_STRATEGIES, default=None)
    parser.add_argument("--repeats",          type=int,   default=DEFAULT_REPEATS)
    parser.add_argument("--top-k",            type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--extra-dims",       type=int,   default=DEFAULT_EXTRA_DIMS)
    parser.add_argument("--dim-weight",       type=float, default=DEFAULT_DIM_WEIGHT)
    parser.add_argument("--normalize-before", action="store_true")
    parser.add_argument("--normalize-after",  action="store_true")
    parser.add_argument("--quiet", "-q",      action="store_true")
    args = parser.parse_args()

    run_benchmark(
        gt_path          = args.gt,
        strategies       = args.strategies,
        n_repeats        = args.repeats,
        top_k            = args.top_k,
        extra_dims       = args.extra_dims,
        dim_weight       = args.dim_weight,
        normalize_before = args.normalize_before,
        normalize_after  = args.normalize_after,
        verbose          = not args.quiet,
    )
