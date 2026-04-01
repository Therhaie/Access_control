"""
ground_truth_collector.py
=========================
Establishes a "ground-truth retrieval set" for each query in the test dataset
by running each question N times against ChromaDB and keeping only the chunks
that appear consistently across all runs (or above a configurable threshold).

Why multiple runs?
------------------
ChromaDB's HNSW index is approximate — consecutive identical queries can
occasionally return slightly different neighbour sets.  Running each query
N times and taking the intersection (or majority-vote set) gives a stable,
reproducible ground-truth that the rotation experiment can rely on.

Output file  (JSON)
-------------------
results/ground_truth_retrievals.json
[
  {
    "query_id":      "triplet_42",
    "question":      "What is …?",
    "triplet_index": "42",
    "runs":          5,
    "threshold":     1.0,          # fraction of runs a chunk must appear in
    "stable_chunks": [
      {
        "content":       "…",
        "triplet_index": "42",
        "document_id":   "7",
        "phrase_seq":    "b",
        "bge_score":     0.8821,   # mean cosine similarity across runs
        "bge_score_std": 0.0003,   # std — low = very stable
        "seen_in_runs":  5         # out of N runs
      },
      …
    ]
  },
  …
]

Usage
-----
python ground_truth_collector.py --dataset example_dataset.json
python ground_truth_collector.py --dataset example_dataset.json --runs 10 --threshold 0.8
python ground_truth_collector.py --dataset example_dataset.json --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from query_pipeline import retrieve          # returns (candidates, meta)

RESULTS_DIR   = Path("results")
OUTPUT_FILE   = RESULTS_DIR / "ground_truth_retrievals.json"

DEFAULT_RUNS      = 5
DEFAULT_THRESHOLD = 1.0    # chunk must appear in ALL runs to be "stable"
DEFAULT_TOP_K     = 20     # how many chunks to retrieve per run


# ═══════════════════════════════════════════════════════════════════════════════
# Core logic
# ═══════════════════════════════════════════════════════════════════════════════

def _chunk_key(chunk: dict) -> str:
    """
    Unique identifier for a chunk.
    We use (triplet_index, document_id, phrase_seq) — this maps exactly to
    the metadata stored during ingestion.
    """
    return f"{chunk.get('source','?')}|{chunk.get('page','?')}|{chunk.get('phrase_seq','?')}"
    # return f"{chunk.get('triplet_index','?')}|{chunk.get('document_id','?')}|{chunk.get('phrase_seq','?')}"
    # return f"{chunk.get('source','?')}|{chunk.get('page','?')}|{chunk.get('phrase_seq','?')}"
    # can also return the 'bge_score' 

def collect_stable_chunks(
    question: str,
    n_runs: int       = DEFAULT_RUNS,
    threshold: float  = DEFAULT_THRESHOLD,
    top_k: int        = DEFAULT_TOP_K,
    verbose: bool     = True,
) -> list[dict]:
    """
    Run `question` against ChromaDB `n_runs` times.
    Return only chunks whose appearance rate >= threshold.

    Each returned chunk carries:
        content, triplet_index, document_id, phrase_seq,
        bge_score (mean), bge_score_std, seen_in_runs
    """
    # key → list of bge_scores across runs in which this chunk appeared
    seen: dict[str, list[float]] = defaultdict(list)
    # key → chunk payload (we only need to store it once)
    payloads: dict[str, dict] = {}

    for run_idx in range(n_runs):
        candidates, _ = retrieve(question, top_k_retrieve=top_k)
        for c in candidates:
            key = _chunk_key(c)
            seen[key].append(c.get("bge_score", 0.0))
            if key not in payloads:
                payloads[key] = {
                    "content":       c.get("content", ""),
                    "triplet_index": str(c.get("triplet_index", c.get("source", "?"))),
                    "document_id":   str(c.get("document_id",   "?")),
                    "phrase_seq":    str(c.get("phrase_seq",     "?")),
                }
                # to modify accordingly to _chunk_key changes done
        if verbose:
            print(f"      run {run_idx+1}/{n_runs} — {len(candidates)} candidates")

    # Filter by threshold
    stable = []
    for key, scores in seen.items():
        appearance_rate = len(scores) / n_runs
        if appearance_rate >= threshold:
            import statistics
            chunk = dict(payloads[key])
            chunk["bge_score"]     = round(sum(scores) / len(scores), 6)
            chunk["bge_score_std"] = round(
                statistics.stdev(scores) if len(scores) > 1 else 0.0, 6
            )
            chunk["seen_in_runs"]  = len(scores)
            stable.append(chunk)

    # Sort by mean bge_score descending
    stable.sort(key=lambda x: x["bge_score"], reverse=True)
    return stable


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_collection(
    dataset_path: str,
    n_runs: int      = DEFAULT_RUNS,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int       = DEFAULT_TOP_K,
    limit: int | None = None,
    verbose: bool    = True,
) -> list[dict]:
    """
    Iterate over each entry in the dataset JSON and collect stable chunks.

    Returns the full list of ground-truth records (also saved to disk).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, encoding="utf-8") as fh:
        dataset: list[dict] = json.load(fh)
        # dataset_like : list[list[dict]] =json.load(fh)

    if limit:
        dataset = dataset[:limit]

    print(f"\n{'═'*62}")
    print(f"  Ground-Truth Retrieval Collector")
    print(f"  Dataset : {dataset_path}  ({len(dataset)} entries)")
    print(f"  Runs/query: {n_runs}   Threshold: {threshold}   Top-K: {top_k}")
    print(f"{'═'*62}\n")

    records: list[dict] = []

    # Version without the for loop to iterate on the sentences of the dataset
    for i, entry in enumerate(dataset, 1):
        # for question in entry["sentences"]:
        question      = entry["question"]
        triplet_index = str(entry.get("id_triplets", entry.get("id", i)))
        query_id      = f"triplet_{triplet_index}"

        print(f"[{i}/{len(dataset)}] {query_id}  Q: {question[:70]}…")
        t0 = time.time()

        stable = collect_stable_chunks(
            question  = question,
            n_runs    = n_runs,
            threshold = threshold,
            top_k     = top_k,
            verbose   = verbose,
        )

        record = {
            "query_id":      query_id,
            "question":      question,
            "triplet_index": triplet_index,
            "ground_truth":  entry.get("response", ""),
            "runs":          n_runs,
            "threshold":     threshold,
            "stable_chunks": stable,
            "elapsed_s":     round(time.time() - t0, 2),
        }
        records.append(record)

    print(f"  → {len(stable)} stable chunks  "
            f"(from {top_k} retrieved × {n_runs} runs)  "
            f"[{record['elapsed_s']}s]\n")
    
    # for i, entry in enumerate(dataset, 1):
    #     question      = entry["question"]
    #     triplet_index = str(entry.get("triplet_index", entry.get("id", i)))
    #     query_id      = f"triplet_{triplet_index}"

    #     print(f"[{i}/{len(dataset)}] {query_id}  Q: {question[:70]}…")
    #     t0 = time.time()

    #     stable = collect_stable_chunks(
    #         question  = question,
    #         n_runs    = n_runs,
    #         threshold = threshold,
    #         top_k     = top_k,
    #         verbose   = verbose,
    #     )

    #     record = {
    #         "query_id":      query_id,
    #         "question":      question,
    #         "triplet_index": triplet_index,
    #         "ground_truth":  entry.get("ground_truth", ""),
    #         "runs":          n_runs,
    #         "threshold":     threshold,
    #         "stable_chunks": stable,
    #         "elapsed_s":     round(time.time() - t0, 2),
    #     }
    #     records.append(record)

    #     print(f"  → {len(stable)} stable chunks  "
    #           f"(from {top_k} retrieved × {n_runs} runs)  "
    #           f"[{record['elapsed_s']}s]\n")

    # ── Persist ───────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    print(f"\n✅  Saved to {OUTPUT_FILE}")
    total_chunks = sum(len(r["stable_chunks"]) for r in records)
    print(f"   {len(records)} queries  |  {total_chunks} total stable chunks\n")

    return records


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect stable ground-truth retrieval sets for each query"
    )
    parser.add_argument("--dataset",   "-d", required=True,
                        help="Path to test dataset JSON")
    parser.add_argument("--runs",      "-r", type=int,   default=DEFAULT_RUNS,
                        help=f"Number of retrieval runs per query (default {DEFAULT_RUNS})")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Min appearance fraction to keep a chunk (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--top-k",     "-k", type=int,   default=DEFAULT_TOP_K,
                        help=f"Top-K chunks retrieved per run (default {DEFAULT_TOP_K})")
    parser.add_argument("--limit",     "-n", type=int,   default=None,
                        help="Only process first N dataset entries")
    parser.add_argument("--quiet",     "-q", action="store_true")
    args = parser.parse_args()

    run_collection(
        dataset_path = args.dataset,
        n_runs       = args.runs,
        threshold    = args.threshold,
        top_k        = args.top_k,
        limit        = args.limit,
        verbose      = not args.quiet,
    )
