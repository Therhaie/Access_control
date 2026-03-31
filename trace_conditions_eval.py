"""
trace_conditions_eval.py
========================
TRACE evaluation across all four pipeline conditions:

  1. baseline          — no rotation, no extra dims, no filtering
  2. rotation          — chunks rotated per (triplet, doc_id) group
  3. extra_dim         — security dims appended (configurable weight / normalisation)
  4. metadata_filter   — restricted chunks blocked via Chroma where-filter

For each condition the standard four TRACe metrics are scored against the
same ground-truth answers:
  T  context_relevance
  R  answer_faithfulness
  A  context_utilization
  C  answer_completeness

Plus a cross-condition answer comparison table showing whether different
security strategies meaningfully change the answers produced.

How each condition retrieves chunks
------------------------------------
baseline      : standard retrieve() from query_pipeline.py
rotation      : loads rotation registry → applies per-group rotation to query
                → queries the rotated Chroma collection
extra_dim     : loads dim config → augments query with auth/unauth signal
                → queries the appropriate dim Chroma collection
metadata_filter: standard retrieve() with a Chroma where-filter
                { "restricted": { "$ne": True } }

All conditions use the same judge LLM (vLLM, port 8001).

Output
------
  results/trace_conditions_results.json   — full per-sample, per-condition scores
  results/trace_conditions_summary.json   — run-level means per condition

Usage
-----
python trace_conditions_eval.py --dataset test_set.json
python trace_conditions_eval.py --dataset test_set.json --conditions baseline rotation
python trace_conditions_eval.py --dataset test_set.json --dim-weight 1.0 --extra-dims 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from ingestion_pipeline import get_embedding_model
from query_pipeline import (
    BGE_QUERY_PREFIX, ORIGINAL_CHROMA,
    _messages, build_context, get_llm,
    LLM_MODEL, MAX_TOKENS, TEMPERATURE,
)
from rotation_experiment import (
    RotationRegistry, apply_rotation,
    ROTATED_CHROMA, ROTATED_COLLECTION,
    ROTATION_REGISTRY_F,
)
from dim_experiment import (
    ExtraDimConfig, augment_vector,
    DIM_CHROMA_BASE, DIM_REGISTRY_FILE,
)
from trace_eval import (
    judge_binary,
    CONTEXT_RELEVANCE_PROMPT,
    ANSWER_FAITHFULNESS_PROMPT,
    CONTEXT_UTILIZATION_PROMPT,
    ANSWER_COMPLETENESS_PROMPT,
    token_f1,
    DEFAULT_JUDGE_URL,
    DEFAULT_JUDGE_MODEL,
)

RESULTS_DIR              = Path("results")
CONDITIONS_RESULTS_FILE  = RESULTS_DIR / "trace_conditions_results.json"
CONDITIONS_SUMMARY_FILE  = RESULTS_DIR / "trace_conditions_summary.json"

ORIGINAL_COLLECTION = "langchain"
DEFAULT_TOP_K       = 20

ALL_CONDITIONS = ["baseline", "rotation", "extra_dim", "metadata_filter"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Retrieval helpers — one per condition
# ═══════════════════════════════════════════════════════════════════════════════

def _chroma_client(path: str):
    return chromadb.PersistentClient(
        path=path, settings=Settings(anonymized_telemetry=False)
    )


def _retrieve_baseline(
    query_vec: np.ndarray,
    top_k: int,
) -> list[dict]:
    """Standard retrieval from the original collection, no modifications."""
    coll = _chroma_client(ORIGINAL_CHROMA).get_collection(ORIGINAL_COLLECTION)
    res  = coll.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "content":       doc,
            "triplet_index": meta.get("triplet_index", "?"),
            "document_id":   meta.get("document_id",   "?"),
            "phrase_seq":    meta.get("phrase_seq",     "?"),
            "bge_score":     round(1.0 - float(dist), 4),
        }
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def _retrieve_rotation(
    query_vec: np.ndarray,
    registry: RotationRegistry,
    group_keys: list[str],
    top_k: int,
) -> list[dict]:
    """
    Query the rotated Chroma collection.
    Uses the mean of all group-rotated query vectors (same logic as rotation_experiment.py).
    """
    try:
        coll = _chroma_client(ROTATED_CHROMA).get_collection(ROTATED_COLLECTION)
    except Exception:
        return []   # rotated collection not yet built

    rot_vecs = np.stack([
        apply_rotation(query_vec, registry.get_or_create(*gk.split("|")))
        for gk in group_keys
    ]) if group_keys else query_vec[np.newaxis, :]
    mean_rot = rot_vecs.mean(axis=0)
    mean_rot /= np.linalg.norm(mean_rot) + 1e-10

    res = coll.query(
        query_embeddings=[mean_rot.tolist()],
        n_results=min(top_k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "content":       doc,
            "triplet_index": meta.get("triplet_index", "?"),
            "document_id":   meta.get("document_id",   "?"),
            "phrase_seq":    meta.get("phrase_seq",     "?"),
            "bge_score":     round(1.0 - float(dist), 4),
        }
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def _retrieve_extra_dim(
    query_vec: np.ndarray,
    cfg: ExtraDimConfig,
    is_authorised: bool,
    top_k: int,
) -> list[dict]:
    """Query the dim-augmented Chroma collection with the appropriate query vector."""
    try:
        client = _chroma_client(DIM_CHROMA_BASE)
        coll   = client.get_collection(f"dim_{cfg.config_id}")
    except Exception:
        return []

    aug_q = augment_vector(query_vec, cfg, is_restricted=is_authorised)

    res = coll.query(
        query_embeddings=[aug_q.tolist()],
        n_results=min(top_k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "content":       doc,
            "triplet_index": meta.get("triplet_index", "?"),
            "document_id":   meta.get("document_id",   "?"),
            "phrase_seq":    meta.get("phrase_seq",     "?"),
            "bge_score":     round(1.0 - float(dist), 4),
        }
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def _retrieve_metadata_filter(
    query_vec: np.ndarray,
    top_k: int,
    use_pre_filter: bool = True,
) -> list[dict]:
    """
    Retrieve with restricted chunks blocked.
    use_pre_filter=True  → Chroma where-filter (hard block before ANN search)
    use_pre_filter=False → retrieve then discard
    """
    coll = _chroma_client(ORIGINAL_CHROMA).get_collection(ORIGINAL_COLLECTION)

    if use_pre_filter:
        res = coll.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(top_k, coll.count()),
            where={"restricted": {"$ne": True}},
            include=["documents", "metadatas", "distances"],
        )
    else:
        res = coll.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(top_k * 3, coll.count()),
            include=["documents", "metadatas", "distances"],
        )

    chunks = [
        {
            "content":       doc,
            "triplet_index": meta.get("triplet_index", "?"),
            "document_id":   meta.get("document_id",   "?"),
            "phrase_seq":    meta.get("phrase_seq",     "?"),
            "bge_score":     round(1.0 - float(dist), 4),
            "restricted":    meta.get("restricted", False),
        }
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]

    if not use_pre_filter:
        chunks = [c for c in chunks if not c.get("restricted")]
        chunks = chunks[:top_k]

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Generate answer from retrieved chunks
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "⚠ No context retrieved."
    context  = build_context(chunks)
    messages = _messages(question, context)
    client   = get_llm()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=False,
    )
    return response.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TRACE scoring for one (question, chunks, answer) triple
# ═══════════════════════════════════════════════════════════════════════════════

def _score_trace(
    question: str,
    ground_truth: str,
    chunks: list[dict],
    answer: str,
    judge_url: str,
    judge_model: str,
) -> dict:
    context = "\n\n---\n\n".join(c.get("content", "") for c in chunks)
    return {
        "context_relevance":   judge_binary(
            CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context),
            judge_url=judge_url, model=judge_model,
        ),
        "answer_faithfulness": judge_binary(
            ANSWER_FAITHFULNESS_PROMPT.format(context=context, answer=answer),
            judge_url=judge_url, model=judge_model,
        ),
        "context_utilization": judge_binary(
            CONTEXT_UTILIZATION_PROMPT.format(
                question=question, context=context, answer=answer
            ),
            judge_url=judge_url, model=judge_model,
        ),
        "answer_completeness": judge_binary(
            ANSWER_COMPLETENESS_PROMPT.format(
                question=question, ground_truth=ground_truth, answer=answer
            ),
            judge_url=judge_url, model=judge_model,
        ),
        "answer_f1": token_f1(answer, ground_truth),
        "n_chunks":  len(chunks),
        "context":   context[:500],   # truncated for storage
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Per-sample evaluation across all conditions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConditionResult:
    condition:   str
    chunks:      list[dict]
    answer:      str
    trace:       dict
    t_retrieve_s:float
    t_generate_s:float


def evaluate_sample_all_conditions(
    sample: dict,
    conditions: list[str],
    embedder,
    registry: Optional[RotationRegistry],
    dim_cfg: Optional[ExtraDimConfig],
    gt_stable_chunks: list[dict],
    top_k: int,
    judge_url: str,
    judge_model: str,
    verbose: bool,
) -> dict:
    question      = sample["question"]
    ground_truth  = sample["ground_truth"]
    triplet_index = str(sample.get("triplet_index", sample.get("id", "?")))

    # Identify rotation groups for this query
    group_keys = list({
        f"{c['triplet_index']}|{c['document_id']}"
        for c in gt_stable_chunks
    })

    raw_query_vec = np.array(
        embedder.embed_query(BGE_QUERY_PREFIX + question), dtype=np.float32
    )

    condition_results: dict[str, dict] = {}

    for condition in conditions:
        if verbose:
            print(f"    → {condition} …", end=" ", flush=True)

        t0 = time.time()

        if condition == "baseline":
            chunks = _retrieve_baseline(raw_query_vec, top_k)

        elif condition == "rotation":
            if registry is None:
                print("(registry not loaded, skipping)")
                continue
            chunks = _retrieve_rotation(raw_query_vec, registry, group_keys, top_k)

        elif condition == "extra_dim":
            if dim_cfg is None:
                print("(dim config not loaded, skipping)")
                continue
            # Authorised query — sees restricted chunks
            chunks = _retrieve_extra_dim(raw_query_vec, dim_cfg, is_authorised=True, top_k=top_k)

        elif condition == "metadata_filter":
            chunks = _retrieve_metadata_filter(raw_query_vec, top_k, use_pre_filter=True)

        else:
            continue

        t_retrieve = time.time() - t0

        t1     = time.time()
        answer = _generate_answer(question, chunks)
        t_gen  = time.time() - t1

        trace  = _score_trace(question, ground_truth, chunks, answer, judge_url, judge_model)
        trace_mean = sum(
            v for v in [
                trace["context_relevance"], trace["answer_faithfulness"],
                trace["context_utilization"], trace["answer_completeness"],
            ] if v is not None
        ) / max(1, sum(
            1 for v in [
                trace["context_relevance"], trace["answer_faithfulness"],
                trace["context_utilization"], trace["answer_completeness"],
            ] if v is not None
        ))

        condition_results[condition] = {
            "answer":        answer,
            "trace":         trace,
            "trace_mean":    round(trace_mean, 4),
            "t_retrieve_s":  round(t_retrieve, 3),
            "t_generate_s":  round(t_gen, 3),
            "n_chunks":      len(chunks),
        }
        if verbose:
            print(f"TRACe={trace_mean:.3f}  t_ret={t_retrieve:.2f}s")

    return {
        "question":      question,
        "ground_truth":  ground_truth,
        "triplet_index": triplet_index,
        "conditions":    condition_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate_conditions(
    all_sample_results: list[dict],
    conditions: list[str],
) -> dict:
    summary: dict[str, dict] = {}
    for cond in conditions:
        cr_vals, af_vals, cu_vals, ac_vals, f1_vals, tm_vals = [], [], [], [], [], []
        t_ret_vals = []
        for s in all_sample_results:
            cr = s["conditions"].get(cond)
            if cr is None:
                continue
            t = cr["trace"]
            if t["context_relevance"]   is not None: cr_vals.append(t["context_relevance"])
            if t["answer_faithfulness"] is not None: af_vals.append(t["answer_faithfulness"])
            if t["context_utilization"] is not None: cu_vals.append(t["context_utilization"])
            if t["answer_completeness"] is not None: ac_vals.append(t["answer_completeness"])
            f1_vals.append(t["answer_f1"])
            tm_vals.append(cr["trace_mean"])
            t_ret_vals.append(cr["t_retrieve_s"])

        def _m(lst): return round(sum(lst)/len(lst), 4) if lst else None

        summary[cond] = {
            "context_relevance_mean":   _m(cr_vals),
            "answer_faithfulness_mean": _m(af_vals),
            "context_utilization_mean": _m(cu_vals),
            "answer_completeness_mean": _m(ac_vals),
            "trace_mean":               _m(tm_vals),
            "answer_f1_mean":           _m(f1_vals),
            "t_retrieve_mean":          _m(t_ret_vals),
            "n_samples":                len(f1_vals),
        }
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Main runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_conditions_eval(
    dataset_path: str,
    conditions: list[str]     = None,
    top_k: int                = DEFAULT_TOP_K,
    judge_url: str            = DEFAULT_JUDGE_URL,
    judge_model: str          = DEFAULT_JUDGE_MODEL,
    dim_weight: float         = 1.0,
    extra_dims: int           = 4,
    normalize_before: bool    = False,
    normalize_after: bool     = False,
    limit: int | None         = None,
    gt_path: str              = str(RESULTS_DIR / "ground_truth_retrievals.json"),
    verbose: bool             = True,
) -> dict:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conditions = conditions or ALL_CONDITIONS

    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(dataset_path, encoding="utf-8") as fh:
        dataset: list[dict] = json.load(fh)
    if limit:
        dataset = dataset[:limit]

    # ── Load ground-truth retrievals for group-key resolution ─────────────────
    gt_lookup: dict[str, list[dict]] = {}
    if Path(gt_path).exists():
        with open(gt_path, encoding="utf-8") as fh:
            for rec in json.load(fh):
                gt_lookup[rec["triplet_index"]] = rec.get("stable_chunks", [])

    # ── Load rotation registry ────────────────────────────────────────────────
    registry: Optional[RotationRegistry] = None
    if "rotation" in conditions:
        embedder_probe = get_embedding_model()
        dim            = len(embedder_probe.embed_query("probe"))
        if ROTATION_REGISTRY_F.exists():
            with open(ROTATION_REGISTRY_F, encoding="utf-8") as fh:
                registry = RotationRegistry.from_serialisable(json.load(fh), dim=dim)
            print(f"  Rotation registry loaded: {len(registry.all_keys())} groups")
        else:
            print("  ⚠  Rotation registry not found — rotation condition will be skipped.")

    # ── Build dim config ──────────────────────────────────────────────────────
    dim_cfg: Optional[ExtraDimConfig] = None
    if "extra_dim" in conditions:
        dim_cfg = ExtraDimConfig(
            extra_dims=extra_dims,
            weight=dim_weight,
            normalize_before=normalize_before,
            normalize_after=normalize_after,
        )
        print(f"  Dim config: {dim_cfg.config_id}")

    embedder = get_embedding_model()

    print(f"\n{'═'*62}")
    print(f"  TRACE Conditions Evaluation")
    print(f"  Dataset: {dataset_path}  ({len(dataset)} samples)")
    print(f"  Conditions: {conditions}")
    print(f"{'═'*62}\n")

    all_sample_results: list[dict] = []

    for i, sample in enumerate(dataset, 1):
        tid = str(sample.get("triplet_index", sample.get("id", str(i))))
        gt_chunks = gt_lookup.get(tid, [])
        print(f"[{i}/{len(dataset)}] triplet={tid}  Q: {sample['question'][:60]}…")

        result = evaluate_sample_all_conditions(
            sample           = sample,
            conditions       = conditions,
            embedder         = embedder,
            registry         = registry,
            dim_cfg          = dim_cfg,
            gt_stable_chunks = gt_chunks,
            top_k            = top_k,
            judge_url        = judge_url,
            judge_model      = judge_model,
            verbose          = verbose,
        )
        all_sample_results.append(result)

    summary = _aggregate_conditions(all_sample_results, conditions)

    # ── Persist ───────────────────────────────────────────────────────────────
    with open(CONDITIONS_RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_sample_results, fh, indent=2, ensure_ascii=False)

    full_output = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset":       dataset_path,
        "conditions":    conditions,
        "n_samples":     len(dataset),
        "summary":       summary,
    }
    with open(CONDITIONS_SUMMARY_FILE, "w", encoding="utf-8") as fh:
        json.dump(full_output, fh, indent=2, ensure_ascii=False)

    # ── Print summary table ───────────────────────────────────────────────────
    bar = "═" * 72
    print(f"\n{bar}")
    print(f"  {'Condition':<18} {'CR':>6} {'AF':>6} {'CU':>6} {'AC':>6} "
          f"{'TRACe':>7} {'F1':>6} {'t_ret':>7}")
    print(f"  {'─'*68}")
    for cond, s in summary.items():
        def _fmt(v): return f"{v:.3f}" if v is not None else "  N/A"
        print(f"  {cond:<18} "
              f"{_fmt(s['context_relevance_mean']):>6} "
              f"{_fmt(s['answer_faithfulness_mean']):>6} "
              f"{_fmt(s['context_utilization_mean']):>6} "
              f"{_fmt(s['answer_completeness_mean']):>6} "
              f"{_fmt(s['trace_mean']):>7} "
              f"{_fmt(s['answer_f1_mean']):>6} "
              f"{_fmt(s['t_retrieve_mean']):>7}s")
    print(f"{bar}\n")
    print(f"  Results → {CONDITIONS_RESULTS_FILE}")
    print(f"  Summary → {CONDITIONS_SUMMARY_FILE}\n")

    return full_output


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TRACE evaluation across all security conditions"
    )
    parser.add_argument("--dataset",          "-d", required=True)
    parser.add_argument("--conditions",       "-c", nargs="+",
                        choices=ALL_CONDITIONS, default=None,
                        help="Which conditions to evaluate (default: all)")
    parser.add_argument("--top-k",            type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--judge-url",        default=DEFAULT_JUDGE_URL)
    parser.add_argument("--judge-model",      default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--dim-weight",       type=float, default=1.0)
    parser.add_argument("--extra-dims",       type=int,   default=4)
    parser.add_argument("--normalize-before", action="store_true")
    parser.add_argument("--normalize-after",  action="store_true")
    parser.add_argument("--limit",            type=int,   default=None)
    parser.add_argument("--gt",               default=str(RESULTS_DIR / "ground_truth_retrievals.json"))
    parser.add_argument("--quiet", "-q",      action="store_true")
    args = parser.parse_args()

    run_conditions_eval(
        dataset_path     = args.dataset,
        conditions       = args.conditions,
        top_k            = args.top_k,
        judge_url        = args.judge_url,
        judge_model      = args.judge_model,
        dim_weight       = args.dim_weight,
        extra_dims       = args.extra_dims,
        normalize_before = args.normalize_before,
        normalize_after  = args.normalize_after,
        limit            = args.limit,
        gt_path          = args.gt,
        verbose          = not args.quiet,
    )
